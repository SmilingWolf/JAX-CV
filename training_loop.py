import argparse
import json
from datetime import datetime
from typing import Any, Callable, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
import wandb
from clu import metrics
from flax import jax_utils
from flax.training import orbax_utils, train_state
from tqdm import tqdm

import Models
from Generators.WDTaggerGen import DataGenerator
from Metrics.ConfusionMatrix import f1score, mcc


@flax.struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Metric
    f1score: metrics.Metric
    mcc: metrics.Metric


class TrainState(train_state.TrainState):
    metrics: Metrics
    constants: Any


def create_optimizer_tx(
    module,
    params,
    learning_rate: Union[float, Callable],
    optimizer_eps: float,
    grad_clip: float,
    weight_decay: float,
    freeze_model_body: bool,
):
    def should_freeze(path, _):
        return "trainable" if "head" in path else "frozen"

    wd_mask = jax.tree_util.tree_map_with_path(module.should_decay, params)
    tx = optax.lamb(
        learning_rate,
        weight_decay=weight_decay,
        eps=optimizer_eps,
        mask=wd_mask,
    )
    tx = optax.chain(optax.clip_by_global_norm(grad_clip), tx)

    if freeze_model_body:
        partition_optimizers = {"trainable": tx, "frozen": optax.set_to_zero()}
        param_partitions = flax.traverse_util.path_aware_map(should_freeze, params)
        tx = optax.multi_transform(partition_optimizers, param_partitions)
    return tx


def create_train_state(
    module,
    params_key,
    target_size: int,
    num_classes: int,
    learning_rate: Union[float, Callable],
    optimizer_eps: float,
    grad_clip: float,
    weight_decay: float,
    freeze_model_body: bool = False,
):
    """Creates an initial 'TrainState'."""
    # initialize parameters by passing a template image
    variables = module.init(
        params_key,
        jnp.ones([1, target_size, target_size, 3]),
        train=False,
    )
    params = variables["params"]
    del variables["params"]
    constants = variables

    loss = metrics.Average.from_output("loss")
    f1score_metric = f1score(
        threshold=0.4,
        averaging="macro",
        num_classes=num_classes,
        from_logits=True,
    )
    mcc_metric = mcc(
        threshold=0.4,
        averaging="macro",
        num_classes=num_classes,
        from_logits=True,
    )
    collection = Metrics.create(loss=loss, f1score=f1score_metric, mcc=mcc_metric)

    tx = create_optimizer_tx(
        module,
        params,
        learning_rate,
        optimizer_eps,
        grad_clip,
        weight_decay,
        freeze_model_body,
    )

    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=collection.empty(),
        constants=constants,
    )


def train_step(state, batch, weights, dropout_key):
    """Train for a single step."""
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params, weights, **kwargs):
        logits = state.apply_fn(
            {"params": params, **kwargs},
            batch["images"],
            train=True,
            rngs={"dropout": dropout_train_key},
        )
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch["labels"])
        loss = loss * weights
        loss = loss.sum() / batch["labels"].shape[0]
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, weights, **state.constants)
    grads = jax.lax.pmean(grads, axis_name="batch")
    state = state.apply_gradients(grads=grads)

    metric_updates = state.metrics.gather_from_model_output(
        logits=logits,
        labels=batch["labels"],
        loss=loss,
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


def eval_step(*, state, batch):
    logits = state.apply_fn(
        {"params": state.params, **state.constants},
        batch["images"],
        train=False,
    )

    loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch["labels"])
    loss = loss.sum() / batch["labels"].shape[0]
    metric_updates = state.metrics.gather_from_model_output(
        logits=logits,
        labels=batch["labels"],
        loss=loss,
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


model_parser = argparse.ArgumentParser(
    description="Model variant to train",
    add_help=False,
)
model_parser.add_argument(
    "--model-name",
    default="vit_small",
    help="Model variant to train",
    type=str,
)

parser = argparse.ArgumentParser(description="Train a network")
parser.add_argument(
    "--run-name",
    default=None,
    help="Run name. If left empty it gets autogenerated",
    type=str,
)
parser.add_argument(
    "--wandb-project",
    default="tpu-tracking",
    help="WandB project",
    type=str,
)
parser.add_argument(
    "--wandb-run-id",
    default=None,
    help="WandB run ID (8 chars code) to resume interrupted run",
    type=str,
)
parser.add_argument(
    "--wandb-tags",
    nargs="*",
    help="Space separated list of tags for WandB",
)
parser.add_argument(
    "--restore-params-ckpt",
    default="",
    help="Restore the parameters from the last step of the given orbax checkpoint. Must be an absolute path. WARNING: restores params only!",
    type=str,
)
parser.add_argument(
    "--restore-simmim-ckpt",
    default="",
    help="Restore the parameters from the last step of the given SimMIM-pretrained orbax checkpoint. Must be an absolute path",
    type=str,
)
parser.add_argument(
    "--freeze-model-body",
    action="store_true",
    help="Freeze the feature extraction layers, train classifier head only",
)
parser.add_argument(
    "--reset-head",
    action="store_true",
    help="Reinit the head weights when loading a pretrained model",
)
parser.add_argument(
    "--clip-pretrained-weights",
    action="store_true",
    help="Clip the pretrained weights around the mean",
)
parser.add_argument(
    "--clip-stddev",
    default=2.0,
    help="Clip the values of pretrained weights to within x standard deviations from the mean",
    type=float,
)
parser.add_argument(
    "--dataset-file",
    default="datasets/aibooru.json",
    help="JSON file with dataset specs",
    type=str,
)
parser.add_argument(
    "--dataset-root",
    default="/home/smilingwolf/datasets",
    help="Dataset root, where the record_shards_train and record_shards_val folders are stored",
    type=str,
)
parser.add_argument(
    "--checkpoints-root",
    default="/mnt/c/Users/SmilingWolf/Desktop/TFKeras/JAX/checkpoints",
    help="Checkpoints root, where the checkpoints will be stored following a <ckpt_root>/<run_name>/<epoch> structure",
    type=str,
)
parser.add_argument(
    "--checkpoints-keep",
    default=2,
    help="Number of best (by val_loss) checkpoints to keep. -1 to always keep the last checkpoint",
    type=int,
)
parser.add_argument(
    "--epochs",
    default=50,
    help="Number of epochs to train for",
    type=int,
)
parser.add_argument(
    "--warmup-epochs",
    default=5,
    help="Number of epochs to dedicate to linear warmup",
    type=int,
)
parser.add_argument(
    "--batch-size",
    default=64,
    help="Per-device batch size",
    type=int,
)
parser.add_argument(
    "--image-size",
    default=256,
    help="Image resolution in input to the network",
    type=int,
)
parser.add_argument(
    "--patch-size",
    default=16,
    help="Size of the image patches",
    type=int,
)
parser.add_argument(
    "--learning-rate",
    default=0.001,
    help="Max learning rate",
    type=float,
)
parser.add_argument(
    "--optimizer-eps",
    default=1e-6,
    help="Optimizer epsilon",
    type=float,
)
parser.add_argument(
    "--grad-clip",
    default=1.0,
    help="Gradient clipping",
    type=float,
)
parser.add_argument(
    "--weight-decay",
    default=0.0001,
    help="Weight decay",
    type=float,
)
parser.add_argument(
    "--loss-weights-file",
    default=None,
    help="Numpy dump of weights to apply to the training loss",
    type=str,
)
parser.add_argument(
    "--mixup-alpha",
    default=0.8,
    help="MixUp alpha",
    type=float,
)
parser.add_argument(
    "--rotation-ratio",
    default=0.0,
    help="Rotation ratio as a fraction of PI",
    type=float,
)
parser.add_argument(
    "--cutout-max-pct",
    default=0.1,
    help="Cutout area as a fraction of the total image area",
    type=float,
)
parser.add_argument(
    "--cutout-patches",
    default=1,
    help="Number of cutout patches",
    type=int,
)
model_arg, remaining = model_parser.parse_known_args()

model_name = model_arg.model_name
model_builder = Models.model_registry[model_name]()
parser = model_builder.extend_parser(parser=parser)

args = parser.parse_args(remaining)

run_name = args.run_name
if run_name is None:
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%Hh%Mm%Ss")
    run_name = f"{model_name}_{date_time}"

checkpoints_root = args.checkpoints_root
dataset_root = args.dataset_root
with open(args.dataset_file) as f:
    dataset_specs = json.load(f)

# Run params
num_epochs = args.epochs
warmup_epochs = args.warmup_epochs
batch_size = args.batch_size
compute_units = jax.device_count()
global_batch_size = batch_size * compute_units
restore_params_ckpt = args.restore_params_ckpt
restore_simmim_ckpt = args.restore_simmim_ckpt

# Dataset params
image_size = args.image_size
num_classes = dataset_specs["num_classes"]
train_samples = dataset_specs["train_samples"]
val_samples = dataset_specs["val_samples"]

# Model hyperparams
patch_size = args.patch_size
learning_rate = args.learning_rate
optimizer_eps = args.optimizer_eps
grad_clip = args.grad_clip
weight_decay = args.weight_decay
loss_weights_file = args.loss_weights_file
freeze_model_body = args.freeze_model_body
reset_head = args.reset_head
clip_pretrained_weights = args.clip_pretrained_weights
clip_stddev = args.clip_stddev

# Augmentations hyperparams
noise_level = 2
mixup_alpha = args.mixup_alpha
rotation_ratio = args.rotation_ratio
cutout_max_pct = args.cutout_max_pct
cutout_patches = args.cutout_patches
random_resize_method = True

# WandB tracking
train_config = {}
train_config["model_name"] = model_name
train_config["checkpoints_root"] = checkpoints_root
train_config["dataset_root"] = dataset_root
train_config["dataset_file"] = args.dataset_file
train_config["num_epochs"] = num_epochs
train_config["warmup_epochs"] = warmup_epochs
train_config["batch_size"] = batch_size
train_config["compute_units"] = compute_units
train_config["global_batch_size"] = global_batch_size
train_config["image_size"] = image_size
train_config["num_classes"] = num_classes
train_config["train_samples"] = train_samples
train_config["val_samples"] = val_samples
train_config["patch_size"] = patch_size
train_config["learning_rate"] = learning_rate
train_config["optimizer_eps"] = optimizer_eps
train_config["grad_clip"] = grad_clip
train_config["weight_decay"] = weight_decay
train_config["loss_weights_file"] = loss_weights_file
train_config["noise_level"] = noise_level
train_config["mixup_alpha"] = mixup_alpha
train_config["rotation_ratio"] = rotation_ratio
train_config["cutout_max_pct"] = cutout_max_pct
train_config["cutout_patches"] = cutout_patches
train_config["random_resize_method"] = random_resize_method
train_config["restore_params_ckpt"] = restore_params_ckpt
train_config["restore_simmim_ckpt"] = restore_simmim_ckpt
train_config["freeze_model_body"] = freeze_model_body
train_config["reset_head"] = reset_head
train_config["clip_pretrained_weights"] = clip_pretrained_weights
train_config["clip_stddev"] = clip_stddev

# Add model specific arguments to WandB dict
args_dict = vars(args)
model_config = {key: args_dict[key] for key in args_dict if key not in train_config}
del model_config["wandb_tags"]
del model_config["run_name"]
del model_config["epochs"]
train_config.update(model_config)

# WandB tracking
wandb_args = dict(
    entity="smilingwolf",
    project=args.wandb_project,
    config=train_config,
    name=run_name,
    tags=args.wandb_tags,
)

if args.wandb_run_id:
    wandb_args["id"] = args.wandb_run_id
    wandb_args["resume"] = "must"

    wandb_entity = wandb_args["entity"]
    wandb_project = wandb_args["project"]
    wandb_run_id = wandb_args["id"]
    wandb_run_path = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"
    run_name = wandb.Api().run(wandb_run_path).name
    wandb_args["name"] = run_name

wandb.init(**wandb_args)

tf.random.set_seed(0)
root_key = jax.random.key(0)
params_key, dropout_key = jax.random.split(key=root_key, num=2)
dropout_keys = jax.random.split(key=dropout_key, num=jax.device_count())
del root_key, dropout_key

training_generator = DataGenerator(
    f"{dataset_root}/record_shards_train/*",
    num_classes=num_classes,
    image_size=image_size,
    batch_size=batch_size,
    num_devices=compute_units,
    noise_level=noise_level,
    mixup_alpha=mixup_alpha,
    rotation_ratio=rotation_ratio,
    cutout_max_pct=cutout_max_pct,
    cutout_patches=cutout_patches,
    random_resize_method=random_resize_method,
)
train_ds = training_generator.genDS()
train_ds = jax_utils.prefetch_to_device(train_ds.as_numpy_iterator(), size=2)

validation_generator = DataGenerator(
    f"{dataset_root}/record_shards_val/*",
    num_classes=num_classes,
    image_size=image_size,
    batch_size=batch_size,
    num_devices=compute_units,
    noise_level=0,
    mixup_alpha=0.0,
    rotation_ratio=0.0,
    cutout_max_pct=0.0,
    random_resize_method=False,
)
val_ds = validation_generator.genDS()
val_ds = jax_utils.prefetch_to_device(val_ds.as_numpy_iterator(), size=2)

model = model_builder.build(
    config=model_builder,
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    dtype=jnp.bfloat16,
    **model_config,
)
# tab_img = jnp.ones([1, image_size, image_size, 3])
# print(model.tabulate(jax.random.key(0), tab_img, train=False))

num_steps_per_epoch = train_samples // global_batch_size
learning_rate = optax.warmup_cosine_decay_schedule(
    init_value=learning_rate * 0.1,
    peak_value=learning_rate,
    warmup_steps=num_steps_per_epoch * warmup_epochs,
    decay_steps=num_steps_per_epoch * num_epochs,
    end_value=learning_rate * 0.01,
)

state = create_train_state(
    model,
    params_key,
    image_size,
    num_classes,
    learning_rate,
    optimizer_eps,
    grad_clip,
    weight_decay,
    freeze_model_body,
)
del params_key

metrics_history = {
    "train_loss": [],
    "train_f1score": [],
    "train_mcc": [],
    "val_loss": [],
    "val_f1score": [],
    "val_mcc": [],
}
ckpt = {"model": state, "metrics_history": metrics_history}

options_dict = dict(
    max_to_keep=args.checkpoints_keep,
    best_fn=lambda metrics: metrics["val_loss"],
    best_mode="min",
)
if args.checkpoints_keep == -1:
    options_dict = dict(max_to_keep=1)

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
options = orbax.checkpoint.CheckpointManagerOptions(
    **options_dict,
    create=True,
)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    f"{checkpoints_root}/{run_name}",
    orbax_checkpointer,
    options,
)


def clipper(param, max_stddev=clip_stddev):
    new_val = param
    new_val_stddev = new_val.std()
    new_val_mean = new_val.mean()
    max_val = new_val_mean + new_val_stddev * max_stddev
    min_val = new_val_mean - new_val_stddev * max_stddev
    new_val = jnp.clip(new_val, min_val, max_val)
    return new_val


if restore_params_ckpt or restore_simmim_ckpt:
    ckpt_path = restore_params_ckpt if restore_params_ckpt else restore_simmim_ckpt

    throwaway_manager = orbax.checkpoint.CheckpointManager(
        ckpt_path,
        orbax_checkpointer,
    )
    latest_epoch = throwaway_manager.latest_step()
    restored = throwaway_manager.restore(latest_epoch)

    if restore_params_ckpt and reset_head:
        del restored["model"]["params"]["head"]

    if clip_pretrained_weights:
        restored["model"]["params"] = jax.tree_util.tree_map(
            clipper,
            restored["model"]["params"],
        )

    transforms = {}
    if restore_simmim_ckpt:
        tx_pairs = model.get_simmim_orbax_txs()
        for tx_regex, tx_action in tx_pairs:
            tx_action = orbax.checkpoint.Transform(original_key=tx_action)
            transforms[tx_regex] = tx_action

    restored = orbax.checkpoint.apply_transformations(restored, transforms, ckpt)

    state = state.replace(params=restored["model"].params)
    del throwaway_manager

latest_epoch = checkpoint_manager.latest_step()
if latest_epoch is not None:
    restored = checkpoint_manager.restore(latest_epoch, items=ckpt)
    state = restored["model"]
    metrics_history = restored["metrics_history"]
    state = state.replace(metrics=state.metrics.empty())
else:
    latest_epoch = 0

# TODO: maybe the weights should be included in the TrainState?
if loss_weights_file:
    label_weights = np.load(loss_weights_file, allow_pickle=False)
else:
    label_weights = np.array([1.0]).astype(np.float32)
label_weights = jax_utils.replicate(label_weights)

step = int(state.step)
state = jax_utils.replicate(state)
p_train_step = jax.pmap(train_step, axis_name="batch")
p_eval_step = jax.pmap(eval_step, axis_name="batch")

epochs = step // num_steps_per_epoch
pbar = tqdm(total=num_steps_per_epoch)
for batch in train_ds:
    # Run optimization steps over training batches and compute batch metrics
    # get updated train state (which contains the updated parameters)
    state = p_train_step(
        state=state,
        batch=batch,
        weights=label_weights,
        dropout_key=dropout_keys,
    )

    if step % 224 == 0:
        merged_metrics = jax_utils.unreplicate(state.metrics)
        merged_metrics = jax.device_get(merged_metrics.loss.compute())
        pbar.set_postfix(loss=f"{merged_metrics:.04f}")

    pbar.update(1)

    # one training epoch has passed
    if (step + 1) % num_steps_per_epoch == 0:
        # compute metrics
        merged_metrics = jax_utils.unreplicate(state.metrics)
        merged_metrics = jax.device_get(merged_metrics.compute())
        for metric, value in merged_metrics.items():
            # record metrics
            metrics_history[f"train_{metric}"].append(value)

        # reset train_metrics for validation
        empty_metrics = state.metrics.empty()
        empty_metrics = jax_utils.replicate(empty_metrics)
        state = state.replace(metrics=empty_metrics)

        # Compute metrics on the validation set after each training epoch
        for val_step, val_batch in enumerate(val_ds):
            state = p_eval_step(state=state, batch=val_batch)
            if val_step == val_samples // global_batch_size:
                break

        merged_metrics = jax_utils.unreplicate(state.metrics)
        merged_metrics = jax.device_get(merged_metrics.compute())
        for metric, value in merged_metrics.items():
            metrics_history[f"val_{metric}"].append(value)

        print(
            f"train epoch: {(step+1) // num_steps_per_epoch}, "
            f"loss: {metrics_history['train_loss'][-1]:.04f}, "
            f"f1score: {metrics_history['train_f1score'][-1]*100:.02f}, "
            f"mcc: {metrics_history['train_mcc'][-1]*100:.02f}"
        )
        print(
            f"val epoch: {(step+1) // num_steps_per_epoch}, "
            f"loss: {metrics_history['val_loss'][-1]:.04f}, "
            f"f1score: {metrics_history['val_f1score'][-1]*100:.02f}, "
            f"mcc: {metrics_history['val_mcc'][-1]*100:.02f}"
        )

        # Log Metrics to Weights & Biases
        wandb.log(
            {
                "train_loss": metrics_history["train_loss"][-1],
                "train_f1score": metrics_history["train_f1score"][-1] * 100,
                "train_mcc": metrics_history["train_mcc"][-1] * 100,
                "val_loss": metrics_history["val_loss"][-1],
                "val_f1score": metrics_history["val_f1score"][-1] * 100,
                "val_mcc": metrics_history["val_mcc"][-1] * 100,
            },
            step=(step + 1) // num_steps_per_epoch,
            commit=True,
        )

        if args.checkpoints_keep > 0:
            ckpt["model"] = jax.device_get(jax_utils.unreplicate(state))
            ckpt["metrics_history"] = metrics_history
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(
                epochs,
                ckpt,
                save_kwargs={"save_args": save_args},
                metrics={"val_loss": float(metrics_history["val_loss"][-1])},
            )

        # reset train_metrics for next training epoch
        empty_metrics = state.metrics.empty()
        empty_metrics = jax_utils.replicate(empty_metrics)
        state = state.replace(metrics=empty_metrics)

        epochs += 1
        if epochs == num_epochs:
            break

        pbar.reset()
    step += 1

pbar.close()
