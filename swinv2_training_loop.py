import argparse
import json
from typing import Any, Callable, Union

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
import tensorflow as tf
from clu import metrics
from flax import jax_utils, struct
from flax.training import orbax_utils, train_state
from tqdm import tqdm

from Generators.WDTaggerGen import DataGenerator
from Metrics.ConfusionMatrix import f1score, mcc
from Models import SwinV2


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Metric
    f1score: metrics.Metric
    mcc: metrics.Metric


class TrainState(train_state.TrainState):
    metrics: Metrics
    constants: Any


def create_train_state(
    module,
    params_key,
    target_size: int,
    num_classes: int,
    learning_rate: Union[float, Callable],
    weight_decay: float,
):
    """Creates an initial 'TrainState'."""
    # initialize parameters by passing a template image
    variables = module.init(
        params_key,
        jnp.ones([1, target_size, target_size, 3]),
        train=False,
    )
    params = variables["params"]
    constants = variables["swinv2_constants"]

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

    def should_decay(path, _):
        is_kernel = path[-1].key == "kernel"
        is_cpb = "attention_bias" in [x.key for x in path]
        return is_kernel and not is_cpb

    wd_mask = jax.tree_util.tree_map_with_path(should_decay, params)
    tx = optax.lamb(learning_rate, weight_decay=weight_decay, mask=wd_mask)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=collection.empty(),
        constants=constants,
    )


def train_step(state, batch, dropout_key):
    """Train for a single step."""
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params, constants):
        logits = state.apply_fn(
            {"params": params, "swinv2_constants": constants},
            batch["images"],
            train=True,
            rngs={"dropout": dropout_train_key},
        )
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch["labels"])
        loss = loss.sum() / batch["labels"].shape[0]
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, state.constants)
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
        {"params": state.params, "swinv2_constants": state.constants},
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


parser = argparse.ArgumentParser(description="Train a network")
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
    default="/tmp/checkpoints/checkpoints",
    help="Checkpoints root, where the checkpoints will be stored following a <ckpt_root>/<network_name>/<epoch> structure",
    type=str,
)
parser.add_argument(
    "--epochs",
    default=50,
    help="Number of epochs to train for",
    type=int,
)
parser.add_argument(
    "--batch-size",
    default=64,
    help="Per-device batch size",
    type=int,
)
parser.add_argument(
    "--learning-rate",
    default=0.001,
    help="Max learning rate",
    type=float,
)
parser.add_argument(
    "--weight-decay",
    default=0.0001,
    help="Weight decay",
    type=float,
)
parser.add_argument(
    "--dropout-rate",
    default=0.1,
    help="Stochastic depth rate",
    type=float,
)
parser.add_argument(
    "--mixup-alpha",
    default=0.8,
    help="MixUp alpha (wow much explanation, so clear)",
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
args = parser.parse_args()

checkpoints_root = args.checkpoints_root
dataset_root = args.dataset_root
with open(args.dataset_file) as f:
    dataset_specs = json.load(f)

# Run params
num_epochs = args.epochs
batch_size = args.batch_size
compute_units = jax.device_count()
global_batch_size = batch_size * compute_units

# Dataset params
image_size = 256
num_classes = dataset_specs["num_classes"]
train_samples = dataset_specs["train_samples"]
val_samples = dataset_specs["val_samples"]

# Model hyperparams
learning_rate = args.learning_rate
weight_decay = args.weight_decay
dropout_rate = args.dropout_rate

# Augmentations hyperparams
noise_level = 2
mixup_alpha = args.mixup_alpha
rotation_ratio = args.rotation_ratio
cutout_max_pct = args.cutout_max_pct
random_resize_method = True

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

model = SwinV2.swinv2_tiny_window8_256(
    img_size=image_size,
    num_classes=num_classes,
    drop_path_rate=dropout_rate,
    dtype=jnp.bfloat16,
)
# print(
#     model.tabulate(
#         jax.random.key(0), jnp.ones([1, image_size, image_size, 3]), train=False
#     )
# )

num_steps_per_epoch = train_samples // global_batch_size
learning_rate = optax.warmup_cosine_decay_schedule(
    init_value=learning_rate * 0.1,
    peak_value=learning_rate,
    warmup_steps=num_steps_per_epoch * 5,
    decay_steps=num_steps_per_epoch * num_epochs,
    end_value=learning_rate * 0.01,
)

state = create_train_state(
    model,
    params_key,
    image_size,
    num_classes,
    learning_rate,
    weight_decay,
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

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
options = orbax.checkpoint.CheckpointManagerOptions(
    max_to_keep=2,
    best_fn=lambda metrics: metrics["val_loss"],
    best_mode="min",
    create=True,
)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    f"{checkpoints_root}/SwinV2",
    orbax_checkpointer,
    options,
)

latest_epoch = checkpoint_manager.latest_step()
if latest_epoch is not None:
    ckpt = {"model": state, "metrics_history": metrics_history}
    restored = checkpoint_manager.restore(latest_epoch, items=ckpt)
    state = state.replace(params=restored["model"].params)
    metrics_history = restored["metrics_history"]
else:
    latest_epoch = 0

state = jax_utils.replicate(state)
p_train_step = jax.pmap(train_step, axis_name="batch")
p_eval_step = jax.pmap(eval_step, axis_name="batch")

epochs = 0
pbar = tqdm(total=num_steps_per_epoch)
for step, batch in enumerate(train_ds):
    # Run optimization steps over training batches and compute batch metrics
    # get updated train state (which contains the updated parameters)
    state = p_train_step(state=state, batch=batch, dropout_key=dropout_keys)

    if step % 192 == 0:
        merged_metrics = jax_utils.unreplicate(state.metrics)
        pbar.set_postfix(loss=f"{merged_metrics.loss.compute():.04f}")

    pbar.update(1)

    # one training epoch has passed
    if (step + 1) % num_steps_per_epoch == 0:
        # compute metrics
        merged_metrics = jax_utils.unreplicate(state.metrics)
        for metric, value in merged_metrics.compute().items():
            # record metrics
            metrics_history[f"train_{metric}"].append(value)

        # reset train_metrics for next training epoch
        empty_metrics = state.metrics.empty()
        empty_metrics = jax_utils.replicate(empty_metrics)
        state = state.replace(metrics=empty_metrics)

        # Compute metrics on the validation set after each training epoch
        val_state = state
        for val_step, val_batch in enumerate(val_ds):
            val_state = p_eval_step(state=val_state, batch=val_batch)
            if val_step == val_samples // global_batch_size:
                break

        val_state = jax_utils.unreplicate(val_state)
        for metric, value in val_state.metrics.compute().items():
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

        ckpt = {"model": val_state, "metrics_history": metrics_history}
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(
            epochs + latest_epoch,
            ckpt,
            save_kwargs={"save_args": save_args},
            metrics={"val_loss": float(metrics_history["val_loss"][-1])},
        )

        epochs += 1
        if epochs == num_epochs:
            break

        pbar.reset()

pbar.close()
