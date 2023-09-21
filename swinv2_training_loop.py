import argparse
from typing import Any, Callable, Union

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from clu import metrics
from flax import jax_utils, struct
from flax.training import train_state
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
    dropout_key: Any
    constants: Any


def create_train_state(
    module,
    params_key,
    dropout_key,
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

    tx = optax.lamb(learning_rate, weight_decay=weight_decay)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=collection.empty(),
        dropout_key=dropout_key,
        constants=constants,
    )


@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    dropout_train_key = jax.random.fold_in(key=state.dropout_key, data=state.step)

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


@jax.jit
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


def prepare_tf_data(xs):
    local_device_count = jax.local_device_count()

    def _prepare(x):
        x = x._numpy()
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


def create_input_iter(ds):
    it = map(prepare_tf_data, ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


parser = argparse.ArgumentParser(description="Train a network")
parser.add_argument(
    "--learning-rate",
    default=0.0005,
    help="Max learning rate",
    type=float,
)
args = parser.parse_args()

# Run params
num_epochs = 50
batch_size = 64
compute_units = jax.device_count()
global_batch_size = batch_size * compute_units

# Dataset params
image_size = 256
num_classes = 5384
train_samples = 24576
val_samples = 11264

# Model hyperparams
learning_rate = args.learning_rate
weight_decay = 0.0005
dropout_rate = 0.1

# Augmentations hyperparams
noise_level = 2
mixup_alpha = 0.8
cutout_max_pct = 0.1
random_resize_method = True

tf.random.set_seed(0)
root_key = jax.random.PRNGKey(0)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

training_generator = DataGenerator(
    "/home/smilingwolf/datasets/record_shards_train/*",
    num_classes=num_classes,
    image_size=image_size,
    batch_size=global_batch_size,
    noise_level=noise_level,
    mixup_alpha=mixup_alpha,
    cutout_max_pct=cutout_max_pct,
    random_resize_method=random_resize_method,
)
train_ds = training_generator.genDS()
train_ds = create_input_iter(train_ds)

validation_generator = DataGenerator(
    "/home/smilingwolf/datasets/record_shards_val/*",
    num_classes=num_classes,
    image_size=image_size,
    batch_size=global_batch_size,
    noise_level=0,
    mixup_alpha=0.0,
    cutout_max_pct=0.0,
    random_resize_method=False,
)
test_ds = validation_generator.genDS()
test_ds = create_input_iter(test_ds)

model = SwinV2.swinv2_tiny_window8_256(
    img_size=image_size,
    num_classes=num_classes,
    drop_path_rate=dropout_rate,
    dtype=jnp.bfloat16,
)
# print(
#     model.tabulate(
#         jax.random.PRNGKey(0), jnp.ones([1, image_size, image_size, 3]), train=False
#     )
# )
state = create_train_state(
    model,
    params_key,
    dropout_key,
    image_size,
    num_classes,
    learning_rate,
    weight_decay,
)
del params_key

state = jax_utils.replicate(state)
p_train_step = jax.pmap(train_step, axis_name="batch")
p_eval_step = jax.pmap(eval_step, axis_name="batch")

metrics_history = {
    "train_loss": [],
    "train_f1score": [],
    "train_mcc": [],
    "test_loss": [],
    "test_f1score": [],
    "test_mcc": [],
}

epochs = 0
num_steps_per_epoch = train_samples // global_batch_size
pbar = tqdm(total=num_steps_per_epoch)
for step, batch in enumerate(train_ds):
    # Run optimization steps over training batches and compute batch metrics
    # get updated train state (which contains the updated parameters)
    state = p_train_step(state=state, batch=batch)

    if step % 32 == 0:
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

        # Compute metrics on the test set after each training epoch
        test_state = state
        for val_step, test_batch in enumerate(test_ds):
            test_state = p_eval_step(state=test_state, batch=test_batch)
            if val_step == val_samples // global_batch_size:
                break

        merged_metrics = jax_utils.unreplicate(test_state.metrics)
        for metric, value in merged_metrics.compute().items():
            metrics_history[f"test_{metric}"].append(value)

        print(
            f"train epoch: {(step+1) // num_steps_per_epoch}, "
            f"loss: {metrics_history['train_loss'][-1]:.04f}, "
            f"f1score: {metrics_history['train_f1score'][-1]*100:.02f}, "
            f"mcc: {metrics_history['train_mcc'][-1]*100:.02f}"
        )
        print(
            f"test epoch: {(step+1) // num_steps_per_epoch}, "
            f"loss: {metrics_history['test_loss'][-1]:.04f}, "
            f"f1score: {metrics_history['test_f1score'][-1]*100:.02f}, "
            f"mcc: {metrics_history['test_mcc'][-1]*100:.02f}"
        )

        epochs += 1
        if epochs == num_epochs:
            break

        pbar.reset()

pbar.close()
