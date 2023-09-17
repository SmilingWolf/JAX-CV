from typing import Any

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from clu import metrics
from flax import struct
from flax.training import train_state
from tqdm import tqdm

from Generators.WDTaggerGen import DataGenerator
from Models import SwinV2

# from Metrics.Recall import RecallV2


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")
    # recall: RecallV2.with_threshold(0.5)


class TrainState(train_state.TrainState):
    metrics: Metrics
    constants: Any


def create_train_state(module, params_key, target_size, learning_rate, weight_decay):
    """Creates an initial `TrainState`."""
    # initialize parameters by passing a template image
    variables = module.init(
        params_key,
        jnp.ones([1, target_size, target_size, 3]),
        train=False,
    )
    params = variables["params"]
    constants = variables["swinv2_constants"]

    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty(),
        constants=constants,
    )


@jax.jit
def train_step(state, batch, dropout_key):
    """Train for a single step."""
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params, constants):
        logits = state.apply_fn(
            {"params": params, "swinv2_constants": constants},
            batch[0],
            train=True,
            rngs={"dropout": dropout_train_key},
        )
        loss = optax.sigmoid_binary_cross_entropy(
            logits=logits, labels=batch[1]
        ).sum() / len(batch[1])
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params, state.constants)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn(
        {"params": state.params, "swinv2_constants": state.constants},
        batch[0],
        train=False,
    )
    metrics_labels = jnp.argmax(batch[1], axis=-1)

    loss = optax.sigmoid_binary_cross_entropy(
        logits=logits, labels=batch[1]
    ).sum() / len(batch[1])
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=metrics_labels, loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


# Run params
num_epochs = 40
batch_size = 64
compute_units = 1
global_batch_size = batch_size * compute_units

# Dataset params
image_size = 256
total_labels = 4
train_samples = 24576
val_samples = 10240

# Model hyperparams
learning_rate = 0.0005
weight_decay = 0.0
dropout_rate = 0.1

# Augmentations hyperparams
noise_level = 2
mixup_alpha = 0.1
cutout_max_pct = 0.1
random_resize_method = True

tf.random.set_seed(0)
root_key = jax.random.PRNGKey(0)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

training_generator = DataGenerator(
    "/home/smilingwolf/datasets/record_shards_train/*",
    total_labels=total_labels,
    image_size=image_size,
    batch_size=global_batch_size,
    noise_level=noise_level,
    mixup_alpha=mixup_alpha,
    cutout_max_pct=cutout_max_pct,
    random_resize_method=random_resize_method,
)
train_ds = training_generator.genDS()

validation_generator = DataGenerator(
    "/home/smilingwolf/datasets/record_shards_val/*",
    total_labels=total_labels,
    image_size=image_size,
    batch_size=global_batch_size,
    noise_level=0,
    mixup_alpha=0.0,
    cutout_max_pct=0.0,
    random_resize_method=False,
)
test_ds = validation_generator.genDS()

model = SwinV2.SwinTransformerV2(
    img_size=image_size,
    num_classes=total_labels,
    embed_dim=96,
    window_size=8,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    drop_path_rate=dropout_rate,
    dtype=jnp.bfloat16,
)
# print(
#     model.tabulate(
#         jax.random.PRNGKey(0), jnp.ones([1, image_size, image_size, 3]), train=False
#     )
# )
state = create_train_state(model, params_key, image_size, learning_rate, weight_decay)
del params_key

metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "train_recall": [],
    "test_loss": [],
    "test_accuracy": [],
    "test_recall": [],
}

epochs = 0
num_steps_per_epoch = train_samples // batch_size
pbar = tqdm(total=num_steps_per_epoch)
for step, batch in enumerate(train_ds.as_numpy_iterator()):
    # Run optimization steps over training batches and compute batch metrics
    # get updated train state (which contains the updated parameters)
    state = train_step(state, batch, dropout_key)

    # aggregate batch metrics
    state = compute_metrics(state=state, batch=batch)

    pbar.update(1)

    # one training epoch has passed
    if (step + 1) % num_steps_per_epoch == 0:
        # compute metrics
        for metric, value in state.metrics.compute().items():
            # record metrics
            metrics_history[f"train_{metric}"].append(value)

        # reset train_metrics for next training epoch
        state = state.replace(metrics=state.metrics.empty())

        # Compute metrics on the test set after each training epoch
        test_state = state
        for val_step, test_batch in enumerate(test_ds.as_numpy_iterator()):
            test_state = compute_metrics(state=test_state, batch=test_batch)
            if val_step == val_samples // batch_size:
                break

        for metric, value in test_state.metrics.compute().items():
            metrics_history[f"test_{metric}"].append(value)

        print(
            f"train epoch: {(step+1) // num_steps_per_epoch}, "
            f"loss: {metrics_history['train_loss'][-1]}, "
            f"accuracy: {metrics_history['train_accuracy'][-1] * 100:.02f}"
        )
        print(
            f"test epoch: {(step+1) // num_steps_per_epoch}, "
            f"loss: {metrics_history['test_loss'][-1]}, "
            f"accuracy: {metrics_history['test_accuracy'][-1] * 100:.02f}"
        )

        epochs += 1
        if epochs == num_epochs:
            break

        pbar.reset()

pbar.close()
