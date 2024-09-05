import dataclasses

import jax.numpy as jnp
import jax.typing as jt
from flax import linen


class VGGStage(linen.Module):
    filters: int = 64
    kernel_sizes: tuple[int] = (3,)
    dtype: jt.DTypeLike = jnp.float32

    @linen.compact
    def __call__(self, x):
        for kernel_size in self.kernel_sizes:
            k_init = linen.initializers.normal(stddev=(10e-2) ** 0.5)
            x = linen.Conv(
                self.filters,
                kernel_size=(kernel_size, kernel_size),
                strides=(1, 1),
                kernel_init=k_init,
                dtype=self.dtype,
            )(x)
            x = linen.relu(x)
        x = linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class VGGNetwork(linen.Module):
    patch_size: int = 3
    num_classes: int = 1000

    filters: tuple[int] = (64, 128, 256, 512, 512)
    kernel_sizes: tuple[tuple[int]] = ((3,), (3,), (3, 3), (3, 3), (3, 3))

    dtype: jt.DTypeLike = jnp.float32

    def setup(self):
        if self.num_classes > 0:
            k_init = linen.initializers.normal(stddev=(10e-2) ** 0.5)
            self.head = linen.Dense(
                self.num_classes,
                kernel_init=k_init,
                dtype=self.dtype,
            )
        else:
            self.head = lambda x: x

    @linen.compact
    def __call__(self, x, train: bool = False):
        for filters, kernel_sizes in zip(self.filters, self.kernel_sizes):
            x = VGGStage(
                filters=filters,
                kernel_sizes=kernel_sizes,
                dtype=self.dtype,
            )(x)

        b, h, w, c = x.shape
        x = jnp.reshape(x, (b, h * w * c))

        k_init = linen.initializers.normal(stddev=(10e-2) ** 0.5)
        x = linen.Dense(4096, kernel_init=k_init, dtype=self.dtype)(x)
        x = linen.relu(x)
        x = linen.Dropout(0.5, deterministic=not train)(x)
        x = linen.Dense(4096, kernel_init=k_init, dtype=self.dtype)(x)
        x = linen.relu(x)
        x = linen.Dropout(0.5, deterministic=not train)(x)

        x = self.head(x)
        return x

    @classmethod
    def build(cls, config, **kwargs):
        config = dataclasses.asdict(config)
        config = {key: kwargs[key] if key in kwargs else config[key] for key in config}
        return cls(**config)

    def extend_parser(self, parser):
        parser.set_defaults(patch_size=self.patch_size)
        return parser

    @staticmethod
    def get_simmim_orbax_txs():
        # SimMIM checkpoint have no head params - don't try to restore them.
        # All the other params we care about are under the "encoder" subsection
        regex = r"(?!model/params/head)model/params/(.*)"
        action = r"model/params/encoder/\1"
        return [(regex, action)]

    def should_decay(self, path, _):
        is_kernel = path[-1].key == "kernel"
        verdict = is_kernel
        return verdict


def vgg11():
    config = {
        "kernel_sizes": (
            (3,),
            (3,),
            (3, 3),
            (3, 3),
            (3, 3),
        )
    }
    return VGGNetwork(**config)


def vgg13():
    config = {
        "kernel_sizes": (
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
        )
    }
    return VGGNetwork(**config)


def vgg19():
    config = {
        "kernel_sizes": (
            (3, 3),
            (3, 3),
            (3, 3, 3, 3),
            (3, 3, 3, 3),
            (3, 3, 3, 3),
        )
    }
    return VGGNetwork(**config)
