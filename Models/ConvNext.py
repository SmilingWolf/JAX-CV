import dataclasses
from functools import partial
from typing import Any, Callable, Tuple

import jax.numpy as jnp
import numpy as np
from flax import linen


class LayerScale(linen.Module):
    dim: int
    layer_scale_init_value: float = 1e-6

    dtype: Any = jnp.float32

    def setup(self):
        self.gamma = self.variable(
            "params",
            "gamma",
            lambda x: self.layer_scale_init_value * jnp.ones((x,)),
            self.dim,
        ).value

    def __call__(self, x):
        return x * self.dtype(self.gamma)


class ConvNextBlock(linen.Module):
    drop_path_ratio: float

    bottleneck_ratio: float = 4.0
    layer_scale_init_value: float = 1e-6
    use_conv_bias: bool = True

    norm_layer: Callable = linen.LayerNorm

    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x, train: bool = False):
        _, _, _, C = x.shape
        hidden_size = int(C * self.bottleneck_ratio)
        kernel_init = linen.initializers.truncated_normal(0.02)

        shortcut = x
        x = linen.Conv(
            features=C,
            kernel_size=(7, 7),
            feature_group_count=C,
            kernel_init=kernel_init,
            use_bias=self.use_conv_bias,
            dtype=self.dtype,
        )(x)
        x = self.norm_layer()(x)
        x = linen.Conv(
            features=hidden_size,
            kernel_size=(1, 1),
            kernel_init=kernel_init,
            use_bias=self.use_conv_bias,
            dtype=self.dtype,
        )(x)
        x = linen.gelu(x)
        x = linen.Conv(
            features=C,
            kernel_size=(1, 1),
            kernel_init=kernel_init,
            use_bias=self.use_conv_bias,
            dtype=self.dtype,
        )(x)
        x = LayerScale(
            dim=C,
            layer_scale_init_value=self.layer_scale_init_value,
            dtype=self.dtype,
        )(x)
        x = linen.Dropout(
            rate=self.drop_path_ratio,
            broadcast_dims=(1, 2, 3),
        )(x, deterministic=not train)
        x = shortcut + x
        return x


class BasicLayer(linen.Module):
    depth: int
    embed_dim: int

    drop_path_ratio: Tuple[float]

    downsample: bool = True
    bottleneck_ratio: float = 4.0
    layer_scale_init_value: float = 1e-6
    use_conv_bias: bool = True

    norm_layer: Callable = linen.LayerNorm

    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x, train: bool = False):
        if self.downsample:
            kernel_init = linen.initializers.truncated_normal(0.02)

            x = self.norm_layer()(x)
            x = linen.Conv(
                features=self.embed_dim,
                kernel_size=(2, 2),
                strides=(2, 2),
                kernel_init=kernel_init,
                use_bias=self.use_conv_bias,
                dtype=self.dtype,
            )(x)

        for i in range(self.depth):
            x = ConvNextBlock(
                drop_path_ratio=self.drop_path_ratio[i],
                bottleneck_ratio=self.bottleneck_ratio,
                layer_scale_init_value=self.layer_scale_init_value,
                use_conv_bias=self.use_conv_bias,
                norm_layer=self.norm_layer,
                dtype=self.dtype,
            )(x, train=train)
        return x


class PatchEmbed(linen.Module):
    r"""Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    patch_size: int = 4
    embed_dim: int = 96
    use_conv_bias: bool = True
    norm_layer: Callable = linen.LayerNorm
    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x):
        B, _, _, _ = x.shape
        patch_size = (self.patch_size, self.patch_size)

        kernel_init = linen.initializers.truncated_normal(0.02)
        x = linen.Conv(
            self.embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            kernel_init=kernel_init,
            use_bias=self.use_conv_bias,
            dtype=self.dtype,
        )(x)
        x = self.norm_layer()(x)
        return x


# Cfr. arXiv:2103.17239. Found this to work better
# than the paper default (1e-6) especially for the tiny variant.
def cait_layer_scale_eps(depth):
    if depth <= 18:
        return 0.1
    elif depth <= 24:
        return 1e-4
    else:
        return 1e-5


class ConvNext(linen.Module):
    image_size: int = 224
    patch_size: int = 4
    num_classes: int = 1000

    depths: Tuple[int] = (3, 3, 27, 3)
    embed_dims: Tuple[int] = (128, 256, 512, 1024)

    drop_path_rate: float = 0.1

    use_norm_bias: bool = True
    use_conv_bias: bool = True

    norm_layer: Callable = linen.LayerNorm

    layer_norm_eps: float = 1e-6
    dtype: Any = jnp.float32

    def setup(self):
        depths = self.depths
        num_layers = len(depths)
        norm_layer = partial(
            self.norm_layer,
            use_bias=self.use_norm_bias,
            epsilon=self.layer_norm_eps,
            dtype=self.dtype,
        )

        layer_scale_init_value = cait_layer_scale_eps(sum(depths))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dims[0],
            use_conv_bias=self.use_conv_bias,
            norm_layer=norm_layer,
            dtype=self.dtype,
        )

        # stochastic depth with linear decay
        dpr = [float(x) for x in np.linspace(0, self.drop_path_rate, sum(depths))]

        # build layers
        convnext_body = []
        for i_layer in range(num_layers):
            layer = BasicLayer(
                depth=depths[i_layer],
                embed_dim=self.embed_dims[i_layer],
                drop_path_ratio=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=i_layer > 0,
                layer_scale_init_value=layer_scale_init_value,
                use_conv_bias=self.use_conv_bias,
                norm_layer=norm_layer,
                dtype=self.dtype,
            )
            convnext_body.append(layer)
        self.convnext_body = convnext_body

        self.norm = norm_layer()
        self.head = (
            linen.Dense(self.num_classes, dtype=self.dtype)
            if self.num_classes > 0
            else lambda x: x
        )

    def __call__(self, x, train: bool = False):
        x = self.patch_embed(x)

        for layer in self.convnext_body:
            x = layer(x, train=train)

        x = jnp.mean(x, axis=(1, 2))
        x = self.norm(x)
        x = self.head(x)
        return x

    @classmethod
    def build(cls, config, **kwargs):
        config = dataclasses.asdict(config)
        config = {key: kwargs[key] if key in kwargs else config[key] for key in config}
        return cls(**config)

    def extend_parser(self, parser):
        parser.set_defaults(image_size=self.image_size)
        parser.set_defaults(patch_size=self.patch_size)
        parser.add_argument(
            "--drop-path-rate",
            default=self.drop_path_rate,
            help="Stochastic depth rate",
            type=float,
        )

        parser.add_argument(
            "--enable-conv-bias",
            dest="use_conv_bias",
            help="Enable conv layers bias",
            action="store_true",
        )
        parser.add_argument(
            "--disable-conv-bias",
            dest="use_conv_bias",
            help="Disable conv layers bias",
            action="store_false",
        )
        parser.set_defaults(use_conv_bias=self.use_conv_bias)

        parser.add_argument(
            "--enable-norm-bias",
            dest="use_norm_bias",
            help="Enable norm layers bias",
            action="store_true",
        )
        parser.add_argument(
            "--disable-norm-bias",
            dest="use_norm_bias",
            help="Disable norm layers bias",
            action="store_false",
        )
        parser.set_defaults(use_norm_bias=self.use_norm_bias)
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
        is_scale = path[-1].key == "scale"
        is_gamma = path[-1].key == "gamma"
        verdict = is_kernel or is_scale or is_gamma
        return verdict


def convnext_tiny():
    config = {
        "embed_dims": (96, 192, 384, 768),
        "depths": (3, 3, 9, 3),
    }
    return ConvNext(**config)


def convnext_small():
    config = {
        "embed_dims": (96, 192, 384, 768),
        "depths": (3, 3, 27, 3),
    }
    return ConvNext(**config)


def convnext_base():
    config = {
        "embed_dims": (128, 256, 512, 1024),
        "depths": (3, 3, 27, 3),
    }
    return ConvNext(**config)
