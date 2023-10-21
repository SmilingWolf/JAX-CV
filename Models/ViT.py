from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from flax import linen


class MLP(linen.Module):
    hidden_features: int
    act_layer: Callable = linen.gelu
    drop_ratio: float = 0.0

    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x, train: bool):
        out_dim = x.shape[-1]

        x = linen.Dense(self.hidden_features, dtype=self.dtype)(x)
        x = self.act_layer(x)
        x = linen.Dropout(self.drop_ratio)(x, deterministic=not train)
        x = linen.Dense(out_dim, dtype=self.dtype)(x)
        return x


class PatchEmbed(linen.Module):
    r"""Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 16.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    patch_size: int = 16
    embed_dim: int = 768

    norm_layer: Callable = None

    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x):
        B, _, _, _ = x.shape
        patch_size = (self.patch_size, self.patch_size)

        x = linen.Conv(
            self.embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            dtype=self.dtype,
        )(x)
        x = jnp.reshape(x, (B, -1, self.embed_dim))
        if self.norm_layer is not None:
            x = self.norm_layer()(x)
        return x


class VisionTransformerBlock(linen.Module):
    mlp_dim: int
    num_heads: int
    drop_path_ratio: float

    norm_layer: Callable

    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x, train: bool = False):
        shortcut = x

        x = self.norm_layer()(x)
        x = linen.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
        )(x, x)
        x = linen.Dropout(
            rate=self.drop_path_ratio,
            broadcast_dims=(1, 2),
        )(x, deterministic=not train)
        x = shortcut + x

        shortcut = x
        x = self.norm_layer()(x)
        x = MLP(hidden_features=self.mlp_dim, dtype=self.dtype)(x, train=train)
        x = linen.Dropout(
            rate=self.drop_path_ratio,
            broadcast_dims=(1, 2),
        )(x, deterministic=not train)
        x = shortcut + x
        return x


class VisionTransformer(linen.Module):
    patch_size: int
    num_classes: int = 1000

    num_layers: int = 12
    embed_dim: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12

    drop_path_rate: float = 0.1

    norm_layer: Callable = linen.LayerNorm

    layer_norm_eps: float = 1e-5
    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x, train: bool = False):
        norm_layer = partial(
            self.norm_layer,
            epsilon=self.layer_norm_eps,
            dtype=self.dtype,
        )

        # stochastic depth with linear decay
        dpr = np.linspace(0, self.drop_path_rate, self.num_layers)
        dpr = [float(x) for x in dpr]

        x = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            dtype=self.dtype,
        )(x)

        pos_emb_init = linen.initializers.normal(stddev=0.02)
        pos_emb = self.param("pos_emb", pos_emb_init, (1, 1, self.embed_dim))
        pos_emb = linen.dtypes.promote_dtype(pos_emb, dtype=self.dtype)[0]
        x = x + pos_emb

        for i in range(self.num_layers):
            x = VisionTransformerBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                drop_path_ratio=dpr[i],
                norm_layer=norm_layer,
                dtype=self.dtype,
            )(x, train=train)

        x = norm_layer(name="norm_layer")(x)
        x = jnp.mean(x, axis=(1,))

        if self.num_classes > 0:
            x = linen.Dense(self.num_classes, name="head", dtype=self.dtype)(x)
        return x


def vit_small(**kwargs):
    model = partial(
        VisionTransformer,
        num_layers=12,
        embed_dim=384,
        mlp_dim=1536,
        num_heads=6,
    )
    model = model(**kwargs)
    return model


def vit_base(**kwargs):
    model = partial(
        VisionTransformer,
        num_layers=12,
        embed_dim=768,
        mlp_dim=3072,
        num_heads=12,
    )
    model = model(**kwargs)
    return model


def vit_large(**kwargs):
    model = partial(
        VisionTransformer,
        num_layers=24,
        embed_dim=1024,
        mlp_dim=4096,
        num_heads=16,
    )
    model = model(**kwargs)
    return model
