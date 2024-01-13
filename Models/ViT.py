import dataclasses
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from flax import linen


class Attention(linen.Module):
    dim: int
    num_heads: int
    qkv_bias: bool = True
    attn_drop_ratio: float = 0.0
    proj_drop_ratio: float = 0.0

    dtype: Any = jnp.float32

    def setup(self):
        self.qkv = linen.Dense(self.dim * 3, use_bias=self.qkv_bias, dtype=self.dtype)
        self.attn_drop = linen.Dropout(self.attn_drop_ratio)
        self.proj = linen.Dense(self.dim, dtype=self.dtype)
        self.proj_drop = linen.Dropout(self.proj_drop_ratio)
        self.softmax = partial(linen.activation.softmax, axis=-1)

    def __call__(self, x, train: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = (qkv[0], qkv[1], qkv[2])

        q = q / jnp.sqrt(q.shape[-1]).astype(q.dtype)
        attn = q @ jnp.transpose(k, (0, 1, 3, 2))

        attn = self.softmax(attn.astype(jnp.float32)).astype(self.dtype)
        attn = self.attn_drop(attn, deterministic=not train)

        x = jnp.transpose(attn @ v, (0, 2, 1, 3))
        x = jnp.reshape((x), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=not train)
        return x


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


class PosEmbed(linen.Module):
    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x):
        _, L, C = x.shape
        pos_emb_init = linen.initializers.normal(stddev=1 / np.sqrt(C))
        pos_emb = self.param("pos_emb", pos_emb_init, (1, L, C))
        pos_emb = linen.dtypes.promote_dtype(pos_emb, dtype=self.dtype)[0]
        x = x + pos_emb
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
        x = Attention(
            dim=x.shape[-1],
            num_heads=self.num_heads,
            dtype=self.dtype,
        )(x, train=train)
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
    patch_size: int = 16
    num_classes: int = 1000

    num_layers: int = 12
    embed_dim: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12

    drop_path_rate: float = 0.1

    norm_layer: Callable = linen.LayerNorm

    layer_norm_eps: float = 1e-5
    dtype: Any = jnp.float32

    def setup(self):
        norm_layer = partial(
            self.norm_layer,
            epsilon=self.layer_norm_eps,
            dtype=self.dtype,
        )

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            dtype=self.dtype,
        )

        self.pos_emb = PosEmbed(dtype=self.dtype)

        # stochastic depth with linear decay
        dpr = np.linspace(0, self.drop_path_rate, self.num_layers)
        dpr = [float(x) for x in dpr]

        vit_body = []
        for i in range(self.num_layers):
            layer = VisionTransformerBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                drop_path_ratio=dpr[i],
                norm_layer=norm_layer,
                dtype=self.dtype,
            )
            vit_body.append(layer)
        self.vit_body = vit_body

        self.norm = norm_layer()
        self.head = (
            linen.Dense(self.num_classes, dtype=self.dtype)
            if self.num_classes > 0
            else lambda x: x
        )

    def __call__(self, x, train: bool = False):
        x = self.patch_embed(x)

        x = self.pos_emb(x)

        for layer in self.vit_body:
            x = layer(x, train=train)

        x = self.norm(x)
        x = jnp.mean(x, axis=(1,))
        x = self.head(x)
        return x

    @classmethod
    def build(cls, config, **kwargs):
        config = dataclasses.asdict(config)
        config = {key: kwargs[key] if key in kwargs else config[key] for key in config}
        return cls(**config)

    def extend_parser(self, parser):
        parser.set_defaults(patch_size=self.patch_size)
        parser.add_argument(
            "--drop-path-rate",
            default=self.drop_path_rate,
            help="Stochastic depth rate",
            type=float,
        )
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
        verdict = is_kernel or is_scale
        return verdict


def vit_small():
    config = {
        "num_layers": 12,
        "embed_dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
    }
    return VisionTransformer(**config)


def vit_base():
    config = {
        "num_layers": 12,
        "embed_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
    }
    return VisionTransformer(**config)


def vit_large():
    config = {
        "num_layers": 24,
        "embed_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    }
    return VisionTransformer(**config)
