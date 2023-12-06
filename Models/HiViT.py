import dataclasses
from functools import partial
from typing import Any, Callable, Tuple, Union

import jax.numpy as jnp
import numpy as np
from flax import linen


class RelativePositionBias(linen.Module):
    input_size: int
    num_heads: int

    dtype: Any = jnp.float32

    def get_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.input_size)
        coords_w = np.arange(self.input_size)

        # 2, Wh, Ww
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))

        # 2, Wh*Ww
        coords_flatten = np.reshape(coords, (2, -1))

        # 2, Wh*Ww, Wh*Ww
        coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

        # Wh*Ww, Wh*Ww, 2
        coords = np.transpose(coords, (1, 2, 0))

        # shift to start from 0
        coords[:, :, 0] = coords[:, :, 0] + (self.input_size - 1)
        coords[:, :, 1] = coords[:, :, 1] + (self.input_size - 1)
        coords[:, :, 0] = coords[:, :, 0] * (2 * self.input_size - 1)

        # Wh*Ww, Wh*Ww
        position_index = np.sum(coords, axis=-1)
        return position_index

    def setup(self):
        self.relative_position_bias_table = self.param(
            "relative_position_bias_table",
            linen.initializers.truncated_normal(stddev=0.02),
            ((2 * self.input_size - 1) * (2 * self.input_size - 1), self.num_heads),
        )

        self.relative_position_index = self.variable(
            "hivit_constants",
            "relative_position_index",
            self.get_relative_position_index,
        ).value

    def __call__(self, x):
        rpe_index = jnp.reshape(self.relative_position_index, (-1,))
        relative_position_bias = self.relative_position_bias_table[rpe_index]

        relative_position_bias = jnp.reshape(
            relative_position_bias,
            (self.input_size**2, self.input_size**2, -1),
        )
        relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))
        relative_position_bias = jnp.expand_dims(relative_position_bias, 0)

        x = x + relative_position_bias
        return x


class Attention(linen.Module):
    input_size: int
    dim: int
    num_heads: int
    qkv_bias: bool = True
    qk_scale: Union[None, float] = None
    attn_drop_ratio: float = 0.0
    proj_drop_ratio: float = 0.0
    rpe_enabled: bool = True

    dtype: Any = jnp.float32

    def setup(self):
        self.attention_bias = (
            RelativePositionBias(
                self.input_size,
                self.num_heads,
                dtype=self.dtype,
            )
            if self.rpe_enabled
            else lambda x: x
        )

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

        attn = self.attention_bias(attn)

        attn = self.softmax(attn).astype(self.dtype)
        attn = self.attn_drop(attn, deterministic=not train)

        x = jnp.transpose(attn @ v, (0, 2, 1, 3))
        x = jnp.reshape((x), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=not train)
        return x


class PatchMerging(linen.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    input_resolution: Tuple[int]
    norm_layer: Callable = linen.LayerNorm

    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        x = jnp.reshape(x, (B, H // 2, 2, W // 2, 2, C))  # B H/2 nH W/2 nW C
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))  # B H/2 W/2 nH nW C
        x = jnp.reshape(x, (B, (H // 2) * (W // 2), 4 * C))  # B H/2*W/2 4*C

        x = self.norm_layer()(x)
        x = linen.Dense(2 * C, use_bias=False, dtype=self.dtype)(x)
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
        patch_size (int): Patch token size. Default: 4.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    patch_size: int = 4
    embed_dim: int = 96
    internal_patches: int = 4

    norm_layer: Callable = None

    dtype: Any = jnp.float32

    def patches_reshape(self, x):
        B, H, W, C = x.shape
        nH = nW = self.internal_patches
        H = H // self.internal_patches
        W = W // self.internal_patches
        x = jnp.reshape(x, (B, H, nH, W, nW, C))
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = jnp.reshape(x, (B, H * W * nH * nW, C))
        return x

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

        x = self.patches_reshape(x)

        if self.norm_layer is not None:
            x = self.norm_layer()(x)
        return x


class HierarchicalViTBlock(linen.Module):
    mlp_dim: int
    num_heads: int
    drop_path_ratio: float

    norm_layer: Callable

    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x, train: bool = False):
        shortcut = x

        x = self.norm_layer()(x)
        if self.num_heads:
            _, L, C = x.shape
            x = Attention(
                input_size=int(L**0.5),
                dim=C,
                num_heads=self.num_heads,
                dtype=self.dtype,
            )(x, train=train)
        else:
            x = MLP(hidden_features=self.mlp_dim, dtype=self.dtype)(x, train=train)

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


class BasicLayer(linen.Module):
    depth: int

    num_heads: int = 12
    mlp_ratio: float = 4.0
    pos_emb_enabled: bool = False
    drop_path_ratio: Union[float, Tuple[float]] = 0.1

    downsample: Union[None, Callable] = None

    norm_layer: Callable = linen.LayerNorm

    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x, train: bool = False):
        B, L, C = x.shape

        if self.pos_emb_enabled:
            x = PosEmbed(dtype=self.dtype)(x)

        mlp_dim = int(C * self.mlp_ratio)
        for i in range(self.depth):
            x = HierarchicalViTBlock(
                mlp_dim=mlp_dim,
                num_heads=self.num_heads,
                drop_path_ratio=self.drop_path_ratio[i],
                norm_layer=self.norm_layer,
                dtype=self.dtype,
            )(x, train=train)

        # patch merging layer
        if self.downsample is not None:
            H = W = int(L**0.5)
            x = self.downsample(
                input_resolution=(H, W),
                norm_layer=self.norm_layer,
                dtype=self.dtype,
            )(x)

        return x


class HierarchicalViT(linen.Module):
    patch_size: int = 4
    num_classes: int = 1000

    depths: Tuple[int] = (2, 2, 20)
    embed_dim: int = 192
    mlp_ratio: Tuple[float] = (3.0, 3.0, 4.0)
    num_heads: Tuple[int] = (None, None, 12)
    pos_emb_delay: int = 2

    drop_path_rate: float = 0.1

    norm_layer: Callable = linen.LayerNorm

    layer_norm_eps: float = 1e-6
    dtype: Any = jnp.float32

    def setup(self):
        num_layers = len(self.depths)
        norm_layer = partial(
            self.norm_layer,
            epsilon=self.layer_norm_eps,
            dtype=self.dtype,
        )

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer,
            dtype=self.dtype,
        )

        # stochastic depth with linear decay
        dpr = np.linspace(0, self.drop_path_rate, sum(self.depths))
        dpr = [float(x) for x in dpr]

        vit_body = []
        for i in range(num_layers):
            dpr_slice = tuple(dpr[sum(self.depths[:i]) : sum(self.depths[: i + 1])])
            layer = BasicLayer(
                depth=self.depths[i],
                mlp_ratio=self.mlp_ratio[i],
                num_heads=self.num_heads[i],
                pos_emb_enabled=i == self.pos_emb_delay,
                drop_path_ratio=dpr_slice,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i < num_layers - 1) else None,
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

        for layer in self.vit_body:
            x = layer(x, train=train)

        x = jnp.mean(x, axis=(1,))
        x = self.norm(x)
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


def hivit_tiny():
    config = {
        "depths": (1, 1, 10),
        "embed_dim": 96,
        "mlp_ratio": (3.0, 3.0, 4.0),
        "num_heads": (None, None, 6),
    }
    return HierarchicalViT(**config)


def hivit_small():
    config = {
        "depths": (2, 2, 20),
        "embed_dim": 96,
        "mlp_ratio": (3.0, 3.0, 4.0),
        "num_heads": (None, None, 6),
    }
    return HierarchicalViT(**config)


def hivit_base():
    config = {
        "depths": (2, 2, 20),
        "embed_dim": 128,
        "mlp_ratio": (3.0, 3.0, 4.0),
        "num_heads": (None, None, 6),
    }
    return HierarchicalViT(**config)
