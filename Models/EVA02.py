import dataclasses
from functools import partial
from typing import Callable, Optional

import einops
import jax.numpy as jnp
import jax.typing as jt
import numpy as np
from flax import linen


class VisionRotaryEmbeddingFast(linen.Module):
    """Apply Rotary Position Embeddings (RoPE)

    Most of the code comes from the original repo:
    https://github.com/baaivision/EVA/blob/master/EVA-02/asuka/rope.py
    """

    dim: int
    seq_len: int = 16
    theta: int = 10000

    @staticmethod
    def broadcat(tensors, dim=-1):
        num_tensors = len(tensors)
        shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
        assert (
            len(shape_lens) == 1
        ), "tensors must all have the same number of dimensions"
        shape_len = list(shape_lens)[0]
        dim = (dim + shape_len) if dim < 0 else dim
        dims = list(zip(*map(lambda t: list(t.shape), tensors)))
        expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
        assert all(
            [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
        ), "invalid dimensions for broadcastable concatentation"
        max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
        expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
        expanded_dims.insert(dim, (dim, dims[dim]))
        expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
        tensors = list(
            map(lambda t: np.broadcast_to(t[0], t[1]), zip(tensors, expandable_shapes))
        )
        return np.concatenate(tensors, axis=dim)

    @staticmethod
    def rotate_half(x):
        x = einops.rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x[..., 0], x[..., 1]
        x = jnp.stack((-x2, x1), axis=-1)
        x = einops.rearrange(x, "... d r -> ... (d r)")
        return x

    def setup(self):
        exp = np.arange(0, self.dim, 2) / -self.dim
        freqs = self.theta**exp

        t = np.arange(self.seq_len)

        freqs = np.einsum("..., f -> ... f", t, freqs)
        freqs = einops.repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = self.broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = np.reshape(np.cos(freqs), (-1, freqs.shape[-1]))
        freqs_sin = np.reshape(np.sin(freqs), (-1, freqs.shape[-1]))

        self.freqs_cos = self.variable(
            "eva02_constants",
            "freqs_cos",
            lambda: np.float32(freqs_cos),
        ).value
        self.freqs_sin = self.variable(
            "eva02_constants",
            "freqs_sin",
            lambda: np.float32(freqs_sin),
        ).value

    def __call__(self, x):
        return x * self.freqs_cos + self.rotate_half(x) * self.freqs_sin


class Attention(linen.Module):
    dim: int
    num_heads: int

    rope: Callable

    qkv_bias: bool = True
    proj_bias: bool = True

    attn_drop_ratio: float = 0.0
    proj_drop_ratio: float = 0.0

    dtype: jt.DTypeLike = jnp.float32

    def setup(self):
        self.qkv = linen.Dense(self.dim * 3, use_bias=self.qkv_bias, dtype=self.dtype)
        self.attn_drop = linen.Dropout(self.attn_drop_ratio)
        self.proj = linen.Dense(self.dim, use_bias=self.proj_bias, dtype=self.dtype)
        self.proj_drop = linen.Dropout(self.proj_drop_ratio)
        self.softmax = partial(linen.softmax, axis=-1)

    def __call__(self, x, train: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = (qkv[0], qkv[1], qkv[2])

        q_cls = q[:, :, :1, :]
        q_seq = q[:, :, 1:, :]
        q_seq = self.rope(q_seq).astype(v.dtype)
        q = jnp.concatenate([q_cls, q_seq], axis=2)

        k_cls = k[:, :, :1, :]
        k_seq = k[:, :, 1:, :]
        k_seq = self.rope(k_seq).astype(v.dtype)
        k = jnp.concatenate([k_cls, k_seq], axis=2)

        q = q / jnp.sqrt(q.shape[-1]).astype(q.dtype)
        attn = q @ jnp.transpose(k, (0, 1, 3, 2))

        attn = self.softmax(attn.astype(jnp.float32)).astype(self.dtype)
        attn = self.attn_drop(attn, deterministic=not train)

        x = jnp.transpose(attn @ v, (0, 2, 1, 3))
        x = jnp.reshape((x), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=not train)
        return x


class SwiGLU(linen.Module):
    hidden_features: int
    scale_mlp: bool
    norm_layer: Callable

    use_bias: bool = True

    act_layer: Callable = linen.silu
    drop_ratio: float = 0.0

    dtype: jt.DTypeLike = jnp.float32

    @linen.compact
    def __call__(self, x, train: bool):
        out_dim = x.shape[-1]

        x = linen.Dense(
            self.hidden_features * 2,
            use_bias=self.use_bias,
            dtype=self.dtype,
        )(x)
        x1 = x[..., : self.hidden_features]
        x2 = x[..., self.hidden_features :]

        x = self.act_layer(x1) * x2
        x = linen.Dropout(self.drop_ratio)(x, deterministic=not train)

        if self.scale_mlp:
            x = self.norm_layer()(x)

        x = linen.Dense(out_dim, use_bias=self.use_bias, dtype=self.dtype)(x)
        return x


class PosEmbed(linen.Module):
    dtype: jt.DTypeLike = jnp.float32

    @linen.compact
    def __call__(self, x):
        _, L, C = x.shape
        pos_emb_init = linen.initializers.normal(stddev=1 / np.sqrt(C))
        pos_emb = self.param("pos_emb", pos_emb_init, (1, L, C))
        pos_emb = pos_emb.astype(self.dtype)
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

    use_bias: bool = True

    norm_layer: Optional[Callable] = None

    dtype: jt.DTypeLike = jnp.float32

    @linen.compact
    def __call__(self, x):
        B, _, _, _ = x.shape
        patch_size = (self.patch_size, self.patch_size)

        x = linen.Conv(
            self.embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=self.use_bias,
            dtype=self.dtype,
        )(x)
        x = jnp.reshape(x, (B, -1, self.embed_dim))
        if self.norm_layer is not None:
            x = self.norm_layer()(x)
        return x


class EVA02TransformerBlock(linen.Module):
    mlp_dim: int
    num_heads: int
    drop_path_ratio: float

    scale_mlp: bool

    use_bias: bool

    norm_layer: Callable
    rope: Callable

    dtype: jt.DTypeLike = jnp.float32

    @linen.compact
    def __call__(self, x, train: bool = False):
        shortcut = x

        x = self.norm_layer()(x)
        x = Attention(
            dim=x.shape[-1],
            num_heads=self.num_heads,
            rope=self.rope,
            qkv_bias=self.use_bias,
            proj_bias=self.use_bias,
            dtype=self.dtype,
        )(x, train=train)
        x = linen.Dropout(
            rate=self.drop_path_ratio,
            broadcast_dims=(1, 2),
        )(x, deterministic=not train)
        x = shortcut + x

        shortcut = x
        x = self.norm_layer()(x)
        x = SwiGLU(
            hidden_features=self.mlp_dim,
            scale_mlp=self.scale_mlp,
            use_bias=self.use_bias,
            norm_layer=self.norm_layer,
            dtype=self.dtype,
        )(x, train=train)
        x = linen.Dropout(
            rate=self.drop_path_ratio,
            broadcast_dims=(1, 2),
        )(x, deterministic=not train)
        x = shortcut + x
        return x


class EVA02Transformer(linen.Module):
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 1000

    num_layers: int = 12
    embed_dim: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12

    scale_mlp: bool = True

    drop_path_rate: float = 0.1

    use_norm_bias: bool = True
    use_linear_bias: bool = True

    norm_layer: Callable = linen.LayerNorm

    layer_norm_eps: float = 1e-6
    dtype: jt.DTypeLike = jnp.float32

    def setup(self):
        norm_layer = partial(
            self.norm_layer,
            epsilon=self.layer_norm_eps,
            use_bias=self.use_norm_bias,
            dtype=self.dtype,
        )

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            use_bias=self.use_linear_bias,
            dtype=self.dtype,
        )

        cls_token_init = linen.initializers.truncated_normal(stddev=0.02)
        self.cls_token = self.param("cls_token", cls_token_init, (1, 1, self.embed_dim))

        self.pos_emb = PosEmbed(dtype=self.dtype)

        half_head_dim = self.embed_dim // self.num_heads // 2
        hw_seq_len = self.image_size // self.patch_size

        self.rope_emb = VisionRotaryEmbeddingFast(dim=half_head_dim, seq_len=hw_seq_len)

        # stochastic depth with linear decay
        dpr = np.linspace(0, self.drop_path_rate, self.num_layers)
        dpr = [float(x) for x in dpr]

        eva02_body = []
        for i_layer in range(self.num_layers):
            layer = EVA02TransformerBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                scale_mlp=self.scale_mlp,
                drop_path_ratio=dpr[i_layer],
                use_bias=self.use_linear_bias,
                norm_layer=norm_layer,
                rope=self.rope_emb,
                dtype=self.dtype,
            )
            eva02_body.append(layer)
        self.eva02_body = eva02_body

        self.norm = norm_layer()
        self.head = (
            linen.Dense(
                self.num_classes,
                use_bias=self.use_linear_bias,
                dtype=self.dtype,
            )
            if self.num_classes > 0
            else lambda x: x
        )

    def __call__(self, x, train: bool = False):
        x = self.patch_embed(x)

        B, N, C = x.shape
        b_cls = self.cls_token.astype(x.dtype)
        b_cls = jnp.broadcast_to(b_cls, (B, 1, C))
        x = jnp.concatenate([b_cls, x], axis=1)

        x = self.pos_emb(x)

        for layer in self.eva02_body:
            x = layer(x, train=train)

        x = x[:, 1:]
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

        parser.add_argument(
            "--enable-linear-bias",
            dest="use_linear_bias",
            help="Enable linear layers bias",
            action="store_true",
        )
        parser.add_argument(
            "--disable-linear-bias",
            dest="use_linear_bias",
            help="Disable linear layers bias",
            action="store_false",
        )
        parser.set_defaults(use_linear_bias=self.use_linear_bias)

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
        verdict = is_kernel
        return verdict


def eva02_small():
    config = {
        "num_layers": 12,
        "embed_dim": 384,
        "mlp_dim": (384 * 4 * 2) // 3,
        "num_heads": 6,
        "scale_mlp": False,
    }
    return EVA02Transformer(**config)


def eva02_base():
    config = {
        "num_layers": 12,
        "embed_dim": 768,
        "mlp_dim": (768 * 4 * 2) // 3,
        "num_heads": 12,
        "scale_mlp": True,
    }
    return EVA02Transformer(**config)


def eva02_large():
    config = {
        "num_layers": 24,
        "embed_dim": 1024,
        "mlp_dim": (1024 * 4 * 2) // 3,
        "num_heads": 16,
        "scale_mlp": True,
    }
    return EVA02Transformer(**config)
