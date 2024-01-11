import dataclasses
from typing import Any, Tuple

import einops
import jax.numpy as jnp
from flax import linen

from .ConvNext import ConvNext
from .HiViT import HierarchicalViT
from .SwinV2 import SwinTransformerV2
from .ViT import VisionTransformer


class WindowedNorm(linen.Module):
    target_size: Tuple[int]
    window_size: int = 47

    def get_targets_count(self):
        window_shape = (self.window_size, self.window_size)
        padding = (
            (self.window_size // 2, self.window_size // 2),
            (self.window_size // 2, self.window_size // 2),
        )

        targets_count = jnp.ones((1, self.target_size[0], self.target_size[1], 1))

        targets_count = linen.avg_pool(
            targets_count,
            window_shape=window_shape,
            strides=(1, 1),
            padding=padding,
            count_include_pad=True,
        )
        targets_count = targets_count * jnp.power(self.window_size, 2.0)
        targets_count = jnp.int32(jnp.rint(targets_count))
        return targets_count

    def setup(self):
        self.targets_count = self.variable(
            "simmim_constants",
            "targets_count",
            self.get_targets_count,
        ).value

    def __call__(self, targets):
        window_size = self.window_size

        window_shape = (window_size, window_size)
        padding = (
            (window_size // 2, window_size // 2),
            (window_size // 2, window_size // 2),
        )

        targets_ = targets

        targets_square = jnp.power(targets, 2.0)

        targets_mean = linen.avg_pool(
            targets,
            window_shape=window_shape,
            strides=(1, 1),
            padding=padding,
            count_include_pad=False,
        )
        targets_square_mean = linen.avg_pool(
            targets_square,
            window_shape=window_shape,
            strides=(1, 1),
            padding=padding,
            count_include_pad=False,
        )

        targets_var = targets_square_mean - jnp.power(targets_mean, 2.0)
        targets_var = targets_var * (self.targets_count / (self.targets_count - 1))
        targets_var = jnp.maximum(targets_var, 0.0)

        targets_ = (targets_ - targets_mean) / jnp.sqrt(targets_var + 1.0e-6)

        return targets_


class SwinTransformerV2ForSimMIM(SwinTransformerV2):
    def setup(self):
        super().setup()

        token_init = linen.initializers.normal(0.02)
        self.mask_token = self.param("mask_token", token_init, (1, 1, self.embed_dim))

    def __call__(self, x, mask, train: bool = False):
        x = self.patch_embed(x)

        B, L, _ = x.shape
        mask_token = linen.dtypes.promote_dtype(self.mask_token, dtype=self.dtype)[0]
        mask_tokens = jnp.broadcast_to(mask_token, (B, L, self.embed_dim))
        mask = jnp.reshape(mask, (B, L, 1)).astype(mask_tokens.dtype)
        x = x * (1.0 - mask) + mask_tokens * mask

        x = self.pos_drop(x, deterministic=not train)

        for layer in self.swin_body:
            x = layer(x, train=train)

        x = self.norm(x)

        B, L, C = x.shape
        H = W = int(L**0.5)
        x = jnp.reshape(x, (B, H, W, C))
        return x

    def get_stride(self):
        return self.patch_size * 2 ** (len(self.depths) - 1)


class VisionTransformerForSimMIM(VisionTransformer):
    def setup(self):
        super().setup()

        token_init = linen.initializers.normal(0.02)
        self.mask_token = self.param("mask_token", token_init, (1, 1, self.embed_dim))

    def __call__(self, x, mask, train: bool = False):
        x = self.patch_embed(x)

        B, L, _ = x.shape
        mask_tokens = jnp.broadcast_to(self.mask_token, (B, L, self.embed_dim))
        mask = jnp.reshape(mask, (B, L, 1)).astype(mask_tokens.dtype)
        x = x * (1.0 - mask) + mask_tokens * mask

        x = self.pos_emb(x)

        for layer in self.vit_body:
            x = layer(x, train=train)

        x = self.norm(x)

        B, L, C = x.shape
        H = W = int(L**0.5)
        x = jnp.reshape(x, (B, H, W, C))
        return x

    def get_stride(self):
        return self.patch_size


class HierarchicalViTForSimMIM(HierarchicalViT):
    def setup(self):
        super().setup()

        token_init = linen.initializers.normal(0.02)
        self.mask_token = self.param("mask_token", token_init, (1, 1, self.embed_dim))

    def __call__(self, x, mask, train: bool = False):
        x = self.patch_embed(x)

        B, L, _ = x.shape
        H = W = int(L**0.5)
        mask_token = linen.dtypes.promote_dtype(self.mask_token, dtype=self.dtype)[0]
        mask_tokens = jnp.broadcast_to(mask_token, (B, L, self.embed_dim))
        mask = jnp.reshape(mask, (B, H, W, 1)).astype(mask_tokens.dtype)
        mask = self.patch_embed.patches_reshape(mask)
        x = x * (1.0 - mask) + mask_tokens * mask

        for layer in self.vit_body:
            x = layer(x, train=train)

        x = self.norm(x)

        B, L, C = x.shape
        H = W = int(L**0.5)
        x = jnp.reshape(x, (B, H, W, C))
        return x

    def get_stride(self):
        return 16


class ConvNextForSimMIM(ConvNext):
    def setup(self):
        super().setup()

        token_init = linen.initializers.normal(0.02)
        self.mask_token = self.param("mask_token", token_init, (self.embed_dims[0],))

    def __call__(self, x, mask, train: bool = False):
        x = self.patch_embed(x)

        B, H, W, _ = x.shape
        mask_tokens = jnp.broadcast_to(self.mask_token, (B, H, W, self.embed_dims[0]))
        mask = jnp.reshape(mask, (B, H, W, 1)).astype(mask_tokens.dtype)
        x = x * (1.0 - mask) + mask_tokens * mask

        for layer in self.convnext_body:
            x = layer(x, train=train)

        x = self.norm(x)
        return x

    def get_stride(self):
        return 32


class SimMIM(linen.Module):
    encoder: linen.Module = SwinTransformerV2ForSimMIM
    encoder_stride: int = 32

    patch_size: int = 4

    enable_windowed_norm: bool = False
    norm_patch_size: int = 47

    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x, mask, train: bool = False):
        z = self.encoder(x, mask, train)
        x_rec = linen.Conv(
            features=self.encoder_stride**2 * 3,
            kernel_size=(1, 1),
            dtype=self.dtype,
        )(z)
        x_rec = einops.rearrange(
            x_rec,
            pattern="... h w (c b1 b2) -> ... (h b1) (w b2) c",
            b1=self.encoder_stride,
            b2=self.encoder_stride,
        )

        mask = jnp.expand_dims(
            jnp.repeat(
                jnp.repeat(mask, self.patch_size, axis=1),
                self.patch_size,
                axis=2,
            ),
            axis=-1,
        )

        B, H, W, C = x.shape
        if self.enable_windowed_norm:
            x = WindowedNorm(target_size=(H, W), window_size=self.norm_patch_size)(x)

        x_rec = linen.dtypes.promote_dtype(x_rec, dtype=x.dtype)[0]
        loss_recon = jnp.abs(x - x_rec)
        loss = jnp.sum(loss_recon * mask) / (jnp.sum(mask) + 1e-5) / C

        return loss, x_rec

    @classmethod
    def build(cls, config, **kwargs):
        encoder = config.encoder.build(config.encoder, **kwargs)

        config = dataclasses.asdict(config)
        config = {key: kwargs[key] if key in kwargs else config[key] for key in config}
        config["encoder"] = encoder
        config["encoder_stride"] = encoder.get_stride()
        return cls(**config)

    def extend_parser(self, parser):
        parser = self.encoder.extend_parser(parser)
        parser.add_argument(
            "--enable-windowed-norm",
            action="store_true",
            help="Use windowed norm of input images as reconstruction target in SimMIM",
        )
        return parser

    def should_decay(self, path, _):
        if path[0].key == "encoder":
            return self.encoder.should_decay(path[1:], _)

        is_kernel = path[-1].key == "kernel"
        verdict = is_kernel
        return verdict


def simmim_swinv2_tiny():
    config = {
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
    }
    encoder = SwinTransformerV2ForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.get_stride(),
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)


def simmim_swinv2_base():
    config = {
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
    }
    encoder = SwinTransformerV2ForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.get_stride(),
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)


def simmim_swinv2_large():
    config = {
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
    }
    encoder = SwinTransformerV2ForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.get_stride(),
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)


def simmim_vit_small():
    config = {
        "num_layers": 12,
        "embed_dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
    }
    encoder = VisionTransformerForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.get_stride(),
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)


def simmim_vit_base():
    config = {
        "num_layers": 12,
        "embed_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
    }
    encoder = VisionTransformerForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.patch_size,
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)


def simmim_vit_large():
    config = {
        "num_layers": 24,
        "embed_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    }
    encoder = VisionTransformerForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.patch_size,
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)


def simmim_hivit_tiny():
    config = {
        "depths": (1, 1, 10),
        "embed_dim": 96,
        "mlp_ratio": (3.0, 3.0, 4.0),
        "num_heads": (None, None, 6),
    }
    encoder = HierarchicalViTForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.get_stride(),
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)


def simmim_hivit_small(**kwargs):
    config = {
        "depths": (2, 2, 20),
        "embed_dim": 96,
        "mlp_ratio": (3.0, 3.0, 4.0),
        "num_heads": (None, None, 6),
    }
    encoder = HierarchicalViTForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.get_stride(),
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)


def simmim_convnext_tiny(**kwargs):
    config = {
        "embed_dims": (96, 192, 384, 768),
        "depths": (3, 3, 9, 3),
    }
    encoder = ConvNextForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.get_stride(),
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)


def simmim_convnext_small(**kwargs):
    config = {
        "embed_dims": (96, 192, 384, 768),
        "depths": (3, 3, 27, 3),
    }
    encoder = ConvNextForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.get_stride(),
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)


def simmim_convnext_base(**kwargs):
    config = {
        "embed_dims": (128, 256, 512, 1024),
        "depths": (3, 3, 27, 3),
    }
    encoder = ConvNextForSimMIM(**config)

    config = {
        "encoder": encoder,
        "encoder_stride": encoder.get_stride(),
        "patch_size": encoder.patch_size,
    }
    return SimMIM(**config)
