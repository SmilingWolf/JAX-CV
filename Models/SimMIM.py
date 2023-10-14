from functools import partial
from typing import Any, Tuple

import einops
import jax.numpy as jnp
from flax import linen

from .SwinV2 import SwinTransformerV2


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

        mask_tokens = jnp.broadcast_to(self.mask_token, (B, L, self.embed_dim))
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


class SimMIM(linen.Module):
    encoder: linen.Module
    encoder_stride: int

    patch_size: int
    in_chans: int = 3

    norm_targets_enabled: bool = True
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
        if self.norm_targets_enabled:
            x = WindowedNorm(target_size=(H, W), window_size=self.norm_patch_size)(x)

        x_rec = linen.dtypes.promote_dtype(x_rec, dtype=x.dtype)[0]
        loss_recon = jnp.abs(x - x_rec)
        loss = jnp.sum(loss_recon * mask) / (jnp.sum(mask) + 1e-5) / self.in_chans

        return loss, x_rec


def simmim_swinv2_tiny_window8_256(**kwargs):
    encoder = partial(
        SwinTransformerV2ForSimMIM,
        embed_dim=96,
        window_size=8,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
    )
    encoder = encoder(**kwargs)
    model = SimMIM(encoder, 32, encoder.patch_size, dtype=encoder.dtype)
    return model


def simmim_swinv2_base_window8_256(**kwargs):
    encoder = partial(
        SwinTransformerV2ForSimMIM,
        embed_dim=128,
        window_size=8,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
    )
    encoder = encoder(**kwargs)
    model = SimMIM(encoder, 32, encoder.patch_size, dtype=encoder.dtype)
    return model
