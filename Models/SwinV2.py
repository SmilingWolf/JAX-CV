import dataclasses
from functools import partial
from typing import Any, Callable, Tuple, Union

import jax
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
        x = linen.Dropout(self.drop_ratio)(x, deterministic=not train)
        return x


class RelativePositionBias(linen.Module):
    window_size: Tuple[int]
    num_heads: int
    pretrained_window_size: Tuple[int]
    dtype: Any = jnp.float32

    def get_relative_coords_table(self):
        coords_h = np.arange(-(self.window_size[0] - 1), self.window_size[0])
        coords_w = np.arange(-(self.window_size[1] - 1), self.window_size[1])

        # 1, 2*Wh-1, 2*Ww-1, 2
        coords_table = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords_table = np.stack(coords_table)
        coords_table = np.transpose(coords_table, (1, 2, 0))
        coords_table = np.expand_dims(coords_table, 0)
        coords_table = np.float32(coords_table)

        if self.pretrained_window_size[0] > 0:
            coords_table[:, :, :, 0] = coords_table[:, :, :, 0] / (
                self.pretrained_window_size[0] - 1
            )
            coords_table[:, :, :, 1] = coords_table[:, :, :, 1] / (
                self.pretrained_window_size[1] - 1
            )
        else:
            coords_table[:, :, :, 0] = coords_table[:, :, :, 0] / (
                self.window_size[0] - 1
            )
            coords_table[:, :, :, 1] = coords_table[:, :, :, 1] / (
                self.window_size[1] - 1
            )

        # normalize to -8, 8
        coords_table = coords_table * 8
        coord_table_sign = np.sign(coords_table)
        coords_table = np.log2(np.abs(coords_table) + 1.0)
        coords_table = coord_table_sign * coords_table / np.log2(8)
        return coords_table

    def get_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])

        # 2, Wh, Ww
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))

        # 2, Wh*Ww
        coords_flatten = np.reshape(coords, (2, -1))

        # 2, Wh*Ww, Wh*Ww
        coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

        # Wh*Ww, Wh*Ww, 2
        coords = np.transpose(coords, (1, 2, 0))

        # shift to start from 0
        coords[:, :, 0] = coords[:, :, 0] + (self.window_size[0] - 1)
        coords[:, :, 1] = coords[:, :, 1] + (self.window_size[1] - 1)
        coords[:, :, 0] = coords[:, :, 0] * (2 * self.window_size[1] - 1)

        # Wh*Ww, Wh*Ww
        position_index = np.sum(coords, axis=-1)
        return position_index

    def setup(self):
        self.relative_coords_table = self.variable(
            "swinv2_constants",
            "relative_coords_table",
            self.get_relative_coords_table,
        ).value

        self.relative_position_index = self.variable(
            "swinv2_constants",
            "relative_position_index",
            self.get_relative_position_index,
        ).value

        # mlp to generate continuous relative position bias
        self.cpb_mlp = linen.Sequential(
            [
                linen.Dense(512, use_bias=True, dtype=self.dtype),
                linen.relu,
                linen.Dense(self.num_heads, use_bias=False, dtype=self.dtype),
            ],
        )

    def __call__(self, x):
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table)
        rpe_index = jnp.reshape(self.relative_position_index, (-1,))

        # Wh*Ww,Wh*Ww,nH
        relative_position_bias_table = jnp.reshape(
            relative_position_bias_table, (-1, self.num_heads)
        )
        relative_position_bias = jnp.reshape(
            relative_position_bias_table[rpe_index],
            (
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ),
        )

        # nH, Wh*Ww, Wh*Ww
        relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))
        relative_position_bias = 16 * linen.sigmoid(relative_position_bias)
        relative_position_bias = jnp.expand_dims(relative_position_bias, 0)
        x = x + relative_position_bias
        return x


def l2_normalize(x):
    rnorm = jax.lax.rsqrt(jnp.maximum(jnp.sum((x * x), axis=-1, keepdims=True), 1e-12))
    return x * rnorm


class WindowAttention(linen.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    dim: int
    window_size: Tuple[int]  # Wh, Ww
    num_heads: int
    qkv_bias: bool = True
    attn_drop_ratio: float = 0.0
    proj_drop_ratio: float = 0.0
    pretrained_window_size: Tuple[int] = (0, 0)
    dtype: Any = jnp.float32

    def setup(self):
        self.logit_scale = self.variable(
            "params",
            "logit_scale",
            lambda x: jnp.log(10 * jnp.ones((x, 1, 1))),
            self.num_heads,
        ).value

        self.qkv = linen.Dense(self.dim * 3, use_bias=False, dtype=self.dtype)
        if self.qkv_bias:
            bias_init = linen.initializers.zeros_init()
            self.q_bias = self.param("q_bias", bias_init, (self.dim,))
            self.v_bias = self.param("v_bias", bias_init, (self.dim,))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attention_bias = RelativePositionBias(
            self.window_size,
            self.num_heads,
            self.pretrained_window_size,
            dtype=self.dtype,
        )

        self.attn_drop = linen.Dropout(self.attn_drop_ratio)
        self.proj = linen.Dense(self.dim, dtype=self.dtype)
        self.proj_drop = linen.Dropout(self.proj_drop_ratio)
        self.softmax = partial(linen.activation.softmax, axis=-1)

    def __call__(self, x, train: bool, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x)
        if self.qkv_bias:
            q_bias, v_bias = linen.dtypes.promote_dtype(
                self.q_bias,
                self.v_bias,
                dtype=self.dtype,
            )
            qkv_bias = jnp.concatenate(
                (
                    q_bias,
                    jnp.zeros_like(v_bias),
                    v_bias,
                )
            )
            qkv = qkv + qkv_bias

        qkv = jnp.reshape(qkv, (B_, N, 3, self.num_heads, -1))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = (qkv[0], qkv[1], qkv[2])

        q_norm = l2_normalize(q)
        k_norm = l2_normalize(k)
        attn = q_norm @ jnp.transpose(k_norm, (0, 1, 3, 2))

        logit_scale = jnp.minimum(self.logit_scale, np.log(100.0))
        logit_scale = linen.dtypes.promote_dtype(logit_scale, dtype=self.dtype)[0]
        logit_scale = jnp.exp(logit_scale)
        attn = attn * logit_scale

        attn = self.attention_bias(attn)

        if mask is not None:
            nW = mask.shape[0]
            mask = linen.dtypes.promote_dtype(mask, dtype=self.dtype)[0]
            mask = jnp.expand_dims(jnp.expand_dims(mask, 1), 0)
            attn = jnp.reshape(attn, (B_ // nW, nW, self.num_heads, N, N)) + mask
            attn = jnp.reshape(attn, (-1, self.num_heads, N, N))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn, deterministic=not train)

        x = jnp.transpose(attn @ v, (0, 2, 1, 3))
        x = jnp.reshape(x, (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=not train)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    windows = jnp.reshape(
        x, (B, H // window_size, window_size, W // window_size, window_size, C)
    )
    windows = jnp.transpose(windows, (0, 1, 3, 2, 4, 5))
    windows = jnp.reshape(windows, (-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = jnp.reshape(
        windows, (B, H // window_size, W // window_size, window_size, window_size, -1)
    )
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    x = jnp.reshape(x, (B, H, W, -1))
    return x


class SwinTransformerBlock(linen.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """
    dim: int
    input_resolution: Tuple[int]
    num_heads: int
    window_size: int = 7
    shift_size: int = 0
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_ratio: float = 0.0
    attn_drop_ratio: float = 0.0
    drop_path_ratio: float = 0.0
    act_layer: Callable = linen.gelu
    norm_layer: Callable = linen.LayerNorm
    pretrained_window_size: int = 0
    dtype: Any = jnp.float32

    def setup(self):
        self.norm1 = self.norm_layer()
        self.attn = WindowAttention(
            self.dim,
            window_size=(self.window_size, self.window_size),
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop_ratio=self.attn_drop_ratio,
            proj_drop_ratio=self.drop_ratio,
            pretrained_window_size=(
                self.pretrained_window_size,
                self.pretrained_window_size,
            ),
            dtype=self.dtype,
        )

        self.drop_path = linen.Dropout(rate=self.drop_path_ratio, broadcast_dims=(1, 2))
        self.norm2 = self.norm_layer()
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(
            hidden_features=mlp_hidden_dim,
            act_layer=self.act_layer,
            drop_ratio=self.drop_ratio,
            dtype=self.dtype,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = jnp.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask = img_mask.at[:, h, w, :].set(cnt)
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = jnp.reshape(
                mask_windows, (-1, self.window_size * self.window_size)
            )
            attn_mask = jnp.expand_dims(mask_windows, 1) - jnp.expand_dims(
                mask_windows, 2
            )
            attn_mask = jnp.where(attn_mask != 0, float(-100.0), attn_mask)
            attn_mask = jnp.where(attn_mask == 0, float(0.0), attn_mask)
        else:
            attn_mask = None

        self.attn_mask = self.variable(
            "swinv2_constants", "attn_mask", lambda: attn_mask
        ).value

    def __call__(self, x, train: bool):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = jnp.reshape(x, (B, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = jnp.roll(
                x,
                shift=(-self.shift_size, -self.shift_size),
                axis=(1, 2),
            )
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)

        # nW*B, window_size*window_size, C
        x_windows = jnp.reshape(x_windows, (-1, self.window_size * self.window_size, C))

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, train=train, mask=self.attn_mask)

        # merge windows
        attn_windows = jnp.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, C),
        )
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = jnp.roll(
                shifted_x,
                shift=(self.shift_size, self.shift_size),
                axis=(1, 2),
            )
        else:
            x = shifted_x
        x = jnp.reshape(x, (B, H * W, C))
        x = shortcut + self.drop_path(self.norm1(x), deterministic=not train)

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x, train)), deterministic=not train)
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
        x = jnp.transpose(x, (0, 1, 3, 4, 2, 5))  # B H/2 W/2 nW nH C
        x = jnp.reshape(x, (B, (H // 2) * (W // 2), 4 * C))  # B H/2*W/2 4*C

        x = linen.Dense(2 * C, use_bias=False, dtype=self.dtype)(x)
        x = self.norm_layer()(x)

        return x


class BasicLayer(linen.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        pretrained_window_size (int): Local window size in pre-training.
    """

    dim: int
    input_resolution: Tuple[int]
    depth: int
    num_heads: int
    window_size: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_ratio: float = 0.0
    attn_drop_ratio: float = 0.0
    drop_path_ratio: Union[float, Tuple[float]] = 0.0
    norm_layer: Callable = linen.LayerNorm
    downsample: Callable = None
    pretrained_window_size: int = 0
    dtype: Any = jnp.float32

    @linen.compact
    def __call__(self, x, train: bool):
        for i in range(self.depth):
            window_size = self.window_size
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            drop_path_ratio = (
                self.drop_path_ratio[i]
                if isinstance(self.drop_path_ratio, tuple)
                else self.drop_path_ratio
            )

            # if window size is larger than input resolution, we don't partition windows
            if min(self.input_resolution) <= window_size:
                shift_size = 0
                window_size = min(self.input_resolution)

            x = SwinTransformerBlock(
                dim=self.dim,
                input_resolution=self.input_resolution,
                num_heads=self.num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_ratio=self.drop_ratio,
                attn_drop_ratio=self.attn_drop_ratio,
                drop_path_ratio=drop_path_ratio,
                norm_layer=self.norm_layer,
                pretrained_window_size=self.pretrained_window_size,
                dtype=self.dtype,
            )(x, train)

        # patch merging layer
        if self.downsample is not None:
            x = self.downsample(
                self.input_resolution,
                norm_layer=self.norm_layer,
                dtype=self.dtype,
            )(x)
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


class SwinTransformerV2(linen.Module):
    r"""Swin Transformer
        A JAX/Flax impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        image_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    image_size: int = 224
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 1000

    embed_dim: int = 96
    depths: Tuple[int] = (2, 2, 6, 2)
    num_heads: Tuple[int] = (3, 6, 12, 24)

    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True

    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    norm_layer: Callable = linen.LayerNorm
    patch_norm: bool = True

    pretrained_window_sizes: Tuple[int] = (0, 0, 0, 0)

    layer_norm_eps: float = 1e-5
    dtype: Any = jnp.float32

    def setup(self):
        depths = self.depths
        num_layers = len(depths)
        norm_layer = partial(
            self.norm_layer,
            epsilon=self.layer_norm_eps,
            dtype=self.dtype,
        )

        patch_resolution = self.image_size // self.patch_size
        patches_resolution = [patch_resolution, patch_resolution]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            dtype=self.dtype,
        )

        self.pos_drop = linen.Dropout(rate=self.drop_rate)

        # stochastic depth with linear decay
        dpr = [float(x) for x in np.linspace(0, self.drop_path_rate, sum(depths))]

        # build layers
        swin_body = []
        for i_layer in range(num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_ratio=self.drop_rate,
                attn_drop_ratio=self.attn_drop_rate,
                drop_path_ratio=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < num_layers - 1) else None,
                pretrained_window_size=self.pretrained_window_sizes[i_layer],
                dtype=self.dtype,
            )
            swin_body.append(layer)
        self.swin_body = swin_body

        self.norm = norm_layer()
        self.head = (
            linen.Dense(self.num_classes, dtype=self.dtype)
            if self.num_classes > 0
            else lambda x: x
        )

    def __call__(self, x, train: bool = False):
        x = self.patch_embed(x)
        x = self.pos_drop(x, deterministic=not train)

        for layer in self.swin_body:
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
        parser.set_defaults(image_size=self.image_size)
        parser.set_defaults(patch_size=self.patch_size)
        parser.add_argument(
            "--window-size",
            default=self.window_size,
            help="SwinV2 window size",
            type=int,
        )
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


def swinv2_tiny():
    config = {
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
    }
    return SwinTransformerV2(**config)


def swinv2_base(**kwargs):
    config = {
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
    }
    return SwinTransformerV2(**config)


def swinv2_large(**kwargs):
    config = {
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
    }
    return SwinTransformerV2(**config)
