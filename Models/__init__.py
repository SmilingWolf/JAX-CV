from .SimMIM import (
    simmim_swinv2_base_window8_256,
    simmim_swinv2_tiny_window8_256,
    simmim_vit_base,
    simmim_vit_small,
)
from .SwinV2 import swinv2_base_window8_256, swinv2_tiny_window8_256
from .ViT import vit_base, vit_large, vit_small

model_registry = {
    "swinv2_tiny": swinv2_tiny_window8_256,
    "swinv2_base": swinv2_base_window8_256,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "simmim_swinv2_tiny": simmim_swinv2_tiny_window8_256,
    "simmim_swinv2_base": simmim_swinv2_base_window8_256,
    "simmim_vit_small": simmim_vit_small,
    "simmim_vit_base": simmim_vit_base,
}
