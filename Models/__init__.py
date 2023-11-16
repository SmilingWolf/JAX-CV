from .SimMIM import (
    simmim_swinv2_base,
    simmim_swinv2_tiny,
    simmim_vit_base,
    simmim_vit_small,
)
from .SwinV2 import swinv2_base, swinv2_tiny
from .ViT import vit_base, vit_small

model_registry = {
    "swinv2_tiny": swinv2_tiny,
    "swinv2_base": swinv2_base,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "simmim_swinv2_tiny": simmim_swinv2_tiny,
    "simmim_swinv2_base": simmim_swinv2_base,
    "simmim_vit_small": simmim_vit_small,
    "simmim_vit_base": simmim_vit_base,
}
