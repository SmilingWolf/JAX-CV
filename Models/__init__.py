from .HiViT import hivit_base, hivit_small, hivit_tiny
from .SimMIM import (
    simmim_hivit_small,
    simmim_hivit_tiny,
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
    "hivit_tiny": hivit_tiny,
    "hivit_small": hivit_small,
    "hivit_base": hivit_base,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "simmim_swinv2_tiny": simmim_swinv2_tiny,
    "simmim_swinv2_base": simmim_swinv2_base,
    "simmim_vit_small": simmim_vit_small,
    "simmim_vit_base": simmim_vit_base,
    "simmim_hivit_tiny": simmim_hivit_tiny,
    "simmim_hivit_small": simmim_hivit_small,
}
