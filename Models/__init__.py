from .SwinV2 import swinv2_base_window8_256, swinv2_tiny_window8_256
from .ViT import vit_base, vit_large, vit_small

model_registry = {
    "swinv2_tiny": swinv2_tiny_window8_256,
    "swinv2_base": swinv2_base_window8_256,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
}
