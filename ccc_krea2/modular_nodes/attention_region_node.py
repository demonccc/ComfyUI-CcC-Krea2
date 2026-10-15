"""Krea2 CcC Attention Region node."""

from typing import Optional, Tuple

from ..attention_regions import AttentionRegionChain, build_attention_region
from ..constants import NODE_CATEGORY


class CcCKrea2AttentionRegion:
    """Declare one tagged rectangular attention region on the target."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_ATTENTION_REGION_CHAIN",)
    RETURN_NAMES = ("attention_regions",)
    FUNCTION = "process"
    DESCRIPTION = (
        "Adds one tagged target attention box. Coordinates are percentages of the target/content image. "
        "Chain multiple nodes for multiple regions. Krea2 CcC Latent carries the boxes through target "
        "geometry transforms; any crop that touches a box is rejected."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tag": (
                    "STRING",
                    {
                        "default": "subject",
                        "tooltip": "Unique name used by Visual Reference to bind a reference to this box.",
                    },
                ),
                "x": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Left edge as percentage of the target/content image.",
                    },
                ),
                "y": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Top edge as percentage of the target/content image.",
                    },
                ),
                "width": (
                    "FLOAT",
                    {
                        "default": 100.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Box width as percentage of the target/content image.",
                    },
                ),
                "height": (
                    "FLOAT",
                    {
                        "default": 100.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Box height as percentage of the target/content image.",
                    },
                ),
            },
            "optional": {
                "previous_regions": ("KREA2_ATTENTION_REGION_CHAIN",),
            },
        }

    def process(
        self,
        tag: str,
        x: float,
        y: float,
        width: float,
        height: float,
        previous_regions: Optional[AttentionRegionChain] = None,
    ) -> Tuple[AttentionRegionChain]:
        chain = previous_regions if previous_regions is not None else AttentionRegionChain()
        region = build_attention_region(tag, x, y, width, height)
        return (chain.append(region),)
