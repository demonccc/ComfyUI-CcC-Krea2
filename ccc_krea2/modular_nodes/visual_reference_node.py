"""Krea2 CcC Visual Reference node."""

from typing import Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import VisualReferenceChain, VisualReferenceEntry


REFERENCE_FIT = ("crop", "resize", "native")
PLACEMENT_GRIDS = ("inside", "outside")
GRID_HORIZONTAL = ("center", "left", "right")
GRID_VERTICAL = ("center", "up", "down")
RESIZE_METHODS = ("lanczos", "bicubic", "bilinear", "area")


class CcCKrea2VisualReference:
    """Declare one ordered visual reference for Krea2 Edit."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_VISUAL_REFERENCE_CHAIN",)
    RETURN_NAMES = ("visual_references",)
    FUNCTION = "process"
    DESCRIPTION = (
        "Adds one ordered Krea2 Edit appearance reference. reference_fit controls only the pixel path sent to VAE: "
        "crop uses the target grid as an inside crop window positioned by the grid controls, resize always scales "
        "proportionally to the target grid longest edge, and native preserves source size except for minimum VAE "
        "alignment. semantic controls whether the same image is also shown to Qwen. prompt_annotation optionally "
        "adds 'Image N: ...' text after the vision prefix."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "reference_fit": (REFERENCE_FIT, {"default": "native"}),
                "placement_grid": (
                    PLACEMENT_GRIDS,
                    {
                        "default": "inside",
                        "tooltip": "RoPE placement grid. crop always forces inside.",
                    },
                ),
                "grid_horizontal_position": (GRID_HORIZONTAL, {"default": "center"}),
                "grid_vertical_position": (GRID_VERTICAL, {"default": "center"}),
                "resize_method": (
                    RESIZE_METHODS,
                    {
                        "default": "lanczos",
                        "tooltip": "Interpolation used only when reference_fit is resize.",
                    },
                ),
                "semantic": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, this appearance reference is also sent to Qwen.",
                    },
                ),
                "semantic_resize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, Qwen input is downscaled only when it exceeds semantic_grounding_px. Never upscales.",
                    },
                ),
                "semantic_grounding_px": (
                    "INT",
                    {"default": 768, "min": 16, "max": 4096, "step": 16},
                ),
                "semantic_resize_method": (
                    RESIZE_METHODS,
                    {
                        "default": "lanczos",
                        "tooltip": "Interpolation used only for semantic downscaling.",
                    },
                ),
                "prompt_annotation": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional text appended as 'Image N: <annotation>' for this Qwen image.",
                    },
                ),
            },
            "optional": {
                "previous_references": ("KREA2_VISUAL_REFERENCE_CHAIN",),
            },
        }

    def process(
        self,
        image: torch.Tensor,
        boost: float = 1.0,
        reference_fit: str = "native",
        placement_grid: str = "inside",
        grid_horizontal_position: str = "center",
        grid_vertical_position: str = "center",
        resize_method: str = "lanczos",
        semantic: bool = True,
        semantic_resize: bool = True,
        semantic_grounding_px: int = 768,
        semantic_resize_method: str = "lanczos",
        prompt_annotation: str = "",
        previous_references: Optional[VisualReferenceChain] = None,
    ) -> Tuple[VisualReferenceChain]:
        chain = previous_references if previous_references is not None else VisualReferenceChain()

        semantic_enabled = bool(semantic)
        entry = VisualReferenceEntry(
            image=image,
            boost=float(boost),
            reference_fit=reference_fit,
            placement_grid="inside" if reference_fit == "crop" else placement_grid,
            grid_horizontal_position=grid_horizontal_position,
            grid_vertical_position=grid_vertical_position,
            resize_method=resize_method,
            semantic=semantic_enabled,
            semantic_resize=bool(semantic_resize) if semantic_enabled else False,
            semantic_grounding_px=int(semantic_grounding_px),
            semantic_resize_method=semantic_resize_method,
            prompt_annotation=prompt_annotation.strip() if semantic_enabled else "",
        )
        return (chain.append(entry),)
