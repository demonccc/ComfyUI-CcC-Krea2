"""Krea2 CcC Visual Reference node."""

from typing import Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import VisualReferenceChain, VisualReferenceEntry


REFERENCE_FIT = ("native", "crop", "contain", "fit")
PLACEMENT_GRIDS = ("inside", "outside")
GRID_HORIZONTAL = ("center", "left", "right")
GRID_VERTICAL = ("center", "up", "down")


def _validate_mask(mask: torch.Tensor, height: int, width: int, name: str, batch_size: int) -> torch.Tensor:
    """Masks share the image pixel grid; never silently resize to disguise misalignment."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or tuple(mask.shape[-2:]) != (height, width):
        raise ValueError(f"[Krea2 CcC Visual Reference] {name} must match its image {width}x{height}; received {tuple(mask.shape)}.")
    if mask.shape[0] not in (1, batch_size):
        raise ValueError(f"[Krea2 CcC Visual Reference] {name} batch must be 1 or {batch_size}, got {mask.shape[0]}.")
    return mask


class CcCKrea2VisualReference:
    """Declare one ordered visual reference for Krea2 Edit."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_REFERENCE_CHAIN",)
    RETURN_NAMES = ("references",)
    FUNCTION = "process"
    DESCRIPTION = (
        "Visual reference geometry: native keeps its original scale, crop selects a target-sized "
        "window without resizing, contain scales proportionally inside target bounds, and fit "
        "scales proportionally to cover the target then center-crops. Resizing uses bicubic "
        "with antialiasing and /16 grid alignment. Placement is independent of VAE geometry."
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
                "rope_mode": (("spatial", "collapsed"), {"default": "spatial"}),
                "reference_stack": (("overlay", "stacked"), {"default": "overlay"}),
                "rope_geometry": (("latent", "source"), {"default": "latent"}),
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
                    {"default": 768, "min": 32, "max": 4096, "step": 32},
                ),
                "prompt_annotation": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional label describing this VAE/RoPE reference (scene, subject, etc.). Used for Image N grounding when this reference is visible to Qwen.",
                    },
                ),
            },
            "optional": {
                "previous_references": ("KREA2_REFERENCE_CHAIN",),
                "reference_boost_mask": ("MASK", {"tooltip": "Optional mask drawn on this reference image. White pixels select reference tokens receiving boost; image geometry is applied to the mask before VAE."}),
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
        rope_mode: str = "spatial",
        reference_stack: str = "overlay",
        rope_geometry: str = "latent",
        semantic: bool = True,
        semantic_resize: bool = True,
        semantic_grounding_px: int = 768,
        prompt_annotation: str = "",
        previous_references: Optional[VisualReferenceChain] = None,
        reference_boost_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[VisualReferenceChain]:
        chain = previous_references if previous_references is not None else VisualReferenceChain()

        semantic_enabled = bool(semantic)
        if reference_boost_mask is not None:
            reference_boost_mask = _validate_mask(reference_boost_mask, int(image.shape[1]), int(image.shape[2]), "reference_boost_mask", int(image.shape[0]))

        from .rope_position import parse_rope_position
        effective_grid = "inside" if reference_fit == "crop" else placement_grid
        effective_geometry = "source" if reference_fit == "native" else rope_geometry
        parse_rope_position(f"{effective_grid}:{grid_horizontal_position}:{grid_vertical_position}:{reference_stack}:{rope_mode}:{effective_geometry}")
        entry = VisualReferenceEntry(
            image=image,
            boost=float(boost),
            reference_boost_mask=reference_boost_mask,
            reference_fit=reference_fit,
            placement_grid=effective_grid,
            rope_mode=rope_mode,
            reference_stack=reference_stack,
            rope_geometry=effective_geometry,
            grid_horizontal_position=grid_horizontal_position,
            grid_vertical_position=grid_vertical_position,
            semantic=semantic_enabled,
            semantic_resize=bool(semantic_resize) if semantic_enabled else False,
            semantic_grounding_px=int(semantic_grounding_px),
            prompt_annotation=prompt_annotation.strip(),
        )
        return (chain.append(entry),)
