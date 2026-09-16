"""Krea2 CcC Visual Reference node."""

from typing import Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import VisualReferenceChain, VisualReferenceEntry


FIT_MODES = ("fit", "crop (legacy)")
ROPE_GRIDS = ("inside", "outside")
ROPE_HORIZONTAL = ("center", "left", "right")
ROPE_VERTICAL = ("center", "up", "down")


class CcCKrea2VisualReference:
    """Declare one ordered visual reference for Krea2 Edit."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_VISUAL_REFERENCE_CHAIN",)
    RETURN_NAMES = ("visual_references",)
    FUNCTION = "process"
    DESCRIPTION = (
        "Adds one visual Krea2 Edit reference. Chaining order is the physical Krea2 reference order. "
        "Fit mirrors the proven Krea2Moodboard fit-inside geometry and preserves the full source for genuine "
        "aspect-ratio mismatches; crop (legacy) center-crops to the target aspect ratio and fills the complete "
        "resolved target grid before VAE encoding. RoPE controls only coordinate placement."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "rope_grid": (ROPE_GRIDS, {"default": "inside"}),
                "rope_horizontal": (ROPE_HORIZONTAL, {"default": "center"}),
                "rope_vertical": (ROPE_VERTICAL, {"default": "center"}),
                "semantic": ("BOOLEAN", {"default": True}),
                "semantic_role": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Metadata label for reports and higher-level routing. Identity appearance references stay "
                            "positional in the Qwen stream; this label is not injected into Identity grounding text."
                        ),
                    },
                ),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Metadata instruction for higher-level routing. It is not injected into the positional "
                            "Identity grounding stream; use Semantic Reference for explicit Qwen-only instructions."
                        ),
                    },
                ),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
            },
            "optional": {
                "previous_references": ("KREA2_VISUAL_REFERENCE_CHAIN",),
                "fit_mode": (
                    FIT_MODES,
                    {
                        "default": "fit",
                        "tooltip": (
                            "fit mirrors Krea2Moodboard: near-matched sources fill the target, while genuine AR "
                            "mismatches keep the complete source on a centered fit-inside reference grid. "
                            "crop (legacy) center-crops to target AR and encodes a full-target-grid reference."
                        ),
                    },
                ),
            },
        }

    def process(
        self,
        image: torch.Tensor,
        boost: float = 1.0,
        rope_grid: str = "inside",
        rope_horizontal: str = "center",
        rope_vertical: str = "center",
        semantic: bool = True,
        semantic_role: str = "",
        instruction: str = "",
        grounding_px: int = 768,
        previous_references: Optional[VisualReferenceChain] = None,
        fit_mode: str = "fit",
    ) -> Tuple[VisualReferenceChain]:
        chain = previous_references if previous_references is not None else VisualReferenceChain()

        semantic_enabled = bool(semantic)
        entry = VisualReferenceEntry(
            image=image,
            boost=float(boost),
            fit_mode=fit_mode,
            rope_grid=rope_grid,
            rope_horizontal=rope_horizontal,
            rope_vertical=rope_vertical,
            semantic=semantic_enabled,
            semantic_role=semantic_role.strip() if semantic_enabled else "",
            instruction=instruction.strip() if semantic_enabled else "",
            grounding_px=int(grounding_px),
        )
        return (chain.append(entry),)
