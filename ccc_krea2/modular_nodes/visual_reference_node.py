"""Krea2 CcC Visual Reference node."""

from typing import Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import VisualReferenceChain, VisualReferenceEntry


ROPE_POSITIONS = ("none", "up", "down", "left", "right")


class CcCKrea2VisualReference:
    """Declare one ordered visual reference for Krea2 Edit."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_VISUAL_REFERENCE_CHAIN",)
    RETURN_NAMES = ("visual_references",)
    FUNCTION = "process"
    DESCRIPTION = (
        "Adds one visual Krea2 Edit reference. Chaining order is the physical Krea2 reference order. "
        "When Semantic is enabled, Semantic Role can optionally name the image for Qwen; an empty role "
        "keeps the original positional Krea2 Edit behavior."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "rope_position": (ROPE_POSITIONS, {"default": "none"}),
                "semantic": ("BOOLEAN", {"default": True}),
                "semantic_role": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional Qwen semantic name for this image, for example 'scene image' or "
                            "'subject image'. Empty keeps positional Krea2 Edit behavior."
                        ),
                    },
                ),
                "instruction": ("STRING", {"multiline": True, "default": ""}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
            },
            "optional": {
                "previous_references": ("KREA2_VISUAL_REFERENCE_CHAIN",),
            },
        }

    def process(
        self,
        image: torch.Tensor,
        boost: float = 1.0,
        rope_position: str = "none",
        semantic: bool = True,
        semantic_role: str = "",
        instruction: str = "",
        grounding_px: int = 768,
        previous_references: Optional[VisualReferenceChain] = None,
    ) -> Tuple[VisualReferenceChain]:
        chain = previous_references if previous_references is not None else VisualReferenceChain()

        semantic_enabled = bool(semantic)
        entry = VisualReferenceEntry(
            image=image,
            boost=float(boost),
            rope_position=rope_position,
            semantic=semantic_enabled,
            semantic_role=semantic_role.strip() if semantic_enabled else "",
            instruction=instruction.strip() if semantic_enabled else "",
            grounding_px=int(grounding_px),
        )
        return (chain.append(entry),)
