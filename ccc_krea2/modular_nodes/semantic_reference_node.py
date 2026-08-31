"""Krea2 CcC Semantic Reference node."""

from typing import Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import SemanticReferenceChain, SemanticReferenceEntry


SEMANTIC_MODES = ("semantic_only", "style_direct", "style_indirect")
STYLE_PROCESSING = ("full", "2x2", "4x4")


class CcCKrea2SemanticReference:
    """Declare one Qwen semantic or style reference."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_SEMANTIC_REFERENCE_CHAIN",)
    RETURN_NAMES = ("semantic_references",)
    FUNCTION = "process"
    DESCRIPTION = "Adds one semantic/style reference using the controls previously exposed by Edit Advanced."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (SEMANTIC_MODES, {"default": "semantic_only"}),
                "instruction": ("STRING", {"multiline": True, "default": ""}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
                "processing": (STYLE_PROCESSING, {"default": "2x2"}),
                "fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "previous_references": ("KREA2_SEMANTIC_REFERENCE_CHAIN",),
            },
        }

    def process(
        self,
        image: torch.Tensor,
        mode: str = "semantic_only",
        instruction: str = "",
        grounding_px: int = 768,
        processing: str = "2x2",
        fidelity: float = 1.0,
        previous_references: Optional[SemanticReferenceChain] = None,
    ) -> Tuple[SemanticReferenceChain]:
        chain = previous_references if previous_references is not None else SemanticReferenceChain()
        entry = SemanticReferenceEntry(
            image=image,
            mode=mode,
            instruction=instruction.strip(),
            grounding_px=int(grounding_px),
            processing=processing,
            fidelity=float(fidelity),
        )
        return (chain.append(entry),)
