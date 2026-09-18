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
    DESCRIPTION = "Adds one Qwen-only semantic/style reference. semantic_only extracts subject/content information without adding a VAE/LoRA reference."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (SEMANTIC_MODES, {"default": "semantic_only"}),
                "instruction": ("STRING", {"multiline": True, "default": ""}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
                "processing": (STYLE_PROCESSING, {"default": "full", "tooltip": "semantic_only always uses full image; crop/tile processing is for style modes."}),
                "fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Krea Mood strength. 1.0 keeps raw Qwen vision rows; lower values apply stronger subject/content extraction."}),
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
        processing: str = "full",
        fidelity: float = 0.5,
        previous_references: Optional[SemanticReferenceChain] = None,
    ) -> Tuple[SemanticReferenceChain]:
        chain = previous_references if previous_references is not None else SemanticReferenceChain()
        # semantic_only extracts subject/content information. It must use the
        # full image so pose/composition/background remain available to Qwen.
        effective_processing = "full" if mode == "semantic_only" else processing
        entry = SemanticReferenceEntry(
            image=image,
            mode=mode,
            instruction=instruction.strip(),
            grounding_px=int(grounding_px),
            processing=effective_processing,
            fidelity=float(fidelity),
        )
        return (chain.append(entry),)
