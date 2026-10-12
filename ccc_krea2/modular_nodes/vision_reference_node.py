"""Krea2 CcC Vision Reference node."""

from typing import Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import VisionReferenceChain, VisionReferenceEntry


VISION_MODES = ("semantic", "visual", "style", "context")
STYLE_PROCESSING = ("full", "2x2", "4x4")

VISION_RUNTIME_MODES = {
    "semantic": "semantic_only",
    "visual": "style_direct",
    "style": "style_indirect",
    "context": "context_indirect",
    "semantic_reference": "semantic_only",
    "visual_reference": "style_direct",
    "style_reference": "style_indirect",
    "semantic_only": "semantic_only",
    "style_direct": "style_direct",
    "style_indirect": "style_indirect",
    "context_indirect": "context_indirect",
}


class CcCKrea2VisionReference:
    """Declare one Qwen/Vision-only reference without an appearance latent."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_VISION_REFERENCE_CHAIN", "STRING")
    RETURN_NAMES = ("vision_references", "vision_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Adds one Vision Reference interpreted as Semantic, Visual, Style, or Context. "
        "Vision References are Qwen-only and never add a VAE/LoRA appearance latent."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (
                    VISION_MODES,
                    {
                        "default": "semantic",
                        "tooltip": (
                            "semantic = content/structure only; visual = direct visual guidance; "
                            "style = aesthetic influence; context = situation/relationships without visual copying."
                        ),
                    },
                ),
                "instruction": ("STRING", {"multiline": True, "default": ""}),
                "style_processing": (
                    STYLE_PROCESSING,
                    {
                        "default": "full",
                        "tooltip": "Used only by Style. Semantic, Visual, and Context always use the full image.",
                    },
                ),
            },
            "optional": {
                "previous_references": ("KREA2_VISION_REFERENCE_CHAIN",),
            },
        }

    def process(
        self,
        image: torch.Tensor,
        mode: str = "semantic",
        instruction: str = "",
        style_processing: str = "full",
        previous_references: Optional[VisionReferenceChain] = None,
    ) -> Tuple[VisionReferenceChain, str]:
        chain = previous_references if previous_references is not None else VisionReferenceChain()
        runtime_mode = VISION_RUNTIME_MODES.get(mode, mode)
        effective_processing = style_processing if runtime_mode == "style_indirect" else "full"
        entry = VisionReferenceEntry(
            image=image,
            mode=runtime_mode,
            instruction=instruction.strip(),
            processing=effective_processing,
        )
        out_chain = chain.append(entry)

        if image.ndim == 3:
            source_height, source_width = int(image.shape[0]), int(image.shape[1])
        else:
            source_height, source_width = int(image.shape[1]), int(image.shape[2])

        physical_images = {
            "full": 1,
            "2x2": 4,
            "4x4": 16,
        }.get(effective_processing, 1)

        if runtime_mode == "semantic_only":
            behavior = "100% semantic/content extraction; direct Vision rows retained"
        elif runtime_mode == "style_direct":
            behavior = "direct Qwen visual representation; direct Vision rows retained"
        elif runtime_mode == "style_indirect":
            behavior = "style influence; direct Vision rows removed after Qwen encoding"
        elif runtime_mode == "context_indirect":
            behavior = "context influence; direct Vision rows removed after Qwen encoding"
        else:
            behavior = "unknown/custom runtime behavior"

        info = "\n".join(
            [
                "=== Krea2 CcC Vision Reference ===",
                f"Public Mode: {mode}",
                f"Runtime Mode: {runtime_mode}",
                f"Instruction: {instruction.strip()!r}" if instruction.strip() else "Instruction: <empty>",
                f"Source Resolution: {source_width} x {source_height}",
                f"Requested Style Processing: {style_processing}",
                f"Effective Processing: {effective_processing}",
                f"Physical Qwen Images: {physical_images}",
                f"Behavior: {behavior}",
                "Appearance Latent: no",
                f"Chain Entries: {len(out_chain.entries)}",
            ]
        )
        return out_chain, info
