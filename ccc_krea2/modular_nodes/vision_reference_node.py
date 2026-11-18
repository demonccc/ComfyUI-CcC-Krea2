"""Qwen Vision reference: identical image processing; optional visual-token retention."""

from typing import Optional, Tuple
import torch
from ..constants import NODE_CATEGORY
from .edit_reference_types import VisionReferenceChain, VisionReferenceEntry

VISION_MODES = ("visual", "semantic")


class CcCKrea2VisionReference:
    """Add a Qwen reference. Semantic removes image rows after multimodal encoding."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_REFERENCE_CHAIN", "STRING")
    RETURN_NAMES = ("references", "vision_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Visual retains Qwen image tokens; Semantic encodes the same image but "
        "removes image tokens. Instruction is free-form and entirely optional."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (VISION_MODES, {"default": "visual"}),
                "instruction": ("STRING", {"multiline": True, "default": ""}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                    "tooltip": "Scale Qwen image rows only when Visual retains them."}),
            },
            "optional": {"previous_references": ("KREA2_REFERENCE_CHAIN",)},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, mode="visual", **kwargs):
        return True if mode in VISION_MODES else f"Unsupported Vision mode: {mode!r}"

    def process(
        self, image: torch.Tensor, mode: str = "visual", instruction: str = "",
        previous_references: Optional[VisionReferenceChain] = None,
        weight: float = 1.0,
    ) -> Tuple[VisionReferenceChain, str]:
        if mode not in VISION_MODES:
            raise ValueError(f"[Krea2 CcC Vision Reference] Unsupported mode: {mode!r}")
        if not 0.0 <= float(weight) <= 2.0:
            raise ValueError("[Krea2 CcC Vision Reference] vision_weight must be in [0, 2].")
        chain = previous_references if previous_references is not None else VisionReferenceChain()
        entry = VisionReferenceEntry(
            image=image, mode=mode, instruction=instruction.strip(), vision_weight=float(weight),
        )
        out = chain.append(entry)
        return out, "\n".join((
            "=== Krea2 CcC Vision Reference ===",
            f"Mode: {mode}",
            f"Visual Tokens: {'retained' if mode == 'visual' else 'removed after encoding'}",
            f"Instruction: {entry.instruction!r}" if entry.instruction else "Instruction: <none>",
            f"Vision Weight: {entry.vision_weight:.2f}",
            f"Chain Entries: {len(out.entries)}",
        ))
