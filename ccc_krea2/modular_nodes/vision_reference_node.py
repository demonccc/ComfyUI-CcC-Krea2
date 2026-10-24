"""Krea2 CcC Vision Reference node."""

import random
from typing import Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from ..qwen_guidance import (
    GUIDANCE_DETAIL_MODES,
    GUIDANCE_SEED_MODES,
    MAX_GUIDANCE_SEED,
    VISION_GUIDANCE_MODES,
)
from .edit_reference_types import VisionReferenceChain, VisionReferenceEntry


VISION_MODES = ("semantic", "visual", "style", "context", "native")
STYLE_PROCESSING = ("full", "2x2", "4x4")

VISION_RUNTIME_MODES = {
    "semantic": "semantic_only",
    "visual": "style_direct",
    "style": "style_indirect",
    "context": "context_indirect",
    "native": "native",
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
    RETURN_TYPES = ("KREA2_REFERENCE_CHAIN", "STRING")
    RETURN_NAMES = ("references", "vision_info")
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
                            "semantic = content/meaning; visual = near-copy/full visual blueprint with strong pose/composition anchoring and no subject duplication; "
                            "style = legacy Visual transfer behavior applied strongly as the dominant visual treatment; "
                            "context = mandatory scene staging with pose, gaze, limbs/hands, gesture, camera/framing, interaction, and spatial relationships."
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
                "guidance_seed_mode": (
                    GUIDANCE_SEED_MODES,
                    {"default": "fixed"},
                ),
                "guidance_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_GUIDANCE_SEED, "step": 1},
                ),
                "guidance_detail": (
                    GUIDANCE_DETAIL_MODES,
                    {
                        "default": "balanced",
                        "tooltip": (
                            "Extraction depth: compact keeps only highest-impact facts; balanced captures all major and "
                            "secondary facts; detailed is exhaustive within the selected Vision mode."
                        ),
                    },
                ),
                "vision_guidance": (
                    VISION_GUIDANCE_MODES,
                    {
                        "default": "balanced",
                        "tooltip": (
                            "Qwen sampling behavior: conservative locks closely to the reference; balanced preserves it stably; "
                            "flexible allows pose/composition and other interpretation to move more freely where the mode permits it."
                        ),
                    },
                ),
            },
            "optional": {
                "previous_references": ("KREA2_REFERENCE_CHAIN",),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        guidance_seed_mode: str = "fixed",
        guidance_seed: int = 0,
        guidance_detail: str = "balanced",
        vision_guidance: str = "balanced",
        **kwargs,
    ):
        if guidance_seed_mode in {"incremental", "random"}:
            return float("nan")
        return (
            f"guidance-seed:{int(guidance_seed)}:"
            f"detail:{guidance_detail}:vision:{vision_guidance}"
        )

    def _resolve_guidance_seed(self, mode: str, configured_seed: int) -> int:
        seed = int(configured_seed)
        if not 0 <= seed <= MAX_GUIDANCE_SEED:
            raise ValueError(
                f"[Krea2 CcC Vision Reference] Guidance seed must be between 0 and {MAX_GUIDANCE_SEED}."
            )

        if mode == "fixed":
            self._guidance_seed_last_mode = mode
            return seed

        if mode == "incremental":
            if getattr(self, "_guidance_seed_last_mode", None) != "incremental":
                self._guidance_seed_counter = seed
            resolved = int(self._guidance_seed_counter)
            self._guidance_seed_counter = (resolved + 1) & MAX_GUIDANCE_SEED
            self._guidance_seed_last_mode = mode
            return resolved

        if mode == "random":
            self._guidance_seed_last_mode = mode
            return random.SystemRandom().randint(0, MAX_GUIDANCE_SEED)

        raise ValueError(
            f"[Krea2 CcC Vision Reference] Unsupported guidance seed mode: {mode!r}."
        )

    def process(
        self,
        image: torch.Tensor,
        mode: str = "semantic",
        instruction: str = "",
        style_processing: str = "full",
        guidance_seed_mode: str = "fixed",
        guidance_seed: int = 0,
        guidance_detail: str = "balanced",
        vision_guidance: str = "balanced",
        previous_references: Optional[VisionReferenceChain] = None,
    ) -> Tuple[VisionReferenceChain, str]:
        chain = previous_references if previous_references is not None else VisionReferenceChain()
        runtime_mode = VISION_RUNTIME_MODES.get(mode, mode)
        effective_processing = style_processing if runtime_mode == "style_indirect" else "full"
        resolved_guidance_seed = self._resolve_guidance_seed(
            guidance_seed_mode,
            guidance_seed,
        )
        entry = VisionReferenceEntry(
            image=image,
            mode=runtime_mode,
            instruction=instruction.strip(),
            processing=effective_processing,
            guidance_seed_mode=guidance_seed_mode,
            guidance_seed_configured=int(guidance_seed),
            guidance_seed=resolved_guidance_seed,
            guidance_detail=guidance_detail,
            vision_guidance=vision_guidance,
        )
        out_chain = chain.append(entry)

        if image.ndim == 3:
            source_height, source_width = int(image.shape[0]), int(image.shape[1])
        else:
            source_height, source_width = int(image.shape[1]), int(image.shape[2])

        info = "\n".join(
            [
                "=== Krea2 CcC Vision Reference ===",
                f"Mode: {mode}",
                f"Internal Mode: {runtime_mode}",
                f"Instruction: {instruction.strip()!r}" if instruction.strip() else "Instruction: <empty>",
                f"Source Resolution: {source_width} x {source_height}",
                f"Requested Style Processing: {style_processing}",
                f"Effective Style Processing: {effective_processing}",
                f"Guidance Seed Mode: {guidance_seed_mode}",
                f"Guidance Seed Configured: {int(guidance_seed)}",
                f"Guidance Seed Used: {resolved_guidance_seed}",
                f"Guidance Detail: {guidance_detail}",
                f"Vision Guidance: {vision_guidance}",
                "Appearance Latent: no",
                "Consumer Processing: deferred to connected node",
                f"Chain Entries: {len(out_chain.entries)}",
            ]
        )
        return out_chain, info
