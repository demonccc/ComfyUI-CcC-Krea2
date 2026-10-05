"""Public Prompt Creator contract with optional visual references."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .edit_prompt_creator_node import CcCKrea2EditPromptCreator as _BasePromptCreator
from .edit_reference_types import VisualReferenceChain


class CcCKrea2EditPromptCreator(_BasePromptCreator):
    """Expose visual_references as optional and isolate create_from_image analysis."""

    @classmethod
    def INPUT_TYPES(cls):
        schema = _BasePromptCreator.INPUT_TYPES()
        required = dict(schema.get("required", {}))
        visual_references = required.pop("visual_references")
        optional = {
            "visual_references": visual_references,
            **dict(schema.get("optional", {})),
        }
        return {
            "required": required,
            "optional": optional,
        }

    def create(
        self,
        clip: Any,
        mode: str = "enhance",
        user_prompt: str = "",
        system_prompt: str = "",
        thinking: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.25,
        top_p: float = 0.90,
        seed: int = 0,
        visual_references: Optional[VisualReferenceChain] = None,
        reference_edit_image: Optional[torch.Tensor] = None,
    ):
        passthrough_references = (
            visual_references if visual_references is not None else VisualReferenceChain()
        )

        # create_from_image has one authoritative visual blueprint: reference_edit_image.
        # Do not expose downstream Visual Reference pixels to Qwen in this mode, otherwise
        # the model can confuse Image 1 / Image 2 with the internal edit blueprint. The
        # original chain is still returned unchanged and remains available to Krea2 Edit.
        analysis_references = (
            VisualReferenceChain()
            if str(mode) == "create_from_image"
            else passthrough_references
        )

        created_prompt, _, creator_info, thinking_text = super().create(
            clip=clip,
            visual_references=analysis_references,
            mode=mode,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            thinking=thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            reference_edit_image=reference_edit_image,
        )

        if str(mode) == "create_from_image":
            creator_info = creator_info.replace(
                "Visual Reference Chain Entries: 0",
                f"Visual Reference Chain Entries: {len(passthrough_references.entries)}",
                1,
            )
            marker = f"Visual Reference Chain Entries: {len(passthrough_references.entries)}"
            creator_info = creator_info.replace(
                marker,
                marker + "\nVisual Reference Analysis: isolated; only reference_edit_image is shown to Prompt Creator",
                1,
            )

        return created_prompt, passthrough_references, creator_info, thinking_text
