"""Public Prompt Creator contract with optional visual references."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .edit_prompt_creator_node import CcCKrea2EditPromptCreator as _BasePromptCreator
from .edit_reference_types import VisualReferenceChain


class CcCKrea2EditPromptCreator(_BasePromptCreator):
    """Expose visual_references as optional while preserving the existing runtime."""

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
        return super().create(
            clip=clip,
            visual_references=(
                visual_references if visual_references is not None else VisualReferenceChain()
            ),
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
