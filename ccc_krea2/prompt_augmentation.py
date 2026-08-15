"""Custom socket types, immutable value objects, and prompt augmentation helpers for CcC Krea2."""

from dataclasses import dataclass
from typing import Tuple, Optional, List

CCC_KREA2_LORA_PROMPT_SETTINGS = "CCC_KREA2_LORA_PROMPT_SETTINGS"
CCC_KREA2_PROMPT_AUGMENTATION = "CCC_KREA2_PROMPT_AUGMENTATION"


@dataclass(frozen=True)
class LoRAPromptSlot:
    """Immutable configuration for a single LoRA prompt slot."""

    enabled: bool = False
    position: str = "append"
    positive_prompt: str = ""
    negative_prompt: str = ""


@dataclass(frozen=True)
class LoRAPromptSettings:
    """Immutable container for LoRA prompt slot settings."""

    enabled: bool = True
    slots: Tuple[LoRAPromptSlot, ...] = ()


@dataclass(frozen=True)
class PromptAugmentation:
    """Immutable bundle of accumulated prompt prepends and appends."""

    positive_prepend: Tuple[str, ...] = ()
    positive_append: Tuple[str, ...] = ()
    negative_prepend: Tuple[str, ...] = ()
    negative_append: Tuple[str, ...] = ()


EMPTY_PROMPT_AUGMENTATION = PromptAugmentation()


def apply_prompt_augmentation(
    positive_prompt: str,
    negative_prompt: str,
    augmentation: Optional[PromptAugmentation] = None,
) -> Tuple[str, str]:
    """Combine base prompts with active prompt augmentations using '\\n\\n' separators.

    Order for positive prompt:
        positive_prepend entries -> original positive prompt -> positive_append entries

    Order for negative prompt:
        negative_prepend entries -> original negative prompt -> negative_append entries

    Blank augmentation entries are excluded.
    Original prompt strings are never modified.
    """
    if augmentation is None:
        return positive_prompt, negative_prompt

    if not isinstance(augmentation, PromptAugmentation):
        raise TypeError(
            f"Invalid prompt augmentation socket object of type {type(augmentation)}. Expected PromptAugmentation."
        )

    # Build positive prompt parts
    pos_parts: List[str] = []
    for entry in augmentation.positive_prepend:
        if entry and entry.strip():
            pos_parts.append(entry.strip())

    if positive_prompt:
        pos_parts.append(positive_prompt)

    for entry in augmentation.positive_append:
        if entry and entry.strip():
            pos_parts.append(entry.strip())

    effective_pos = "\n\n".join(pos_parts) if pos_parts else positive_prompt

    # Build negative prompt parts
    neg_parts: List[str] = []
    for entry in augmentation.negative_prepend:
        if entry and entry.strip():
            neg_parts.append(entry.strip())

    if negative_prompt:
        neg_parts.append(negative_prompt)

    for entry in augmentation.negative_append:
        if entry and entry.strip():
            neg_parts.append(entry.strip())

    effective_neg = "\n\n".join(neg_parts) if neg_parts else negative_prompt

    return effective_pos, effective_neg
