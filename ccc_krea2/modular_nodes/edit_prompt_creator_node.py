"""Optional multimodal prompt creator for Krea2 CcC Edit."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import EditReferenceChain, ImageReferenceEntry, VisualReferenceEntry, VisionReferenceEntry


PROMPT_CREATOR_MODES = ("enhance_t2i", "enhance_i2i", "create_from_image_t2i", "create_from_image_i2i", "create_from_theme_t2i", "create_from_theme_i2i", "none")

PRESET_TEMPLATES = {
    "enhance": "Enhance the instructions into a detailed {kind} prompt for Krea2 by adding relevant and coherent details that develop the original idea while preserving its intent.",
    "create_from_image": "Analyze the provided images thoroughly to identify and understand their visible content, characteristics, details, and relationships. Use the instructions to determine which visual information is relevant and how that information should be used to construct the desired result. Based on this analysis and the instructions, create a detailed and accurate {kind} prompt for Krea2.",
    "create_from_theme": "Create an original {kind} prompt for Krea2 inspired by the instructions, freely developing them into a complete, imaginative scene.",
}

KREA2_PROMPT_GUIDELINES = """Write the final image description in natural language.

Preserve the specified elements, relationships, and style.

Make the description rich and detailed, with coherent subjects, attributes, actions, and surroundings.

Avoid unnecessary specifications that conflict with the intended result.

Return only the final image description as one cohesive English paragraph."""


def _resolve_preset(mode: str) -> str:
    if mode not in PROMPT_CREATOR_MODES:
        raise ValueError(f"[Krea2 CcC Edit Prompt Creator] Invalid mode: {mode!r}.")
    if mode == "none":
        return ""
    family, variant = mode.rsplit("_", 1)
    return PRESET_TEMPLATES[family].format(kind="text-to-image" if variant == "t2i" else "image-to-image")


def _build_generation_prompt(mode: str, instructions: str, mapping_lines: List[str]) -> str:
    preset = _resolve_preset(mode)
    clean = str(instructions or "").strip()
    if not clean and not (mode in ("create_from_image_t2i", "create_from_image_i2i") and mapping_lines):
        raise ValueError("[Krea2 CcC Edit Prompt Creator] instructions must not be empty unless create_from_image has reference images.")
    parts = []
    if preset:
        parts.append(preset)
    if clean:
        parts.append("INSTRUCTIONS:\n" + clean)
    if mapping_lines:
        neutral = [re.sub(r"\. Role: (?:EDIT|REFERENCE) reference\.$", ".", line) for line in mapping_lines]
        parts.append("IMAGE MAPPING:\n" + "\n".join(neutral))
    parts.append("KREA2 PROMPT GUIDELINES:\n" + KREA2_PROMPT_GUIDELINES)
    return "\n\n".join(parts)


def _thinking_llama_template(image_count: int) -> str:
    vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
    images = vision_block * max(0, int(image_count))
    return (
        "<|im_start|>user\n"
        f"{images}{{}}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>"
    )


def _split_generated_text(
    clip: Any,
    generated_ids: Any,
    raw_prompt: str,
    thinking_prefilled: bool = False,
) -> Tuple[str, str]:
    generated_text = str(clip.decode(generated_ids)).strip()
    reasoning, separator, text = generated_text.partition("</think>")

    has_reasoning_prefix = reasoning.lstrip().startswith("<think>")
    if separator and (thinking_prefilled or has_reasoning_prefix or raw_prompt.rstrip().endswith("<think>")):
        cleaned_reasoning = reasoning.replace("<think>", "", 1).strip()
        return text.strip(), cleaned_reasoning

    if not separator and (thinking_prefilled or has_reasoning_prefix):
        cleaned_reasoning = reasoning.replace("<think>", "", 1).strip()
        return "", cleaned_reasoning

    return generated_text, ""


def _normalize_created_prompt(text: str) -> str:
    cleaned = str(text or "").strip()

    if cleaned.startswith("~~~") and cleaned.endswith("~~~"):
        cleaned = cleaned[3:-3].strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()

    # Defensive cleanup for meta-instructions occasionally echoed by the VLM.
    cleaned = re.sub(
        r"(?is)^\s*you are (?:a|an) professional image editor\.\s*",
        "",
        cleaned,
        count=1,
    )
    cleaned = re.sub(
        r"(?is)^\s*your task is to create a new image based on the user's request,\s*"
        r"using the provided visual references and internal reference edit image as guides\.\s*",
        "",
        cleaned,
        count=1,
    )
    cleaned = re.sub(
        r"(?is)^\s*(?:here is|here's)\s+(?:the\s+)?(?:final\s+)?(?:krea2\s+)?(?:edit\s+)?prompt\s*:\s*",
        "",
        cleaned,
        count=1,
    )
    cleaned = re.sub(
        r"(?is)^\s*(?:final\s+)?(?:krea2\s+)?edit\s+prompt\s*:\s*",
        "",
        cleaned,
        count=1,
    )

    cleaned = re.sub(r"(?i)\bKrea\s+Image\s+(\d+)\b", r"Image \1", cleaned)

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'"):
        cleaned = cleaned[1:-1].strip()

    lowered = cleaned.lower()
    if not cleaned or lowered.startswith("you are a professional image editor") or lowered.startswith(
        "your task is to create a new image"
    ):
        raise RuntimeError(
            "[Krea2 CcC Edit Prompt Creator] Qwen returned meta-instructions instead of a final edit prompt. "
            "Try another seed, increase max_tokens, or adjust the selected preset or instructions."
        )

    return cleaned


def _collect_analysis_images(
    references: EditReferenceChain,
    reference_edit_image: Optional[EditReferenceChain],
) -> Tuple[List[torch.Tensor], List[str], List[str]]:
    """Send edit images first, then the remaining references, without paired duplicates."""
    images: List[torch.Tensor] = []
    mapping_lines: List[str] = []
    warnings: List[str] = []
    for role, chain in (
        ("EDIT", reference_edit_image or EditReferenceChain()),
        ("REFERENCE", references),
    ):
        for entry in chain.entries:
            if isinstance(entry, VisionReferenceEntry) and entry.paired_visual:
                continue
            if not isinstance(entry, (ImageReferenceEntry, VisualReferenceEntry, VisionReferenceEntry)):
                continue
            if entry.image is None:
                warnings.append(f"{role} reference has no source image.")
                continue
            images.append(
                entry.source_image if isinstance(entry, VisionReferenceEntry) and entry.source_image is not None
                else entry.image
            )
            index = len(images)
            mapping_lines.append(
                f"Vision input {index} = Image {index}. Role: {role} reference."
            )
    return images, mapping_lines, warnings




class CcCKrea2EditPromptCreator:
    """Create an edit prompt with the same multimodal CLIP used by Krea2."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("STRING", "KREA2_REFERENCE_CHAIN", "STRING", "STRING")
    RETURN_NAMES = ("created_prompt", "references", "creator_info", "thinking")
    FUNCTION = "create"
    DESCRIPTION = (
        "Optional Qwen3-VL prompt creator. Reuses the same multimodal CLIP as Krea2/Generate Text, "
        "reads the edit reference first, then other references, including image-only sources. "
        "It generates text and forwards the ordered chain without changing conditioning flags."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "mode": (PROMPT_CREATOR_MODES, {"default": "enhance_t2i"}),
                "instructions": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "",
                        "tooltip": "Instructions used to generate the Krea2 prompt.",
                    },
                ),
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable Qwen thinking. Reasoning is returned separately on the thinking output.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 32, "max": 32768, "step": 32},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.01, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Sampling seed passed to Qwen3-VL text generation.",
                    },
                ),
            },
            "optional": {
                "references": ("KREA2_REFERENCE_CHAIN",),
                "reference_edit_image": (
                    "KREA2_REFERENCE_CHAIN",
                    {
                        "tooltip": (
                            "Reference for the image being edited; analyzed first. "
                            "Can contain an image even when both conditioning paths are disabled."
                        )
                    },
                ),
            },
        }

    def create(
        self,
        clip: Any,
        references: Optional[EditReferenceChain] = None,
        mode: str = "enhance_t2i",
        instructions: str = "",
        thinking: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.25,
        top_p: float = 0.90,
        seed: int = 0,
        reference_edit_image: Optional[EditReferenceChain] = None,
    ):
        if references is None:
            references = EditReferenceChain()
        if not isinstance(references, EditReferenceChain):
            raise TypeError(
                "[Krea2 CcC Edit Prompt Creator] references must be a EditReferenceChain."
            )

        if reference_edit_image is not None and not isinstance(reference_edit_image, EditReferenceChain):
            raise TypeError("[Krea2 CcC Edit Prompt Creator] reference_edit_image must be a Reference chain.")
        images, mapping_lines, warnings = (
            ([], [], []) if mode in ("enhance_t2i", "enhance_i2i") else _collect_analysis_images(
                references=references,
                reference_edit_image=reference_edit_image,
            )
        )
        combined_references = EditReferenceChain(
            (reference_edit_image.entries if reference_edit_image is not None else ()) + references.entries
        )
        generation_prompt = _build_generation_prompt(str(mode), instructions, mapping_lines)

        tokenize_kwargs = {
            "images": images,
            "min_length": 1,
            "thinking": bool(thinking),
            "system_prompt": "",
        }
        thinking_prefilled = bool(thinking)
        if thinking_prefilled:
            tokenize_kwargs["llama_template"] = _thinking_llama_template(len(images))

        tokens = clip.tokenize(
            generation_prompt,
            **tokenize_kwargs,
        )
        generated_ids = clip.generate(
            tokens,
            do_sample=True,
            max_length=int(max_tokens),
            temperature=float(temperature),
            top_k=64,
            top_p=float(top_p),
            min_p=0.05,
            repetition_penalty=1.05,
            seed=int(seed),
        )
        raw_created_prompt, thinking_text = _split_generated_text(
            clip,
            generated_ids,
            generation_prompt,
            thinking_prefilled=thinking_prefilled,
        )
        if not raw_created_prompt.strip() and thinking_text:
            raise RuntimeError(
                "[Krea2 CcC Edit Prompt Creator] Qwen generation ended inside the thinking block before "
                "producing a final prompt. Increase max_tokens or disable thinking."
            )
        created_prompt = _normalize_created_prompt(raw_created_prompt)

        info_lines = [
            "=== Krea2 CcC Edit Prompt Creator ===",
            f"Mode: {mode}",
            f"Thinking: {'enabled' if thinking else 'disabled'}",
            f"Thinking Output: {'present' if thinking_text else ('empty (model did not emit <think>)' if thinking else 'disabled')}",
            f"Preset: {_resolve_preset(str(mode)) or 'none'}",
            f"Visual Reference Chain Entries: {len(references.entries)}",
            f"Images Analyzed: {len(images)}",
            f"Reference Edit Image: {'yes' if reference_edit_image is not None else 'no'}",
            f"Max Tokens: {int(max_tokens)}",
            f"Temperature: {float(temperature):.2f}",
            f"Top P: {float(top_p):.2f}",
            f"Seed: {int(seed)}",
        ]
        info_lines.extend(["", "Effective Qwen Prompt (preset + instructions + guidelines):", generation_prompt])
        if mapping_lines:
            info_lines.append("")
            info_lines.append("Image Mapping:")
            info_lines.extend(f"  {line}" for line in mapping_lines)
        if warnings:
            info_lines.append("")
            info_lines.append("Warnings:")
            info_lines.extend(f"  - {warning}" for warning in warnings)
        if thinking_text:
            info_lines.extend(["", "Thinking:", thinking_text])
        info_lines.extend(["", "Created Prompt:", created_prompt])

        return created_prompt, combined_references, "\n".join(info_lines), thinking_text
