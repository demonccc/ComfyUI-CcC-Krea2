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
    if clean:
        parts.append("INSTRUCTIONS:\n" + clean)
    if mapping_lines:
        neutral = [re.sub(r"\. Role: (?:EDIT|REFERENCE) reference\.$", ".", line) for line in mapping_lines]
        parts.append("IMAGE MAPPING:\n" + "\n".join(neutral))
    return "\n\n".join(parts)


def _split_generated_text(
    clip: Any,
    generated_ids: Any,
    raw_prompt: str,
    thinking_enabled: bool = False,
) -> Tuple[str, str]:
    """Separate explicit ANALYSIS and FINAL PROMPT sections in one generation.

    Qwen3-VL Instruct does not provide a native ComfyUI thinking toggle. Avoid
    forcing <think> into the assistant prefix: the checkpoint can put its full
    answer there and terminate. Never accept an unlabelled final answer when
    thinking was requested.
    """
    generated_text = str(clip.decode(generated_ids)).strip()
    if not thinking_enabled:
        return generated_text, ""

    analysis_match = re.search(r"(?im)^\s*ANALYSIS\s*:\s*", generated_text)
    final_match = re.search(r"(?im)^\s*FINAL PROMPT\s*:\s*", generated_text)
    if final_match and analysis_match and analysis_match.start() < final_match.start():
        analysis = generated_text[analysis_match.end():final_match.start()].strip()
        final_prompt = generated_text[final_match.end():].strip()
        return final_prompt, analysis
    # Preserve any partial text on the thinking output, but never reinterpret
    # that text as a successful final prompt.
    if analysis_match:
        return "", generated_text[analysis_match.end():].strip()
    return "", generated_text


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
) -> Tuple[List[torch.Tensor], List[str], List[str]]:
    """Analyze every source in the single ordered reference chain once."""
    images: List[torch.Tensor] = []
    mapping_lines: List[str] = []
    warnings: List[str] = []
    for entry in references.entries:
        if isinstance(entry, VisionReferenceEntry) and entry.paired_visual:
            continue
        if not isinstance(entry, (ImageReferenceEntry, VisualReferenceEntry, VisionReferenceEntry)):
            continue
        if entry.image is None:
            warnings.append("Reference has no source image.")
            continue
        images.append(
            entry.source_image if isinstance(entry, VisionReferenceEntry) and entry.source_image is not None
            else entry.image
        )
        index = len(images)
        mapping_lines.append(f"Vision input {index} = Image {index}.")
    return images, mapping_lines, warnings


def _edit_image_slot_map(references: EditReferenceChain) -> List[Optional[int]]:
    """Map analyzed images to Edit's physical Qwen slots, preserving chain order."""
    entries = references.entries
    active = []
    deferred_styles = []
    analysis_positions = []
    for index, entry in enumerate(entries):
        if isinstance(entry, VisionReferenceEntry) and entry.paired_visual:
            continue
        if isinstance(entry, (ImageReferenceEntry, VisualReferenceEntry, VisionReferenceEntry)) and entry.image is not None:
            analysis_positions.append(index)
    for index, entry in enumerate(entries):
        if isinstance(entry, ImageReferenceEntry):
            continue
        if isinstance(entry, VisualReferenceEntry):
            paired = (entries[index + 1] if index + 1 < len(entries)
                      and isinstance(entries[index + 1], VisionReferenceEntry)
                      and entries[index + 1].paired_visual else None)
            tiled = paired is not None and paired.mode == "style_indirect" and paired.processing != "full"
            if entry.semantic and not tiled:
                active.append(index)
            continue
        if isinstance(entry, VisionReferenceEntry):
            if entry.mode == "style_indirect":
                deferred_styles.append((index, 1 if entry.processing == "full" else
                                        16 if entry.processing == "4x4" else 4))
            elif not entry.paired_visual:
                active.append(index)
    slot_map = {index: slot for slot, index in enumerate(active, start=1)}
    next_slot = len(active) + 1
    for index, count in deferred_styles:
        if count == 1:
            slot_map[index] = next_slot
        next_slot += count
    return [slot_map.get(index) for index in analysis_positions]


def _remap_created_image_labels(
    prompt: str, slot_map: List[Optional[int]]
) -> Tuple[str, List[int]]:
    """Replace analysis indices with Edit's physical indices simultaneously."""
    ungrounded = []
    def replace_label(match):
        index = int(match.group(1))
        if index < 1 or index > len(slot_map):
            return match.group(0)
        slot = slot_map[index - 1]
        if slot is None:
            ungrounded.append(index)
            return "the analyzed source image"
        return f"Image {slot}"
    updated = re.sub(r"(?i)\bImage\s+(\d+)\b", replace_label, prompt)
    return updated, sorted(set(ungrounded))




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
    ):
        if references is None:
            references = EditReferenceChain()
        if not isinstance(references, EditReferenceChain):
            raise TypeError(
                "[Krea2 CcC Edit Prompt Creator] references must be a EditReferenceChain."
            )

        images, mapping_lines, warnings = (
            ([], [], []) if mode in ("enhance_t2i", "enhance_i2i")
            else _collect_analysis_images(references)
        )
        # The Prompt Creator does not change which references reach Edit.
        combined_references = references
        generation_prompt = _build_generation_prompt(str(mode), instructions, mapping_lines)
        slot_map = _edit_image_slot_map(references) if images else []
        system_prompt = "\n\n".join(part for part in (
            _resolve_preset(str(mode)),
            KREA2_PROMPT_GUIDELINES,
        ) if part)

        if images:
            generation_prompt += ("\n\nOUTPUT IMAGE REFERENCES:\n"
                "Describe analysis-only images fully in words. Do not refer to them as Image N "
                "in the final prompt. Any Image N in your final prompt must refer only to "
                "a reference actually provided to Krea2 Edit. The analysis image numbering "
                "is not necessarily the Edit conditioning numbering.")
        if thinking:
            # Explicit structured analysis in a SINGLE generation. Qwen3-VL
            # Instruct has no native thinking=True switch in ComfyUI.
            system_prompt = system_prompt.replace(
                "Return only the final image description as one cohesive English paragraph.",
                "Return exactly two sections, ANALYSIS and FINAL PROMPT. "
                "ANALYSIS explains the reference roles and constraints; "
                "FINAL PROMPT contains one cohesive English paragraph describing "
                "the desired final image and no analysis.",
            )
            generation_prompt += (
                "\n\nOUTPUT FORMAT (REQUIRED):\n"
                "ANALYSIS:\n"
                "First, analyze Image 1 and Image 2 separately when present. "
                "For each image, identify its visible subjects and relevant details. "
                "Compare the user instructions with the observations and explicitly "
                "decide which image supplies identity, clothing, pose, scene, "
                "composition, and interactions. Note conflicts or uncertain "
                "details. This analysis must not be just a generic description "
                "of one image.\n\n"
                "FINAL PROMPT:\n"
                "Provide the complete final image description resulting from the "
                "decisions above, following all user instructions. No explanations "
                "or extra formatting in this section.\n\n"
                "Write both sections using the exact headings ANALYSIS: and FINAL PROMPT:."
            )
        tokenize_kwargs = {
            "images": images,
            "min_length": 1,
            "system_prompt": system_prompt,
        }

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
        raw_output = str(clip.decode(generated_ids)).strip()
        raw_created_prompt, thinking_text = _split_generated_text(
            clip, generated_ids, generation_prompt, thinking_enabled=bool(thinking)
        )
        error = ""
        if thinking and not re.search(r"(?im)^\s*ANALYSIS\s*:\s*", raw_output):
            error = (
                "Qwen did not produce the required ANALYSIS: section. "
                "The raw output is preserved on thinking. "
                "Check model and prompt format; max_tokens may not be the cause."
            )
        elif thinking and not re.search(r"(?im)^\s*FINAL PROMPT\s*:\s*", raw_output):
            error = (
                "Qwen produced analysis but did not produce the required "
                "FINAL PROMPT: section. The partial output is preserved on thinking. "
                "This can be a format compliance failure or early EOS."
            )
        elif thinking and not raw_created_prompt.strip():
            error = (
                "Qwen produced the FINAL PROMPT: heading but no final text. "
                "The partial analysis is preserved on thinking."
            )
        if error:
            created_prompt = ""
            analysis_only_mentions = []
        else:
            created_prompt = _normalize_created_prompt(raw_created_prompt)
            created_prompt, analysis_only_mentions = _remap_created_image_labels(created_prompt, slot_map)

        generated_token_count = len(generated_ids) if isinstance(generated_ids, (list, tuple, torch.Tensor)) else "<unknown>"
        last_token_id = int(generated_ids[-1]) if isinstance(generated_ids, (list, tuple)) and generated_ids else "<unknown>"
        info_lines = ([f"ERROR: {error}", ""] if error else []) + [
            "=== Krea2 CcC Edit Prompt Creator ===",
            f"Mode: {mode}",
            f"Thinking: {'enabled' if thinking else 'disabled'}",
            "Thinking Strategy: one-pass explicit ANALYSIS / FINAL PROMPT (no fallback)",
            f"Thinking Output: {'present' if thinking_text else ('empty' if thinking else 'disabled')}",
            f"Qwen Decoded Output Prefix: {raw_output[:200]!r}",
            f"Generated Tokens: {generated_token_count}",
            f"Last Generated Token ID: {last_token_id}",
            f"Preset: {_resolve_preset(str(mode)) or 'none'}",
            f"Visual Reference Chain Entries: {len(references.entries)}",
            f"Images Analyzed: {len(images)}",
            f"Max Tokens: {int(max_tokens)}",
            f"Temperature: {float(temperature):.2f}",
            f"Top P: {float(top_p):.2f}",
            f"Seed: {int(seed)}",
        ]
        info_lines.extend(["", "Effective Qwen System Prompt:", system_prompt,
                           "", "Effective Qwen User Prompt:", generation_prompt])
        if mapping_lines:
            info_lines.append("")
            info_lines.append("Image Mapping:")
            info_lines.extend(f"  {line}" for line in mapping_lines)
        if slot_map:
            info_lines.append("Edit Image Slots: " + ", ".join(
                f"Analysis Image {i} -> " + (f"Edit Image {slot}" if slot is not None else "analysis only")
                for i, slot in enumerate(slot_map, start=1)))
        if analysis_only_mentions:
            info_lines.append("Warning: model referred to analysis-only images by number: "
                              + ", ".join(map(str, analysis_only_mentions)))
        if warnings:
            info_lines.append("")
            info_lines.append("Warnings:")
            info_lines.extend(f"  - {warning}" for warning in warnings)
        if thinking_text:
            info_lines.extend(["", "Thinking:", thinking_text])
        info_lines.extend(["", "Created Prompt:", created_prompt])

        return created_prompt, combined_references, "\n".join(info_lines), thinking_text
