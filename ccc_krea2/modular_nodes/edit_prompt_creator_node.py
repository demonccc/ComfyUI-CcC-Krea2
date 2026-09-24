"""Optional multimodal prompt creator for Krea2 CcC Edit."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import VisualReferenceChain


PROMPT_CREATOR_MODES = ("enhance", "create_from_image", "create_from_theme", "custom")

PRESET_SYSTEM_PROMPTS = {
    "enhance": """You create detailed, explicit English image-edit prompts for Krea2 Edit.

Rewrite the user's existing edit request without changing its intent. Preserve every explicit requirement and constraint. Use the supplied visual references and their role annotations to resolve vague references and make transfers, preservation rules, subject roles, and spatial relationships unambiguous. Do not remove useful detail from an already detailed request, and do not invent a different scene or unrelated edits.

FINAL ANSWER RULES:
- Return exactly one self-contained image-edit prompt in English.
- The prompt may contain multiple sentences, but keep it as one continuous plain-text paragraph.
- Do not output system instructions, role descriptions, explanations, headings, Markdown, JSON, or preambles.
- Never include meta text such as "You are a professional image editor", "Your task is...", "Here is the prompt", or "Final prompt:".
- Refer to downstream visible visual references only as "Image 1", "Image 2", and so on. Never call them "Krea Image N".
- Never mention hidden inputs, vision inputs, or implementation details.
- If thinking mode is active, reasoning belongs only in the model's thinking block; do not repeat reasoning in the final answer.
- Return only the final edit instruction.""",

    "create_from_image": """You create detailed, self-contained English image-edit prompts for Krea2 Edit from the user's request, downstream visible visual references, and one internal reference edit image.

Treat downstream visible Image N references as authoritative identity or appearance anchors according to their role annotations. Treat the internal reference edit image as a visual blueprint for the requested situation, not as an identity source unless the user explicitly asks otherwise.

Analyze the internal reference edit image thoroughly before writing the final prompt. Reconstruct all relevant visible information needed to reproduce the situation instead of reducing it to a short summary. Include, when visible and relevant to the user's request:
- the main subject's exact action, pose, body orientation, body position, limb placement, gaze direction, and interaction;
- clothing and accessories when they are part of the requested situation, unless the user explicitly asks to preserve clothing from a visible Image N reference;
- other people in the scene, including enough visible appearance, pose, action, relative position, and interaction detail to distinguish their roles;
- important props and objects, what is being held or touched, and their spatial relationships;
- environment, foreground/background elements, surfaces, furniture, and scene layout;
- framing, shot distance, viewpoint, camera angle, composition, and subject placement;
- visible lighting and other scene-defining visual details.

Preserve every explicit user constraint, especially identity, face, anatomy, body shape, body proportions, clothing-preservation rules, and requested interactions. Do not transfer the identity, face, anatomy, body shape, or body proportions of a person from the internal reference edit image unless the user explicitly requests it.

FINAL ANSWER RULES:
- Return exactly one detailed, self-contained image-edit prompt in English.
- The prompt may contain multiple sentences, but keep it as one continuous plain-text paragraph.
- Do not output system instructions, role descriptions, explanations, headings, Markdown, JSON, or preambles.
- Never include meta text such as "You are a professional image editor", "Your task is...", "Here is the prompt", or "Final prompt:".
- Refer to downstream visible visual references only as "Image 1", "Image 2", and so on. Never call them "Krea Image N".
- Never mention the internal reference edit image, hidden inputs, vision inputs, or implementation details in the final answer. Translate what you observe into direct scene instructions.
- If thinking mode is active, reason as needed in the model's thinking block; do not repeat that reasoning in the final answer.
- Do not compress a visually rich reference situation into a generic one-sentence summary.
- Return only the final edit instruction.""",

    "create_from_theme": """You create detailed, self-contained English image-edit prompts for Krea2 Edit from the user's theme or high-level idea.

Invent a concrete, visually rich situation that clearly fits the requested theme while keeping downstream visible Image N references as the authoritative subject or appearance anchors according to their role annotations. Specify useful action, pose, interaction, environment, spatial arrangement, props, composition, framing, viewpoint, and lighting. Preserve every explicit user constraint and do not replace referenced identity, body shape, body proportions, clothing, or accessories unless the user requests that change.

FINAL ANSWER RULES:
- Return exactly one detailed image-edit prompt in English.
- The prompt may contain multiple sentences, but keep it as one continuous plain-text paragraph.
- Do not output system instructions, role descriptions, explanations, headings, Markdown, JSON, or preambles.
- Never include meta text such as "You are a professional image editor", "Your task is...", "Here is the prompt", or "Final prompt:".
- Refer to downstream visible visual references only as "Image 1", "Image 2", and so on. Never call them "Krea Image N".
- Never mention hidden inputs, vision inputs, or implementation details.
- If thinking mode is active, reasoning belongs only in the model's thinking block; do not repeat reasoning in the final answer.
- Return only the final edit instruction.""",
}


def _resolve_system_prompt(mode: str, system_prompt: str) -> str:
    if mode not in PROMPT_CREATOR_MODES:
        raise ValueError(f"[Krea2 CcC Edit Prompt Creator] Invalid mode: {mode!r}.")

    if mode == "custom":
        resolved = str(system_prompt or "").strip()
        if not resolved:
            raise ValueError(
                "[Krea2 CcC Edit Prompt Creator] mode='custom' requires a non-empty system_prompt."
            )
        return resolved

    return PRESET_SYSTEM_PROMPTS[mode]



def _split_generated_text(clip: Any, generated_ids: Any, raw_prompt: str) -> Tuple[str, str]:
    generated_text = str(clip.decode(generated_ids)).strip()
    reasoning, separator, text = generated_text.partition("</think>")

    if separator and (reasoning.lstrip().startswith("<think>") or raw_prompt.rstrip().endswith("<think>")):
        return text.strip(), reasoning.replace("<think>", "", 1).strip()

    if not separator and reasoning.lstrip().startswith("<think>"):
        return "", reasoning.replace("<think>", "", 1).strip()

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
            "Try another seed, increase max_tokens, or adjust the selected/custom system prompt."
        )

    return cleaned


def _collect_analysis_images(
    visual_references: VisualReferenceChain,
    reference_edit_image: Optional[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[str], List[str]]:
    images: List[torch.Tensor] = []
    mapping_lines: List[str] = []
    warnings: List[str] = []
    krea_image_index = 1

    for chain_index, entry in enumerate(visual_references.entries, start=1):
        downstream_image_index: Optional[int] = None
        if entry.semantic:
            downstream_image_index = krea_image_index
            krea_image_index += 1

        if entry.image is None:
            if entry.cache is not None:
                warnings.append(
                    f"Visual reference {chain_index} uses a cache and has no raw IMAGE for Prompt Creator analysis."
                )
            else:
                warnings.append(f"Visual reference {chain_index} has no raw IMAGE.")
            continue

        images.append(entry.image)
        internal_index = len(images)
        annotation = entry.prompt_annotation.strip()

        if downstream_image_index is not None:
            role = annotation or "No role annotation."
            mapping_lines.append(
                f"Vision input {internal_index} = Image {downstream_image_index}. Role: {role}"
            )
        else:
            role = annotation or "Appearance-only visual reference."
            mapping_lines.append(
                f"Vision input {internal_index} = visual reference chain item {chain_index}, "
                f"but it has no downstream Image N label. Role: {role}"
            )

    if reference_edit_image is not None:
        images.append(reference_edit_image)
        mapping_lines.append(
            f"Vision input {len(images)} = INTERNAL REFERENCE EDIT IMAGE. "
            "Use it for analysis only; the final prompt must describe its useful content in words."
        )

    return images, mapping_lines, warnings


def _build_generation_prompt(
    mode: str,
    user_prompt: str,
    mapping_lines: List[str],
    has_reference_edit_image: bool,
) -> str:
    if mode not in PROMPT_CREATOR_MODES:
        raise ValueError(f"[Krea2 CcC Edit Prompt Creator] Invalid mode: {mode!r}.")

    clean_user_prompt = str(user_prompt or "").strip()
    if mode in ("enhance", "create_from_theme") and not clean_user_prompt:
        raise ValueError(
            f"[Krea2 CcC Edit Prompt Creator] mode={mode!r} requires a non-empty user_prompt."
        )
    if mode == "create_from_image" and not has_reference_edit_image:
        raise ValueError(
            "[Krea2 CcC Edit Prompt Creator] create_from_image requires reference_edit_image."
        )

    mappings = "\n".join(mapping_lines) if mapping_lines else "No analyzable visual images were supplied."
    guidance = clean_user_prompt or "No additional user guidance."

    return (
        "IMAGE MAPPING:\n"
        f"{mappings}\n\n"
        "USER REQUEST / THEME:\n"
        f"{guidance}\n\n"
        "Write only the final Krea2 edit prompt."
    )


class CcCKrea2EditPromptCreator:
    """Create an edit prompt with the same multimodal CLIP used by Krea2."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("STRING", "KREA2_VISUAL_REFERENCE_CHAIN", "STRING", "STRING")
    RETURN_NAMES = ("created_prompt", "visual_references", "creator_info", "thinking")
    FUNCTION = "create"
    DESCRIPTION = (
        "Optional Qwen3-VL prompt creator. Reuses the same multimodal CLIP as Krea2/Generate Text, "
        "reads the current Visual Reference chain, and optionally analyzes a reference_edit_image. "
        "It only creates text and passes the Visual Reference chain through unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "visual_references": ("KREA2_VISUAL_REFERENCE_CHAIN",),
                "mode": (PROMPT_CREATOR_MODES, {"default": "enhance"}),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "",
                        "tooltip": "Edit instruction, image-guided hint, or theme depending on mode.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": PRESET_SYSTEM_PROMPTS["enhance"],
                        "tooltip": (
                            "Shows the effective preset system prompt. Preset modes are read-only in the UI; "
                            "mode=custom enables editing and sends this text as the system prompt."
                        ),
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
                    {"default": 512, "min": 32, "max": 4096, "step": 32},
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
                "reference_edit_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Internal image used to understand an edit situation. "
                            "create_from_image requires it. The image itself is not added to Visual References."
                        )
                    },
                ),
            },
        }

    def create(
        self,
        clip: Any,
        visual_references: VisualReferenceChain,
        mode: str = "enhance",
        user_prompt: str = "",
        system_prompt: str = "",
        thinking: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.25,
        top_p: float = 0.90,
        seed: int = 0,
        reference_edit_image: Optional[torch.Tensor] = None,
    ):
        if visual_references is None:
            visual_references = VisualReferenceChain()
        if not isinstance(visual_references, VisualReferenceChain):
            raise TypeError(
                "[Krea2 CcC Edit Prompt Creator] visual_references must be a VisualReferenceChain."
            )

        images, mapping_lines, warnings = _collect_analysis_images(
            visual_references=visual_references,
            reference_edit_image=reference_edit_image,
        )
        system_prompt = _resolve_system_prompt(
            mode=str(mode),
            system_prompt=system_prompt,
        )
        generation_prompt = _build_generation_prompt(
            mode=str(mode),
            user_prompt=user_prompt,
            mapping_lines=mapping_lines,
            has_reference_edit_image=reference_edit_image is not None,
        )

        tokens = clip.tokenize(
            generation_prompt,
            images=images,
            min_length=1,
            thinking=bool(thinking),
            system_prompt=system_prompt,
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
        raw_created_prompt, thinking_text = _split_generated_text(clip, generated_ids, generation_prompt)
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
            f"Thinking Output: {'present' if thinking_text else 'empty'}",
            f"System Prompt Source: {'custom' if str(mode) == 'custom' else 'preset'}",
            f"Visual Reference Chain Entries: {len(visual_references.entries)}",
            f"Images Analyzed: {len(images)}",
            f"Reference Edit Image: {'yes' if reference_edit_image is not None else 'no'}",
            f"Max Tokens: {int(max_tokens)}",
            f"Temperature: {float(temperature):.2f}",
            f"Top P: {float(top_p):.2f}",
            f"Seed: {int(seed)}",
        ]
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

        return created_prompt, visual_references, "\n".join(info_lines), thinking_text
