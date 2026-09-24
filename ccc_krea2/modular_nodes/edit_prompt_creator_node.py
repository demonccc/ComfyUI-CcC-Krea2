"""Optional multimodal prompt creator for Krea2 CcC Edit."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import VisualReferenceChain


PROMPT_CREATOR_MODES = ("enhance", "create_from_image", "create_from_theme", "custom")

OUTPUT_CONTRACT = """The result is fed directly into Krea2 Edit.

OUTPUT CONTRACT:
- Output exactly one final image-edit prompt in English.
- Output plain text only.
- Do not output system instructions, role descriptions, analysis, explanations, headings, Markdown, JSON, or preambles.
- Never begin with or include meta text such as "You are a professional image editor", "Your task is...", "Here is the prompt", or "Final prompt:".
- Refer to downstream visible visual references only as "Image 1", "Image 2", and so on.
- Never call a reference "Krea Image N".
- Never mention the internal reference edit image, hidden inputs, vision inputs, or implementation details.
- When an internal reference edit image is supplied, translate its useful visual content into explicit scene/action/pose/interaction/environment/composition wording.
- Return only the final edit instruction."""

PRESET_SYSTEM_PROMPTS = {
    "enhance": """Rewrite the user's edit request as one concise, explicit Krea2 image-edit prompt.
Use the supplied visual references and their role annotations to resolve vague references.
Make requested transfers and preservation constraints clear.
Preserve the user's intent exactly. Do not invent a different scene or unrelated edits.""",
    "create_from_image": """Create a Krea2 image-edit prompt from the user's request and the supplied images.
Use downstream visible Image N references as identity or appearance anchors.
Use the internal reference edit image only as the source of useful situation details such as action, pose, interaction, environment, framing, composition, and spatial context.
Rebuild that situation around the relevant visible reference subject or subjects.
Do not copy identity or appearance from the internal reference edit image unless the user explicitly requests it.""",
    "create_from_theme": """Create a Krea2 image-edit prompt from the user's theme or high-level idea.
Invent a concrete, visually interesting situation, action, environment, and composition that clearly fit the theme.
Use downstream visible Image N references as subject or appearance anchors.
Do not replace referenced identity or appearance unless the user explicitly asks for that.""",
}


def _resolve_system_prompt(mode: str, custom_system_prompt: str) -> str:
    if mode not in PROMPT_CREATOR_MODES:
        raise ValueError(f"[Krea2 CcC Edit Prompt Creator] Invalid mode: {mode!r}.")

    if mode == "custom":
        base = str(custom_system_prompt or "").strip()
        if not base:
            raise ValueError(
                "[Krea2 CcC Edit Prompt Creator] mode='custom' requires a non-empty custom_system_prompt."
            )
    else:
        base = PRESET_SYSTEM_PROMPTS[mode]

    return f"{base}\n\n{OUTPUT_CONTRACT}"


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
                "custom_system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "tooltip": (
                            "Used only when mode=custom. Defines how Qwen should construct the edit prompt; "
                            "the final-output contract remains enforced."
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
        custom_system_prompt: str = "",
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
            custom_system_prompt=custom_system_prompt,
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
            f"Custom System Prompt: {'yes' if str(mode) == 'custom' else 'no'}",
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
        info_lines.extend(["", "Created Prompt:", created_prompt])

        return created_prompt, visual_references, "\n".join(info_lines), thinking_text
