"""Optional multimodal prompt creator for Krea2 CcC Edit."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import VisualReferenceChain


PROMPT_CREATOR_MODES = ("enhance", "create_from_image", "create_from_theme")

SYSTEM_PROMPT = """You create concise, explicit English image-edit prompts for Krea2.

You receive zero or more Krea visual references plus, optionally, one internal reference edit image.
The Krea visual-reference labels given in the user message are authoritative. If a reference has a
Krea Image N label, preserve that exact number in the final prompt. Some appearance-only references
may not have an Image N label; never invent one for them.

The internal reference edit image is visible only to you. It is NOT automatically visible to Krea2
Edit. Never mention it in the final prompt as "the reference edit image", "the base image", or as an
Image N. Instead, translate the useful scene, action, pose, interaction, environment, framing, and
spatial context you observe in it into explicit words.

Return exactly one final edit prompt in English. No title, no explanation, no Markdown, no JSON.
Keep the user's intent. Be concrete about what changes, what transfers, and what must remain
unchanged. Do not invent unrelated edits unless the selected mode explicitly asks you to create a
new situation from a theme."""


MODE_INSTRUCTIONS = {
    "enhance": (
        "ENHANCE MODE: Improve the user's existing edit instruction. Resolve vague references using "
        "the supplied images and role annotations. Make transfer and preservation constraints explicit, "
        "but do not change the requested edit or invent a different scene."
    ),
    "create_from_image": (
        "CREATE FROM IMAGE MODE: Use the internal reference edit image as the source of the situation: "
        "its action, pose, interaction, environment, composition, and useful scene details. Rebuild that "
        "situation around the relevant Krea visual reference subject(s). Convert the reference edit image "
        "into explicit textual scene/action instructions. Do not copy the identity or appearance of a "
        "subject from the internal reference edit image unless the user explicitly asks for it."
    ),
    "create_from_theme": (
        "CREATE FROM THEME MODE: Treat the user's text as a theme or high-level idea. Invent a concrete, "
        "visually interesting new situation, action, and environment that clearly fits that theme, while "
        "anchoring the referenced subject(s) and requested appearance to the Krea visual references. "
        "Do not replace their identity or appearance unless the user asks for that."
    ),
}


def _decode_generated_text(clip: Any, generated_ids: Any, raw_prompt: str) -> str:
    generated_text = str(clip.decode(generated_ids))
    reasoning, separator, text = generated_text.partition("</think>")
    if separator and (reasoning.lstrip().startswith("<think>") or raw_prompt.rstrip().endswith("<think>")):
        return text.strip()
    return generated_text.strip()


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
                f"Vision input {internal_index} = Krea Image {downstream_image_index}. Role: {role}"
            )
        else:
            role = annotation or "Appearance-only visual reference."
            mapping_lines.append(
                f"Vision input {internal_index} = visual reference chain item {chain_index}, "
                f"but it is not shown to Krea Qwen as Image N. Role: {role}"
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
        f"{MODE_INSTRUCTIONS[mode]}\n\n"
        "IMAGE MAPPING:\n"
        f"{mappings}\n\n"
        "USER REQUEST / THEME:\n"
        f"{guidance}\n\n"
        "Write only the final Krea2 edit prompt."
    )


class CcCKrea2EditPromptCreator:
    """Create an edit prompt with the same multimodal CLIP used by Krea2."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("STRING", "KREA2_VISUAL_REFERENCE_CHAIN", "STRING")
    RETURN_NAMES = ("created_prompt", "visual_references", "creator_info")
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
            thinking=False,
            system_prompt=SYSTEM_PROMPT,
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
        created_prompt = _decode_generated_text(clip, generated_ids, generation_prompt)

        info_lines = [
            "=== Krea2 CcC Edit Prompt Creator ===",
            f"Mode: {mode}",
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

        return created_prompt, visual_references, "\n".join(info_lines)
