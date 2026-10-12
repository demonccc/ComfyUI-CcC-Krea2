"""Shared runtime translation for Krea2 CcC Vision References."""

import torch

from .reference_specs import (
    PreparedVisionImage,
    ReferenceChain,
    ReferenceSpec,
    StyleReferenceSpec,
    VisionPrepSpec,
)
from .modular_nodes.edit_reference_types import VisionReferenceChain, VisionReferenceEntry


SEMANTIC_DIRECTIVE = (
    "Use this reference only for semantic content: subjects, objects, attributes, actions, pose, and scene meaning. "
    "Apply that semantic information to the result requested by the user's prompt; the reference does not introduce "
    "an additional independent subject. Ignore identity, exact appearance, rendering, lighting, palette, and style."
)

VISUAL_DIRECTIVE = (
    "Use this reference as direct visual guidance for the subject/result requested by the user's prompt. "
    "Preserve visible appearance and details, but do not duplicate the reference subject as an additional subject."
)

STYLE_DIRECTIVE = (
    "Use this reference only as a style guide: adopt its rendering, line work, shading, color palette, lighting, "
    "texture, medium, tone, and mood. Do not copy or introduce its subjects, people, identities, clothing, objects, "
    "layout, pose, or composition."
)

CONTEXT_DIRECTIVE = (
    "Use this reference only for situation, action, framing, environment, spatial relationships, and scene context. "
    "Do not copy identities, clothing, exact appearance, or style, and do not introduce a reference person as an "
    "additional subject unless the user's prompt explicitly requires that extra role."
)


def _with_user_instruction(base: str, instruction: str) -> str:
    user = (instruction or "").strip()
    if not user:
        return base
    return f"{base} Additional guidance, only within this mode's scope: {user}"


def prepare_tokenizer_native_image(image: torch.Tensor) -> PreparedVisionImage:
    """Keep source pixels untouched until the Qwen tokenizer/vision processor owns native geometry."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image[..., :3].clamp(0.0, 1.0)
    h, w = int(image.shape[1]), int(image.shape[2])
    return PreparedVisionImage(
        original_image=image,
        vision_image=image,
        prep_spec=VisionPrepSpec(
            mode="tokenizer_native",
            semantic_min_mp=0.0,
            semantic_max_mp=0.0,
            semantic_fixed_mp=0.0,
            downscale_method_requested="none",
            upscale_method_requested="none",
            encoder_signature="Qwen tokenizer-native",
            resolved_alignment=None,
            resolved_native_limits=None,
        ),
        debug_metadata={
            "src_hw": (h, w),
            "prep_hw": (h, w),
            "target_hw": (h, w),
            "direction": "none",
            "resolved_method": "none",
            "additional_adjustment": "owned by Qwen tokenizer",
        },
    )


def build_vision_reference_spec(entry: VisionReferenceEntry):
    """Translate one public Vision Reference entry into the proven internal reference contract."""
    prep = prepare_tokenizer_native_image(entry.image)

    if entry.mode == "semantic_only":
        return ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            vision_instruction=_with_user_instruction(SEMANTIC_DIRECTIVE, entry.instruction),
            appearance_reference=False,
            include_in_vision=True,
            vision_reference_mode="semantic",
            vision_summary_rows=0,
            vision_indirect=True,
            semantic_extract="none",
            semantic_strength=1.0,
        )

    if entry.mode == "style_direct":
        return ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            vision_instruction=_with_user_instruction(VISUAL_DIRECTIVE, entry.instruction),
            appearance_reference=False,
            include_in_vision=True,
            vision_reference_mode="visual",
            vision_summary_rows=64,
            vision_indirect=False,
            indirect_style_transfer=False,
        )

    if entry.mode == "style_indirect":
        return StyleReferenceSpec(
            prepared_image=prep,
            vision_instruction=_with_user_instruction(STYLE_DIRECTIVE, entry.instruction),
            appearance_reference=False,
            include_in_vision=True,
            vision_reference_mode="style",
            vision_summary_rows=0,
            vision_indirect=True,
            style_processing=entry.processing,
            style_fidelity=1.0,
            indirect_style_transfer=True,
        )

    if entry.mode == "context_indirect":
        return ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            vision_instruction=_with_user_instruction(CONTEXT_DIRECTIVE, entry.instruction),
            appearance_reference=False,
            include_in_vision=True,
            vision_reference_mode="context",
            vision_summary_rows=0,
            vision_indirect=True,
            semantic_strength=1.0,
            indirect_vision_context=True,
        )

    raise ValueError(f"Unsupported Vision Reference runtime mode: {entry.mode}")


def append_vision_references(
    chain: ReferenceChain,
    vision_references: VisionReferenceChain,
) -> ReferenceChain:
    """Append non-style Vision refs first and style refs last, preserving the Krea2 slot contract."""
    pending_style = []
    for entry in vision_references.entries:
        spec = build_vision_reference_spec(entry)
        if getattr(spec, "reference_path", "") == "style":
            pending_style.append(spec)
        else:
            chain = chain.append(spec)

    for spec in pending_style:
        chain = chain.append(spec)
    return chain
