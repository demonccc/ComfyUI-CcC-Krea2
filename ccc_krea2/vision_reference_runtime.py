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


from .vision_presets import (
    SEMANTIC_DIRECTIVE, VISUAL_DIRECTIVE, STYLE_DIRECTIVE, CONTEXT_DIRECTIVE
)


def _with_user_instruction(base: str, instruction: str, scope: str) -> str:
    """Apply custom guidance without contradicting the selected Vision preset."""
    user = (instruction or "").strip()
    if not user:
        return base
    return (
        f"{base} Additional {scope} guidance (subject to the mode's preservation rules): {user}"
    )


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

    if entry.mode == "original":
        return ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            vision_instruction=entry.instruction.strip(),
            appearance_reference=False,
            include_in_vision=True,
            vision_reference_mode="original",
            vision_summary_rows=0,
            vision_indirect=False,
            semantic_extract="none",
            semantic_strength=1.0,
        )

    if entry.mode == "semantic_only":
        return ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            vision_instruction=_with_user_instruction(SEMANTIC_DIRECTIVE, entry.instruction, "semantic"),
            appearance_reference=False,
            include_in_vision=True,
            vision_reference_mode="semantic",
            vision_summary_rows=0,
            vision_indirect=False,
            semantic_extract="subject",
            semantic_strength=0.25,
        )

    if entry.mode == "style_direct":
        return ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            vision_instruction=_with_user_instruction(VISUAL_DIRECTIVE, entry.instruction, "visual"),
            appearance_reference=False,
            include_in_vision=True,
            vision_reference_mode="visual",
            vision_summary_rows=0,
            vision_indirect=False,
            indirect_style_transfer=False,
        )

    if entry.mode == "style_indirect":
        return StyleReferenceSpec(
            prepared_image=prep,
            vision_instruction=_with_user_instruction(STYLE_DIRECTIVE, entry.instruction, "style"),
            appearance_reference=False,
            include_in_vision=True,
            vision_reference_mode="style",
            vision_summary_rows=0,
            vision_indirect=False,
            style_processing=entry.processing,
            style_tile_resize=entry.tile_resize,
            style_tile_max_edge=entry.tile_max_edge,
            style_fidelity=0.0,
            indirect_style_transfer=False,
        )

    if entry.mode == "context_indirect":
        return ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            vision_instruction=_with_user_instruction(CONTEXT_DIRECTIVE, entry.instruction, "context"),
            appearance_reference=False,
            include_in_vision=True,
            vision_reference_mode="context",
            # Keep spatial rows intact: whitening and subsampling are not a
            # spatial-context extractor and erase gaze/pose/interaction detail.
            vision_summary_rows=0,
            vision_indirect=False,
            semantic_strength=1.0,
            indirect_vision_context=False,
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
