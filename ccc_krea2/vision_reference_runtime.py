"""Translate the two Vision presets to the shared Qwen reference contract."""

import torch
from .reference_specs import PreparedVisionImage, ReferenceChain, ReferenceSpec, VisionPrepSpec
from .modular_nodes.edit_reference_types import VisionReferenceChain, VisionReferenceEntry


def prepare_tokenizer_native_image(image: torch.Tensor) -> PreparedVisionImage:
    """Preserve image pixels until Qwen's processor owns geometry."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image[..., :3].clamp(0.0, 1.0)
    h, w = int(image.shape[1]), int(image.shape[2])
    return PreparedVisionImage(
        original_image=image,
        vision_image=image,
        prep_spec=VisionPrepSpec(
            mode="tokenizer_native", semantic_min_mp=0.0, semantic_max_mp=0.0,
            semantic_fixed_mp=0.0, downscale_method_requested="none",
            upscale_method_requested="none", encoder_signature="Qwen tokenizer-native",
            resolved_alignment=None, resolved_native_limits=None,
        ),
        debug_metadata={
            "src_hw": (h, w), "prep_hw": (h, w), "target_hw": (h, w),
            "direction": "none", "resolved_method": "none",
            "additional_adjustment": "owned by Qwen tokenizer",
        },
    )


def build_vision_reference_spec(entry: VisionReferenceEntry) -> ReferenceSpec:
    """Only token retention differs between Visual and Semantic."""
    if entry.mode not in ("visual", "semantic"):
        raise ValueError(f"Unsupported Vision Reference mode: {entry.mode}")
    return ReferenceSpec(
        reference_path="edit",
        prepared_image=prepare_tokenizer_native_image(entry.image),
        vision_instruction=entry.instruction.strip(),
        appearance_reference=False,
        include_in_vision=True,
        vision_weight=float(entry.vision_weight),
        vision_reference_mode=entry.mode,
    )


def append_vision_references(chain: ReferenceChain, vision_references: VisionReferenceChain) -> ReferenceChain:
    for entry in vision_references.entries:
        chain = chain.append(build_vision_reference_spec(entry))
    return chain
