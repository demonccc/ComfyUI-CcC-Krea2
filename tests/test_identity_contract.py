"""Regression contracts for the Krea2 CcC Edit path."""

import torch

from ccc_krea2.constants import VISION_PAD_TOKEN
from ccc_krea2.identity_contract import (
    build_grounded_negative_user_content,
    build_grounded_positive_user_content,
)
from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry
from ccc_krea2.modular_nodes.edit_node import CcCKrea2Edit, _append_semantic, _prepare_qwen_image
from ccc_krea2.modular_nodes.visual_reference_node import CcCKrea2VisualReference
from ccc_krea2.modular_nodes.semantic_reference_node import CcCKrea2SemanticReference
from ccc_krea2.reference_specs import ReferenceChain
from ccc_krea2.patch import attach_reference_runtime_to_conditioning


class _Spec:
    include_in_vision = True
    reference_path = "edit"
    appearance_reference = True
    alias = "subject image"
    vision_instruction = "Use only the subject face and body."


class _PlainSpec:
    include_in_vision = True
    reference_path = "edit"
    appearance_reference = True
    alias = ""
    vision_instruction = ""


class _SemanticOnlySpec:
    include_in_vision = True
    reference_path = "edit"
    appearance_reference = False
    alias = "target image"
    vision_instruction = "Use this image only as semantic target context."


def test_grounded_positive_supports_optional_visual_prompt_annotation():
    refs = [
        {"spec": _PlainSpec()},
        {"spec": _Spec(), "expanded_aliases": ("subject image",)},
    ]
    text = build_grounded_positive_user_content(refs, "Replace the woman.")
    prefix = VISION_PAD_TOKEN * 2
    assert text.startswith(prefix)
    assert "Image 2: Use only the subject face and body." in text
    assert text.endswith("Replace the woman.")


def test_ccc_semantic_only_extension_may_add_text_after_complete_vision_prefix():
    refs = [
        {"spec": _PlainSpec()},
        {"spec": _SemanticOnlySpec(), "expanded_aliases": ("target image",)},
    ]
    text = build_grounded_positive_user_content(refs, "Change the pose.")
    prefix = VISION_PAD_TOKEN * 2
    assert text.startswith(prefix)
    assert text.count(VISION_PAD_TOKEN) == 2
    assert text.find("target image") >= len(prefix)
    assert text.find("semantic target context") >= len(prefix)
    assert text.find("Change the pose.") >= len(prefix)


def test_grounded_negative_keeps_image_marker_but_drops_positive_annotations():
    text = build_grounded_negative_user_content([{"spec": _Spec()}], "")
    assert text == VISION_PAD_TOKEN
    assert "subject image" not in text
    assert "face and body" not in text


def test_grounded_negative_can_append_explicit_user_negative_without_reference_annotations():
    text = build_grounded_negative_user_content([{"spec": _Spec()}], "bad quality")
    assert text == VISION_PAD_TOKEN + "bad quality"
    assert "subject image" not in text


def test_qwen_grounding_uses_lanczos_cap_without_pre_alignment():
    # Longest-side downscale remains tokenizer-native: 971x2059 at 768 becomes
    # 362x768 here rather than being pre-aligned to a vision patch multiple.
    image = torch.zeros((1, 2059, 971, 3))
    prepared = _prepare_qwen_image(image=image, clip=None, grounding_px=768)
    assert prepared.original_image.shape == (1, 2059, 971, 3)
    assert prepared.vision_image.shape == (1, 768, 362, 3)
    assert prepared.debug_metadata["resolved_method"] == "lanczos"
    assert prepared.debug_metadata["additional_adjustment"] == "owned by Qwen tokenizer"


def test_qwen_grounding_zero_keeps_native_pixels_for_tokenizer():
    image = torch.zeros((1, 1164, 1719, 3))
    prepared = _prepare_qwen_image(image=image, clip=None, grounding_px=0)
    assert prepared.vision_image.shape == image.shape
    assert prepared.debug_metadata["resolved_method"] == "none"


def test_v12_fit_contains_portrait_with_minimal_alignment_crop():
    geom = resolve_krea2edit_geometry(
        src_h=2059,
        src_w=965,
        tgt_h=1408,
        tgt_w=1408,
        fit_mode="fit",
    )
    assert geom.mode_resolved == "contain"
    assert geom.crop_rectangle == (3, 0, 959, 2059)
    assert geom.vae_input_pixel_size == (656, 1408)
    assert geom.vae_latent_grid_size == (82, 176)


def test_conditioning_runtime_keeps_positive_boost_and_negative_neutral():
    base = [[torch.zeros((1, 4, 8)), {}]]
    positive = attach_reference_runtime_to_conditioning(
        base,
        reference_count=1,
        rope_positions=["inside:center:center"],
        reference_boosts=[4.0],
    )
    negative = attach_reference_runtime_to_conditioning(
        base,
        reference_count=1,
        rope_positions=["inside:center:center"],
        reference_boosts=None,
    )

    assert positive[0][1]["reference_fit"] == [True]
    assert positive[0][1]["reference_boosts"] == [4.0]
    assert positive[0][1]["reference_rope_positions"] == ["inside:center:center"]
    assert negative[0][1]["reference_fit"] == [True]
    assert "reference_boosts" not in negative[0][1]


def test_semantic_only_uses_subject_extraction_without_appearance_latent():
    image = torch.zeros((1, 512, 384, 3))
    chain = _append_semantic(
        chain=ReferenceChain(),
        image=image,
        clip=None,
        instruction="Keep the complete scene semantics.",
        grounding_px=768,
        mode="semantic_only",
        processing="2x2",
        fidelity=0.5,
    )

    spec = chain[0]
    assert spec.appearance_reference is False
    assert spec.include_in_vision is True
    assert spec.semantic_extract == "subject"
    assert spec.semantic_strength == 0.5
    assert "pose" in spec.vision_instruction
    assert "surrounding people" in spec.vision_instruction
    assert "background" in spec.vision_instruction
    assert "Keep the complete scene semantics." in spec.vision_instruction


def test_semantic_reference_forces_full_image_for_semantic_only():
    node = CcCKrea2SemanticReference()
    image = torch.zeros((1, 256, 256, 3))
    (chain,) = node.process(
        image=image,
        mode="semantic_only",
        instruction="",
        grounding_px=768,
        processing="4x4",
        fidelity=0.5,
    )
    assert chain.entries[0].processing == "full"

    required = CcCKrea2SemanticReference.INPUT_TYPES()["required"]
    assert required["processing"][1]["default"] == "full"
    assert required["fidelity"][1]["default"] == 0.5


def test_public_edit_and_visual_reference_surfaces_match_refactored_contract():
    visual_required = CcCKrea2VisualReference.INPUT_TYPES()["required"]
    assert list(visual_required) == [
        "image",
        "boost",
        "reference_fit",
        "placement_grid",
        "grid_horizontal_position",
        "grid_vertical_position",
        "resize_method",
        "semantic",
        "semantic_resize",
        "semantic_grounding_px",
        "semantic_resize_method",
        "prompt_annotation",
    ]

    edit_inputs = CcCKrea2Edit.INPUT_TYPES()
    assert list(edit_inputs["required"]) == [
        "model",
        "clip",
        "vae",
        "latent",
        "positive_prompt",
        "apply_krea2_edit_patch",
    ]
    assert list(edit_inputs["optional"]) == ["visual_references", "semantic_references"]
