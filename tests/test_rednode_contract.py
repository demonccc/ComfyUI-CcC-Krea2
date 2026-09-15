"""Regression contracts aligned with RedNode / Krea2 Identity Edit v1.2."""

import torch

from ccc_krea2.constants import VISION_PAD_TOKEN
from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry
from ccc_krea2.modular_nodes.edit_node import CcCKrea2Edit, _prepare_qwen_image
from ccc_krea2.modular_nodes.visual_reference_node import CcCKrea2VisualReference
from ccc_krea2.patch import attach_reference_runtime_to_conditioning
from ccc_krea2.rednode_contract import (
    build_grounded_negative_user_content,
    build_grounded_positive_user_content,
)


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


def test_grounded_positive_matches_rednode_when_no_annotations_are_requested():
    refs = [{"spec": _PlainSpec()}, {"spec": _PlainSpec()}]
    text = build_grounded_positive_user_content(refs, "Change the pose.")
    assert text == VISION_PAD_TOKEN * 2 + "Change the pose."


def test_grounded_positive_keeps_all_vision_blocks_contiguous_before_annotations():
    refs = [
        {"spec": _PlainSpec()},
        {"spec": _Spec(), "expanded_aliases": ("subject image",)},
    ]
    text = build_grounded_positive_user_content(refs, "Replace the woman.")
    prefix = VISION_PAD_TOKEN * 2
    assert text.startswith(prefix)
    assert text.count(VISION_PAD_TOKEN) == 2
    assert text.find("subject image") >= len(prefix)
    assert text.find("face and body") >= len(prefix)
    assert text.find("Replace the woman.") >= len(prefix)


def test_grounded_negative_keeps_image_marker_but_drops_positive_annotations():
    text = build_grounded_negative_user_content([{"spec": _Spec()}], "")
    assert text == VISION_PAD_TOKEN
    assert "subject image" not in text
    assert "face and body" not in text


def test_grounded_negative_can_append_explicit_user_negative_without_reference_annotations():
    text = build_grounded_negative_user_content([{"spec": _Spec()}], "bad quality")
    assert text == VISION_PAD_TOKEN + "bad quality"
    assert "subject image" not in text


def test_qwen_grounding_matches_rednode_area_cap_without_pre_alignment():
    # RedNode downsizes the longest side to grounding_px and lets Qwen own its native
    # patch/merge alignment. 971x2059 at 768 therefore becomes 362x768 here, not 352x768.
    image = torch.zeros((1, 2059, 971, 3))
    prepared = _prepare_qwen_image(image=image, clip=None, grounding_px=768)
    assert prepared.original_image.shape == (1, 2059, 971, 3)
    assert prepared.vision_image.shape == (1, 768, 362, 3)
    assert prepared.debug_metadata["resolved_method"] == "area"
    assert prepared.debug_metadata["additional_adjustment"] == "owned by Qwen tokenizer"


def test_qwen_grounding_zero_keeps_native_pixels_for_tokenizer():
    image = torch.zeros((1, 1164, 1719, 3))
    prepared = _prepare_qwen_image(image=image, clip=None, grounding_px=0)
    assert prepared.vision_image.shape == image.shape
    assert prepared.debug_metadata["resolved_method"] == "none"


def test_v12_fit_preserves_complete_portrait_source_on_square_target():
    geom = resolve_krea2edit_geometry(
        src_h=2059,
        src_w=965,
        tgt_h=1408,
        tgt_w=1408,
        fit_mode="fit",
    )
    assert geom.mode_resolved == "fit"
    assert geom.crop_rectangle == (0, 0, 965, 2059)
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


def test_public_edit_and_visual_reference_surfaces_remain_unchanged():
    visual_required = CcCKrea2VisualReference.INPUT_TYPES()["required"]
    assert list(visual_required) == [
        "image",
        "boost",
        "rope_grid",
        "rope_horizontal",
        "rope_vertical",
        "semantic",
        "semantic_role",
        "instruction",
        "grounding_px",
    ]

    edit_inputs = CcCKrea2Edit.INPUT_TYPES()
    assert list(edit_inputs["required"]) == [
        "model",
        "clip",
        "vae",
        "latent",
        "positive_prompt",
        "negative_prompt",
        "apply_krea2_edit_patch",
    ]
    assert list(edit_inputs["optional"]) == ["visual_references", "semantic_references"]
