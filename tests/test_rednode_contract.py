"""Regression contracts aligned with RedNode / Krea2 Identity Edit v1.2."""

import torch

from ccc_krea2.constants import VISION_PAD_TOKEN
from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry
from ccc_krea2.modular_nodes.edit_node import CcCKrea2Edit
from ccc_krea2.modular_nodes.visual_reference_node import CcCKrea2VisualReference
from ccc_krea2.patch import attach_reference_runtime_to_conditioning
from ccc_krea2.rednode_contract import build_grounded_negative_user_content


class _Spec:
    include_in_vision = True
    reference_path = "edit"
    appearance_reference = True
    alias = "subject image"
    vision_instruction = "Use only the subject face and body."


def test_grounded_negative_keeps_image_marker_but_drops_positive_annotations():
    text = build_grounded_negative_user_content([{"spec": _Spec()}], "")
    assert text == VISION_PAD_TOKEN
    assert "subject image" not in text
    assert "face and body" not in text


def test_grounded_negative_can_append_explicit_user_negative_without_reference_annotations():
    text = build_grounded_negative_user_content([{"spec": _Spec()}], "bad quality")
    assert text == VISION_PAD_TOKEN + "bad quality"
    assert "subject image" not in text


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
