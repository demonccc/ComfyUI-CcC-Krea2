"""Unit tests for visual reference fit modes and Moodboard style processing."""

import torch
from ccc_krea2.krea2edit_geometry import resolve_visual_reference_fit
from ccc_krea2.style_processing import slice_style_image, expand_style_reference_spans
from ccc_krea2.vision_prep import prepare_vision_image
from ccc_krea2.reference_specs import StyleReferenceSpec


def test_visual_fit_exact():
    img = torch.rand(1, 512, 512, 3)
    out_img, _, meta = resolve_visual_reference_fit(img, target_h=512, target_w=512, mode="auto")
    assert meta["mode_resolved"] == "exact"
    assert out_img.shape == img.shape


def test_visual_fit_crop_only():
    img = torch.rand(1, 520, 520, 3)
    out_img, _, meta = resolve_visual_reference_fit(img, target_h=512, target_w=512, mode="auto")
    assert meta["mode_resolved"] == "crop_only"
    assert out_img.shape == (1, 512, 512, 3)


def test_visual_fit_crop_and_resize():
    img = torch.rand(1, 1000, 1024, 3)
    out_img, _, meta = resolve_visual_reference_fit(img, target_h=512, target_w=512, mode="auto")
    assert meta["mode_resolved"] == "crop_and_resize"
    assert out_img.shape == (1, 512, 512, 3)


def test_visual_fit_fit_mode():
    img = torch.rand(1, 1080, 1920, 3)
    out_img, _, meta = resolve_visual_reference_fit(img, target_h=512, target_w=512, mode="fit")
    assert meta["mode_resolved"] == "fit"
    # No black canvas padding, aligned to floor /16 dimensions
    fh, fw = meta["spatial_hw"]
    assert fh % 16 == 0
    assert fw % 16 == 0
    assert out_img.shape[1] == fh
    assert out_img.shape[2] == fw


def test_style_slicing():
    img = torch.rand(1, 1000, 1000, 3)
    crops_full = slice_style_image(img, mode="full")
    assert len(crops_full) == 1

    crops_2x2 = slice_style_image(img, mode="2x2")
    assert len(crops_2x2) == 4

    crops_4x4 = slice_style_image(img, mode="4x4")
    assert len(crops_4x4) == 16


def test_expand_style_reference_spans():
    img = torch.rand(1, 1000, 1000, 3)
    prep = prepare_vision_image(img, clip=None, mode="native")
    spec = StyleReferenceSpec(
        role="style",
        prepared_image=prep,
        requested_vision_slot=None,
        aliases_template="style_image",
        parsed_aliases=("style_image",),
        extra_vision_directive="",
        style_processing="2x2"
    )
    crops, s_start, s_end = expand_style_reference_spans(spec, start_slot=3, clip=None)
    assert len(crops) == 4
    assert s_start == 3
    assert s_end == 6
