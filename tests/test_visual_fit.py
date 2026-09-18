"""Tests for current visual-reference geometry and semantic/style processing."""

import torch

from ccc_krea2.krea2edit_geometry import resolve_visual_reference_fit
from ccc_krea2.reference_specs import StyleReferenceSpec
from ccc_krea2.style_processing import (
    apply_statistical_style_fidelity,
    expand_style_reference_spans,
    slice_style_image,
)
from ccc_krea2.vision_prep import prepare_vision_image


def test_public_crop_uses_positioned_inside_grid_window():
    img = torch.rand(1, 900, 1200, 3)
    out_img, _, meta = resolve_visual_reference_fit(
        img,
        target_h=512,
        target_w=640,
        mode="crop",
        grid_horizontal_position="left",
        grid_vertical_position="down",
    )
    assert meta["mode_resolved"] == "crop_window"
    assert meta["crop_rectangle"] == (0, 388, 640, 512)
    assert out_img.shape == (1, 512, 640, 3)
    assert meta["interpolation_method"] == "none"


def test_public_resize_always_scales_up_or_down_with_selected_method():
    small = torch.rand(1, 256, 128, 3)
    out_small, _, meta_small = resolve_visual_reference_fit(
        small,
        target_h=1024,
        target_w=1024,
        mode="resize",
        resize_method="lanczos",
    )
    assert meta_small["mode_resolved"] == "resize"
    assert out_small.shape == (1, 1024, 512, 3)
    assert meta_small["interpolation_method"] == "lanczos"

    large = torch.rand(1, 2048, 1024, 3)
    out_large, _, meta_large = resolve_visual_reference_fit(
        large,
        target_h=1024,
        target_w=1024,
        mode="resize",
        resize_method="bicubic",
    )
    assert out_large.shape == (1, 1024, 512, 3)
    assert meta_large["interpolation_method"] == "bicubic"


def test_public_native_preserves_pixels_except_vae_alignment():
    img = torch.rand(1, 701, 603, 3)
    out_img, _, meta = resolve_visual_reference_fit(
        img,
        target_h=512,
        target_w=512,
        mode="native",
    )
    assert meta["mode_resolved"] == "native"
    assert out_img.shape == (1, 704, 608, 3)
    assert meta["interpolation_method"] == "pad"


def test_style_slicing():
    img = torch.rand(1, 1000, 1000, 3)
    assert len(slice_style_image(img, mode="full")) == 1
    assert len(slice_style_image(img, mode="2x2")) == 4
    assert len(slice_style_image(img, mode="4x4")) == 16


def test_expand_style_reference_spans():
    img = torch.rand(1, 1000, 1000, 3)
    prep = prepare_vision_image(img, clip=None, mode="native")
    spec = StyleReferenceSpec(
        prepared_image=prep,
        alias="style_image",
        parsed_aliases=("style_image",),
        style_processing="2x2",
    )
    crops, start, end = expand_style_reference_spans(spec, start_slot=3, clip=None)
    assert len(crops) == 4
    assert start == 3
    assert end == 6


def test_apply_statistical_style_fidelity_and_indirect():
    cond = torch.randn(1, 100, 768)

    blended, indirect, removed = apply_statistical_style_fidelity(
        cond,
        spans_info=[((10, 30), 0.5, False)],
    )
    assert blended.shape == cond.shape
    assert indirect is False
    assert removed == []

    sliced, indirect, removed = apply_statistical_style_fidelity(
        cond,
        spans_info=[((10, 30), 0.5, True)],
    )
    assert sliced.shape == (1, 80, 768)
    assert indirect is True
    assert len(removed) == 20

    sliced_multi, indirect_multi, removed_multi = apply_statistical_style_fidelity(
        cond,
        spans_info=[((10, 20), 0.5, True), ((40, 50), 0.8, True)],
    )
    assert sliced_multi.shape == (1, 80, 768)
    assert indirect_multi is True
    assert len(removed_multi) == 20
    assert set(removed_multi) == set(range(10, 20)).union(set(range(40, 50)))
