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
    assert meta["whether_interpolation_occurred"] is False
    assert meta["interpolation_method"] == "none"


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


def test_apply_statistical_style_fidelity_and_indirect():
    from ccc_krea2.style_processing import apply_statistical_style_fidelity
    cond = torch.randn(1, 100, 768)

    # Test fidelity blending
    spans_info_blend = [((10, 30), 0.5, False)]
    blended, ind, removed = apply_statistical_style_fidelity(cond, spans_info=spans_info_blend)
    assert blended.shape == cond.shape
    assert ind is False
    assert removed == []

    # Test indirect style transfer single span
    spans_info_ind = [((10, 30), 0.5, True)]
    sliced, ind, removed = apply_statistical_style_fidelity(cond, spans_info=spans_info_ind)
    assert sliced.shape == (1, 80, 768)
    assert ind is True
    assert len(removed) == 20

    # Test multi-span indirect style transfer in single pass
    spans_info_multi = [
        ((10, 20), 0.5, True),
        ((40, 50), 0.8, True)
    ]
    sliced_multi, ind_m, removed_m = apply_statistical_style_fidelity(cond, spans_info=spans_info_multi)
    assert sliced_multi.shape == (1, 80, 768)
    assert ind_m is True
    assert len(removed_m) == 20
    assert set(removed_m) == set(range(10, 20)).union(set(range(40, 50)))


def test_geometry_parity_floor_vs_round():
    from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry
    # Test genuine aspect ratio mismatch (src_ar = 2.0 vs tgt_ar = 1.0)
    # src = 200x100 (W=200, H=100), tgt = 500x500 (W=500, H=500)
    # sc = min(500/100, 500/200) = 2.5
    # fitted_h = min(500, max(16, int(100 * 2.5) // 16 * 16)) = 240
    # fitted_w = min(500, max(16, int(200 * 2.5) // 16 * 16)) = 500
    geom_fit_mismatch = resolve_krea2edit_geometry(src_h=100, src_w=200, tgt_h=500, tgt_w=500, fit_mode="fit")
    assert geom_fit_mismatch.mode_resolved == "fit"
    assert geom_fit_mismatch.vae_input_pixel_size == (496, 240)
    assert geom_fit_mismatch.vae_input_pixel_size[0] % 16 == 0
    assert geom_fit_mismatch.vae_input_pixel_size[1] % 16 == 0

    # Test manual fit mode with near-matched aspect ratio does not apply crop_only
    geom_fit = resolve_krea2edit_geometry(src_h=520, src_w=520, tgt_h=512, tgt_w=512, fit_mode="fit")
    assert geom_fit.mode_resolved == "crop_and_resize"  # Coverage >= 0.92 -> crop_and_resize, not crop_only

    # Test manual crop mode
    geom_crop = resolve_krea2edit_geometry(src_h=600, src_w=800, tgt_h=512, tgt_w=512, fit_mode="crop")
    assert geom_crop.mode_resolved == "crop"
    assert geom_crop.vae_input_pixel_size == (512, 512)

