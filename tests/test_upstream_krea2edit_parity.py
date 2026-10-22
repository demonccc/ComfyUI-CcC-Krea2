"""Geometry and grounded-encoding contracts matched to upstream Krea2Edit V2."""
from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry
from ccc_krea2.conditioning import KREA2_EDIT_GROUNDED_TEMPLATE


def test_fit_preserves_entire_mismatched_aspect_source():
    g = resolve_krea2edit_geometry(
        src_h=1024, src_w=512, tgt_h=1024, tgt_w=1024, fit_mode="fit"
    )
    assert g.vae_input_pixel_size == (512, 1024)
    assert g.crop_rectangle == (0, 0, 512, 1024)


def test_fit_near_matched_aspect_crops_to_target():
    g = resolve_krea2edit_geometry(
        src_h=1024, src_w=1000, tgt_h=1024, tgt_w=1024, fit_mode="fit"
    )
    assert g.vae_input_pixel_size == (1024, 1024)
    assert g.crop_rectangle[2] <= 1000


def test_fit_uses_multiple_of_16_grids():
    g = resolve_krea2edit_geometry(
        src_h=901, src_w=543, tgt_h=1024, tgt_w=1024, fit_mode="fit"
    )
    w, h = g.vae_input_pixel_size
    assert w % 16 == h % 16 == 0


def test_edit_qwen_template_is_training_aligned():
    text = KREA2_EDIT_GROUNDED_TEMPLATE.format("A test edit")
    assert "Describe the image by detailing the color, shape, size" in text
    assert "spatial relationships of the objects and background" in text
    assert "<|im_start|>system\n" in text
    assert "<|im_start|>user\nA test edit" in text
