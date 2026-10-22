"""Keep project-defined reference framing separate from Qwen grounding."""
from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry
from ccc_krea2.conditioning import KREA2_EDIT_GROUNDED_TEMPLATE


def test_project_fit_covers_target_without_changing_public_geometry():
    geom = resolve_krea2edit_geometry(
        src_h=1024, src_w=512, tgt_h=1024, tgt_w=1024, fit_mode="fit"
    )
    assert geom.vae_input_pixel_size == (1024, 1024)
    assert geom.crop_rectangle[2] <= 512


def test_edit_qwen_grounding_instruction_is_present():
    prompt = KREA2_EDIT_GROUNDED_TEMPLATE.format("A test edit")
    assert "Describe the image by detailing the color, shape, size" in prompt
    assert "spatial relationships of the objects and background" in prompt
