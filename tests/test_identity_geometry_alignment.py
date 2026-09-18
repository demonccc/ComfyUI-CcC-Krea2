import torch

from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry
from ccc_krea2.modular_nodes.rope_position import build_incontext_3d_rope_pos_ids


def test_fit_mismatch_preserves_complete_source_on_snapped_grid():
    geom = resolve_krea2edit_geometry(
        src_h=2059,
        src_w=971,
        tgt_h=1408,
        tgt_w=1408,
        fit_mode="fit",
    )

    assert geom.mode_resolved == "fit"
    assert geom.vae_input_pixel_size == (656, 1408)
    assert geom.crop_rectangle == (0, 0, 971, 2059)
    assert geom.vae_latent_grid_size == (82, 176)


def test_current_portrait_subject_fit_preserves_complete_source_inside_target():
    geom = resolve_krea2edit_geometry(
        src_h=2059,
        src_w=971,
        tgt_h=2064,
        tgt_w=1376,
        fit_mode="fit",
    )

    assert geom.mode_resolved == "fit"
    assert geom.crop_rectangle == (0, 0, 971, 2059)
    assert geom.vae_input_pixel_size == (960, 2064)
    assert geom.vae_latent_grid_size == (120, 258)
    assert geom.target_grid_size == (172, 258)


def test_centered_rope_uses_current_integer_center_offset():
    pos = build_incontext_3d_rope_pos_ids(
        batch_size=1,
        txt_len=0,
        ref_token_grids=[(88, 41)],
        target_grid=(88, 88),
        device=torch.device("cpu"),
        ref_rope_positions=["inside:center:center"],
    )

    # Current CcC behavior uses integer floor division: (88-41)//2 = 23.
    assert float(pos[0, 0, 0]) == 1.0
    assert float(pos[0, 0, 1]) == 0.0
    assert float(pos[0, 0, 2]) == 23.0
