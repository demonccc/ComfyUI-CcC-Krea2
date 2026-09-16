import torch

from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry
from ccc_krea2.modular_nodes.rope_position import build_incontext_3d_rope_pos_ids


def test_fit_mismatch_uses_minimal_crop_to_preserve_snapped_scale():
    geom = resolve_krea2edit_geometry(
        src_h=2059,
        src_w=971,
        tgt_h=1408,
        tgt_w=1408,
        fit_mode="fit",
    )

    assert geom.mode_resolved == "fit"
    assert geom.vae_input_pixel_size == (656, 1408)
    assert geom.crop_rectangle == (6, 0, 959, 2059)
    assert geom.vae_latent_grid_size == (82, 176)


def test_centered_rope_uses_fractional_half_token_offset():
    pos = build_incontext_3d_rope_pos_ids(
        batch_size=1,
        txt_len=0,
        ref_token_grids=[(88, 41)],
        target_grid=(88, 88),
        device=torch.device("cpu"),
        ref_rope_positions=["inside:center:center"],
    )

    # First reference token is frame 1 at y=0, x=(88-41)/2 = 23.5.
    assert float(pos[0, 0, 0]) == 1.0
    assert float(pos[0, 0, 1]) == 0.0
    assert float(pos[0, 0, 2]) == 23.5
