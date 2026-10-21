"""Current Visual Reference geometry and mask alignment (historical fit removed)."""

import torch


def test_current_reference_geometry_modes_use_16_aligned_vae_inputs():
    from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry, process_image_and_mask_geometry

    tgt_h, tgt_w = 512, 512
    src_h, src_w = 1040, 520

    geom_contain = resolve_krea2edit_geometry(src_h, src_w, tgt_h, tgt_w, fit_mode="contain")
    assert geom_contain.mode_resolved == "contain"
    assert geom_contain.vae_input_pixel_size == (256, 512)
    assert geom_contain.vae_input_pixel_size[0] % 16 == 0
    assert geom_contain.vae_input_pixel_size[1] % 16 == 0

    geom_crop = resolve_krea2edit_geometry(src_h, src_w, tgt_h, tgt_w, fit_mode="crop")
    assert geom_crop.mode_resolved == "crop"
    assert geom_crop.vae_input_pixel_size == (512, 512)

    geom_native = resolve_krea2edit_geometry(701, 603, tgt_h, tgt_w, fit_mode="native")
    assert geom_native.vae_input_pixel_size == (592, 688)
    assert geom_native.interpolation_method == "none"

    src_img = torch.zeros((1, src_h, src_w, 3), dtype=torch.float32)
    src_mask = torch.zeros((1, src_h, src_w), dtype=torch.float32)
    res_img, res_mask = process_image_and_mask_geometry(src_img, src_mask, geom_contain)
    assert res_img.shape == (1, 512, 256, 3)
    assert res_mask.shape == (1, 512, 256)
