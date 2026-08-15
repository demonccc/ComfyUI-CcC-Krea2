"""Unit tests for geometry transformations using coordinate-gradient images."""

from typing import Tuple
import torch
from ccc_krea2.geometry import apply_reference_fit_transform


def make_coordinate_gradient_image(height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates a (1, H, W, 3) image with linear y and x coordinate gradients."""
    y = torch.linspace(0.0, 1.0, height).unsqueeze(1).repeat(1, width)
    x = torch.linspace(0.0, 1.0, width).unsqueeze(0).repeat(height, 1)
    z = torch.zeros((height, width))
    img = torch.stack([y, x, z], dim=-1).unsqueeze(0)
    return img, y, x


def test_portrait_to_square_crops_top_bottom():
    # Portrait: H=800, W=400 -> Target 400x400
    ih, iw = 800, 400
    th, tw = 400, 400
    img, _, _ = make_coordinate_gradient_image(ih, iw)

    res_img, _, meta = apply_reference_fit_transform(image=img, target_h=th, target_w=tw, mode="crop")

    # Output spatial dimensions match target
    assert res_img.shape == (1, th, tw, 3)

    # Red channel contains y-gradient. Original y=0..1 mapped across 800px.
    # Center 400px crop spans y=200..600 -> gradient range ~0.25..0.75
    y_red_top = res_img[0, 0, 0, 0].item()
    y_red_bottom = res_img[0, -1, 0, 0].item()

    assert 0.20 <= y_red_top <= 0.30
    assert 0.70 <= y_red_bottom <= 0.80

    # Green channel contains x-gradient (un-cropped full 0..1 range)
    x_green_left = res_img[0, 0, 0, 1].item()
    x_green_right = res_img[0, 0, -1, 1].item()
    assert x_green_left < 0.05
    assert x_green_right > 0.95


def test_landscape_to_square_crops_left_right():
    # Landscape: H=400, W=800 -> Target 400x400
    ih, iw = 400, 800
    th, tw = 400, 400
    img, _, _ = make_coordinate_gradient_image(ih, iw)

    res_img, _, _ = apply_reference_fit_transform(image=img, target_h=th, target_w=tw, mode="crop")

    assert res_img.shape == (1, th, tw, 3)

    # Green channel contains x-gradient. Center 400px crop spans x=200..600 -> gradient ~0.25..0.75
    x_green_left = res_img[0, 0, 0, 1].item()
    x_green_right = res_img[0, 0, -1, 1].item()

    assert 0.20 <= x_green_left <= 0.30
    assert 0.70 <= x_green_right <= 0.80

    # Red channel contains y-gradient (un-cropped full 0..1 range)
    y_red_top = res_img[0, 0, 0, 0].item()
    y_red_bottom = res_img[0, -1, 0, 0].item()
    assert y_red_top < 0.05
    assert y_red_bottom > 0.95


def test_crop_dimensions_never_exceed_source():
    # Test various extreme aspect ratios
    test_cases = [
        (100, 1000, 512, 512),
        (1000, 100, 512, 512),
        (300, 400, 1024, 1024),
    ]
    for ih, iw, th, tw in test_cases:
        img = torch.rand((1, ih, iw, 3))
        res_img, _, _ = apply_reference_fit_transform(image=img, target_h=th, target_w=tw, mode="crop")
        assert res_img.shape == (1, th, tw, 3)


def test_image_and_mask_remain_spatially_aligned():
    ih, iw = 800, 400
    th, tw = 400, 400
    img, _, _ = make_coordinate_gradient_image(ih, iw)

    # Mask active only where y >= 0.5 (bottom half of source image)
    y_grid = torch.linspace(0.0, 1.0, ih).unsqueeze(1).repeat(1, iw)
    mask = (y_grid >= 0.5).float()

    res_img, res_mask, _ = apply_reference_fit_transform(image=img, target_h=th, target_w=tw, mode="crop", mask=mask)

    assert res_img.shape == (1, th, tw, 3)
    assert res_mask.shape == (1, th, tw)

    # In cropped region (y=200..600), mask is 0 for top half (y < 400 -> y_grid < 0.5) and 1 for bottom half (y >= 400 -> y_grid >= 0.5)
    assert res_mask[0, 0, 0].item() == 0.0
    assert res_mask[0, -1, 0].item() == 1.0


def test_near_match_geometry_divisible_by_8_not_16():
    """Assert crop_and_resize equals exact target dimensions when target is divisible by 8 but not 16."""
    from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry, process_image_and_mask_geometry

    # Target: 520x520 (520 % 8 == 0, 520 % 16 == 8)
    tgt_h, tgt_w = 520, 520
    # Near-match source: 528x520 (ratio 1.015 vs 1.0 -> within 0.15 threshold for near-match crop_and_resize in Auto mode)
    src_h, src_w = 528, 520

    # 1. Fit mode with near-match coverage -> resolves to crop_and_resize
    geom_near = resolve_krea2edit_geometry(540, 540, tgt_h, tgt_w, fit_mode="fit")
    assert geom_near.mode_resolved == "crop_and_resize"
    assert geom_near.vae_input_pixel_size == (520, 520)

    # 2. Manual crop mode -> exact target dimensions
    geom_crop = resolve_krea2edit_geometry(src_h, src_w, tgt_h, tgt_w, fit_mode="crop")
    assert geom_crop.mode_resolved == "crop"
    assert geom_crop.vae_input_pixel_size == (520, 520)

    # 3. Genuine mismatch source: 1040x520 (aspect ratio 2.0 vs 1.0 -> resolves to fit)
    geom_fit = resolve_krea2edit_geometry(1040, 520, tgt_h, tgt_w, fit_mode="fit")
    assert geom_fit.mode_resolved == "fit"
    # Genuine fit truncates to /16: 520 // 16 * 16 = 512, 260 // 16 * 16 = 256
    assert geom_fit.vae_input_pixel_size[0] % 16 == 0
    assert geom_fit.vae_input_pixel_size[1] % 16 == 0

    # 4. Image and mask transform parity
    src_img = torch.zeros((1, src_h, src_w, 3), dtype=torch.float32)
    src_mask = torch.zeros((1, src_h, src_w), dtype=torch.float32)

    res_img, res_mask = process_image_and_mask_geometry(src_img, src_mask, geom_near)
    assert res_img.shape == (1, 520, 520, 3)
    assert res_mask.shape == (1, 520, 520)
