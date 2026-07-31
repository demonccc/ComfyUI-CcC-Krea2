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

    res_img, _, meta = apply_reference_fit_transform(
        image=img,
        target_h=th,
        target_w=tw,
        mode="crop"
    )

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

    res_img, _, _ = apply_reference_fit_transform(
        image=img,
        target_h=th,
        target_w=tw,
        mode="crop"
    )

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
        res_img, _, _ = apply_reference_fit_transform(
            image=img,
            target_h=th,
            target_w=tw,
            mode="crop"
        )
        assert res_img.shape == (1, th, tw, 3)


def test_image_and_mask_remain_spatially_aligned():
    ih, iw = 800, 400
    th, tw = 400, 400
    img, _, _ = make_coordinate_gradient_image(ih, iw)

    # Mask active only where y >= 0.5 (bottom half of source image)
    y_grid = torch.linspace(0.0, 1.0, ih).unsqueeze(1).repeat(1, iw)
    mask = (y_grid >= 0.5).float()

    res_img, res_mask, _ = apply_reference_fit_transform(
        image=img,
        target_h=th,
        target_w=tw,
        mode="crop",
        mask=mask
    )

    assert res_img.shape == (1, th, tw, 3)
    assert res_mask.shape == (1, th, tw)

    # In cropped region (y=200..600), mask is 0 for top half (y < 400 -> y_grid < 0.5) and 1 for bottom half (y >= 400 -> y_grid >= 0.5)
    assert res_mask[0, 0, 0].item() == 0.0
    assert res_mask[0, -1, 0].item() == 1.0
