"""Unit tests for Krea2 CcC Latent geometry and image-fit policies."""

import pytest
import torch

from ccc_krea2.modular_nodes.latent_node import (
    _fit_content_image,
    _nearest_krea_geometry,
    _preserve_aspect_krea_bounds,
)


def test_nearest_krea_geometry_uses_known_preset_geometry():
    assert _nearest_krea_geometry(1920, 1080) == (2048, 1152)


def test_preserve_aspect_keeps_already_valid_geometry():
    assert _preserve_aspect_krea_bounds(1536, 1024) == (1536, 1024)


def test_preserve_aspect_scales_small_geometry_into_krea_bounds():
    assert _preserve_aspect_krea_bounds(800, 1200) == (1024, 1536)


def test_preserve_aspect_scales_large_geometry_into_krea_bounds():
    width, height = _preserve_aspect_krea_bounds(4000, 2667)
    assert width == 2048
    assert 1024 <= height <= 2048
    assert width % 16 == 0
    assert height % 16 == 0


def test_preserve_aspect_rejects_geometry_that_cannot_fit_bounds():
    with pytest.raises(ValueError, match="cannot fit this aspect ratio"):
        _preserve_aspect_krea_bounds(3000, 1000)


def test_content_fit_crop_fills_target_without_padding():
    image = torch.rand(1, 100, 200, 3)
    fitted, meta = _fit_content_image(
        image=image,
        target_w=100,
        target_h=100,
        content_fit="crop",
        resize_method="bilinear",
    )
    assert fitted.shape == (1, 100, 100, 3)
    assert meta["padding"] == (0, 0, 0, 0)


def test_content_fit_contain_uses_white_padding():
    image = torch.zeros(1, 100, 200, 3)
    fitted, meta = _fit_content_image(
        image=image,
        target_w=100,
        target_h=100,
        content_fit="contain",
        resize_method="bilinear",
    )
    assert fitted.shape == (1, 100, 100, 3)
    assert meta["padding"] != (0, 0, 0, 0)
    assert torch.all(fitted[:, 0, :, :] == 1.0)
    assert torch.all(fitted[:, -1, :, :] == 1.0)


def test_content_fit_stretch_fills_target_directly():
    image = torch.rand(1, 100, 200, 3)
    fitted, meta = _fit_content_image(
        image=image,
        target_w=96,
        target_h=128,
        content_fit="stretch",
        resize_method="bilinear",
    )
    assert fitted.shape == (1, 128, 96, 3)
    assert meta["fitted_size"] == (96, 128)
