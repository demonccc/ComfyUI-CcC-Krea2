"""CPU-safe unit tests for pixel-space geometric transformations."""

import pytest
import torch
from ccc_krea2.geometry import apply_sampling_transform, apply_reference_fit_transform


def test_sampling_transform_modes():
    img = torch.rand((1, 800, 600, 3))
    mask = torch.rand((1, 800, 600))

    # Fit mode
    fit_img, fit_mask = apply_sampling_transform(img, target_h=1024, target_w=1024, mode="fit", mask=mask)
    assert fit_img.shape == (1, 1024, 1024, 3)
    assert fit_mask.shape == (1, 1024, 1024)

    # Crop mode
    crop_img, crop_mask = apply_sampling_transform(img, target_h=1024, target_w=1024, mode="crop", mask=mask)
    assert crop_img.shape == (1, 1024, 1024, 3)
    assert crop_mask.shape == (1, 1024, 1024)

    # Stretch mode
    str_img, str_mask = apply_sampling_transform(img, target_h=1024, target_w=1024, mode="stretch", mask=mask)
    assert str_img.shape == (1, 1024, 1024, 3)
    assert str_mask.shape == (1, 1024, 1024)


def test_reference_fit_transform_modes():
    img = torch.rand((1, 800, 600, 3))
    mask = torch.rand((1, 800, 600))

    # Fit mode: native AR preserved (800x600 -> 1024x768 aligned to /16 without black canvas padding)
    fit_img, fit_mask, fit_meta = apply_reference_fit_transform(img, target_h=1024, target_w=1024, mode="fit", mask=mask)
    assert fit_img.shape == (1, 1024, 768, 3)
    assert fit_mask.shape == (1, 1024, 768)
    assert fit_meta["spatial_hw"] == (1024, 768)
    assert fit_meta["lat_hw"] == (128, 96)

    # Crop mode: crops to target_h x target_w
    crop_img, crop_mask, crop_meta = apply_reference_fit_transform(img, target_h=1024, target_w=1024, mode="crop", mask=mask)
    assert crop_img.shape == (1, 1024, 1024, 3)
    assert crop_mask.shape == (1, 1024, 1024)
    assert crop_meta["spatial_hw"] == (1024, 1024)

    # Stretch mode must raise ValueError for reference tokens
    with pytest.raises(ValueError, match="Invalid reference_fit_mode"):
        apply_reference_fit_transform(img, target_h=1024, target_w=1024, mode="stretch", mask=mask)
