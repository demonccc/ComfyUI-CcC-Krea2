"""Reference boost masks follow the reference pixels through geometry processing."""

import pytest
import torch

from ccc_krea2.krea2edit_geometry import (
    process_image_and_mask_geometry,
    resolve_krea2edit_geometry,
)


def test_reference_boost_mask_is_cropped_with_the_reference_before_vae():
    image = torch.zeros((1, 128, 256, 3))
    mask = torch.zeros((1, 128, 256))
    image[:, 32:96, 80:112, 0] = 1.0
    mask[:, 32:96, 80:112] = 1.0
    geom = resolve_krea2edit_geometry(
        src_h=128, src_w=256, tgt_h=128, tgt_w=128, fit_mode="fit",
    )
    prepared_image, prepared_mask = process_image_and_mask_geometry(image, mask, geom)
    assert prepared_image.shape == (1, 128, 128, 3)
    assert prepared_mask.shape == (1, 128, 128)
    # No independent mask fit: the marked subject remains at the same pixels.
    torch.testing.assert_close(prepared_image[..., 0], prepared_mask)


def test_native_reference_mask_crops_to_the_same_sixteen_pixel_grid():
    image = torch.zeros((1, 130, 146, 3))
    mask = torch.zeros((1, 130, 146))
    image[:, 24:104, 20:120, 0] = 1.0
    mask[:, 24:104, 20:120] = 1.0
    geom = resolve_krea2edit_geometry(
        src_h=130, src_w=146, tgt_h=128, tgt_w=128, fit_mode="native",
    )
    prepared_image, prepared_mask = process_image_and_mask_geometry(image, mask, geom)
    assert prepared_image.shape[1:3] == (128, 144)
    assert prepared_mask.shape[1:3] == (128, 144)
    torch.testing.assert_close(prepared_image[..., 0], prepared_mask)


