"""Reference/target masks follow their own pixel spaces without a second latent resize."""

import pytest
import torch

from ccc_krea2.krea2edit_geometry import (
    process_image_and_mask_geometry,
    resolve_krea2edit_geometry,
)
from ccc_krea2.modular_nodes.edit_node import _resolve_reference_target_masks
from ccc_krea2.modular_nodes.edit_reference_types import VisualReferenceEntry


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


def test_edit_rejects_target_mask_from_outdated_target_resolution():
    entry = VisualReferenceEntry(
        image=torch.zeros((1, 32, 32, 3)),
        attention_scope="boost in mask",
        target_attention_mask=torch.ones((1, 128, 128)),
    )
    target = {"samples": torch.zeros((1, 16, 24, 24))}
    with pytest.raises(ValueError, match="192x192"):
        _resolve_reference_target_masks((entry,), target)


def test_each_target_mask_retains_reference_chain_order():
    a = torch.ones((1, 128, 128))
    b = torch.zeros((1, 128, 128))
    refs = (
        VisualReferenceEntry(attention_scope="only in mask", target_attention_mask=a),
        VisualReferenceEntry(attention_scope="boost in mask", target_attention_mask=b),
    )
    masks, scopes = _resolve_reference_target_masks(
        refs, {"samples": torch.zeros((1, 16, 16, 16))}
    )
    assert masks[0] is a and masks[1] is b
    assert scopes == ["only in mask", "boost in mask"]
