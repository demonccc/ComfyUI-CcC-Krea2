"""Krea2 CcC Paint Geometry pad/crop/restore contract."""

import pytest
import torch

from ccc_krea2.modular_nodes.paint_geometry_node import (
    prepare_paint_geometry,
    restore_paint_geometry,
)


def _image(height, width):
    image = torch.zeros((1, height, width, 3), dtype=torch.float32)
    image[..., 0] = 0.2
    image[..., 1] = 0.4
    image[..., 2] = 0.6
    return image


def _prepare(
    image,
    mode="pad",
    fill="edge",
    hpos="center",
    vpos="center",
    mask=None,
    expand_left=0,
    expand_top=0,
    expand_right=0,
    expand_bottom=0,
):
    return prepare_paint_geometry(
        image,
        mask,
        geometry_mode=mode,
        padding_fill=fill,
        horizontal_position=hpos,
        vertical_position=vpos,
        expand_left=expand_left,
        expand_top=expand_top,
        expand_right=expand_right,
        expand_bottom=expand_bottom,
    )


def test_pad_selects_containing_krea_geometry_without_resizing_source():
    image = _image(1500, 1000)
    mask = torch.zeros((1, 1500, 1000), dtype=torch.float32)
    context, prepared, prepared_mask = _prepare(image, mask=mask)

    assert (context["target_width"], context["target_height"]) == (1024, 1536)
    assert prepared.shape == (1, 1536, 1024, 3)
    transform = context["transform"]
    y = transform["pad_top"]
    x = transform["pad_left"]
    assert torch.allclose(prepared[:, y:y + 1500, x:x + 1000], image)
    assert torch.all(prepared_mask[:, y:y + 1500, x:x + 1000] == 0)
    assert torch.all(prepared_mask[:, :y] == 1) if y else True


def test_explicit_outpaint_expansion_is_separate_from_temporary_krea_padding():
    image = _image(1024, 1024)
    context, _, prepared_mask = _prepare(
        image,
        mode="pad",
        hpos="left",
        expand_right=128,
    )

    expansion = context["working_expansion_mask"]
    krea_padding = context["working_krea_padding_mask"]
    user = context["working_user_mask"]

    assert prepared_mask.shape == (1, 1024, 1536)
    assert torch.all(user == 0)
    assert torch.all(expansion[:, :, :1024] == 0)
    assert torch.all(expansion[:, :, 1024:1152] == 1)
    assert torch.all(expansion[:, :, 1152:] == 0)
    assert torch.all(krea_padding[:, :, :1152] == 0)
    assert torch.all(krea_padding[:, :, 1152:] == 1)
    assert torch.equal(prepared_mask, torch.maximum(expansion, krea_padding))


def test_padding_fill_modes_are_available_and_edge_is_non_destructive():
    image = _image(1500, 1000)
    for fill in ("edge", "reflect", "neutral", "white"):
        context, prepared, _ = _prepare(image, fill=fill)
        assert prepared.shape[1:3] == (1536, 1024)
        assert context["padding_fill"] == fill


def test_crop_selects_inner_krea_geometry_and_position():
    image = _image(1200, 1600)
    context, prepared, _ = _prepare(image, mode="crop", hpos="right", vpos="bottom")
    assert (context["target_width"], context["target_height"]) == (1536, 1152)
    assert prepared.shape == (1, 1152, 1536, 3)
    assert context["transform"]["crop_x"] == 64
    assert context["transform"]["crop_y"] == 48


def test_crop_guardrail_rejects_image_below_every_krea_target():
    with pytest.raises(ValueError, match="Use padding"):
        _prepare(_image(800, 800), mode="crop")


def test_pad_guardrail_rejects_image_above_available_krea_size():
    with pytest.raises(ValueError, match="Use crop"):
        _prepare(_image(1000, 2050), mode="pad")


def test_pad_restore_removes_only_temporary_krea_padding():
    image = _image(1500, 1000)
    context, prepared, _ = _prepare(image, mode="pad")
    restored = restore_paint_geometry(prepared, context)
    assert restored.shape == image.shape
    assert torch.allclose(restored, image)


def test_crop_restore_composites_only_generated_mask_region():
    image = _image(1200, 1600)
    context, prepared, _ = _prepare(image, mode="crop")
    generated = torch.ones_like(prepared)
    mask = torch.zeros((1, prepared.shape[1], prepared.shape[2]), dtype=torch.float32)
    mask[:, 100:200, 100:200] = 1.0

    restored = restore_paint_geometry(generated, context, mask)
    x = context["transform"]["crop_x"]
    y = context["transform"]["crop_y"]
    assert torch.all(restored[:, y + 100:y + 200, x + 100:x + 200] == 1.0)
    assert torch.allclose(restored[:, y:y + 50, x:x + 50], context["base_image"][:, y:y + 50, x:x + 50])
