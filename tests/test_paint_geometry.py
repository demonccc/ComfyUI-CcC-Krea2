"""Krea2 CcC Paint Geometry pad/restore contract."""

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
    image = _image(1024, 1344)
    context, _, prepared_mask = _prepare(
        image,
        hpos="left",
        expand_right=128,
    )

    expansion = context["working_expansion_mask"]
    krea_padding = context["working_krea_padding_mask"]
    user = context["working_user_mask"]

    assert prepared_mask.shape == (1, 1024, 1536)
    assert torch.all(user == 0)
    assert torch.all(expansion[:, :, :1344] == 0)
    assert torch.all(expansion[:, :, 1344:1472] == 1)
    assert torch.all(expansion[:, :, 1472:] == 0)
    assert torch.all(krea_padding[:, :, :1472] == 0)
    assert torch.all(krea_padding[:, :, 1472:] == 1)
    assert torch.equal(prepared_mask, torch.maximum(expansion, krea_padding))


def test_padding_fill_modes_are_available_and_edge_is_non_destructive():
    image = _image(1500, 1000)
    for fill in ("edge", "reflect", "neutral", "white"):
        context, prepared, _ = _prepare(image, fill=fill)
        assert prepared.shape[1:3] == (1536, 1024)
        assert context["padding_fill"] == fill


def test_pad_guardrail_rejects_source_larger_than_supported_presets():
    with pytest.raises(ValueError, match="Resize the source"):
        _prepare(_image(1000, 2050))


def test_pad_preserves_complete_source_and_mask():
    image = _image(800, 800)
    mask = torch.zeros((1, 800, 800), dtype=torch.float32)
    mask[:, 750:790, 730:790] = 1
    context, prepared, prepared_mask = _prepare(image, mask=mask)
    x, y = context["transform"]["pad_left"], context["transform"]["pad_top"]
    torch.testing.assert_close(prepared[:, y:y + 800, x:x + 800], image)
    torch.testing.assert_close(prepared_mask[:, y:y + 800, x:x + 800], mask)


def test_pad_restore_removes_only_temporary_krea_padding():
    image = _image(1500, 1000)
    context, prepared, _ = _prepare(image)
    restored = restore_paint_geometry(prepared, context)
    assert restored.shape == image.shape
    assert torch.allclose(restored, image)


def test_restore_rejects_removed_crop_mode():
    image = _image(800, 800)
    context, prepared, _ = _prepare(image)
    context["mode"] = "crop"
    with pytest.raises(ValueError, match="only pad is supported"):
        restore_paint_geometry(prepared, context)
