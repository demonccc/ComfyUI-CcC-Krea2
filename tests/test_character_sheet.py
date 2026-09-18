"""Tests for the deterministic Krea2 CcC Character Sheet composer."""

import pytest
import torch

from ccc_krea2.modular_nodes.character_sheet_node import (
    CcCKrea2CharacterSheet,
    PRESET_4_HEADS_1_BODY,
    PRESET_3_HEADS_1_BODY,
    PRESET_4_HEADS,
    _layout_rects,
)


def _solid(rgb, height=64, width=64):
    image = torch.zeros((1, height, width, 3), dtype=torch.float32)
    image[..., 0] = rgb[0]
    image[..., 1] = rgb[1]
    image[..., 2] = rgb[2]
    return image


def _center_pixel(image, rect):
    x, y, w, h = rect
    return image[0, y + h // 2, x + w // 2]


def test_character_sheet_inputs_use_semantic_view_names():
    optional = CcCKrea2CharacterSheet.INPUT_TYPES()["optional"]
    assert tuple(optional) == (
        "front_view",
        "three_quarter_view",
        "profile_view",
        "extra_view",
        "full_body_view",
    )


def test_four_heads_plus_body_composes_square_output():
    node = CcCKrea2CharacterSheet()
    colors = {
        "front_view": (1.0, 0.0, 0.0),
        "three_quarter_view": (0.0, 1.0, 0.0),
        "profile_view": (0.0, 0.0, 1.0),
        "extra_view": (1.0, 1.0, 0.0),
        "full_body_view": (1.0, 0.0, 1.0),
    }
    kwargs = {name: _solid(rgb) for name, rgb in colors.items()}

    (sheet,) = node.compose(
        layout_preset=PRESET_4_HEADS_1_BODY,
        resolution=256,
        slot_fit="cover",
        padding=8,
        outer_margin=8,
        background="black",
        **kwargs,
    )

    assert tuple(sheet.shape) == (1, 256, 256, 3)
    rects = _layout_rects(PRESET_4_HEADS_1_BODY, 256, 8, 8)
    for name, rgb in colors.items():
        assert torch.allclose(
            _center_pixel(sheet, rects[name]),
            torch.tensor(rgb, dtype=sheet.dtype),
            atol=1e-5,
        )


def test_three_heads_plus_body_does_not_require_extra_view():
    node = CcCKrea2CharacterSheet()
    (sheet,) = node.compose(
        layout_preset=PRESET_3_HEADS_1_BODY,
        resolution=256,
        slot_fit="contain",
        padding=4,
        outer_margin=4,
        background="white",
        front_view=_solid((1.0, 0.0, 0.0), 80, 60),
        three_quarter_view=_solid((0.0, 1.0, 0.0), 80, 60),
        profile_view=_solid((0.0, 0.0, 1.0), 80, 60),
        full_body_view=_solid((1.0, 0.0, 1.0), 120, 40),
    )
    assert tuple(sheet.shape) == (1, 256, 256, 3)


def test_four_heads_preset_does_not_require_full_body():
    node = CcCKrea2CharacterSheet()
    (sheet,) = node.compose(
        layout_preset=PRESET_4_HEADS,
        resolution=256,
        slot_fit="cover",
        padding=0,
        outer_margin=0,
        background="gray",
        front_view=_solid((1.0, 0.0, 0.0)),
        three_quarter_view=_solid((0.0, 1.0, 0.0)),
        profile_view=_solid((0.0, 0.0, 1.0)),
        extra_view=_solid((1.0, 1.0, 0.0)),
    )
    assert tuple(sheet.shape) == (1, 256, 256, 3)


def test_missing_required_semantic_view_fails_clearly():
    node = CcCKrea2CharacterSheet()
    with pytest.raises(ValueError, match="full_body_view"):
        node.compose(
            layout_preset=PRESET_3_HEADS_1_BODY,
            resolution=256,
            slot_fit="contain",
            padding=8,
            outer_margin=8,
            background="white",
            front_view=_solid((1.0, 0.0, 0.0)),
            three_quarter_view=_solid((0.0, 1.0, 0.0)),
            profile_view=_solid((0.0, 0.0, 1.0)),
        )


def test_each_semantic_view_accepts_exactly_one_image():
    node = CcCKrea2CharacterSheet()
    batched_front = torch.zeros((2, 64, 64, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="batch size 2"):
        node.compose(
            layout_preset=PRESET_4_HEADS,
            resolution=256,
            slot_fit="cover",
            padding=0,
            outer_margin=0,
            background="white",
            front_view=batched_front,
            three_quarter_view=_solid((0.0, 1.0, 0.0)),
            profile_view=_solid((0.0, 0.0, 1.0)),
            extra_view=_solid((1.0, 1.0, 0.0)),
        )
