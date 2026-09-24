"""Tests for the deterministic Krea2 CcC Character Sheet composer."""

import pytest
import torch

from ccc_krea2.modular_nodes.character_sheet_node import (
    CHARACTER_SHEET_PRESETS,
    CHARACTER_SHEET_RESIZE_METHODS,
    PRESET_1_BODY,
    PRESET_1_PORTRAIT,
    PRESET_1_PORTRAIT_2_BODIES,
    PRESET_2_BODIES,
    PRESET_2_PORTRAITS,
    PRESET_3_PORTRAITS,
    PRESET_3_PORTRAITS_1_BODY,
    PRESET_4_PORTRAITS,
    PRESET_4_PORTRAITS_1_BODY,
    CcCKrea2CharacterSheet,
    _layout_rects,
)
from ccc_krea2.modular_nodes.latent_node import (
    DEFAULT_KREA_PRESET_SIZE,
    KREA_PRESET_SIZES,
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


def _compose(node, sheet_type, **images):
    return node.compose(
        sheet_type=sheet_type,
        sheet_geometry=DEFAULT_KREA_PRESET_SIZE,
        resize_method="lanczos",
        padding=8,
        outer_margin=8,
        background="white",
        **images,
    )


def test_character_sheet_exposes_generic_portrait_and_body_inputs():
    optional = CcCKrea2CharacterSheet.INPUT_TYPES()["optional"]
    assert tuple(optional) == (
        "portrait_1",
        "portrait_2",
        "portrait_3",
        "portrait_4",
        "body_1",
        "body_2",
    )


def test_character_sheet_presets_match_public_contract():
    assert CHARACTER_SHEET_PRESETS == (
        "1 portrait",
        "2 portraits",
        "3 portraits",
        "4 portraits",
        "1 body",
        "2 bodies",
        "1 portrait + 2 bodies",
        "3 portraits + 1 body",
        "4 portraits + 1 body",
    )


def test_character_sheet_geometry_reuses_latent_krea_presets():
    required = CcCKrea2CharacterSheet.INPUT_TYPES()["required"]
    choices, options = required["sheet_geometry"]
    assert tuple(choices) == tuple(KREA_PRESET_SIZES)
    assert options["default"] == DEFAULT_KREA_PRESET_SIZE
    assert "resolution" not in required
    assert "slot_fit" not in required
    resize_choices, resize_options = required["resize_method"]
    assert tuple(resize_choices) == CHARACTER_SHEET_RESIZE_METHODS
    assert resize_options["default"] == "lanczos"


def test_four_portraits_plus_body_composes_selected_krea_geometry():
    node = CcCKrea2CharacterSheet()
    geometry = "1536 x 1024 | 3:2 | ~1.57 MP"
    colors = {
        "portrait_1": (1.0, 0.0, 0.0),
        "portrait_2": (0.0, 1.0, 0.0),
        "portrait_3": (0.0, 0.0, 1.0),
        "portrait_4": (1.0, 1.0, 0.0),
        "body_1": (1.0, 0.0, 1.0),
    }

    (sheet,) = node.compose(
        sheet_type=PRESET_4_PORTRAITS_1_BODY,
        sheet_geometry=geometry,
        resize_method="lanczos",
        padding=8,
        outer_margin=8,
        background="black",
        **{name: _solid(rgb) for name, rgb in colors.items()},
    )

    assert tuple(sheet.shape) == (1, 1024, 1536, 3)
    rects = _layout_rects(
        PRESET_4_PORTRAITS_1_BODY,
        width=1536,
        height=1024,
        outer_margin=8,
        padding=8,
    )
    for name, rgb in colors.items():
        assert torch.allclose(
            _center_pixel(sheet, rects[name]),
            torch.tensor(rgb, dtype=sheet.dtype),
            atol=1e-5,
        )


def test_single_portrait_fit_keeps_complete_source_without_crop():
    node = CcCKrea2CharacterSheet()
    source = torch.zeros((1, 64, 128, 3), dtype=torch.float32)
    source[:, :, :64, 0] = 1.0
    source[:, :, 64:, 2] = 1.0

    (sheet,) = node.compose(
        sheet_type=PRESET_1_PORTRAIT,
        sheet_geometry=DEFAULT_KREA_PRESET_SIZE,
        resize_method="lanczos",
        padding=8,
        outer_margin=0,
        background="white",
        portrait_1=source,
    )

    assert torch.allclose(sheet[0, 0, 512], torch.ones(3), atol=1e-5)
    assert sheet[0, 512, 64, 0] > 0.9
    assert sheet[0, 512, 960, 2] > 0.9


@pytest.mark.parametrize(
    ("preset", "images"),
    (
        (PRESET_1_PORTRAIT, {"portrait_1": _solid((1.0, 0.0, 0.0))}),
        (
            PRESET_2_PORTRAITS,
            {
                "portrait_1": _solid((1.0, 0.0, 0.0)),
                "portrait_2": _solid((0.0, 1.0, 0.0)),
            },
        ),
        (
            PRESET_3_PORTRAITS,
            {
                "portrait_1": _solid((1.0, 0.0, 0.0)),
                "portrait_2": _solid((0.0, 1.0, 0.0)),
                "portrait_3": _solid((0.0, 0.0, 1.0)),
            },
        ),
        (
            PRESET_4_PORTRAITS,
            {
                "portrait_1": _solid((1.0, 0.0, 0.0)),
                "portrait_2": _solid((0.0, 1.0, 0.0)),
                "portrait_3": _solid((0.0, 0.0, 1.0)),
                "portrait_4": _solid((1.0, 1.0, 0.0)),
            },
        ),
        (PRESET_1_BODY, {"body_1": _solid((1.0, 0.0, 1.0), 120, 48)}),
        (
            PRESET_2_BODIES,
            {
                "body_1": _solid((1.0, 0.0, 1.0), 120, 48),
                "body_2": _solid((0.0, 1.0, 1.0), 120, 48),
            },
        ),
        (
            PRESET_1_PORTRAIT_2_BODIES,
            {
                "portrait_1": _solid((1.0, 0.0, 0.0)),
                "body_1": _solid((1.0, 0.0, 1.0), 120, 48),
                "body_2": _solid((0.0, 1.0, 1.0), 120, 48),
            },
        ),
        (
            PRESET_3_PORTRAITS_1_BODY,
            {
                "portrait_1": _solid((1.0, 0.0, 0.0)),
                "portrait_2": _solid((0.0, 1.0, 0.0)),
                "portrait_3": _solid((0.0, 0.0, 1.0)),
                "body_1": _solid((1.0, 0.0, 1.0), 120, 48),
            },
        ),
        (
            PRESET_4_PORTRAITS_1_BODY,
            {
                "portrait_1": _solid((1.0, 0.0, 0.0)),
                "portrait_2": _solid((0.0, 1.0, 0.0)),
                "portrait_3": _solid((0.0, 0.0, 1.0)),
                "portrait_4": _solid((1.0, 1.0, 0.0)),
                "body_1": _solid((1.0, 0.0, 1.0), 120, 48),
            },
        ),
    ),
)
def test_all_presets_compose(preset, images):
    node = CcCKrea2CharacterSheet()
    (sheet,) = _compose(node, preset, **images)
    assert tuple(sheet.shape) == (1, 1024, 1024, 3)


@pytest.mark.parametrize("resize_method", CHARACTER_SHEET_RESIZE_METHODS)
def test_character_sheet_accepts_all_resize_methods(resize_method):
    node = CcCKrea2CharacterSheet()
    (sheet,) = node.compose(
        sheet_type=PRESET_1_PORTRAIT,
        sheet_geometry=DEFAULT_KREA_PRESET_SIZE,
        resize_method=resize_method,
        padding=8,
        outer_margin=8,
        background="white",
        portrait_1=_solid((0.3, 0.5, 0.7), 73, 119),
    )
    assert tuple(sheet.shape) == (1, 1024, 1024, 3)


def test_missing_required_generic_view_fails_clearly():
    node = CcCKrea2CharacterSheet()
    with pytest.raises(ValueError, match="body_2"):
        _compose(
            node,
            PRESET_1_PORTRAIT_2_BODIES,
            portrait_1=_solid((1.0, 0.0, 0.0)),
            body_1=_solid((1.0, 0.0, 1.0)),
        )


def test_each_view_accepts_exactly_one_image():
    node = CcCKrea2CharacterSheet()
    batched = torch.zeros((2, 64, 64, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="batch size 2"):
        _compose(node, PRESET_1_PORTRAIT, portrait_1=batched)
