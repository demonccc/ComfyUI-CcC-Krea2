"""Functional tests for RoPE layout, coordinates and annotated preview."""

import numpy as np
import pytest
import torch
from PIL import Image

from ccc_krea2.modular_nodes.rope_position import (
    build_incontext_3d_rope_pos_ids, plan_rope_positions, parse_rope_position,
)
from ccc_krea2.modular_nodes.edit_reference_types import VisualReferenceEntry
from ccc_krea2.modular_nodes.rope_preview import render_rope_preview, _boost_visibility_preview, _render_boost_gallery


def position(grid, h, v, stack="overlay", mode="spatial", geometry="latent", source=None):
    value = f"{grid}:{h}:{v}:{stack}:{mode}:{geometry}"
    return value if source is None else f"{value}:{source[0]}:{source[1]}"


def test_invalid_outside_center_center_fails():
    with pytest.raises(ValueError, match="outside"):
        parse_rope_position(position("outside", "center", "center"))


def test_native_source_position_shifts_by_own_width_or_height():
    # 200x100 conceptual coordinates are represented by a 20x10 token grid.
    grids = [(10, 20)]
    left = plan_rope_positions(grids, (10, 10), [position("outside", "left", "center", geometry="source")])[0]
    up = plan_rope_positions(grids, (10, 10), [position("outside", "center", "up", geometry="source")])[0]
    assert left.x == -20 and left.y == 0
    assert up.x == (10 - 20) / 2 and up.y == -10


def test_outside_right_stack_moves_each_reference_by_previous_width():
    config = position("outside", "right", "center", stack="stacked")
    layout = plan_rope_positions([(6, 12), (8, 9), (6, 11)], (10, 10), [config] * 3)
    assert [ref.x for ref in layout] == [10, 22, 31]


def test_inside_stack_may_overflow_without_implicit_resize():
    config = position("inside", "left", "center", stack="stacked")
    layout = plan_rope_positions([(6, 12), (6, 12)], (10, 10), [config] * 2)
    assert [ref.x for ref in layout] == [0, 12]
    assert layout[-1].x + layout[-1].width > 10


def test_collapsed_rope_keeps_frame_and_same_spatial_coordinates():
    spec = position("outside", "right", "center", mode="collapsed")
    values = build_incontext_3d_rope_pos_ids(
        batch_size=1, txt_len=0, ref_token_grids=[(2, 3)],
        target_grid=(4, 4), device=torch.device("cpu"),
        ref_rope_positions=[spec],
    )[0]
    ref = values[:6]
    assert torch.all(ref[:, 0] == 1)
    assert torch.unique(ref[:, 1]).numel() == 1
    assert torch.unique(ref[:, 2]).numel() == 1
    assert torch.all(values[6:, 0] == 0)


def test_preview_is_rgb_and_draws_only_visual_entries():
    latent = {"samples": torch.zeros((1, 16, 16, 16))}
    entry = VisualReferenceEntry(image=torch.zeros((1, 256, 256, 3)))
    img = render_rope_preview([entry], latent)
    assert img.ndim == 4 and img.shape[-1] == 3 and img.shape[0] == 4
    assert img.min().item() >= 0 and img.max().item() <= 1
    assert torch.unique(img).numel() > 3

def test_three_views_keep_identical_canvas_and_use_processed_target_content():
    latent = {
        "samples": torch.zeros((1, 16, 16, 16)),
        "ccc_krea2_target_preview_image": torch.ones((1, 128, 128, 3)),
    }
    reference = VisualReferenceEntry(image=torch.zeros((1, 128, 128, 3)), reference_fit="native")
    preview = render_rope_preview([reference], latent)
    assert preview.shape[0] == 4
    assert preview.shape[1:] == render_rope_preview([reference], {
        "samples": latent["samples"]
    }).shape[1:]
    # White source target is visible in both overlay views but not the layout.
    assert float(preview[1].mean()) > float(preview[0].mean())



def test_preview_inverse_opacity_0_to_10():
    source = Image.new("RGB", (3, 1), (200, 200, 200))
    output = _boost_visibility_preview(source, np.array([[0.0, 5.0, 10.0]]))
    pixels = np.asarray(output)
    assert pixels.shape == (1, 3, 3)
    assert pixels[0, 0, 0] == 20  # 90 percent black at Boost 0.
    assert pixels[0, 1, 0] > pixels[0, 0, 0]
    assert pixels[0, 2, 0] == 190  # 5 percent black at Boost 10.


def test_regional_gallery_displays_all_four_refs_when_rope_overlaps():
    latent = {"samples": torch.zeros((1, 16, 16, 16))}
    entries = [
        VisualReferenceEntry(
            image=torch.full((1, 128, 128, 3), 1.0),
            boost=10.0 if index == 3 else float(index),
            reference_fit="native",
        )
        for index in range(4)
    ]
    preview = render_rope_preview(entries, latent)
    assert preview.shape[0] == 4
    assert preview.shape[-1] == 3
    # The fourth view is a separate 2x2 gallery rather than a RoPE overlap.
    regional = preview[3]
    assert regional.shape[0] >= 430
    assert regional.shape[1] >= 500
    # Ref 4 uses Boost 10 and remains much brighter than Ref 1 Boost 0.
    h, w = regional.shape[:2]
    top_left = regional[70:min(h // 2 - 5, 100), 30:70].mean().item()
    bottom_right = regional[h // 2 + 55:h // 2 + 85, w // 2 + 40:w // 2 + 80].mean().item()
    assert bottom_right > top_left + 0.3


def test_gallery_accepts_three_independent_regional_maps():
    previews = [
        _boost_visibility_preview(Image.new("RGB", (64, 64), (200, 200, 200)),
                                  np.full((64, 64), boost))
        for boost in (0, 5, 10)
    ]
    gallery = _render_boost_gallery((600, 450), previews, [(v, v) for v in (0, 5, 10)])
    assert gallery.size == (600, 450)
    assert len(np.unique(np.asarray(gallery).reshape(-1, 3), axis=0)) > 4
