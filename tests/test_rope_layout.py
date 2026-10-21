"""Functional tests for RoPE layout, coordinates and annotated preview."""

import pytest
import torch

from ccc_krea2.modular_nodes.rope_position import (
    build_incontext_3d_rope_pos_ids, plan_rope_positions, parse_rope_position,
)
from ccc_krea2.modular_nodes.edit_reference_types import VisualReferenceEntry
from ccc_krea2.modular_nodes.rope_preview import render_rope_preview


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
    assert img.ndim == 4 and img.shape[-1] == 3 and img.shape[0] == 1
    assert img.min().item() >= 0 and img.max().item() <= 1
    assert torch.unique(img).numel() > 3
