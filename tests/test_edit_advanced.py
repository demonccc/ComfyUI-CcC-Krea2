"""Contracts for the single-node Edit Advanced geometry and RoPE laboratory."""

import torch

from ccc_krea2.modular_nodes.edit_advanced_node import (
    CcCKrea2EditAdvanced,
    _channel_alias,
    _resolve_target_geometry,
)
from ccc_krea2.patch import _build_incontext_3d_rope_pos_ids
from ccc_krea2.reference_slots import resolve_reference_slots_and_aliases
from ccc_krea2.reference_specs import ReferenceChain, ReferenceSpec


def test_advanced_node_exposes_manual_lab_controls():
    required = CcCKrea2EditAdvanced.INPUT_TYPES()["required"]
    for name in (
        "latent",
        "reference_1",
        "reference_2",
        "reference_1_boost",
        "reference_2_boost",
        "aspect_ratio",
        "resolution",
        "grid_size_source",
        "grid_geometry_source",
        "reference_1_rope_position",
        "reference_2_rope_position",
        "latent_semantic",
        "semantic_1_source",
        "semantic_2_source",
    ):
        assert name in required

    for forbidden in ("preset", "use_default_prompt", "outfit_source", "style_source"):
        assert forbidden not in required


def test_grid_size_and_geometry_sources_are_independent():
    sources = {
        "subject": torch.zeros((1, 1600, 1400, 3)),
        "scene": torch.zeros((1, 640, 512, 3)),
        "outfit": None,
        "style": None,
    }
    width, height, size_label, geometry_label = _resolve_target_geometry(
        latent_source="empty",
        aspect_ratio="from source",
        resolution="from source",
        grid_size_source="subject",
        grid_geometry_source="scene",
        sources=sources,
    )
    # Subject contributes its 2.24 MP pixel budget; Scene contributes its 4:5 geometry.
    assert (width, height) == (1344, 1680)
    assert size_label == "subject"
    assert geometry_label == "scene"


def test_same_image_can_be_reused_by_multiple_advanced_channels():
    channels = (
        ("reference 1", True),
        ("reference 2", True),
        ("latent semantic", False),
        ("semantic 1", False),
        ("semantic 2", False),
    )
    chain = ReferenceChain(
        tuple(
            ReferenceSpec(
                alias=_channel_alias(channel, "scene"),
                appearance_reference=appearance,
                include_in_vision=True,
            )
            for channel, appearance in channels
        )
    )

    resolved, _ = resolve_reference_slots_and_aliases(chain)
    aliases = [alias for item in resolved for alias in item["expanded_aliases"]]
    assert len(aliases) == len(channels)
    assert len(set(aliases)) == len(channels)


def test_rope_directions_place_reference_fully_outside_target():
    target = (20, 30)
    ref = (10, 12)
    expected_first_yx = {
        "none": (5.0, 9.0),
        "up": (-10.0, 9.0),
        "down": (20.0, 9.0),
        "left": (5.0, -12.0),
        "right": (5.0, 30.0),
    }
    for position, expected in expected_first_yx.items():
        positions = _build_incontext_3d_rope_pos_ids(
            batch_size=1,
            txt_len=0,
            ref_token_grids=[ref],
            target_grid=target,
            device=torch.device("cpu"),
            ref_rope_positions=[position],
        )
        first_ref = positions[0, 0]
        assert tuple(first_ref[1:].tolist()) == expected
        assert first_ref[0].item() == 1.0
