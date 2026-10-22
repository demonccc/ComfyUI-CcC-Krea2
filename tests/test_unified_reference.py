"""Unified Reference node contract and Visual/Vision isolation tests."""
import torch
import pytest

from ccc_krea2.modular_nodes.reference_node import CcCKrea2Reference
from ccc_krea2.modular_nodes.edit_reference_types import (
    VisualReferenceEntry, VisionReferenceEntry, EditReferenceChain,
)


def image():
    return torch.ones((1, 32, 32, 3), dtype=torch.float32)


def test_schema_has_visual_before_vision():
    keys = list(CcCKrea2Reference.INPUT_TYPES()["required"])
    assert keys[0:2] == ["image", "visual_enable"]
    assert keys.index("visual_reference_fit") < keys.index("vision_enable")
    assert keys.index("vision_enable") < keys.index("vision_mode")
    assert "semantic" not in keys
    assert "visual_semantic" not in keys


def test_both_enabled_by_default():
    schema = CcCKrea2Reference.INPUT_TYPES()["required"]
    assert schema["visual_enable"][1]["default"] is True
    assert schema["vision_enable"][1]["default"] is True
    chain, _ = CcCKrea2Reference().process(image())
    assert len(chain.entries) == 2


def test_visual_only_has_no_qwen_image():
    chain, info = CcCKrea2Reference().process(image(), visual_enable=True, vision_enable=False)
    assert len(chain.entries) == 1
    assert isinstance(chain.entries[0], VisualReferenceEntry)
    assert chain.entries[0].semantic is False
    assert "Vision: disabled" in info


def test_vision_only_has_no_appearance_reference():
    chain, _ = CcCKrea2Reference().process(image(), visual_enable=False, vision_enable=True)
    assert len(chain.entries) == 1
    assert isinstance(chain.entries[0], VisionReferenceEntry)


def test_both_paths_share_image_and_keep_chain_order():
    previous = EditReferenceChain()
    node = CcCKrea2Reference()
    chain, _ = node.process(image(), previous_references=previous,
                           visual_enable=True, vision_enable=True,
                           vision_mode="semantic")
    assert len(chain.entries) == 2
    assert isinstance(chain.entries[0], VisualReferenceEntry)
    assert isinstance(chain.entries[1], VisionReferenceEntry)
    assert chain.entries[0].image is chain.entries[1].image
    assert chain.entries[0].semantic is True
    assert chain.entries[1].paired_visual is True


def test_both_disabled_rejected():
    with pytest.raises(ValueError, match="Enable Visual, Vision"):
        CcCKrea2Reference().process(image(), visual_enable=False, vision_enable=False)


def test_vision_resize_limits_edge_without_touching_visual_pixels():
    original = torch.rand((1, 1200, 1600, 3))
    chain, _ = CcCKrea2Reference().process(
        original, vision_resize=True, vision_max_edge=768
    )
    assert chain.entries[0].image is original
    assert tuple(chain.entries[1].image.shape) == (1, 576, 768, 3)


def test_vision_resize_disabled_passes_source_unchanged():
    original = torch.rand((1, 1200, 1600, 3))
    chain, _ = CcCKrea2Reference().process(
        original, vision_resize=False
    )
    assert chain.entries[1].image is original


def test_public_ui_has_no_ineffective_guidance_presets():
    fields = CcCKrea2Reference.INPUT_TYPES()["required"]
    assert "vision_guidance_detail" not in fields
    assert "vision_vision_guidance" not in fields


def test_visual_semantic_follows_vision_enable():
    image = torch.zeros((1, 32, 32, 3))
    node = CcCKrea2Reference()
    combined, _ = node.process(image, visual_enable=True, vision_enable=True)
    assert combined.entries[0].semantic is True
    assert combined.entries[0].semantic_resize is True
    assert combined.entries[0].semantic_grounding_px == 768
    assert len(combined.entries) == 2

    visual_only, _ = node.process(image, visual_enable=True, vision_enable=False)
    assert visual_only.entries[0].semantic is False
    assert len(visual_only.entries) == 1

    vision_only, _ = node.process(image, visual_enable=False, vision_enable=True)
    assert len(vision_only.entries) == 1
    assert isinstance(vision_only.entries[0], VisionReferenceEntry)


def test_combined_semantic_resize_controls_match_vision_options():
    image = torch.zeros((1, 32, 32, 3))
    chain, _ = CcCKrea2Reference().process(
        image, vision_resize=False, vision_max_edge=1024
    )
    assert chain.entries[0].semantic
    assert chain.entries[0].semantic_resize is False
    assert chain.entries[0].semantic_grounding_px == 1024
