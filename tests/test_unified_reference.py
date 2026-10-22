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
    assert chain.entries[0].semantic is False


def test_both_disabled_rejected():
    with pytest.raises(ValueError, match="Enable Visual, Vision"):
        CcCKrea2Reference().process(image(), visual_enable=False, vision_enable=False)
