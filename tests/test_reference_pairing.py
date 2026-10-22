"""Ensure unified Visual+Vision references use one Edit Qwen slot."""
from unittest.mock import patch
import torch
import pytest

from ccc_krea2.modular_nodes.reference_node import CcCKrea2Reference
from ccc_krea2.modular_nodes.edit_node import _combine_reference_chains
from ccc_krea2.modular_nodes.edit_reference_types import VisionReferenceEntry
from ccc_krea2.vision_reference_runtime import prepare_tokenizer_native_image
from ccc_krea2.reference_slots import resolve_reference_slots_and_aliases


@pytest.mark.parametrize("mode", ["semantic", "visual", "style", "context"])
def test_combined_reference_is_one_logical_qwen_slot(mode):
    image = torch.rand(1, 64, 96, 3)
    refs, _ = CcCKrea2Reference().process(
        image, vision_mode=mode, vision_style_processing="full"
    )
    assert len(refs.entries) == 2
    assert refs.entries[0].semantic is True
    assert refs.entries[1].paired_visual is True
    with patch(
        "ccc_krea2.modular_nodes.edit_node._prepare_qwen_image",
        side_effect=lambda image, **kwargs: prepare_tokenizer_native_image(image),
    ):
        resolved = _combine_reference_chains(None, refs, {"samples": torch.empty(1, 4, 8, 12)})
    assert len(resolved.references) == 1
    assert resolved.references[0].appearance_reference is True
    assert resolved.references[0].vision_reference_mode == mode
    slots, _ = resolve_reference_slots_and_aliases(resolved)
    assert slots[0]["vae_reference_frame"] == 1
    assert slots[0]["physical_qwen_range"] == (1, 1)


def test_visual_only_and_vision_only_are_not_implicitly_combined():
    image = torch.rand(1, 64, 96, 3)
    node = CcCKrea2Reference()
    visual, _ = node.process(image, visual_enable=True, vision_enable=False)
    vision, _ = node.process(image, visual_enable=False, vision_enable=True)
    assert not visual.entries[0].semantic
    assert isinstance(vision.entries[0], VisionReferenceEntry)
    assert not vision.entries[0].paired_visual


def test_vision_resize_does_not_resize_visual_vae_source():
    image = torch.rand(1, 1000, 1600, 3)
    refs, _ = CcCKrea2Reference().process(image, vision_max_edge=768)
    with patch(
        "ccc_krea2.modular_nodes.edit_node._prepare_qwen_image",
        side_effect=lambda image, **kwargs: prepare_tokenizer_native_image(image),
    ):
        chain = _combine_reference_chains(None, refs, {"samples": torch.empty(1, 4, 8, 12)})
    prepared = chain.references[0].prepared_image
    assert prepared.original_image is image
    assert tuple(prepared.vision_image.shape) == (1, 480, 768, 3)


@pytest.mark.parametrize("processing,expected", [("2x2", 4), ("4x4", 16)])
def test_combined_tiled_style_preserves_all_crops(processing, expected):
    image = torch.rand(1, 128, 128, 3)
    refs, _ = CcCKrea2Reference().process(
        image, vision_mode="style", vision_style_processing=processing
    )
    with patch(
        "ccc_krea2.modular_nodes.edit_node._prepare_qwen_image",
        side_effect=lambda image, **kwargs: prepare_tokenizer_native_image(image),
    ):
        chain = _combine_reference_chains(None, refs, {"samples": torch.empty(1, 4, 8, 8)})
    assert len(chain.references) == 2
    assert chain.references[0].appearance_reference is True
    assert chain.references[1].style_processing == processing
    slots, _ = resolve_reference_slots_and_aliases(chain)
    assert slots[1]["physical_qwen_range"] == (2, expected + 1)
