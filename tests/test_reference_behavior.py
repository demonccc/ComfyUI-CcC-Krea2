"""Behavior-based unit tests for the unified Edit reference chain.

These tests feed actual tensors through the node methods rather than freezing
widget lists or checked-in workflow files.
"""

import pytest
import torch

from ccc_krea2.modular_nodes.edit_reference_types import (
    EditReferenceChain, VisualReferenceEntry, VisionReferenceEntry,
)
from ccc_krea2.modular_nodes.visual_reference_node import CcCKrea2VisualReference
from ccc_krea2.modular_nodes.vision_reference_node import CcCKrea2VisionReference
from ccc_krea2.modular_nodes.edit_node import _combine_reference_chains
from ccc_krea2.modular_nodes.edit_prompt_creator_node import _collect_analysis_images


@pytest.mark.parametrize("order", [
    ("visual", "vision", "visual"),
    ("vision", "visual", "vision"),
    ("visual", "visual", "vision"),
])
def test_mixed_reference_chain_preserves_append_order_and_input_images(order):
    images = [torch.full((1, 48, 64, 3), float(i + 1) / 10)
              for i in range(len(order))]
    nodes = {
        "visual": CcCKrea2VisualReference(),
        "vision": CcCKrea2VisionReference(),
    }
    chain = None
    for kind, image in zip(order, images):
        if kind == "visual":
            chain = nodes[kind].process(
                image=image, semantic=False, previous_references=chain,
            )[0]
        else:
            chain = nodes[kind].process(
                image=image, previous_references=chain,
            )[0]

    assert isinstance(chain, EditReferenceChain)
    assert len(chain.entries) == len(images)
    assert all(entry.image is image for entry, image in zip(chain.entries, images))
    assert [
        isinstance(entry, VisualReferenceEntry) for entry in chain.entries
    ] == [kind == "visual" for kind in order]

    prepared = _combine_reference_chains(clip=None, references=chain, latent={})
    assert len(prepared.references) == len(images)
    assert [spec.appearance_reference for spec in prepared.references] == [
        kind == "visual" for kind in order
    ]


def test_prompt_creator_reads_only_visual_entries_without_modifying_chain():
    images = [torch.full((1, 64, 64, 3), float(i) / 5) for i in range(1, 5)]
    chain = EditReferenceChain((
        VisionReferenceEntry(image=images[0], mode="semantic_only"),
        VisualReferenceEntry(image=images[1], semantic=True),
        VisionReferenceEntry(image=images[2], mode="context_indirect"),
        VisualReferenceEntry(image=images[3], semantic=False),
    ))

    original = chain.entries
    analyzed, mapping, warnings = _collect_analysis_images(
        references=chain, reference_edit_image=None,
    )
    assert len(analyzed) == 2
    assert analyzed[0] is images[1] and analyzed[1] is images[3]
    assert any("Image 1" in row for row in mapping)
    assert not any("Image 2" in row for row in mapping)
    assert not warnings
    assert chain.entries is original


def test_reference_chain_append_is_immutable():
    image = torch.zeros((1, 32, 32, 3))
    first = EditReferenceChain().append(VisualReferenceEntry(image=image))
    second = first.append(VisionReferenceEntry(image=image))
    assert len(first.entries) == 1
    assert len(second.entries) == 2
    assert second.entries[0] is first.entries[0]


def test_visual_reference_rejects_missing_region_tag():
    node = CcCKrea2VisualReference()
    with pytest.raises(ValueError, match="region_tag"):
        node.process(
            image=torch.zeros((1, 64, 64, 3)),
            attention_scope="only in region",
            region_tag="",
        )
