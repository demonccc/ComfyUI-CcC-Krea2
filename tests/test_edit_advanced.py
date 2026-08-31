"""Contracts for the split Krea2 Edit nodes."""

import torch

from ccc_krea2.modular_nodes.edit_node import CcCKrea2Edit, _resolve_target_geometry
from ccc_krea2.modular_nodes.semantic_reference_node import CcCKrea2SemanticReference
from ccc_krea2.modular_nodes.visual_reference_node import CcCKrea2VisualReference


def test_visual_reference_exposes_only_existing_reference_controls_plus_semantic_role():
    inputs = CcCKrea2VisualReference.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]

    assert list(required) == [
        "image",
        "boost",
        "rope_position",
        "semantic",
        "semantic_role",
        "instruction",
        "grounding_px",
    ]
    assert list(optional) == ["previous_references"]
    assert required["boost"][1]["default"] == 1.0
    assert required["boost"][1]["max"] == 10.0
    assert required["rope_position"][0] == ("none", "up", "down", "left", "right")
    assert required["semantic"][1]["default"] is True
    assert required["semantic_role"][1]["default"] == ""
    assert required["grounding_px"][1]["default"] == 768


def test_visual_reference_semantic_role_is_ignored_when_semantic_is_disabled():
    image = torch.zeros((1, 64, 64, 3))
    node = CcCKrea2VisualReference()

    (chain,) = node.process(
        image=image,
        semantic=False,
        semantic_role="subject image",
        instruction="Use subject identity",
        grounding_px=768,
    )

    entry = chain.entries[0]
    assert entry.semantic is False
    assert entry.semantic_role == ""
    assert entry.instruction == ""


def test_visual_reference_chain_preserves_krea2_physical_order():
    scene = torch.zeros((1, 64, 96, 3))
    subject = torch.zeros((1, 96, 64, 3))
    node = CcCKrea2VisualReference()

    (scene_chain,) = node.process(image=scene, semantic_role="scene image")
    (chain,) = node.process(
        image=subject,
        boost=4.0,
        semantic_role="subject image",
        previous_references=scene_chain,
    )

    assert len(chain.entries) == 2
    assert chain.entries[0].semantic_role == "scene image"
    assert chain.entries[1].semantic_role == "subject image"
    assert chain.entries[1].boost == 4.0


def test_semantic_reference_exposes_advanced_semantic_controls():
    required = CcCKrea2SemanticReference.INPUT_TYPES()["required"]
    assert list(required) == [
        "image",
        "mode",
        "instruction",
        "grounding_px",
        "processing",
        "fidelity",
    ]
    assert required["mode"][0] == ("semantic_only", "style_direct", "style_indirect")
    assert required["processing"][0] == ("full", "2x2", "4x4")
    assert required["fidelity"][1]["default"] == 1.0


def test_edit_exposes_global_and_target_controls_without_advanced_slots():
    inputs = CcCKrea2Edit.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]

    assert list(required) == [
        "model",
        "clip",
        "vae",
        "positive_prompt",
        "negative_prompt",
        "aspect_ratio",
        "resolution",
        "latent_semantic",
        "latent_semantic_instruction",
        "latent_grounding_px",
        "batch_size",
        "apply_krea2_edit_patch",
    ]
    assert list(optional) == [
        "target_image",
        "grid_size_image",
        "grid_geometry_image",
        "visual_references",
        "semantic_references",
    ]

    for removed in (
        "reference_1",
        "reference_2",
        "semantic_1_source",
        "semantic_2_source",
        "latent",
        "grid_size_source",
        "grid_geometry_source",
    ):
        assert removed not in required
        assert removed not in optional


def test_grid_size_and_geometry_images_are_independent():
    size_image = torch.zeros((1, 1600, 1400, 3))
    geometry_image = torch.zeros((1, 640, 512, 3))

    width, height, size_label, geometry_label = _resolve_target_geometry(
        target_image=None,
        aspect_ratio="from source",
        resolution="from source",
        grid_size_image=size_image,
        grid_geometry_image=geometry_image,
    )

    assert (width, height) == (1344, 1680)
    assert size_label == "grid size image"
    assert geometry_label == "grid geometry image"
