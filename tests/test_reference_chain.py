"""Unit tests for ReferenceChain, slot resolution, alias parsing, and role nodes."""

import torch
import pytest
from ccc_krea2.vision_prep import prepare_vision_image
from ccc_krea2.reference_slots import parse_aliases, resolve_reference_slots_and_aliases
from ccc_krea2.modular_nodes.subject_node import CcCKrea2SubjectImage
from ccc_krea2.modular_nodes.scene_node import CcCKrea2SceneImage


def test_alias_parsing():
    parsed = parse_aliases(" scene_image , Image {slot}, scene_image ")
    assert parsed == ("scene_image", "Image {slot}")


def test_reference_chain_immutable_append():
    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=None, mode="native")

    subj_node = CcCKrea2SubjectImage()
    (chain1,) = subj_node.process(prepared_image=prep, vision_slot="auto")
    assert len(chain1.references) == 1

    scene_node = CcCKrea2SceneImage()
    (chain2,) = scene_node.process(prepared_image=prep, vision_slot="auto", previous_references=chain1)
    assert len(chain1.references) == 1
    assert len(chain2.references) == 2


def test_slot_assignment_auto_consecutive():
    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=None, mode="native")

    subj_node = CcCKrea2SubjectImage()
    (chain1,) = subj_node.process(prepared_image=prep, vision_slot="auto")
    scene_node = CcCKrea2SceneImage()
    (chain2,) = scene_node.process(prepared_image=prep, vision_slot="auto", previous_references=chain1)

    resolved, warnings = resolve_reference_slots_and_aliases(chain2)
    assert resolved[0]["resolved_slot"] == 1
    assert resolved[1]["resolved_slot"] == 2
    assert "Image 1" in resolved[0]["expanded_aliases"]
    assert "Image 2" in resolved[1]["expanded_aliases"]


def test_slot_gap_raises_error():
    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=None, mode="native")

    subj_node = CcCKrea2SubjectImage()
    (chain1,) = subj_node.process(prepared_image=prep, vision_slot="1")
    scene_node = CcCKrea2SceneImage()
    (chain2,) = scene_node.process(prepared_image=prep, vision_slot="3", previous_references=chain1)

    with pytest.raises(ValueError, match="gaps detected in resolved logical slots"):
        resolve_reference_slots_and_aliases(chain2)


def test_slot_sorting_order():
    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=None, mode="native")

    subj_node = CcCKrea2SubjectImage()
    (chain1,) = subj_node.process(prepared_image=prep, vision_slot="2")
    scene_node = CcCKrea2SceneImage()
    (chain2,) = scene_node.process(prepared_image=prep, vision_slot="1", previous_references=chain1)

    resolved, warnings = resolve_reference_slots_and_aliases(chain2)
    assert len(resolved) == 2
    assert resolved[0]["resolved_slot"] == 1
    assert resolved[0]["spec"].role == "scene"
    assert resolved[1]["resolved_slot"] == 2
    assert resolved[1]["spec"].role == "subject"


def test_internal_masked_anchor_dataclass_defaults():
    """Assert SubjectReferenceSpec and SceneReferenceSpec masked directive anchor defaults are 0.0."""
    from ccc_krea2.reference_specs import SubjectReferenceSpec, SceneReferenceSpec

    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=None, mode="native")

    default_subj = SubjectReferenceSpec(
        role="subject",
        prepared_image=prep,
        requested_vision_slot=None,
        aliases_template="",
        parsed_aliases=(),
        extra_vision_directive="",
    )
    assert default_subj.masked_identity_anchor == 0.0

    explicit_subj = SubjectReferenceSpec(
        role="subject",
        prepared_image=prep,
        requested_vision_slot=None,
        aliases_template="",
        parsed_aliases=(),
        extra_vision_directive="",
        masked_identity_anchor=0.8,
    )
    assert explicit_subj.masked_identity_anchor == 0.8

    default_scene = SceneReferenceSpec(
        role="scene",
        prepared_image=prep,
        requested_vision_slot=None,
        aliases_template="",
        parsed_aliases=(),
        extra_vision_directive="",
    )
    assert default_scene.masked_region_anchor == 0.0

    explicit_scene = SceneReferenceSpec(
        role="scene",
        prepared_image=prep,
        requested_vision_slot=None,
        aliases_template="",
        parsed_aliases=(),
        extra_vision_directive="",
        masked_region_anchor=0.9,
    )
    assert explicit_scene.masked_region_anchor == 0.9

