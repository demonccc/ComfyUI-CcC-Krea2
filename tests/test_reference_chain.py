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


def test_target_vision_context_ordering_and_vae_frames():
    from ccc_krea2.target_latent import TargetVisionContext, should_include_target_in_vision
    from ccc_krea2.reference_specs import ReferenceSpec, ReferenceChain

    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=None, mode="native")

    target_vctx = TargetVisionContext(
        include_in_vision="yes",
        target_vision_slot=None,
        target_alias="target",
        target_image=prep,
    )

    subj_spec = ReferenceSpec(reference_path="edit", prepared_image=prep, alias="subject", appearance_reference=True)
    chain = ReferenceChain((subj_spec,))

    target_spec = ReferenceSpec(
        reference_path="edit",
        prepared_image=target_vctx.target_image,
        requested_vision_slot=target_vctx.target_vision_slot,
        alias="target",
        appearance_reference=False,
    )

    effective_chain = ReferenceChain((target_spec,) + chain.references)
    resolved, warnings = resolve_reference_slots_and_aliases(effective_chain)

    assert len(resolved) == 2
    # Target is slot 1, appearance_reference False -> VAE frame None
    assert resolved[0]["resolved_slot"] == 1
    assert resolved[0]["vae_reference_frame"] is None
    # Subject is slot 2, appearance_reference True -> VAE frame 1
    assert resolved[1]["resolved_slot"] == 2
    assert resolved[1]["vae_reference_frame"] == 1


def test_target_vision_context_style_ordering():
    from ccc_krea2.target_latent import TargetVisionContext
    from ccc_krea2.reference_specs import ReferenceSpec, ReferenceChain

    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=None, mode="native")

    target_spec = ReferenceSpec(
        reference_path="edit",
        prepared_image=prep,
        requested_vision_slot=None,
        alias="target",
        appearance_reference=False,
    )
    style_spec = ReferenceSpec(
        reference_path="style",
        prepared_image=prep,
        requested_vision_slot=None,
        alias="style",
        appearance_reference=False,
    )

    # Prepending target_spec before Style maintains Target slot 1, Style slot 2 without failing "Style last" rule
    effective_chain = ReferenceChain((target_spec, style_spec))
    resolved, warnings = resolve_reference_slots_and_aliases(effective_chain)

    assert len(resolved) == 2
    assert resolved[0]["resolved_slot"] == 1
    assert resolved[1]["resolved_slot"] == 2


def test_target_manual_slot_collision_raises():
    from ccc_krea2.reference_specs import ReferenceSpec, ReferenceChain

    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=None, mode="native")

    target_spec = ReferenceSpec(
        reference_path="edit",
        prepared_image=prep,
        requested_vision_slot=1,
        alias="target",
        appearance_reference=False,
    )
    subj_spec = ReferenceSpec(
        reference_path="edit",
        prepared_image=prep,
        requested_vision_slot=1,
        alias="subject",
        appearance_reference=True,
    )

    effective_chain = ReferenceChain((target_spec, subj_spec))
    with pytest.raises(ValueError, match="Duplicate vision slot 1 specified"):
        resolve_reference_slots_and_aliases(effective_chain)


