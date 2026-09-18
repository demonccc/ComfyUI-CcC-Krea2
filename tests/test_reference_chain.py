"""Tests for the current CcC reference chain and slot resolution."""

import pytest
import torch

from ccc_krea2.reference_slots import parse_aliases, resolve_reference_slots_and_aliases
from ccc_krea2.reference_specs import ReferenceChain, ReferenceSpec
from ccc_krea2.vision_prep import prepare_vision_image


def _prepared_image():
    image = torch.rand(1, 512, 512, 3)
    return prepare_vision_image(image=image, clip=None, mode="native")


def test_alias_parsing():
    assert parse_aliases(" scene_image , Image {slot}, scene_image ") == ("scene_image", "Image {slot}")


def test_reference_chain_immutable_append():
    prep = _prepared_image()
    first = ReferenceChain().append(ReferenceSpec(prepared_image=prep, alias="first"))
    second = first.append(ReferenceSpec(prepared_image=prep, alias="second"))

    assert len(first.references) == 1
    assert len(second.references) == 2


def test_slot_assignment_auto_consecutive():
    prep = _prepared_image()
    chain = ReferenceChain(
        (
            ReferenceSpec(prepared_image=prep, alias="first"),
            ReferenceSpec(prepared_image=prep, alias="second"),
        )
    )
    resolved, _ = resolve_reference_slots_and_aliases(chain)

    assert [item["resolved_slot"] for item in resolved] == [1, 2]
    assert [item["vae_reference_frame"] for item in resolved] == [1, 2]


def test_slot_gap_raises_error():
    prep = _prepared_image()
    chain = ReferenceChain(
        (
            ReferenceSpec(prepared_image=prep, requested_vision_slot=1),
            ReferenceSpec(prepared_image=prep, requested_vision_slot=3),
        )
    )

    with pytest.raises(ValueError, match="gaps detected"):
        resolve_reference_slots_and_aliases(chain)


def test_manual_slot_collision_raises():
    prep = _prepared_image()
    chain = ReferenceChain(
        (
            ReferenceSpec(prepared_image=prep, requested_vision_slot=1),
            ReferenceSpec(prepared_image=prep, requested_vision_slot=1),
        )
    )

    with pytest.raises(ValueError, match="Duplicate vision slot 1"):
        resolve_reference_slots_and_aliases(chain)


def test_appearance_reference_can_skip_qwen_without_losing_vae_frame():
    prep = _prepared_image()
    chain = ReferenceChain(
        (
            ReferenceSpec(prepared_image=prep, alias="vae only", include_in_vision=False),
            ReferenceSpec(prepared_image=prep, alias="qwen and vae", include_in_vision=True),
        )
    )

    resolved, _ = resolve_reference_slots_and_aliases(chain)

    assert resolved[0]["vae_reference_frame"] == 1
    assert resolved[0]["physical_qwen_range"] is None
    assert resolved[1]["vae_reference_frame"] == 2
    assert resolved[1]["physical_qwen_range"] == (1, 1)


def test_semantic_reference_has_no_vae_frame():
    prep = _prepared_image()
    chain = ReferenceChain(
        (
            ReferenceSpec(
                prepared_image=prep,
                alias="semantic",
                appearance_reference=False,
                include_in_vision=True,
            ),
        )
    )

    resolved, _ = resolve_reference_slots_and_aliases(chain)

    assert resolved[0]["vae_reference_frame"] is None
    assert resolved[0]["physical_qwen_range"] == (1, 1)
