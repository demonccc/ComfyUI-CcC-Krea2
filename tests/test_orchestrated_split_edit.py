"""Regression coverage for the split CcC Krea2 Edit contract."""

import inspect

import torch

from ccc_krea2.identity_contract import (
    build_grounded_negative_user_content,
    build_grounded_positive_user_content,
)
from ccc_krea2.modular_nodes import edit_node
from ccc_krea2.modular_nodes.edit_reference_types import VisualReferenceEntry
from ccc_krea2.patch import attach_reference_runtime_to_conditioning


def _extras(conditioning):
    return conditioning[0][1]


def test_split_edit_uses_positional_qwen_builders():
    assert edit_node.edit_engine_runtime.build_krea2_user_content is build_grounded_positive_user_content
    assert edit_node.edit_engine_runtime.build_krea2_negative_user_content is build_grounded_negative_user_content


def test_split_edit_negative_prompt_is_fixed_empty_and_not_user_editable():
    required = edit_node.CcCKrea2Edit.INPUT_TYPES()["required"]
    signature = inspect.signature(edit_node.CcCKrea2Edit.process)

    assert edit_node.KREA2_EDIT_NEGATIVE_PROMPT == ""
    assert "negative_prompt" not in required
    assert "negative_prompt" not in signature.parameters

    source = inspect.getsource(edit_node.CcCKrea2Edit.process)
    assert "negative_prompt=KREA2_EDIT_NEGATIVE_PROMPT" in source


def test_reference_fit_modes_expose_simple_public_contract():
    native = VisualReferenceEntry(image=torch.zeros((1, 64, 32, 3)), reference_fit="native")
    resize = VisualReferenceEntry(image=torch.zeros((1, 64, 32, 3)), reference_fit="resize")
    crop = VisualReferenceEntry(image=torch.zeros((1, 64, 32, 3)), reference_fit="crop")

    assert native.resolved_fit_mode == "native"
    assert resize.resolved_fit_mode == "resize"
    assert crop.resolved_fit_mode == "crop"


def test_positive_and_negative_runtime_boosts_are_independent():
    refs = [
        torch.zeros((1, 16, 1, 176, 116)),
        torch.zeros((1, 16, 1, 176, 82)),
    ]
    base = [[torch.zeros((1, 4, 8)), {"reference_latents": refs}]]
    ropes = ["inside:center:center", "inside:center:center"]

    positive = attach_reference_runtime_to_conditioning(
        base,
        reference_count=2,
        rope_positions=ropes,
        reference_boosts=[2.0, 6.0],
    )
    negative = attach_reference_runtime_to_conditioning(
        base,
        reference_count=2,
        rope_positions=ropes,
        reference_boosts=None,
    )

    assert _extras(positive)["reference_fit"] == [True, True]
    assert _extras(positive)["reference_rope_positions"] == ropes
    assert _extras(positive)["reference_boosts"] == [2.0, 6.0]

    assert _extras(negative)["reference_fit"] == [True, True]
    assert _extras(negative)["reference_rope_positions"] == ropes
    assert "reference_boosts" not in _extras(negative)


def test_split_edit_uses_single_ccc_runtime():
    source = inspect.getsource(edit_node.CcCKrea2Edit.process)
    assert "run_krea2_edit_orchestrator" in source
    assert "attach_reference_runtime_to_conditioning" in source
    assert "patch_krea2_model" in source
    assert 'runtime_mode = "ccc"' in source
    assert "_external_" not in source
