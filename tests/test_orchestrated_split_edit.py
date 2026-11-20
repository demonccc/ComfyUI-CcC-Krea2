"""Regression coverage for the split Krea2 CcC Edit contract."""

import inspect

import torch

from ccc_krea2.identity_contract import (
    build_grounded_negative_user_content,
    build_grounded_positive_user_content,
)
from ccc_krea2.modular_nodes import edit_node
from ccc_krea2.patch import attach_reference_runtime_to_conditioning


def _extras(conditioning):
    return conditioning[0][1]


def test_split_edit_uses_positional_qwen_builders():
    assert edit_node.edit_engine_runtime.build_grounded_positive_user_content is build_grounded_positive_user_content
    assert edit_node.edit_engine_runtime.build_grounded_negative_user_content is build_grounded_negative_user_content


def test_split_edit_negative_prompt_is_fixed_empty_and_not_user_editable():
    required = edit_node.CcCKrea2Edit.INPUT_TYPES()["required"]
    signature = inspect.signature(edit_node.CcCKrea2Edit.process)

    assert edit_node.KREA2_EDIT_NEGATIVE_PROMPT == ""
    assert "negative_prompt" not in required
    assert "negative_prompt" not in signature.parameters

    source = inspect.getsource(edit_node.CcCKrea2Edit.process)
    assert "negative_prompt=KREA2_EDIT_NEGATIVE_PROMPT" in source


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

    assert "reference_fit" not in _extras(positive)
    assert _extras(positive)["reference_rope_positions"] == ropes
    assert _extras(positive)["reference_boosts"] == [2.0, 6.0]

    assert "reference_fit" not in _extras(negative)
    assert _extras(negative)["reference_rope_positions"] == ropes
    assert "reference_boosts" not in _extras(negative)


def test_split_edit_uses_single_ccc_runtime():
    source = inspect.getsource(edit_node.CcCKrea2Edit.process)
    assert "run_krea2_edit_orchestrator" in source
    assert "attach_reference_runtime_to_conditioning" in source
    assert "patch_krea2_model" in source
    assert 'runtime_mode = "ccc"' in source
    assert "_external_" not in source


def test_latent_semantic_preserves_target_composition_directive_after_vision_refactor():
    source = inspect.getsource(edit_node._combine_reference_chains)
    assert "SEMANTIC_SUBJECT_DIRECTIVE" in source
    assert 'if mode == "semantic":' in source
    assert "instruction=semantic_instruction" in source
    assert "build_vision_reference_spec(entry)" in source
