"""Regression coverage for split Edit parity with the Krea2 Identity conditioning contract."""

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


def test_split_edit_uses_positional_identity_qwen_builders():
    assert edit_node.edit_engine_runtime.build_krea2_user_content is build_grounded_positive_user_content
    assert edit_node.edit_engine_runtime.build_krea2_negative_user_content is build_grounded_negative_user_content


def test_reference_fit_mode_exposes_full_target_crop_baseline():
    fit = VisualReferenceEntry(image=torch.zeros((1, 64, 32, 3)), fit_mode="fit")
    crop = VisualReferenceEntry(image=torch.zeros((1, 64, 32, 3)), fit_mode="crop (legacy)")

    assert fit.resolved_fit_mode == "fit"
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


def test_split_edit_prepares_with_conditioning_transport_then_patches_model():
    source = inspect.getsource(edit_node.CcCKrea2Edit.process)
    assert "run_krea2_edit_orchestrator" in source
    assert "apply_model_patch=False" in source
    assert "attach_reference_runtime_to_conditioning" in source
    assert "patch_krea2_model" in source
    assert "patch_krea2_orchestrated_model" not in source
