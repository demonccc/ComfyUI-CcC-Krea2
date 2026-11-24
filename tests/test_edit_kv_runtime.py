"""Contracts for experimental Edit K/V reference attention."""

import math

import pytest
import torch

from ccc_krea2.modular_nodes.edit_node import CcCKrea2Edit
from ccc_krea2.edit_kv_runtime import _edit_kv_reference_bias
from ccc_krea2.patch import patch_krea2_model


def _bias(boosts, maps=None):
    return _edit_kv_reference_bias(
        batch_size=1, txt_len=2, target_len=3,
        ref_grids=[(2, 2)] * len(boosts),
        boosts=boosts, attention_maps=maps or [None] * len(boosts),
        device=torch.device("cpu"), dtype=torch.float32,
    )


def test_kv_toggle_is_off_by_default_and_describes_experimental_runtime():
    schema = CcCKrea2Edit.INPUT_TYPES()["required"]
    assert schema["use_kv_reference"][0] == "BOOLEAN"
    assert schema["use_kv_reference"][1]["default"] is False


def test_no_boost_or_mask_does_not_allocate_bias():
    assert _bias([0.0]) is None


def test_kv_boost_weights_only_target_to_reference_attention():
    bias = _bias([5.0])
    assert bias.shape == (1, 1, 5, 9)
    assert torch.all(bias[:, :, :2] == 0)
    assert torch.all(bias[:, :, 2:, :5] == 0)
    torch.testing.assert_close(bias[:, :, 2:, 5:], torch.full((1, 1, 3, 4), math.log(10.0)))


def test_kv_minus_five_excludes_reference_keys_without_masking_known_target():
    bias = _bias([-5.0])
    assert torch.isneginf(bias[:, :, 2:, 5:]).all()
    assert torch.isfinite(bias[:, :, :, :5]).all()


def test_kv_regional_mask_overrides_global_boost():
    weights = torch.tensor([[[-5.0, 2.0], [0.0, 5.0]]])
    bias = _bias([4.0], [weights])
    assert torch.isneginf(bias[0, 0, 2:, 5]).all()
    assert bias[0, 0, 2, 6].item() == pytest.approx(2.0 * math.log(10.0) / 5.0)
    assert bias[0, 0, 2, 7].item() == pytest.approx(0.0)
    assert bias[0, 0, 2, 8].item() == pytest.approx(math.log(10.0))


class FakeModel:
    def clone(self):
        return FakeModel()

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrapper = wrapper


def test_joint_and_kv_both_register_edit_wrapper_without_mutating_original():
    original = FakeModel()
    joint = patch_krea2_model(original)
    cached = patch_krea2_model(original, use_kv_reference=True)
    assert joint._ccc_patch_key == cached._ccc_patch_key == "ccc_krea2_edit"
    assert hasattr(joint, "wrapper")
    assert hasattr(cached, "wrapper")
    assert not hasattr(original, "wrapper")
