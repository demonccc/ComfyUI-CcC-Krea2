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
    schema = CcCKrea2Edit.INPUT_TYPES()["optional"]
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



def test_kv_runtime_caches_reference_and_passes_boost_maps(monkeypatch):
    from ccc_krea2 import edit_kv_runtime

    class FakeDit:
        patch = 2

    source = torch.ones((1, 16, 8, 8))
    state = {"last_timestep": None, "entries": {}}
    calls = {"precompute": 0, "forward": 0}

    def fake_precompute(*args, **kwargs):
        calls["precompute"] += 1
        assert kwargs["rope_positions"] == ["inside:center:center"]
        return [("keys", "values")]

    def fake_forward(*args, **kwargs):
        calls["forward"] += 1
        bias = kwargs["reference_bias"](txt_len=2, target_len=16)
        assert bias.shape == (1, 1, 18, 34)
        assert torch.isneginf(bias[:, :, 2:, 18:]).all()
        return args[1]

    monkeypatch.setattr(edit_kv_runtime, "_precompute_registered_reference_kv", fake_precompute)
    monkeypatch.setattr(edit_kv_runtime, "_forward_with_registered_reference_kv", fake_forward)

    for _ in range(2):
        result = edit_kv_runtime.krea2_dit_edit_kv_forward(
            dit_model=FakeDit(), x=source, timesteps=torch.tensor([1.0]),
            context=torch.zeros((1, 2, 16)), ref_latents=[source],
            ref_boosts=[-5.0], ref_rope_positions=["inside:center:center"],
            kv_state=state,
        )
        torch.testing.assert_close(result, source)
    assert calls == {"precompute": 1, "forward": 2}


def test_edit_wrapper_dispatches_to_selected_path(monkeypatch):
    from ccc_krea2 import patch
    from ccc_krea2 import edit_kv_runtime

    calls = []

    def joint(**kwargs):
        calls.append("joint")
        return kwargs["x"]

    def kv(**kwargs):
        calls.append("kv")
        return kwargs["x"]

    monkeypatch.setattr(patch, "krea2_dit_incontext_forward", joint)
    monkeypatch.setattr(edit_kv_runtime, "krea2_dit_edit_kv_forward", kv)

    class Executor:
        class_obj = object()

    x = torch.zeros((1, 16, 8, 8))
    context = torch.zeros((1, 2, 16))
    ref = torch.ones_like(x)
    for enabled in (False, True):
        model = patch_krea2_model(FakeModel(), use_kv_reference=enabled)
        output = model.wrapper(
            Executor(), x, torch.tensor([1.0]), context,
            ref_latents=[ref], transformer_options={},
        )
        torch.testing.assert_close(output, x)
    assert calls == ["joint", "kv"]
