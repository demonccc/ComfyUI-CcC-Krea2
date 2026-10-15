"""Regression tests for opt-in Qwen Vision weighting."""
import torch

from ccc_krea2.vision_attention import install_vision_attention

class _FakeModel:
    def __init__(self):
        self.patch = None
    def clone(self):
        return _FakeModel()
    def set_model_attn1_patch(self, fn):
        self.patch = fn

def test_off_by_default_without_spans():
    model = _FakeModel()
    out, count = install_vision_attention(model, [[torch.zeros(1, 4, 12), {}]])
    assert out is model
    assert count == 0

def test_only_vision_rows_weighted():
    model = _FakeModel()
    cond = [[torch.zeros(1, 6, 12), {"ccc_vision_weight_spans": ((1, 3, 0.5),)}]]
    patched, count = install_vision_attention(model, cond)
    assert patched is not model
    assert count == 2
    v = torch.ones(1, 1, 6, 8)
    out = patched.patch(v, v, v, extra_options={"cond_or_uncond": [0]})
    observed = out["v"]
    assert torch.all(observed[:, :, 1:3] == 0.5)
    assert torch.all(observed[:, :, :1] == 1)
    assert torch.all(observed[:, :, 3:] == 1)

def test_never_weight_unconditional_cfg_rows():
    model = _FakeModel()
    cond = [[torch.zeros(1, 4, 12), {"ccc_vision_weight_spans": ((0, 2, 0.35),)}]]
    patched, _ = install_vision_attention(model, cond)
    v = torch.ones(2, 1, 4, 8)
    result = patched.patch(v, v, v, extra_options={"cond_or_uncond": [0, 1]})["v"]
    assert torch.all(result[0, :, :2] == 0.35)
    assert torch.all(result[1] == 1)

def test_strength_one_does_not_patch_model():
    model = _FakeModel()
    out, n = install_vision_attention(
        model, [[torch.zeros(1, 4, 12), {"ccc_vision_weight_spans": ((0, 2, 1.0),)}]]
    )
    assert out is model and n == 0
