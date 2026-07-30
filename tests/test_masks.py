"""CPU-safe unit tests for mask processing and attention boost numerical stability."""

import pytest
import torch
import math
from ccc_krea2.masks import process_inpaint_mask, process_attention_mask
from ccc_krea2.patch import _compute_ref_attention_bias


def test_inpaint_mask_processing():
    mask = torch.zeros((100, 100))
    mask[40:60, 40:60] = 1.0

    # Invert
    inv = process_inpaint_mask(mask, invert=True)
    assert inv[0, 0, 0].item() == 1.0
    assert inv[0, 50, 50].item() == 0.0

    # Grow
    grown = process_inpaint_mask(mask, grow=5)
    assert grown.sum() > mask.sum()

    # Blur
    blurred = process_inpaint_mask(mask, blur=5)
    assert 0.0 < blurred[0, 38, 38].item() < 1.0


def test_attention_mask_hard_vs_soft():
    mask = torch.tensor([[0.2, 0.7], [0.4, 0.9]])

    # Hard threshold
    hard = process_attention_mask(mask, mode="hard")
    assert torch.equal(hard[0], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

    # Soft mode
    soft = process_attention_mask(mask, mode="soft")
    assert torch.allclose(soft[0], mask)


def test_ref_attention_bias_numerical_safety():
    device = torch.device("cpu")
    dtype = torch.float32

    # Boost = 1.0 -> No bias added (returns None)
    bias_off = _compute_ref_attention_bias(
        boosts=[1.0], txt_len=10, ref_lens=[64], tgt_len=64,
        ref_masks=[None], device=device, dtype=dtype
    )
    assert bias_off is None

    # Boost > 1.0 (e.g. 2.5) -> Positive log bias
    bias_high = _compute_ref_attention_bias(
        boosts=[2.5], txt_len=10, ref_lens=[64], tgt_len=64,
        ref_masks=[None], device=device, dtype=dtype
    )
    assert bias_high is not None
    assert not torch.isnan(bias_high).any()
    assert not torch.isinf(bias_high).any()
    expected_val = math.log(2.5)
    assert abs(bias_high[0, 0, 74, 10].item() - expected_val) < 1e-4

    # Boost < 1.0 (e.g. 0.5) -> Negative log bias
    bias_low = _compute_ref_attention_bias(
        boosts=[0.5], txt_len=10, ref_lens=[64], tgt_len=64,
        ref_masks=[None], device=device, dtype=dtype
    )
    assert bias_low is not None
    assert not torch.isnan(bias_low).any()
    assert not torch.isinf(bias_low).any()
    assert bias_low[0, 0, 74, 10].item() < 0.0

    # Zero boost (0.0) -> Clamped to 1e-4, no log(0) -inf exception or NaN
    bias_zero = _compute_ref_attention_bias(
        boosts=[0.0], txt_len=10, ref_lens=[64], tgt_len=64,
        ref_masks=[None], device=device, dtype=dtype
    )
    assert bias_zero is not None
    assert not torch.isnan(bias_zero).any()
    assert not torch.isinf(bias_zero).any()
