"""Contract-level comparisons of Krea2Edit-native and optional CcC conditioning.

These tests do not claim GPU image-fidelity parity with upstream.
"""
import math

import pytest
import torch

from ccc_krea2.style_processing import StyleSpanOperation, apply_statistical_style_fidelity
from ccc_krea2.patch import _compute_ref_attention_bias_patchified


def _transform(mode: str, value: float):
    raw = torch.arange(1 * 8 * 24, dtype=torch.float32).reshape(1, 8, 24) + 0.5
    if mode in ("native", "context"):
        return raw, raw
    op = StyleSpanOperation(
        logical_reference_id="test",
        logical_vision_slot=1,
        physical_qwen_index=1,
        row_start=1,
        row_end=7,
        style_fidelity=value,
        indirect_style_transfer=False,
        extract="subject",
        summary_rows=0,
    )
    transformed, _, _ = apply_statistical_style_fidelity(raw, [op])
    return raw, transformed


def test_native_does_not_change_qwen_vision_rows():
    original, output = _transform("native", 1.0)
    assert torch.equal(original, output)


def test_context_without_extraction_does_not_change_qwen_vision_rows():
    original, output = _transform("context", 1.0)
    assert torch.equal(original, output)


def test_semantic_whitens_seventy_five_percent_of_vision_rows():
    original, output = _transform("semantic", 0.25)
    source = original[:, 1:7].reshape(1, 6, 12, 2)
    mean = source.mean(dim=1, keepdim=True)
    sigma = source.std(dim=1, keepdim=True, correction=1) + 1e-6
    expected = 0.25 * source + 0.75 * (source - mean) / sigma
    torch.testing.assert_close(output[:, 1:7].reshape(1, 6, 12, 2), expected)
    assert not torch.equal(original[:, 1:7], output[:, 1:7])
    assert torch.equal(original[:, :1], output[:, :1])
    assert torch.equal(original[:, 7:], output[:, 7:])


@pytest.mark.parametrize("boost", [0.5, 1.0, 4.0, 6.0])
def test_global_boost_matches_upstream_log_attention_formula(boost):
    # Upstream _ref_attn_bias applies log(boost) only to target-query to
    # reference-key positions; this is a direct arithmetic contract check.
    bias = _compute_ref_attention_bias_patchified(
        boosts=[boost], txt_len=2, ref_token_lens=[4], tgt_len=3,
        ref_token_grids=[(2, 2)], target_grid=(2, 2),
        device=torch.device("cpu"), dtype=torch.float32, batch_size=1,
    )
    if boost == 1.0:
        assert bias is None
        return
    assert bias.shape == (1, 1, 9, 9)
    torch.testing.assert_close(
        bias[0, 0, 6:, 2:6], torch.full((3, 4), math.log(boost))
    )
    assert torch.count_nonzero(bias[0, 0, :6]) == 0
    assert torch.count_nonzero(bias[0, 0, 6:, :2]) == 0
    assert torch.count_nonzero(bias[0, 0, 6:, 6:]) == 0
