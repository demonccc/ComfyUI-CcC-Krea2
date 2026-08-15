"""Unit tests for batched reference attention masks and bias computation."""

import math
import torch
from ccc_krea2.patch import _compute_ref_attention_bias_patchified


def test_batch_1_mask_applies_expected_boost():
    # 1 reference with boost 2.5 and 10x10 mask (half 1.0, half 0.0)
    boost = 2.5
    expected_b_val = math.log(boost)

    # Spatial mask 1x10x10
    mask = torch.zeros((1, 10, 10))
    mask[:, :5, :] = 1.0  # Top half active

    bias = _compute_ref_attention_bias_patchified(
        boosts=[boost],
        txt_len=10,
        ref_token_lens=[25],  # 5x5 token grid
        tgt_len=64,
        ref_masks=[mask],
        ref_token_grids=[(5, 5)],
        mask_modes=["hard"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert bias is not None
    # Sequence layout: text (0..10), ref (10..35), target (35..99)
    # Target tokens (35..99) attending to ref tokens (10..35)
    tgt_to_ref_bias = bias[0, 0, 35:, 10:35]  # shape (64, 25)

    # Active tokens should have expected_b_val; inactive tokens should have 0.0
    assert torch.allclose(tgt_to_ref_bias[:, :15], torch.tensor(expected_b_val), atol=1e-4)
    assert torch.allclose(tgt_to_ref_bias[:, 15:], torch.tensor(0.0), atol=1e-4)


def test_batch_2_mask_is_not_silently_ignored_and_uses_first_item():
    boost = 2.0
    expected_b_val = math.log(boost)

    # Batch of 2 masks: item 0 has top half active, item 1 has bottom half active
    mask_b2 = torch.zeros((2, 20, 20))
    mask_b2[0, :10, :] = 1.0
    mask_b2[1, 10:, :] = 1.0

    bias = _compute_ref_attention_bias_patchified(
        boosts=[boost],
        txt_len=10,
        ref_token_lens=[100],  # 10x10 grid
        tgt_len=64,
        ref_masks=[mask_b2],
        ref_token_grids=[(10, 10)],
        mask_modes=["hard"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert bias is not None
    # Sequence layout: text (0..10), ref (10..110), target (110..174)
    tgt_to_ref_bias = bias[0, 0, 110:, 10:110]  # shape (64, 100)

    # Must NOT be silently ignored (not all zeros)
    assert not torch.all(tgt_to_ref_bias == 0.0)

    # First mask item (top half active = first 50 tokens) must be used
    assert torch.allclose(tgt_to_ref_bias[:, :50], torch.tensor(expected_b_val), atol=1e-4)
    assert torch.allclose(tgt_to_ref_bias[:, 50:], torch.tensor(0.0), atol=1e-4)


def test_hard_and_soft_mask_modes():
    boost = 3.0
    expected_b_val = math.log(boost)

    # Soft mask with gradient values: 0.2, 0.6, 0.8
    mask_soft = torch.tensor([[[0.2, 0.6], [0.8, 0.0]]])

    # 1. Hard mode: threshold > 0.5 -> [0.0, 1.0, 1.0, 0.0]
    bias_hard = _compute_ref_attention_bias_patchified(
        boosts=[boost],
        txt_len=5,
        ref_token_lens=[4],
        tgt_len=10,
        ref_masks=[mask_soft],
        ref_token_grids=[(2, 2)],
        mask_modes=["hard"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    tgt_to_ref_hard = bias_hard[0, 0, 9:, 5:9]
    assert tgt_to_ref_hard[0, 0].item() == 0.0
    assert abs(tgt_to_ref_hard[0, 1].item() - expected_b_val) < 1e-4
    assert abs(tgt_to_ref_hard[0, 2].item() - expected_b_val) < 1e-4
    assert tgt_to_ref_hard[0, 3].item() == 0.0

    # 2. Soft mode: clamp(0, 1) -> [0.2 * b_val, 0.6 * b_val, 0.8 * b_val, 0.0]
    bias_soft = _compute_ref_attention_bias_patchified(
        boosts=[boost],
        txt_len=5,
        ref_token_lens=[4],
        tgt_len=10,
        ref_masks=[mask_soft],
        ref_token_grids=[(2, 2)],
        mask_modes=["soft"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    tgt_to_ref_soft = bias_soft[0, 0, 9:, 5:9]
    assert abs(tgt_to_ref_soft[0, 0].item() - (0.2 * expected_b_val)) < 1e-4
    assert abs(tgt_to_ref_soft[0, 1].item() - (0.6 * expected_b_val)) < 1e-4
    assert abs(tgt_to_ref_soft[0, 2].item() - (0.8 * expected_b_val)) < 1e-4
    assert tgt_to_ref_soft[0, 3].item() == 0.0


def test_regions_outside_mask_retain_zero_additional_bias():
    boost = 4.0
    # Fully unmasked region (mask == 0.0)
    mask_zero = torch.zeros((1, 10, 10))

    bias = _compute_ref_attention_bias_patchified(
        boosts=[boost],
        txt_len=5,
        ref_token_lens=[25],
        tgt_len=10,
        ref_masks=[mask_zero],
        ref_token_grids=[(5, 5)],
        mask_modes=["hard"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert bias is not None
    # All attention bias values must be exactly 0.0 (NO -1e4 penalty)
    assert torch.all(bias == 0.0)
