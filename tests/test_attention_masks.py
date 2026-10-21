"""Behavioral tests for source and target masks on ordered visual references."""

import math

import pytest
import torch

from ccc_krea2.patch import _compute_ref_attention_bias_patchified


def bias_for(*, boosts, source_masks=None, target_masks=None, scopes=None, grids=None,
             target_grid=(2, 4), batch_size=1):
    grids = grids or [(2, 2)] * len(boosts)
    return _compute_ref_attention_bias_patchified(
        boosts=boosts,
        txt_len=3,
        ref_token_lens=[h * w for h, w in grids],
        tgt_len=target_grid[0] * target_grid[1],
        ref_masks=source_masks,
        ref_token_grids=grids,
        ref_target_masks=target_masks,
        ref_attention_scopes=scopes,
        target_grid=target_grid,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=batch_size,
    )


def test_unmasked_reference_boost_is_global_and_neutral_is_noop():
    assert bias_for(boosts=[1.0]) is None
    bias = bias_for(boosts=[4.0])
    # Three text tokens, four reference tokens, eight target tokens.
    torch.testing.assert_close(
        bias[0, 0, 7:, 3:7], torch.full((8, 4), math.log(4.0))
    )


def test_reference_mask_boosts_only_selected_reference_tokens():
    # Reference image: 16x16, select its left half, keeping the VAE geometry.
    source_mask = torch.zeros((1, 16, 16))
    source_mask[:, :, :8] = 1.0
    bias = bias_for(boosts=[4.0], source_masks=[source_mask])
    selected = bias[0, 0, 7:, 3:7]
    torch.testing.assert_close(selected[:, [0, 2]], torch.full((8, 2), math.log(4.0)))
    assert torch.count_nonzero(selected[:, [1, 3]]) == 0


def test_target_mask_boosts_only_query_tokens_inside_target():
    target = torch.zeros((1, 16, 32))
    target[:, :, :16] = 1.0
    bias = bias_for(boosts=[2.0], target_masks=[target], scopes=["boost in mask"])
    selected = bias[0, 0, 7:, 3:7]
    rows = [0, 1, 4, 5]
    other = [2, 3, 6, 7]
    torch.testing.assert_close(selected[rows], torch.full((4, 4), math.log(2.0)))
    assert torch.count_nonzero(selected[other]) == 0


def test_source_and_target_masks_intersect_not_overwrite_each_other():
    source = torch.tensor([[[1., 0.], [0., 1.]]])
    target = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])
    bias = bias_for(boosts=[3.0], source_masks=[source],
                    target_masks=[target], scopes=["boost in mask"])
    selected = bias[0, 0, 7:, 3:7]
    assert selected[0, 0].item() == pytest.approx(math.log(3.0))
    assert selected[0, 3].item() == pytest.approx(math.log(3.0))
    assert selected[0, 1].item() == 0.0
    assert selected[1, 0].item() == 0.0


def test_only_in_mask_blocks_direct_attention_outside_target_even_with_neutral_boost():
    target = torch.zeros((1, 16, 32))
    target[:, :, :16] = 1.0
    bias = bias_for(boosts=[1.0], target_masks=[target], scopes=["only in mask"])
    selected = bias[0, 0, 7:, 3:7]
    assert (selected[[0, 1, 4, 5]] == 0.0).all()
    assert torch.isneginf(selected[[2, 3, 6, 7]]).all()


def test_mask_batches_are_independent():
    target = torch.zeros((2, 16, 32))
    target[0, :, :16] = 1.0
    target[1, :, 16:] = 1.0
    bias = bias_for(boosts=[2.0], target_masks=[target],
                    scopes=["boost in mask"], batch_size=2)
    a, b = bias[:, 0, 7:, 3:7]
    assert a[0, 0].item() == pytest.approx(math.log(2.0))
    assert b[0, 0].item() == 0.0
    assert a[3, 0].item() == 0.0
    assert b[3, 0].item() == pytest.approx(math.log(2.0))


def test_source_and_target_masks_match_their_own_reference_blocks():
    source_one = torch.tensor([[[1., 0.], [0., 0.]]])
    source_two = torch.tensor([[[0., 0.], [0., 1.]]])
    bias = bias_for(boosts=[2.0, 3.0], source_masks=[source_one, source_two])
    # text 0..2, first ref 3..6, second ref 7..10, target from 11 onward
    block1 = bias[0, 0, 11:, 3:7]
    block2 = bias[0, 0, 11:, 7:11]
    torch.testing.assert_close(block1[:, 0], torch.full((8,), math.log(2.0)))
    assert torch.count_nonzero(block1[:, 1:]) == 0
    torch.testing.assert_close(block2[:, 3], torch.full((8,), math.log(3.0)))
    assert torch.count_nonzero(block2[:, :3]) == 0


def test_missing_target_mask_and_wrong_batch_fail_clearly():
    with pytest.raises(ValueError, match="target_attention_mask"):
        bias_for(boosts=[2.0], scopes=["boost in mask"])
    bad = torch.ones((3, 16, 16))
    with pytest.raises(ValueError, match="batch"):
        bias_for(boosts=[2.0], source_masks=[bad], batch_size=2)
