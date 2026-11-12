"""Behavioral tests for source masks and regional boosts on visual references."""

import math

import pytest
import torch

from ccc_krea2.patch import _compute_ref_attention_bias_patchified


def bias_for(*, boosts, source_masks=None, grids=None,
             target_grid=(2, 4), batch_size=1):
    grids = grids or [(2, 2)] * len(boosts)
    return _compute_ref_attention_bias_patchified(
        boosts=boosts,
        txt_len=3,
        ref_token_lens=[h * w for h, w in grids],
        tgt_len=target_grid[0] * target_grid[1],
        ref_masks=source_masks,
        ref_token_grids=grids,
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


def test_source_masks_match_their_own_reference_blocks():
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


def test_wrong_source_mask_batch_fails_clearly():
    bad = torch.ones((3, 16, 16))
    with pytest.raises(ValueError, match="batch"):
        bias_for(boosts=[2.0], source_masks=[bad], batch_size=2)


def test_zero_boost_hard_blocks_target_attention_to_reference():
    bias = bias_for(boosts=[0.0])
    assert torch.isneginf(bias[0, 0, 7:, 3:7]).all()
    assert (bias[0, 0, :7, :] == 0).all()
    assert (bias[0, 0, 7:, :3] == 0).all()
    assert (bias[0, 0, 7:, 7:] == 0).all()


def test_zero_boost_blocks_only_masked_source_tokens():
    source_mask = torch.tensor([[[1., 0.], [0., 1.]]])
    bias = bias_for(boosts=[0.0], source_masks=[source_mask])
    selected = bias[0, 0, 7:, 3:7]
    assert torch.isneginf(selected[:, [0, 3]]).all()
    assert (selected[:, [1, 2]] == 0).all()


def test_zero_boost_only_affects_its_reference_when_others_use_ten():
    bias = bias_for(boosts=[0.0, 10.0])
    assert torch.isneginf(bias[0, 0, 11:, 3:7]).all()
    torch.testing.assert_close(
        bias[0, 0, 11:, 7:11],
        torch.full((8, 4), math.log(10.0)),
    )


def test_zero_boost_with_region_map_blocks_zero_regions():
    weight_map = torch.tensor([[[0., 10.], [1., 0.]]])
    bias = _compute_ref_attention_bias_patchified(
        boosts=[1.0], txt_len=3, ref_token_lens=[4], tgt_len=8,
        ref_token_grids=[(2, 2)], ref_attention_maps=[weight_map],
        target_grid=(2, 4), device=torch.device("cpu"), dtype=torch.float32,
    )
    selected = bias[0, 0, 7:, 3:7]
    assert torch.isneginf(selected[:, [0, 3]]).all()
    torch.testing.assert_close(selected[:, 1], torch.full((8,), math.log(10.0)))
    assert (selected[:, 2] == 0).all()


def test_negative_boost_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        bias_for(boosts=[-0.5])


def test_mask_composite_effective_boost_last_region_wins():
    from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask
    source = torch.ones((1, 4, 4, 3))
    full = torch.ones((1, 4, 4))
    left = torch.zeros((1, 4, 4))
    left[:, :, :2] = 1.0
    center = torch.zeros((1, 4, 4))
    center[:, 1:3, 1:3] = 1.0
    node = CcCKrea2Mask()
    _, _, first, _, _ = node.process(source, source="mask", mask=full, boost=2.0)
    _, _, second, _, _ = node.process(
        source, source="mask", mask=left, boost=10.0,
        previous_mask_attention=first,
    )
    original, union, third, composite, _ = node.process(
        source, source="mask", mask=center, boost=0.0,
        previous_mask_attention=second, composite_color="#000000",
    )
    torch.testing.assert_close(original, source)
    assert torch.all(union == 1)
    assert len(third.regions) == 3
    # The last region overrides even Boost 10; outside it is Boost 2 or 10.
    assert composite[0, 1, 1, 0].item() == pytest.approx(0.10)
    assert composite[0, 0, 0, 0].item() == pytest.approx(0.95)
    assert composite[0, 0, 3, 0].item() == pytest.approx(0.27)


def test_mask_composite_color_and_unselected_pixels():
    from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask
    source = torch.ones((1, 2, 2, 3))
    selection = torch.zeros((1, 2, 2))
    selection[0, 0, 0] = 1.0
    _, _, _, composite, _ = CcCKrea2Mask().process(
        source, source="mask", mask=selection, boost=10,
        composite_color="#FF0000",
    )
    # Selected Boost 10 remains nearly original; unselected defaults Boost 0.
    torch.testing.assert_close(composite[0, 0, 0], torch.tensor([1.0, 0.95, 0.95]))
    torch.testing.assert_close(composite[0, 1, 1], torch.tensor([1.0, 0.1, 0.1]))


def test_mask_composite_rejects_invalid_hex():
    from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask
    with pytest.raises(ValueError, match="composite_color"):
        CcCKrea2Mask().process(
            torch.ones((1, 2, 2, 3)), source="mask",
            mask=torch.ones((1, 2, 2)), composite_color="red",
        )
