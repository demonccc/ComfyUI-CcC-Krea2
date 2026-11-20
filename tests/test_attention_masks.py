"""Behavioral tests for global boosts and regional attention masks."""

import math

import pytest
import torch

from ccc_krea2.patch import _compute_ref_attention_bias_patchified


def bias_for(*, boosts, grids=None,
             target_grid=(2, 4), batch_size=1):
    grids = grids or [(2, 2)] * len(boosts)
    return _compute_ref_attention_bias_patchified(
        boosts=boosts,
        txt_len=3,
        ref_token_lens=[h * w for h, w in grids],
        tgt_len=target_grid[0] * target_grid[1],
        ref_token_grids=grids,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=batch_size,
    )


def test_unmasked_reference_boost_is_global_and_neutral_is_noop():
    assert bias_for(boosts=[0.0]) is None
    bias = bias_for(boosts=[2.0])
    # Three text tokens, four reference tokens, eight target tokens.
    torch.testing.assert_close(
        bias[0, 0, 7:, 3:7], torch.full((8, 4), 2.0 * math.log(10.0) / 5.0)
    )


def test_boost_inputs_are_signed_and_increment_by_tenth():
    from ccc_krea2.modular_nodes.visual_reference_node import CcCKrea2VisualReference
    from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask

    for spec in (
        CcCKrea2VisualReference.INPUT_TYPES()["required"]["boost"],
        CcCKrea2Mask.INPUT_TYPES()["required"]["boost"],
    ):
        assert spec[1]["default"] == 0.0
        assert spec[1]["min"] == -5.0
        assert spec[1]["max"] == 5.0
        assert spec[1]["step"] == 0.1

    bias = bias_for(boosts=[0.1])
    torch.testing.assert_close(
        bias[0, 0, 7:, 3:7],
        torch.full((8, 4), 0.1 * math.log(10.0) / 5.0),
    )


def test_signed_boost_preserves_previous_multiplier_range():
    # Check the agreed symmetric exponential mapping at every integer step.
    for value in range(-4, 6):
        expected = 10.0 ** (value / 5.0)
        bias = bias_for(boosts=[value])
        if value == 0:
            assert bias is None
        else:
            torch.testing.assert_close(
                bias[0, 0, 7:, 3:7],
                torch.full((8, 4), math.log(expected)),
            )


def test_minus_five_boost_blocks_direct_reference_attention():
    bias = bias_for(boosts=[-5.0])
    assert torch.isneginf(bias[0, 0, 7:, 3:7]).all()
    assert (bias[0, 0, :7, :] == 0).all()
    assert (bias[0, 0, 7:, :3] == 0).all()
    assert (bias[0, 0, 7:, 7:] == 0).all()


def test_minus_five_blocks_only_its_reference():
    bias = bias_for(boosts=[-5.0, 2.0])
    assert torch.isneginf(bias[0, 0, 11:, 3:7]).all()
    torch.testing.assert_close(
        bias[0, 0, 11:, 7:11],
        torch.full((8, 4), 2.0 * math.log(10.0) / 5.0),
    )


def test_regional_minus_five_blocks_while_zero_remains_neutral():
    weight_map = torch.tensor([[[-5., 2.], [0., -5.]]])
    bias = _compute_ref_attention_bias_patchified(
        boosts=[0.0], txt_len=3, ref_token_lens=[4], tgt_len=8,
        ref_token_grids=[(2, 2)], ref_attention_maps=[weight_map],
        device=torch.device("cpu"), dtype=torch.float32,
    )
    selected = bias[0, 0, 7:, 3:7]
    assert torch.isneginf(selected[:, [0, 3]]).all()
    torch.testing.assert_close(selected[:, 1], torch.full((8,), 2.0 * math.log(10.0) / 5.0))
    assert (selected[:, 2] == 0).all()


@pytest.mark.parametrize("boost", [-5.0, -4.9, -4.0, -3.0, -2.0, -1.0, 0.0, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0])
def test_uniform_regional_boost_matches_global(boost):
    regional = _compute_ref_attention_bias_patchified(
        boosts=[0.0], txt_len=3, ref_token_lens=[4], tgt_len=8,
        ref_token_grids=[(2, 2)],
        ref_attention_maps=[torch.full((1, 2, 2), boost)],
        device=torch.device("cpu"), dtype=torch.float32,
    )
    global_bias = bias_for(boosts=[boost])
    if global_bias is None:
        assert (regional == 0).all()
    else:
        torch.testing.assert_close(regional, global_bias)


def test_negative_boost_reduces_attention():
    bias = bias_for(boosts=[-2.0])
    torch.testing.assert_close(bias[0, 0, 7:, 3:7], torch.full((8, 4), -2.0 * math.log(10.0) / 5.0))


def test_out_of_range_boost_is_rejected():
    with pytest.raises(ValueError, match=r"\[-5, 5\]"):
        bias_for(boosts=[-5.1])


def test_mask_composite_effective_boost_last_region_wins():
    from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask
    source = torch.ones((1, 4, 4, 3))
    full = torch.ones((1, 4, 4))
    left = torch.zeros((1, 4, 4))
    left[:, :, :2] = 1.0
    center = torch.zeros((1, 4, 4))
    center[:, 1:3, 1:3] = 1.0
    node = CcCKrea2Mask()
    _, _, first, _, _ = node.process(source, source="mask", mask=full, boost=-3.0)
    _, _, second, _, _ = node.process(
        source, source="mask", mask=left, boost=5.0,
        previous_mask_attention=first,
    )
    original, union, third, composite, _ = node.process(
        source, source="mask", mask=center, boost=-5.0,
        previous_mask_attention=second, composite_color="#000000",
    )
    torch.testing.assert_close(original, source)
    assert torch.all(union == 1)
    assert len(third.regions) == 3
    # The last region overrides even Boost +5; outside it is Boost -3 or +5.
    assert composite[0, 1, 1, 0].item() == pytest.approx(0.10)
    assert composite[0, 0, 0, 0].item() == pytest.approx(0.95)
    assert composite[0, 0, 3, 0].item() == pytest.approx(0.27)


def test_mask_composite_color_and_unselected_pixels():
    from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask
    source = torch.ones((1, 2, 2, 3))
    selection = torch.zeros((1, 2, 2))
    selection[0, 0, 0] = 1.0
    _, _, _, composite, _ = CcCKrea2Mask().process(
        source, source="mask", mask=selection, boost=5,
        composite_color="#FF0000",
    )
    # Selected Boost +5 is tinted slightly; unselected pixels remain original.
    torch.testing.assert_close(composite[0, 0, 0], torch.tensor([1.0, 0.95, 0.95]))
    torch.testing.assert_close(composite[0, 1, 1], torch.ones(3))


def test_mask_composite_rejects_invalid_hex():
    from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask
    with pytest.raises(ValueError, match="composite_color"):
        CcCKrea2Mask().process(
            torch.ones((1, 2, 2, 3)), source="mask",
            mask=torch.ones((1, 2, 2)), composite_color="red",
        )


def test_composite_unselected_pixels_remain_exact_for_minus_five_boost():
    from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask
    source = torch.tensor([[[[0.2, 0.3, 0.4], [0.7, 0.6, 0.5]]]])
    selection = torch.tensor([[[1.0, 0.0]]])
    original, _, _, composite, _ = CcCKrea2Mask().process(
        source, source="mask", mask=selection, boost=-5,
        composite_color="#000000",
    )
    torch.testing.assert_close(composite[0, 0, 1], original[0, 0, 1])
    torch.testing.assert_close(composite[0, 0, 0], source[0, 0, 0] * 0.10)


def test_unified_reference_boost_regions_override_global_only_inside_masks():
    from ccc_krea2.modular_nodes.reference_node import CcCKrea2Reference
    from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask

    image = torch.ones((1, 4, 4, 3))
    selected = torch.zeros((1, 4, 4))
    selected[:, 1:3, 1:3] = 1
    mask_chain = CcCKrea2Mask().process(
        image, source="mask", mask=selected, boost=-5,
    )[2]
    node = CcCKrea2Reference()
    assert "mask_attention" in node.INPUT_TYPES()["optional"]
    assert "reference_boost_mask" not in node.INPUT_TYPES()["optional"]
    refs, _ = node.process(
        image=image, vae_rope_enable=True, vision_enable=False,
        vae_rope_boost=2.0, mask_attention=mask_chain,
    )
    entry = refs.entries[0]
    assert entry.boost == pytest.approx(2.0)
    assert entry.attention_weight_map[0, 0, 0].item() == pytest.approx(2.0)
    assert entry.attention_weight_map[0, 1, 1].item() == pytest.approx(-5.0)


def test_unified_reference_regional_masks_last_wins():
    from ccc_krea2.modular_nodes.reference_node import CcCKrea2Reference
    from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask

    image = torch.ones((1, 4, 4, 3))
    left = torch.zeros((1, 4, 4))
    left[:, :, :2] = 1
    center = torch.zeros((1, 4, 4))
    center[:, 1:3, 1:3] = 1
    mask_node = CcCKrea2Mask()
    earlier = mask_node.process(image, source="mask", mask=left, boost=4)[2]
    latest = mask_node.process(
        image, source="mask", mask=center, boost=-2,
        previous_mask_attention=earlier,
    )[2]
    refs, _ = CcCKrea2Reference().process(
        image=image, vae_rope_enable=True, vision_enable=False,
        vae_rope_boost=0, mask_attention=latest,
    )
    entry = refs.entries[0]
    effective = entry.attention_weight_map[0]
    assert effective[0, 0].item() == pytest.approx(4)
    assert effective[1, 1].item() == pytest.approx(-2)
    assert effective[0, 3].item() == pytest.approx(0)
