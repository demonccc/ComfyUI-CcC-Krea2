"""Regional reference attention and reusable Paint mask union."""
import math
import torch
from ccc_krea2.patch import _compute_ref_attention_bias_patchified
from ccc_krea2.modular_nodes.mask_node import CcCKrea2Mask


def _bias(weight_map, boost=3.0):
    return _compute_ref_attention_bias_patchified(
        boosts=[boost], txt_len=1, ref_token_lens=[4], tgt_len=4,
        ref_token_grids=[(2, 2)], ref_attention_maps=[weight_map],
        target_grid=(2, 2), device=torch.device("cpu"), dtype=torch.float32,
    )[0, 0, 5:, 1:5]


def test_regions_replace_not_multiply_global_boost():
    regions = torch.tensor([[[2., 3.], [0.5, 3.]]])
    bias = _bias(regions, boost=3.0)
    torch.testing.assert_close(bias[0], regions[0].flatten().log())


def test_zero_region_blocks_direct_attention():
    regions = torch.tensor([[[2., 0.], [3., 3.]]])
    bias = _bias(regions, boost=3.0)
    assert torch.isneginf(bias[:, 1]).all()
    assert bias[0, 0].item() == torch.tensor(2.).log().item()


def test_mask_chain_union_and_last_region_priority():
    image = torch.zeros((1, 8, 8, 3))
    first = torch.zeros((1, 8, 8))
    first[:, :5, :5] = 1
    second = torch.zeros((1, 8, 8))
    second[:, 3:, 3:] = 1
    node = CcCKrea2Mask()
    image_out, m1, c1, _, _ = node.process(
        image, source="external_mask", external_mask=first, boost=2.
    )
    image_out2, merged, c2, _, _ = node.process(
        image_out, source="external_mask", external_mask=second,
        previous_masks=c1, boost=0.
    )
    assert image_out2 is image
    torch.testing.assert_close(merged, torch.maximum(first, second))
    assert [r.boost for r in c2.regions] == [2., 0.]
    # Later region wins for overlap (3,3).
    weights = torch.full((1, 8, 8), 1.)
    for region in c2.regions:
        weights = torch.where(region.mask > 0.5, region.boost, weights)
    assert weights[0, 3, 3] == 0.
    assert weights[0, 0, 0] == 2.
    assert weights[0, 0, 7] == 1.


def test_external_mask_requires_no_sam_dependency():
    image = torch.zeros((1, 16, 16, 3))
    mask = torch.ones((1, 16, 16))
    _, merged, _, _, _ = CcCKrea2Mask().process(
        image, external_mask=mask, source="external_mask"
    )
    torch.testing.assert_close(merged, mask)

