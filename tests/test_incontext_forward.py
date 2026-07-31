"""Unit tests for Krea2 in-context forward pass, 3D RoPE position IDs, attention steering, and ModelPatcher registration."""

import math
import pytest
import torch
from ccc_krea2.patch import (
    patch_krea2_model,
    krea2_dit_incontext_forward,
    _build_ref_3d_rope_pos_ids,
    _compute_ref_attention_bias
)
from ccc_krea2.references import PreparedReference, ReferenceRole


class MockDiTWithKwargs:
    def __init__(self):
        self.last_kwargs = {}

    def forward(self, x, timesteps, context, ref_latents=None, ref_pos_ids=None, attn_bias=None, transformer_options=None):
        self.last_kwargs = {
            "x": x,
            "timesteps": timesteps,
            "context": context,
            "ref_latents": ref_latents,
            "ref_pos_ids": ref_pos_ids,
            "attn_bias": attn_bias,
            "transformer_options": transformer_options
        }
        # Return sequence matching text + ref + target (e.g. 77 + 256 + 1024 = 1357 tokens)
        seq_len = 77 + 256 + 1024
        return torch.rand((1, seq_len, 16))


class MockModelWithWrappersMP:
    def __init__(self):
        self.registered_calls = []

    def clone(self):
        m = MockModelWithWrappersMP()
        m.registered_calls = self.registered_calls.copy()
        return m

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.registered_calls.append((wrapper_type, key, wrapper))


def test_modelpatcher_wrappersmp_registration():
    model = MockModelWithWrappersMP()
    prep_ref = PreparedReference(
        role=ReferenceRole.SUBJECT,
        grounding_image=torch.rand((1, 768, 768, 3)),
        vae_latent=torch.rand((1, 16, 16, 16)),
        token_attention_mask=None,
        boost=2.5,
        spatial_hw=(1024, 1024),
        lat_hw=(16, 16)
    )

    patched = patch_krea2_model(model, [prep_ref])
    assert len(patched.registered_calls) == 1

    wrapper_type, key, wrapper = patched.registered_calls[0]
    assert key == "ccc_krea2_edit"
    assert callable(wrapper)


def test_boost_changes_logit_bias():
    txt_len = 77
    ref_lens = [256]
    tgt_len = 1024
    device = torch.device("cpu")
    dtype = torch.float32

    bias_default = _compute_ref_attention_bias(
        boosts=[1.0],
        txt_len=txt_len,
        ref_lens=ref_lens,
        tgt_len=tgt_len,
        ref_masks=[None],
        device=device,
        dtype=dtype
    )
    assert bias_default is None

    bias_boosted = _compute_ref_attention_bias(
        boosts=[2.5],
        txt_len=txt_len,
        ref_lens=ref_lens,
        tgt_len=tgt_len,
        ref_masks=[None],
        device=device,
        dtype=dtype
    )
    assert bias_boosted is not None
    expected_b_val = math.log(2.5)

    ref_start = 77
    ref_end = 77 + 256
    target_start = 77 + 256

    target_ref_slice = bias_boosted[0, 0, target_start:, ref_start:ref_end]
    assert torch.allclose(target_ref_slice, torch.tensor(expected_b_val))


def test_ref_fit_offsets_change_rope_pos_ids():
    device = torch.device("cpu")

    pos_zero = _build_ref_3d_rope_pos_ids(
        frame_idx=1,
        lat_h=16,
        lat_w=16,
        y_offset=0.0,
        x_offset=0.0,
        device=device
    )

    pos_offset = _build_ref_3d_rope_pos_ids(
        frame_idx=1,
        lat_h=16,
        lat_w=16,
        y_offset=64.0,
        x_offset=32.0,
        device=device
    )

    # Frame dimension (row 0) should be identical
    assert torch.equal(pos_zero[0], pos_offset[0])

    # Y grid (row 1) should be offset by 64.0 / 8.0 = +8.0
    assert torch.allclose(pos_offset[1] - pos_zero[1], torch.tensor(8.0))

    # X grid (row 2) should be offset by 32.0 / 8.0 = +4.0
    assert torch.allclose(pos_offset[2] - pos_zero[2], torch.tensor(4.0))


def test_per_reference_mask_affects_only_matching_block():
    txt_len = 77
    ref_lens = [100, 100]
    tgt_len = 400
    device = torch.device("cpu")
    dtype = torch.float32

    mask1 = torch.zeros(100)  # masked out
    mask2 = None

    bias = _compute_ref_attention_bias(
        boosts=[1.0, 1.0],
        txt_len=txt_len,
        ref_lens=ref_lens,
        tgt_len=tgt_len,
        ref_masks=[mask1, mask2],
        device=device,
        dtype=dtype
    )

    ref1_slice = bias[0, 0, 77+200:, 77:177]
    ref2_slice = bias[0, 0, 77+200:, 177:277]

    assert torch.allclose(ref1_slice, torch.tensor(-1e4))
    assert torch.allclose(ref2_slice, torch.tensor(0.0))


def test_returned_token_range_contains_target_only():
    dit = MockDiTWithKwargs()

    x = torch.rand((1, 16, 32, 32))  # 1024 target tokens
    timesteps = torch.tensor([1.0])
    context = torch.rand((1, 77, 2048))
    ref_lat = torch.rand((1, 16, 16, 16))

    out = krea2_dit_incontext_forward(
        dit_model=dit,
        x=x,
        timesteps=timesteps,
        context=context,
        ref_latents=[ref_lat],
        ref_boosts=[1.0],
        ref_masks=[None],
        ref_fit=[{"y_offset": 0.0, "x_offset": 0.0}],
        transformer_options={}
    )

    # Output shape must match x target shape [1, 16, 32, 32]
    assert out.shape == (1, 16, 32, 32)
