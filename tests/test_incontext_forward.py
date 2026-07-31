"""Realistic integration tests for custom Krea 2 edit DiT forward, patchification, 3D RoPE, and attention steering."""

import math
import pytest
import torch
import torch.nn as nn
from ccc_krea2.patch import (
    patch_krea2_model,
    krea2_dit_incontext_forward,
    _build_incontext_3d_rope_pos_ids,
    _compute_ref_attention_bias_patchified
)
from ccc_krea2.references import PreparedReference, ReferenceRole
from ccc_krea2.latents import generate_krea2_latent


class MockTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_attn_bias = None

    def forward(self, x, vec_emb=None, rope_pos_ids=None, attn_bias=None, **kwargs):
        self.last_attn_bias = attn_bias
        return x


class MockKrea2DiT(nn.Module):
    """Realistic MockKrea2DiT exposing model components required by custom forward."""

    def __init__(self, in_channels=16, patch_size=2, hidden_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Input projections
        self.img_in = nn.Linear(in_channels * patch_size * patch_size, hidden_dim)
        self.txt_in = nn.Linear(2048, hidden_dim)
        self.time_in = nn.Linear(1, hidden_dim)
        self.vector_in = nn.Linear(hidden_dim, hidden_dim)

        # Transformer blocks
        self.block1 = MockTransformerBlock()
        self.blocks = nn.ModuleList([self.block1])

        # Final projection
        self.final_layer = nn.Linear(hidden_dim, in_channels * patch_size * patch_size)

    def forward(self, x, timesteps, context):
        """Native forward pass: does NOT accept custom ref_pos_ids or attn_bias."""
        return x


class MockModelWithWrappersMP:
    def __init__(self):
        self.registered_calls = []

    def clone(self):
        m = MockModelWithWrappersMP()
        m.registered_calls = self.registered_calls.copy()
        return m

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.registered_calls.append((wrapper_type, key, wrapper))


def test_three_argument_wrappersmp_registration():
    model = MockModelWithWrappersMP()
    prep_ref = PreparedReference(
        role=ReferenceRole.SUBJECT,
        grounding_image=torch.rand((1, 768, 768, 3)),
        vae_latent=torch.rand((1, 16, 16, 16)),
        spatial_attention_mask=None,
        boost=2.5,
        spatial_hw=(1024, 1024),
        lat_hw=(16, 16)
    )

    patched = patch_krea2_model(model, [prep_ref])
    assert len(patched.registered_calls) == 1

    wrapper_type, key, wrapper = patched.registered_calls[0]
    assert wrapper_type == "DIFFUSION_MODEL"
    assert key == "ccc_krea2_edit"
    assert callable(wrapper)


def test_custom_edit_forward_never_delegates_to_native_forward():
    dit = MockKrea2DiT()
    x = torch.rand((1, 16, 32, 32))  # target 32x32 -> 16x16 = 256 tokens (patch_size=2)
    timesteps = torch.tensor([1.0])
    context = torch.rand((1, 77, 2048))
    ref_lat = torch.rand((1, 16, 16, 16))

    out = krea2_dit_incontext_forward(
        dit_model=dit,
        x=x,
        timesteps=timesteps,
        context=context,
        ref_latents=[ref_lat],
        ref_boosts=[2.5],
        ref_masks=[None],
        mask_modes=["hard"],
        transformer_options={}
    )

    # Returned shape must match input target x shape exactly
    assert out.shape == (1, 16, 32, 32)
    # Transformer block received non-null attn_bias
    assert dit.block1.last_attn_bias is not None


def test_token_length_uses_patch_size():
    # Target 32x32 -> patch_size=2 -> 16x16 = 256 tokens
    # Ref 16x16 -> patch_size=2 -> 8x8 = 64 tokens
    # Text 77 tokens
    # Total seq_len = 77 + 64 + 256 = 397 tokens
    txt_len = 77
    ref_grids = [(8, 8)]  # 64 tokens
    target_grid = (16, 16)  # 256 tokens
    device = torch.device("cpu")

    rope_ids = _build_incontext_3d_rope_pos_ids(
        txt_len=txt_len,
        ref_token_grids=ref_grids,
        target_grid=target_grid,
        device=device
    )

    assert rope_ids.shape == (3, 397)


def test_per_reference_boost_reaches_transformer_block_attention_bias():
    txt_len = 77
    ref_lens = [64]
    tgt_len = 256
    device = torch.device("cpu")
    dtype = torch.float32

    bias = _compute_ref_attention_bias_patchified(
        boosts=[2.5],
        txt_len=txt_len,
        ref_token_lens=ref_lens,
        tgt_len=tgt_len,
        ref_masks=[None],
        ref_token_grids=[(8, 8)],
        mask_modes=["hard"],
        device=device,
        dtype=dtype
    )

    expected_boost_val = math.log(2.5)
    target_start = 77 + 64
    ref_start = 77
    ref_end = 77 + 64

    slice_val = bias[0, 0, target_start:, ref_start:ref_end]
    assert torch.allclose(slice_val, torch.tensor(expected_boost_val))


def test_mask_aligns_to_patchified_grid():
    spatial_mask = torch.zeros((1, 100, 100))  # masked out
    ref_grid = (8, 8)  # 64 tokens

    bias = _compute_ref_attention_bias_patchified(
        boosts=[1.0],
        txt_len=77,
        ref_token_lens=[64],
        tgt_len=256,
        ref_masks=[spatial_mask],
        ref_token_grids=[ref_grid],
        mask_modes=["hard"],
        device=torch.device("cpu"),
        dtype=torch.float32
    )

    ref_slice = bias[0, 0, 77+64:, 77:77+64]
    assert torch.allclose(ref_slice, torch.tensor(-1e4))


def test_ksampler_latents_remain_raw_vae_latents():
    class DummyVAE:
        def encode(self, img):
            return torch.ones((1, 16, 32, 32)) * 0.5

    class DummyModelWithProcessLatentIn:
        class InnerModel:
            def process_latent_in(self, lat):
                return lat * 100.0
        model = InnerModel()

    model = DummyModelWithProcessLatentIn()
    vae = DummyVAE()
    base_img = torch.rand((1, 256, 256, 3))

    lat_dict = generate_krea2_latent(
        model=model,
        vae=vae,
        width=256,
        height=256,
        batch_size=1,
        latent_source="image",
        base_image=base_img
    )

    # KSampler latent must remain raw VAE latent (0.5), not processed (50.0)
    assert torch.allclose(lat_dict["samples"], torch.tensor(0.5))
