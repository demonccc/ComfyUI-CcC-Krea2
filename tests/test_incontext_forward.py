"""Realistic integration tests for custom Krea 2 SingleStreamDiT forward execution."""

import math
import torch
import torch.nn as nn
from ccc_krea2.patch import (
    patch_krea2_model,
    krea2_dit_incontext_forward,
    _compute_ref_attention_bias_patchified
)
from ccc_krea2.references import PreparedReference, ReferenceRole
from ccc_krea2.latents import generate_krea2_latent


class MockRealKrea2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_freqs = None
        self.last_tvec = None
        self.last_attn_bias = None
        self.last_transformer_options = None

    def forward(self, x, freqs=None, tvec=None, attn_bias=None, transformer_options=None):
        self.last_freqs = freqs
        self.last_tvec = tvec
        self.last_attn_bias = attn_bias
        self.last_transformer_options = transformer_options
        return x


class MockKrea2DiT(nn.Module):
    """MockKrea2DiT exposing real Krea 2 SingleStreamDiT members."""

    def __init__(self, in_channels=16, patch_size=2, hidden_dim=64):
        super().__init__()
        self.patch = patch_size
        self.channels = in_channels
        self.hidden_dim = hidden_dim

        self.unpack_context_called = False
        self.first_called_count = 0
        self.pe_embedder_called_with = None
        self.last_called = False

        # Real Krea 2 input layers
        self.first_layer = nn.Linear(in_channels * patch_size * patch_size, hidden_dim)
        self.tproj_layer = nn.Linear(1, hidden_dim)
        self.tmlp_layer = nn.Linear(hidden_dim, hidden_dim)

        self.block1 = MockRealKrea2Block()
        self.blocks = nn.ModuleList([self.block1])

        self.last_layer = nn.Linear(hidden_dim, in_channels * patch_size * patch_size)

    def _unpack_context(self, context):
        self.unpack_context_called = True
        return context

    def first(self, x_patch):
        self.first_called_count += 1
        return self.first_layer(x_patch)

    def tproj(self, t):
        return self.tproj_layer(t.unsqueeze(-1) if t.ndim == 1 else t)

    def tmlp(self, t_emb):
        return self.tmlp_layer(t_emb)

    def pe_embedder(self, position_ids):
        self.pe_embedder_called_with = position_ids
        return torch.rand((position_ids.shape[1], self.hidden_dim))

    def last(self, h_seq):
        self.last_called = True
        return self.last_layer(h_seq)

    def forward(self, x, timesteps, context):
        raise RuntimeError("Native text-to-image forward must never be called during custom edit forward!")


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


def test_real_krea2_member_execution_order():
    dit = MockKrea2DiT()
    x = torch.rand((1, 16, 32, 32))  # 32x32 -> 16x16 = 256 target tokens
    timesteps = torch.tensor([1.0])
    context = torch.rand((1, 77, 64))
    ref_lat = torch.rand((1, 16, 16, 16))  # 16x16 -> 8x8 = 64 ref tokens

    opts = {"option_a": True}

    out = krea2_dit_incontext_forward(
        dit_model=dit,
        x=x,
        timesteps=timesteps,
        context=context,
        ref_latents=[ref_lat],
        ref_boosts=[2.5],
        ref_masks=[None],
        mask_modes=["hard"],
        transformer_options=opts
    )

    # 1. Assert _unpack_context was called
    assert dit.unpack_context_called is True

    # 2. Assert first was called for target and reference (2 times)
    assert dit.first_called_count == 2

    # 3. Assert pe_embedder received full position sequence [3, 77 + 64 + 256] = [3, 397]
    assert dit.pe_embedder_called_with is not None
    assert dit.pe_embedder_called_with.shape == (3, 397)

    # 4. Assert block received freqs, tvec, attn_bias, transformer_options
    blk = dit.block1
    assert blk.last_freqs is not None
    assert blk.last_tvec is not None
    assert blk.last_attn_bias is not None
    assert blk.last_transformer_options == opts

    # 5. Assert last was called
    assert dit.last_called is True

    # 6. Assert target tokens alone are returned with 4D target shape
    assert out.shape == (1, 16, 32, 32)


def test_5d_latent_shape_preservation():
    dit = MockKrea2DiT()
    x_5d = torch.rand((1, 16, 1, 32, 32))
    timesteps = torch.tensor([1.0])
    context = torch.rand((1, 77, 64))
    ref_lat_5d = torch.rand((1, 16, 1, 16, 16))

    out_5d = krea2_dit_incontext_forward(
        dit_model=dit,
        x=x_5d,
        timesteps=timesteps,
        context=context,
        ref_latents=[ref_lat_5d],
        ref_boosts=[1.0],
        ref_masks=[None],
        mask_modes=["hard"],
        transformer_options={}
    )

    assert out_5d.ndim == 5
    assert out_5d.shape == (1, 16, 1, 32, 32)


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

    assert torch.allclose(lat_dict["samples"], torch.tensor(0.5))
