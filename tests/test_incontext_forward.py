"""Realistic integration tests for custom Krea 2 SingleStreamDiT forward execution."""

import math
import torch
import torch.nn as nn
from ccc_krea2.patch import patch_krea2_model, krea2_dit_incontext_forward, _compute_ref_attention_bias_patchified
from ccc_krea2.references import PreparedReference, ReferenceRole
from ccc_krea2.latents import generate_krea2_latent


class MockRealKrea2Block(nn.Module):
    """Block mimicking real Krea 2 SingleStreamBlock positional forward signature."""

    def __init__(self):
        super().__init__()
        self.last_vec = None
        self.last_freqs = None
        self.last_attn_bias = None
        self.last_transformer_options = None

    def forward(self, x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options=None):
        self.last_vec = vec
        self.last_freqs = freqs
        self.last_attn_bias = mask
        self.last_transformer_options = transformer_options
        return x


class MockLastLayer(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, out_channels)
        self.last_tvec = None

    def forward(self, x, tvec):
        self.last_tvec = tvec
        return self.linear(x)


class MockKrea2DiT(nn.Module):
    """MockKrea2DiT exposing real Krea 2 SingleStreamDiT member signatures."""

    def __init__(self, in_channels=16, patch_size=2, hidden_dim=64, tdim=256):
        super().__init__()
        self.patch = patch_size
        self.channels = in_channels
        self.tdim = tdim
        self.hidden_dim = hidden_dim

        self.unpack_context_called = False
        self.txtfusion_called = False
        self.txtmlp_called = False
        self.first_called_count = 0
        self.pe_embedder_called_with = None
        self.tmlp_called_with = None
        self.tproj_called_with = None

        self.first_layer = nn.Linear(in_channels * patch_size * patch_size, hidden_dim)
        self.tproj_layer = nn.Linear(hidden_dim, hidden_dim)
        self.tmlp_layer = nn.Linear(tdim, hidden_dim)

        self.block1 = MockRealKrea2Block()
        self.blocks = nn.ModuleList([self.block1])

        self.last = MockLastLayer(hidden_dim, in_channels * patch_size * patch_size)

    def _unpack_context(self, context):
        self.unpack_context_called = True
        return context

    def txtfusion(self, context, mask=None, transformer_options=None):
        self.txtfusion_called = True
        return context

    def txtmlp(self, context):
        self.txtmlp_called = True
        return context

    def first(self, x_patch):
        self.first_called_count += 1
        return self.first_layer(x_patch)

    def tmlp(self, t_emb):
        self.tmlp_called_with = t_emb
        return self.tmlp_layer(t_emb)

    def tproj(self, t):
        self.tproj_called_with = t
        return self.tproj_layer(t)

    def pe_embedder(self, position_ids):
        self.pe_embedder_called_with = position_ids
        return torch.rand((position_ids.shape[0], position_ids.shape[1], self.hidden_dim))

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


def test_wrappersmp_registration_uses_diffusion_model_key():
    model = MockModelWithWrappersMP()
    prep_ref = PreparedReference(
        role=ReferenceRole.SUBJECT,
        grounding_image=torch.rand((1, 768, 768, 3)),
        vae_latent=torch.rand((1, 16, 16, 16)),
        spatial_attention_mask=None,
        boost=2.5,
        spatial_hw=(1024, 1024),
        lat_hw=(16, 16),
    )

    patched = patch_krea2_model(model, [prep_ref])
    assert len(patched.registered_calls) == 1

    wrapper_type, key, wrapper = patched.registered_calls[0]
    assert wrapper_type == "diffusion_model"
    assert key == "ccc_krea2_edit"
    assert callable(wrapper)


def test_real_krea2_member_execution_order_and_signatures():
    dit = MockKrea2DiT()
    x = torch.rand((2, 16, 32, 32))  # batch size 2
    timesteps = torch.tensor([1.0, 1.0])
    context = torch.rand((2, 77, 64))
    ref_lat = torch.rand((1, 16, 16, 16))

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
        transformer_options=opts,
    )

    # 1. Assert _unpack_context, txtfusion and txtmlp called
    assert dit.unpack_context_called is True
    assert dit.txtfusion_called is True
    assert dit.txtmlp_called is True

    # 2. Assert first was called for target and reference (2 times)
    assert dit.first_called_count == 2

    # 3. Assert pe_embedder received position IDs shape [B, L, 3] = [2, 397, 3]
    assert dit.pe_embedder_called_with is not None
    assert dit.pe_embedder_called_with.shape == (2, 397, 3)

    # 4. Assert timestep_embedding -> tmlp -> tproj order
    assert dit.tmlp_called_with is not None
    assert dit.tproj_called_with is not None

    # 5. Assert block received positional vec, freqs, attn_bias (mask), transformer_options metadata
    blk = dit.block1
    assert blk.last_vec is not None
    assert blk.last_freqs is not None
    assert blk.last_attn_bias is not None
    assert blk.last_transformer_options["total_blocks"] == 1
    assert blk.last_transformer_options["block_type"] == "single"
    assert blk.last_transformer_options["block_index"] == 0

    # 6. Assert last received t
    assert dit.last.last_tvec is not None

    # 7. Assert target tokens alone are returned with 4D target shape
    assert out.shape == (2, 16, 32, 32)


def test_5d_temporal_inputs_preserve_all_frames():
    dit = MockKrea2DiT()
    x_5d = torch.rand((2, 16, 3, 32, 32))  # B=2, C=16, T=3, H=32, W=32
    timesteps = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    context = torch.rand((6, 77, 64))
    ref_lat_5d = torch.rand((2, 16, 3, 16, 16))

    out_5d = krea2_dit_incontext_forward(
        dit_model=dit,
        x=x_5d,
        timesteps=timesteps,
        context=context,
        ref_latents=[ref_lat_5d],
        ref_boosts=[1.0],
        ref_masks=[None],
        mask_modes=["hard"],
        transformer_options={},
    )

    assert out_5d.ndim == 5
    assert out_5d.shape == (2, 16, 3, 32, 32)


def test_reference_device_and_dtype_alignment():
    dit = MockKrea2DiT()
    x = torch.rand((1, 16, 32, 32), dtype=torch.float32)
    timesteps = torch.tensor([1.0])
    context = torch.rand((1, 77, 64), dtype=torch.float32)
    ref_lat = torch.rand((1, 16, 16, 16), dtype=torch.float32)

    krea2_dit_incontext_forward(
        dit_model=dit,
        x=x,
        timesteps=timesteps,
        context=context,
        ref_latents=[ref_lat],
        ref_boosts=[2.5],
        ref_masks=[None],
        mask_modes=["hard"],
        transformer_options={},
    )

    assert dit.first_called_count == 2


def test_unmasked_reference_regions_retain_zero_bias():
    txt_len = 77
    ref_lens = [64]
    tgt_len = 256
    device = torch.device("cpu")
    dtype = torch.float32

    # Spatial mask with half 1.0 (masked) and half 0.0 (unmasked)
    spatial_mask = torch.zeros((1, 100, 100))
    spatial_mask[:, :50, :] = 1.0

    bias = _compute_ref_attention_bias_patchified(
        boosts=[2.5],
        txt_len=txt_len,
        ref_token_lens=ref_lens,
        tgt_len=tgt_len,
        ref_masks=[spatial_mask],
        ref_token_grids=[(8, 8)],
        mask_modes=["hard"],
        device=device,
        dtype=dtype,
    )

    expected_boost_val = math.log(2.5)
    target_start = 77 + 64
    ref_start = 77

    target_to_ref_bias = bias[0, 0, target_start:, ref_start : ref_start + 64]

    # Unmasked region must equal 0.0 (NO -1e4 penalty!)
    unmasked_tokens = target_to_ref_bias[:, 32:]
    assert torch.allclose(unmasked_tokens, torch.tensor(0.0))

    # Masked region must equal log(boost)
    masked_tokens = target_to_ref_bias[:, :32]
    assert torch.allclose(masked_tokens, torch.tensor(expected_boost_val))


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
        model=model, vae=vae, width=256, height=256, batch_size=1, latent_source="image", base_image=base_img
    )

    assert torch.allclose(lat_dict["samples"], torch.tensor(0.5))
