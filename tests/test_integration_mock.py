"""Integration-style unit tests using MockKrea2DiT exposing real Krea 2 members."""

import torch
import torch.nn as nn
from ccc_krea2.patch import patch_krea2_model
from ccc_krea2.references import PreparedReference, ReferenceRole, _process_latent_in_if_available
from ccc_krea2.latents import generate_krea2_latent
from ccc_krea2.engine import Krea2EditEngine, NodeExecutionRequest
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS


class MockRealKrea2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_attn_bias = None

    def forward(self, x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options=None):
        self.last_attn_bias = mask
        return x


class MockLastLayer(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, tvec):
        return self.linear(x)


class MockKrea2DiT(nn.Module):
    def __init__(self, in_channels=16, patch_size=2, hidden_dim=64, tdim=256):
        super().__init__()
        self.patch = patch_size
        self.channels = in_channels
        self.tdim = tdim
        self.hidden_dim = hidden_dim

        self.first_layer = nn.Linear(in_channels * patch_size * patch_size, hidden_dim)
        self.tproj_layer = nn.Linear(hidden_dim, hidden_dim)
        self.tmlp_layer = nn.Linear(tdim, hidden_dim)

        self.block1 = MockRealKrea2Block()
        self.blocks = nn.ModuleList([self.block1])
        self.last = MockLastLayer(hidden_dim, in_channels * patch_size * patch_size)

    def _unpack_context(self, context):
        return context

    def txtfusion(self, context, mask=None, transformer_options=None):
        return context

    def txtmlp(self, context):
        return context

    def first(self, x_patch):
        return self.first_layer(x_patch)

    def tmlp(self, t_emb):
        return self.tmlp_layer(t_emb)

    def tproj(self, t):
        return self.tproj_layer(t)

    def pe_embedder(self, position_ids):
        return torch.rand((position_ids.shape[0], position_ids.shape[1], self.hidden_dim))

    def forward(self, x, timesteps, context):
        raise RuntimeError("Native forward must never be called during custom edit forward!")


class MockWrapperExecutor:
    def __init__(self, dit_model):
        self.class_obj = dit_model

    def __call__(self, x, timesteps, context, *wargs, **kwargs):
        return self.class_obj.forward(x, timesteps, context)


class MockInnerModel:
    def __init__(self):
        self.load_device = torch.device("cpu")
        self.model_dtype = torch.float32

    def process_latent_in(self, latent):
        return latent * 0.13025


class MockModel:
    def __init__(self):
        self.model = MockInnerModel()
        self.model_options = {}

    def clone(self):
        m = MockModel()
        m.model_options = self.model_options.copy()
        return m


class MockVAE:
    def encode(self, image):
        if image.ndim == 4 and image.shape[-1] == 3:
            b, h, w, _ = image.shape
            return torch.ones((b, 16, h // 8, w // 8))
        return torch.ones((1, 16, 16, 16))


def test_diffusion_model_wrapper_execution_signature():
    model = MockModel()
    dit = MockKrea2DiT()
    executor = MockWrapperExecutor(dit)

    ref_lat = torch.rand((1, 16, 16, 16))
    prep_ref = PreparedReference(
        role=ReferenceRole.SUBJECT,
        grounding_image=torch.rand((1, 768, 768, 3)),
        vae_latent=ref_lat,
        spatial_attention_mask=None,
        boost=2.5,
        spatial_hw=(1024, 1024),
        lat_hw=(16, 16)
    )

    patched = patch_krea2_model(model, [prep_ref])

    wrappers = patched.model_options["transformer_options"]["wrappers"]
    if isinstance(wrappers, dict):
        diff_wrappers = wrappers.get("diffusion_model", wrappers)
        if isinstance(diff_wrappers, dict):
            stored = diff_wrappers["ccc_krea2_edit"]
            wrapper = stored[0] if isinstance(stored, list) else stored
        else:
            wrapper = diff_wrappers[0]
    else:
        wrapper = wrappers[0]

    x = torch.rand((1, 16, 32, 32))
    timesteps = torch.tensor([1.0])
    context = torch.rand((1, 77, 64))

    out = wrapper(executor, x, timesteps, context)

    assert dit.block1.last_attn_bias is not None
    assert out.shape == (1, 16, 32, 32)


def test_process_latent_in_execution_only_for_references():
    model = MockModel()
    raw_latent = torch.ones((1, 16, 16, 16))

    processed = _process_latent_in_if_available(model, raw_latent)
    assert torch.equal(processed, raw_latent * 0.13025)


def test_explicit_inpaint_base_role_selection():
    img_src = torch.rand((1, 500, 500, 3))
    img_sub = torch.rand((1, 400, 400, 3))
    img_scn = torch.rand((1, 600, 600, 3))

    req_inpaint = NodeExecutionRequest(
        node_name="CcC Krea2 - Inpaint",
        model=MockModel(),
        clip=None,
        prompt="test",
        vae=MockVAE(),
        source_image=img_src,
        subject_image=img_sub,
        inpaint_base_role=ReferenceRole.SOURCE
    )
    base_src = Krea2EditEngine._resolve_base_image(req_inpaint)
    assert torch.equal(base_src, img_src)

    req_inpaint_sub = NodeExecutionRequest(
        node_name="CcC Krea2 - Inpaint Subject + Outfit",
        model=MockModel(),
        clip=None,
        prompt="test",
        vae=MockVAE(),
        subject_image=img_sub,
        inpaint_base_role=ReferenceRole.SUBJECT
    )
    base_sub = Krea2EditEngine._resolve_base_image(req_inpaint_sub)
    assert torch.equal(base_sub, img_sub)

    req_inpaint_scn = NodeExecutionRequest(
        node_name="CcC Krea2 - Inpaint Subject + Scene",
        model=MockModel(),
        clip=None,
        prompt="test",
        vae=MockVAE(),
        scene_image=img_scn,
        inpaint_base_role=ReferenceRole.SCENE
    )
    base_scn = Krea2EditEngine._resolve_base_image(req_inpaint_scn)
    assert torch.equal(base_scn, img_scn)


def test_image_latent_batching_repetition():
    model = MockModel()
    vae = MockVAE()
    base_img = torch.rand((1, 512, 512, 3))

    lat_dict = generate_krea2_latent(
        model=model,
        vae=vae,
        width=1024,
        height=1024,
        batch_size=4,
        latent_source="image",
        base_image=base_img
    )

    assert lat_dict["samples"].shape[0] == 4


def test_all_node_registrations():
    expected = {
        "CcCKrea2Subject",
        "CcCKrea2SubjectOutfit",
        "CcCKrea2SubjectScene",
        "CcCKrea2SubjectSceneOutfit",
        "CcCKrea2Inpaint",
        "CcCKrea2InpaintSubjectOutfit",
        "CcCKrea2InpaintSubjectScene",
        "CcCKrea2ImageAdvancedSettings",
        "CcCKrea2EditAdvancedSettings",
    }
    assert set(NODE_CLASS_MAPPINGS.keys()) == expected
