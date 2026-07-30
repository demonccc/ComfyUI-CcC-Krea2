"""Integration-style unit tests using mocked WrapperExecutor and current ComfyUI signatures."""

import pytest
import torch
from ccc_krea2.patch import patch_krea2_model, krea2_dit_incontext_forward
from ccc_krea2.references import PreparedReference, ReferenceRole, _process_latent_in_if_available
from ccc_krea2.latents import generate_krea2_latent
from ccc_krea2.engine import Krea2EditEngine, NodeExecutionRequest
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS


class MockSingleStreamDiT:
    def __init__(self):
        self.last_kwargs = {}
        self.called = False

    def forward(self, x, timesteps, context, ref_latents=None, attn_bias=None, **kwargs):
        self.called = True
        self.last_kwargs = kwargs
        self.last_kwargs["ref_latents"] = ref_latents
        self.last_kwargs["attn_bias"] = attn_bias
        return x


class MockWrapperExecutor:
    def __init__(self, dit_model):
        self.class_obj = dit_model

    def __call__(self, x, timesteps, context, *wargs, **kwargs):
        return self.class_obj.forward(x, timesteps, context, **kwargs)


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
            return torch.rand((b, 16, h // 8, w // 8))
        return torch.rand((1, 16, 16, 16))


def test_diffusion_model_wrapper_execution_signature():
    model = MockModel()
    dit = MockSingleStreamDiT()
    executor = MockWrapperExecutor(dit)

    ref_lat = torch.rand((1, 16, 16, 16))
    prep_ref = PreparedReference(
        role=ReferenceRole.SUBJECT,
        grounding_image=torch.rand((1, 768, 768, 3)),
        vae_latent=ref_lat,
        token_attention_mask=None,
        boost=2.5,
        spatial_hw=(1024, 1024),
        lat_hw=(16, 16)
    )

    # Canonical patch call: patch_krea2_model(model, prepared_refs)
    patched = patch_krea2_model(model, [prep_ref])

    wrappers = patched.model_options["transformer_options"]["wrappers"]
    assert len(wrappers) == 1
    wrapper = wrappers[0]

    x = torch.rand((1, 16, 128, 128))
    timesteps = torch.tensor([1.0])
    context = torch.rand((1, 77, 2048))

    out = wrapper(executor, x, timesteps, context)

    assert dit.called
    assert dit.last_kwargs["attn_bias"] is not None
    assert torch.equal(out, x)


def test_process_latent_in_execution():
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


def test_all_seven_node_registrations():
    expected = {
        "CcCKrea2Subject",
        "CcCKrea2SubjectOutfit",
        "CcCKrea2SubjectScene",
        "CcCKrea2SubjectSceneOutfit",
        "CcCKrea2Inpaint",
        "CcCKrea2InpaintSubjectOutfit",
        "CcCKrea2InpaintSubjectScene",
    }
    assert set(NODE_CLASS_MAPPINGS.keys()) == expected
