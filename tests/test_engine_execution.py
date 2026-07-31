"""Realistic engine execution tests covering Krea2EditEngine.execute for key node workflows."""

from typing import List, Tuple
import torch
import torch.nn as nn

from ccc_krea2.engine import Krea2EditEngine, NodeExecutionRequest
from ccc_krea2.constants import ReferenceRole
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS
from ccc_krea2.patch import patch_krea2_model


class MockCLIPTokenizer:
    def __init__(self):
        self.tokenize_calls = []

    def tokenize(self, text: str, images: List[torch.Tensor] = None, llama_template: str = None) -> dict:
        self.tokenize_calls.append({
            "text": text,
            "images": images,
            "llama_template": llama_template
        })
        return {"text": text, "num_images": len(images) if images else 0}

    def encode_from_tokens_scheduled(self, tokens: dict) -> List[Tuple[torch.Tensor, dict]]:
        cond_tensor = torch.zeros((1, 77, 4096))
        return [[cond_tensor, {"prompt_text": tokens["text"]}]]


class MockKrea2SingleStreamDiT(nn.Module):
    def __init__(self, in_channels=16, patch_size=2, hidden_dim=64):
        super().__init__()
        self.patch = patch_size
        self.channels = in_channels

    def pe_embedder(self, pos_ids):
        return None

    def txtfusion(self, ctx, mask=None, transformer_options=None):
        return ctx


class MockInnerKrea2Model:
    def __init__(self):
        self.load_device = torch.device("cpu")
        self.model_dtype = torch.float32
        self.diffusion_model = MockKrea2SingleStreamDiT()

    def process_latent_in(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * 0.13025


class MockModel:
    def __init__(self):
        self.model = MockInnerKrea2Model()
        self.model_options = {}

    def clone(self):
        m = MockModel()
        m.model_options = self.model_options.copy()
        return m


class MockVAE:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4 and image.shape[-1] == 3:
            b, h, w, _ = image.shape
            val = float(image.mean().item())
            return torch.full((b, 16, h // 8, w // 8), fill_value=val)
        return torch.ones((1, 16, 16, 16))


def test_engine_execute_subject_workflow():
    model = MockModel()
    clip = MockCLIPTokenizer()
    vae = MockVAE()
    subject_img = torch.rand((1, 512, 512, 3))

    req = NodeExecutionRequest(
        node_name="CcC Krea2 - Subject",
        model=model,
        clip=clip,
        prompt="photo of subject person standing in park",
        negative_prompt="blurry, distorted",
        vae=vae,
        subject_image=subject_img,
        subject_boost=2.5,
        role_order=[ReferenceRole.SUBJECT]
    )

    patched_model, positive, negative, latent_dict = Krea2EditEngine.execute(req)

    # 1. Assert exactly 4 outputs returned
    assert patched_model is not None
    assert positive is not None
    assert negative is not None
    assert isinstance(latent_dict, dict) and "samples" in latent_dict

    # 2. Assert model is patched
    assert "transformer_options" in patched_model.model_options

    # 3. Assert positive and negative tokenize with identical image references
    assert len(clip.tokenize_calls) == 2
    pos_call = clip.tokenize_calls[0]
    neg_call = clip.tokenize_calls[1]

    assert pos_call["text"] == "photo of subject person standing in park"
    assert neg_call["text"] == "blurry, distorted"
    assert len(pos_call["images"]) == 1
    assert len(neg_call["images"]) == 1
    assert torch.equal(pos_call["images"][0], neg_call["images"][0])
    assert pos_call["llama_template"] == neg_call["llama_template"]

    # 4. Assert conditionings do not contain duplicated reference_latents metadata
    pos_extra = positive[0][1]
    assert "reference_latents" not in pos_extra
    assert "reference_boosts" not in pos_extra


def test_engine_execute_subject_scene_workflow():
    model = MockModel()
    clip = MockCLIPTokenizer()
    vae = MockVAE()
    subject_img = torch.rand((1, 512, 512, 3))
    scene_img = torch.rand((1, 600, 400, 3))

    req = NodeExecutionRequest(
        node_name="CcC Krea2 - Subject + Scene",
        model=model,
        clip=clip,
        prompt="subject person in modern futuristic room",
        negative_prompt="low quality",
        vae=vae,
        subject_image=subject_img,
        scene_image=scene_img,
        subject_boost=2.5,
        scene_boost=1.0,
        role_order=[ReferenceRole.SCENE, ReferenceRole.SUBJECT]
    )

    patched_model, positive, negative, latent_dict = Krea2EditEngine.execute(req)

    assert patched_model is not None
    assert len(positive) == 1
    assert len(negative) == 1
    assert latent_dict["samples"].shape[0] == 1

    # Two grounding images for Scene + Subject order
    pos_call = clip.tokenize_calls[0]
    assert len(pos_call["images"]) == 2
    assert "reference_latents" not in positive[0][1]


def test_engine_execute_inpaint_subject_scene_workflow():
    model = MockModel()
    clip = MockCLIPTokenizer()
    vae = MockVAE()
    subject_img = torch.full((1, 512, 512, 3), 0.2)
    scene_img = torch.full((1, 600, 400, 3), 0.8)
    inpaint_mask = torch.ones((1, 512, 512))

    req = NodeExecutionRequest(
        node_name="CcC Krea2 - Inpaint Subject + Scene",
        model=model,
        clip=clip,
        prompt="inpaint subject person",
        negative_prompt="deformed",
        vae=vae,
        subject_image=subject_img,
        scene_image=scene_img,
        inpaint_mask=inpaint_mask,
        inpaint_base_role=ReferenceRole.SCENE,
        sampling_resize_mode="crop",
        role_order=[ReferenceRole.SCENE, ReferenceRole.SUBJECT]
    )

    patched_model, positive, negative, latent_dict = Krea2EditEngine.execute(req)

    assert patched_model is not None
    assert positive is not None
    assert negative is not None
    assert "samples" in latent_dict
    assert "noise_mask" in latent_dict

    # Assert VAE latent was generated from scene_image (mean 0.8) and NOT subject_image (mean 0.2)
    samples = latent_dict["samples"]
    assert abs(float(samples.mean().item()) - 0.8) < 1e-3


def test_wrapper_compatibility_fallback_nested_structure():
    """Verify that when ModelPatcher lacks add_wrapper_with_key, patch_krea2_model creates exact nested structure expected by ComfyUI."""
    class MockLegacyModel:
        def __init__(self):
            self.model_options = {}

        def clone(self):
            m = MockLegacyModel()
            m.model_options = self.model_options.copy()
            return m

    legacy_model = MockLegacyModel()
    patched = patch_krea2_model(legacy_model, prepared_refs=[])

    # Verify exact nested structure: model_options -> transformer_options -> wrappers -> diffusion_model -> ccc_krea2_edit
    assert "transformer_options" in patched.model_options
    t_opts = patched.model_options["transformer_options"]
    assert "wrappers" in t_opts
    wrappers = t_opts["wrappers"]
    assert "diffusion_model" in wrappers
    diff_wrappers = wrappers["diffusion_model"]
    assert "ccc_krea2_edit" in diff_wrappers
    assert callable(diff_wrappers["ccc_krea2_edit"])


def test_nodes_via_mappings():
    node_cls = NODE_CLASS_MAPPINGS["CcCKrea2Subject"]
    node_instance = node_cls()
    assert hasattr(node_instance, "process")
