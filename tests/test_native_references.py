"""Test suite for native ComfyUI reference latents plumbing."""

import torch
from ccc_krea2.edit_engine import run_krea2_edit_orchestrator
from ccc_krea2.reference_specs import ReferenceChain, ReferenceSpec
from ccc_krea2.vision_prep import prepare_image_for_qwen
from ccc_krea2.patch import is_model_already_patched


class DummyModel:
    def clone(self):
        return DummyModel()


class DummyClip:
    def tokenize(self, prompt, images=None, **kwargs):
        tok_pairs = []
        if images:
            for img in images:
                tok_pairs.append([{"type": "image", "data": img}, None])
        else:
            tok_pairs.append([100, None])
        return {"qwen3vl": [tok_pairs]}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.randn(1, 257, 1536), {}]]


class DummyVAE:
    def encode(self, image):
        return {"samples": torch.zeros((1, 16, 1, 32, 32))}


def test_native_reference_method_attaches_reference_latents():
    model = DummyModel()
    clip = DummyClip()
    vae = DummyVAE()

    img = torch.zeros((1, 256, 256, 3))
    prep = prepare_image_for_qwen(img, clip)

    spec = ReferenceSpec(
        reference_path="edit",
        prepared_image=prep,
        alias="subject",
    )
    chain = ReferenceChain().append(spec)

    target_latent = {
        "samples": torch.zeros((1, 16, 1, 32, 32))
    }

    out_model, pos, neg, lat, info = run_krea2_edit_orchestrator(
        model=model,
        clip=clip,
        vae=vae,
        references=chain,
        target_latent=target_latent,
        positive_prompt="test prompt",
        negative_prompt="",
        reference_method="native",
    )

    assert out_model is model
    assert not is_model_already_patched(out_model, "ccc_krea2_edit")
    assert "Native Default Reference Method: none" in info
    assert "reference_latents were attached, but no native default reference method was detected" in info


def test_native_reference_method_with_model_default_method():
    model = DummyModel()
    model.default_ref_method = "index_timestep_zero"
    clip = DummyClip()
    vae = DummyVAE()

    img = torch.zeros((1, 256, 256, 3))
    prep = prepare_image_for_qwen(img, clip)

    spec = ReferenceSpec(
        reference_path="edit",
        prepared_image=prep,
        alias="subject",
    )
    chain = ReferenceChain().append(spec)
    target_latent = {"samples": torch.zeros((1, 16, 1, 32, 32))}

    out_model, pos, neg, lat, info = run_krea2_edit_orchestrator(
        model=model,
        clip=clip,
        vae=vae,
        references=chain,
        target_latent=target_latent,
        positive_prompt="test prompt",
        negative_prompt="",
        reference_method="native",
    )

    assert out_model is model
    assert "Native Default Reference Method: index_timestep_zero" in info
    # Should not warn when default method is present
    assert "no native default reference method was detected" not in info

