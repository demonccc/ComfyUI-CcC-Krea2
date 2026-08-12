"""Tests verifying negative conditioning behavior and negative Qwen image streams."""

import torch
from ccc_krea2.edit_engine import run_krea2_edit_orchestrator
from ccc_krea2.reference_specs import ReferenceChain, ReferenceSpec, StyleReferenceSpec
from ccc_krea2.vision_prep import prepare_vision_image


class DummyClip:
    def __init__(self):
        self.tokenize_calls = []

    def clone(self):
        return self

    def tokenize(self, prompt, images=None, **kwargs):
        self.tokenize_calls.append({"prompt": prompt, "images": images or []})
        tok_pairs = []
        if images:
            for img in images:
                tok_pairs.append([{"type": "image", "data": img}, None])
        else:
            tok_pairs.append([100, None])
        return {"qwen3vl": [tok_pairs]}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.randn(1, 2000, 1536), {}]]


class DummyVAE:
    def encode(self, image):
        return {"samples": torch.zeros((1, 16, 64, 64))}


class DummyModel:
    def clone(self):
        return self


def test_negative_conditioning_image_streams():
    """Verify that negative Qwen stream correctly isolates appearance refs (no style, no semantic, no target)."""
    model = DummyModel()
    clip = DummyClip()
    vae = DummyVAE()
    target_latent = {"samples": torch.zeros((1, 16, 1, 32, 32))}

    img = torch.rand(1, 256, 256, 3)
    prep = prepare_vision_image(image=img, clip=clip, mode="native")

    # 1. Appearance Ref
    app_spec = ReferenceSpec(
        reference_path="edit",
        prepared_image=prep,
        alias="subject",
        vision_instruction="Subject identity",
        appearance_reference=True,
    )

    # 2. Semantic-only Ref
    sem_spec = ReferenceSpec(
        reference_path="edit",
        prepared_image=prep,
        alias="scene",
        vision_instruction="Scene composition",
        appearance_reference=False,
    )

    # 3. Style Ref
    style_spec = StyleReferenceSpec(
        reference_path="style",
        prepared_image=prep,
        alias="style",
        style_fidelity=1.0,
        style_processing="2x2",
    )

    chain = ReferenceChain((app_spec, sem_spec, style_spec))

    # Run Krea2 Edit
    run_krea2_edit_orchestrator(
        model=model,
        clip=clip,
        vae=vae,
        references=chain,
        target_latent=target_latent,
        positive_prompt="test",
        negative_prompt="blurry",
        reference_method="krea2_edit",
    )

    # tokenize_calls[0] is positive, tokenize_calls[1] is negative
    assert len(clip.tokenize_calls) == 2

    pos_call = clip.tokenize_calls[0]
    neg_call = clip.tokenize_calls[1]

    # Positive images: 1 appearance + 1 semantic + 4 style crops = 6 images
    assert len(pos_call["images"]) == 6

    # Negative images: ONLY the 1 appearance ref
    assert len(neg_call["images"]) == 1


def test_ostris_negative_stream_has_zero_images():
    """Verify that Ostris negative Qwen stream has NO vision markers and NO images."""
    model = DummyModel()
    clip = DummyClip()
    vae = DummyVAE()
    target_latent = {"samples": torch.zeros((1, 16, 1, 32, 32))}

    img = torch.rand(1, 256, 256, 3)
    prep = prepare_vision_image(image=img, clip=clip, mode="native")

    app_spec = ReferenceSpec(
        reference_path="edit",
        prepared_image=prep,
        alias="subject",
        appearance_reference=True,
    )

    chain = ReferenceChain((app_spec,))

    run_krea2_edit_orchestrator(
        model=model,
        clip=clip,
        vae=vae,
        references=chain,
        target_latent=target_latent,
        positive_prompt="test",
        negative_prompt="blurry",
        reference_method="ostris_edit",
    )

    assert len(clip.tokenize_calls) == 2
    pos_call = clip.tokenize_calls[0]
    neg_call = clip.tokenize_calls[1]

    assert len(pos_call["images"]) == 1

    # Negative images must be zero for ostris edit
    assert len(neg_call["images"]) == 0
    assert "<|vision_start|>" not in neg_call["prompt"]
