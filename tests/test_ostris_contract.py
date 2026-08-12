"""Test suite for Ostris Edit backend contracts and preprocessors."""

import torch
import pytest
from ccc_krea2.ostris_backend import (
    build_ostris_qwen_prompt,
    preprocess_ostris_vision_image,
    preprocess_ostris_ref_pixel_image,
    patch_ostris_model,
)


def test_build_ostris_qwen_prompt_format():
    resolved_refs = [
        {"spec": type("Spec", (), {"alias": "subject", "vision_instruction": "blue eyes"})()},
        {"spec": type("Spec", (), {"alias": "outfit", "vision_instruction": ""})()},
    ]

    prompt = build_ostris_qwen_prompt(resolved_refs, user_prompt="a cat wearing sunglasses")

    assert "Picture 1: <|vision_start|><|image_pad|><|vision_end|> (subject): blue eyes" in prompt
    assert "Picture 2: <|vision_start|><|image_pad|><|vision_end|> (outfit)" in prompt
    assert prompt.endswith("a cat wearing sunglasses")


def test_preprocess_ostris_vision_image_no_upscale():
    small_img = torch.zeros((1, 100, 100, 3))
    out_small = preprocess_ostris_vision_image(small_img)

    assert out_small.shape == (1, 100, 100, 3)


def test_preprocess_ostris_vision_image_area_downscale_preserves_exact_ratio():
    large_img = torch.zeros((1, 1000, 1000, 3))
    out_large = preprocess_ostris_vision_image(large_img)

    _, h, w, _ = out_large.shape
    assert h * w <= 384 * 384 + 100
    assert h == w  # Exact 1:1 aspect ratio preserved without /16 snapping


def test_preprocess_ostris_ref_pixel_image_snaps_16():
    large_pixel_img = torch.zeros((1, 2000, 2000, 3))
    out_ref = preprocess_ostris_ref_pixel_image(large_pixel_img)

    _, h, w, _ = out_ref.shape
    assert h * w <= 1024 * 1024 + 100
    assert h % 16 == 0
    assert w % 16 == 0


class DummyModel:
    def clone(self):
        return DummyModel()


class DummyClip:
    def __init__(self):
        self.tokenize_calls = []

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
        return [[torch.randn(1, 257, 1536), {}]]


class DummyVAE:
    def encode(self, image):
        return {"samples": torch.zeros((1, 16, 1, 32, 32))}


def test_ostris_backend_execution_path():
    from ccc_krea2.edit_engine import run_krea2_edit_orchestrator
    from ccc_krea2.reference_specs import ReferenceChain, ReferenceSpec
    from ccc_krea2.vision_prep import prepare_image_for_qwen

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
    target_latent = {"samples": torch.zeros((1, 16, 1, 32, 32))}

    # Case 1: apply_model_patch=True
    out_model, pos, neg, lat, info = run_krea2_edit_orchestrator(
        model=model,
        clip=clip,
        vae=vae,
        references=chain,
        target_latent=target_latent,
        positive_prompt="test prompt",
        negative_prompt="neg prompt",
        reference_method="ostris_edit",
        apply_model_patch=True,
    )

    assert out_model is model
    assert "Negative Physical Qwen Image Count: 0" in info
    assert "Reference Latents Method: index_timestep_zero" in info
    assert "Ostris Reference Method: explicitly applied via conditioning" in info

    # Case 2: apply_model_patch=False
    _, _, _, _, info_unpatched = run_krea2_edit_orchestrator(
        model=model,
        clip=clip,
        vae=vae,
        references=chain,
        target_latent=target_latent,
        positive_prompt="test prompt",
        negative_prompt="neg prompt",
        reference_method="ostris_edit",
        apply_model_patch=False,
    )
    assert "Ostris reference method was not explicitly applied" in info_unpatched


def test_ostris_kv_cache_raises_not_implemented():
    from ccc_krea2.edit_engine import run_krea2_edit_orchestrator
    from ccc_krea2.reference_specs import ReferenceChain

    model = DummyModel()
    clip = DummyClip()
    vae = DummyVAE()
    target_latent = {"samples": torch.zeros((1, 16, 1, 32, 32))}

    with pytest.raises(NotImplementedError, match="ostris_kv_cache=True is currently unsupported"):
        run_krea2_edit_orchestrator(
            model=model,
            clip=clip,
            vae=vae,
            references=ReferenceChain(),
            target_latent=target_latent,
            positive_prompt="test prompt",
            negative_prompt="",
            reference_method="ostris_edit",
            ostris_kv_cache=True,
        )

    with pytest.raises(NotImplementedError, match="ostris_kv_cache"):
        patch_ostris_model(model, ostris_kv_cache=True)


def test_ostris_semantic_vlm_preprocessing():
    """Verify that Ostris backend applies VLM preprocessing (<= 384x384) to semantic-only references."""
    from ccc_krea2.edit_engine import run_krea2_edit_orchestrator
    from ccc_krea2.reference_specs import ReferenceChain, ReferenceSpec
    from ccc_krea2.vision_prep import prepare_vision_image
    
    model = DummyModel()
    clip = DummyClip()
    vae = DummyVAE()
    target_latent = {"samples": torch.zeros((1, 16, 1, 32, 32))}

    # Create a large image (1024x1024)
    large_img = torch.rand(1, 1024, 1024, 3)
    prep = prepare_vision_image(image=large_img, clip=clip, mode="native")
    
    # Semantic-only reference
    semantic_spec = ReferenceSpec(
        reference_path="edit",
        prepared_image=prep,
        alias="subject",
        vision_instruction="Subject identity",
        appearance_reference=False,
    )
    chain = ReferenceChain((semantic_spec,))
    
    run_krea2_edit_orchestrator(
        model=model,
        clip=clip,
        vae=vae,
        references=chain,
        target_latent=target_latent,
        positive_prompt="test prompt",
        negative_prompt="",
        reference_method="ostris_edit",
    )
    
    # Verify the image sent to CLIP tokenize is area-downscaled
    assert len(clip.tokenize_calls) == 2
    call = clip.tokenize_calls[0]
    images_sent = call["images"]
    assert len(images_sent) == 1
    
    vlm_img = images_sent[0]
    vlm_h, vlm_w = vlm_img.shape[1], vlm_img.shape[2]
    
    # Should be downscaled to ~384x384 (area <= 147456)
    assert vlm_h * vlm_w <= 384 * 384 + 1000
    assert vlm_h < 1024
    
    # Verify NO Picture N is in the prompt for semantic-only refs
    prompt = call["prompt"]
    assert "Picture 1:" not in prompt
