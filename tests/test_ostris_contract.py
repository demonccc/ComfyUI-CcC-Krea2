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
