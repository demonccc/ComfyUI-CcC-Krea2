"""CPU-safe unit tests for Qwen3-VL grounding image preprocessor."""

import torch
from ccc_krea2.grounding import resize_grounding_image, resolve_grounding_px


def test_resolve_grounding_px():
    assert resolve_grounding_px("balanced", 512) == 768
    assert resolve_grounding_px("max_identity", 512) == 1024
    assert resolve_grounding_px("custom", 512) == 512


def test_grounding_resize_none():
    img = torch.rand((1, 500, 800, 3))
    out = resize_grounding_image(img, resize_mode="none")
    assert torch.equal(img, out)


def test_grounding_resize_normalize_preset_balanced():
    # Longest edge 1600 -> should normalize to 768
    img = torch.rand((1, 1600, 800, 3))
    out = resize_grounding_image(img, resize_mode="normalize", grounding_preset="balanced")
    assert out.shape == (1, 768, 384, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_grounding_resize_normalize_preset_max_identity():
    # Longest edge 1600 -> should normalize to 1024
    img = torch.rand((1, 1600, 800, 3))
    out = resize_grounding_image(img, resize_mode="normalize", grounding_preset="max_identity")
    assert out.shape == (1, 1024, 512, 3)


def test_grounding_resize_downscale_only():
    # Small image (400x200) -> downscale_only should NOT upscale to 768
    small_img = torch.rand((1, 400, 200, 3))
    out_small = resize_grounding_image(small_img, resize_mode="downscale_only", grounding_px=768)
    assert out_small.shape == (1, 400, 200, 3)

    # Large image (1500x1000) -> downscale_only should reduce to 768
    large_img = torch.rand((1, 1500, 1000, 3))
    out_large = resize_grounding_image(large_img, resize_mode="downscale_only", grounding_px=768)
    assert out_large.shape[1] == 768


def test_grounding_resize_clamp():
    # Tiny image (200x100) -> clamp to min 512
    tiny_img = torch.rand((1, 200, 100, 3))
    out_tiny = resize_grounding_image(tiny_img, resize_mode="clamp", grounding_min_px=512, grounding_max_px=1024)
    assert max(out_tiny.shape[1], out_tiny.shape[2]) == 512

    # Huge image (2000x1000) -> clamp to max 1024
    huge_img = torch.rand((1, 2000, 1000, 3))
    out_huge = resize_grounding_image(huge_img, resize_mode="clamp", grounding_min_px=512, grounding_max_px=1024)
    assert max(out_huge.shape[1], out_huge.shape[2]) == 1024
