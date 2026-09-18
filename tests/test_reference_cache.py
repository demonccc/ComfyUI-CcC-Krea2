"""Reference-cache contracts for Krea2 CcC Edit."""

from types import SimpleNamespace

import torch

from ccc_krea2.modular_nodes.cache_nodes import CcCKrea2CachedVisualReference
from ccc_krea2.reference_cache import (
    CachedQwenImage,
    build_qwen_visual_cache,
    create_reference_cache,
    load_reference_cache,
    save_reference_cache,
    use_cached_qwen_images,
)


class _FakeTransformer:
    model_type = "qwen3vl_4b"

    def preprocess_embed(self, embed, device):
        image = embed["data"]
        tokens = max(1, int(image.shape[1] * image.shape[2] // (32 * 32)))
        merged = torch.arange(tokens * 8, dtype=torch.float32).reshape(tokens, 8)
        grid = torch.tensor([[1, 2, tokens * 2]], dtype=torch.int64)
        deepstack = [merged + 1, merged + 2, merged + 3]
        return merged.to(device), {"grid": grid, "deepstack": [t.to(device) for t in deepstack]}


class _FakeClip:
    def __init__(self):
        transformer = _FakeTransformer()
        self.cond_stage_model = SimpleNamespace(
            qwen3vl_4b=SimpleNamespace(transformer=transformer)
        )
        self.patcher = SimpleNamespace(load_device=torch.device("cpu"))

    def load_model(self):
        return None


class _FakeVAE:
    def encode(self, image):
        batch, height, width, _ = image.shape
        return torch.zeros((batch, 16, height // 8, width // 8), dtype=torch.float32)


def test_qwen_visual_cache_captures_prompt_independent_payload():
    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    cache = build_qwen_visual_cache(
        clip=_FakeClip(),
        image=image,
        semantic_resize=False,
        grounding_px=768,
    )

    assert cache.model_type == "qwen3vl_4b"
    assert cache.input_size == (64, 64)
    assert cache.merged.ndim == 2
    assert cache.grid.shape == (1, 3)
    assert len(cache.deepstack) == 3


def test_cached_qwen_preprocess_returns_cache_without_visual_execution():
    clip = _FakeClip()
    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    cache = build_qwen_visual_cache(clip, image, semantic_resize=False)
    payload = CachedQwenImage(cache)

    transformer = clip.cond_stage_model.qwen3vl_4b.transformer
    with use_cached_qwen_images(clip, [payload]):
        merged, extra = transformer.preprocess_embed(
            {"type": "image", "data": payload},
            device=torch.device("cpu"),
        )

    assert torch.equal(merged, cache.merged)
    assert torch.equal(extra["grid"], cache.grid)
    assert len(extra["deepstack"]) == 3


def test_reference_cache_roundtrip_is_lossless(tmp_path):
    image = torch.zeros((1, 128, 128, 3), dtype=torch.float32)
    target = {"samples": torch.zeros((1, 16, 16, 16), dtype=torch.float32)}
    cache = create_reference_cache(
        image=image,
        clip=_FakeClip(),
        vae=_FakeVAE(),
        target_latent=target,
        reference_fit="native",
        semantic_resize=False,
    )

    path = tmp_path / "reference.safetensors"
    save_reference_cache(cache, path)
    loaded = load_reference_cache(path)

    assert torch.equal(loaded.appearance_latent, cache.appearance_latent)
    assert torch.equal(loaded.qwen_visual.merged, cache.qwen_visual.merged)
    assert torch.equal(loaded.qwen_visual.grid, cache.qwen_visual.grid)
    assert all(
        torch.equal(left, right)
        for left, right in zip(loaded.qwen_visual.deepstack, cache.qwen_visual.deepstack)
    )
    assert loaded.geometry == cache.geometry
    assert dict(loaded.metadata) == dict(cache.metadata)


def test_cached_visual_reference_keeps_runtime_controls():
    image = torch.zeros((1, 128, 128, 3), dtype=torch.float32)
    target = {"samples": torch.zeros((1, 16, 16, 16), dtype=torch.float32)}
    cache = create_reference_cache(
        image=image,
        clip=_FakeClip(),
        vae=_FakeVAE(),
        target_latent=target,
        reference_fit="native",
        semantic_resize=False,
    )

    (chain,) = CcCKrea2CachedVisualReference().process(
        cache=cache,
        boost=3.5,
        placement_grid="inside",
        grid_horizontal_position="right",
        grid_vertical_position="down",
        semantic=True,
        prompt_annotation="cached subject",
    )

    entry = chain.entries[0]
    assert entry.cache is cache
    assert entry.image is None
    assert entry.boost == 3.5
    assert entry.rope_position == "inside:right:down"
    assert entry.prompt_annotation == "cached subject"
