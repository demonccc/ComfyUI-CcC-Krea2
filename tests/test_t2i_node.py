"""Runtime behavior tests for CcCKrea2TextToImage node."""

from types import SimpleNamespace

import pytest
import torch

from ccc_krea2.constants import VISION_PAD_TOKEN
from ccc_krea2.t2i import CcCKrea2TextToImage, _encode_t2i_vision, calculate_t2i_resolution
from ccc_krea2.modular_nodes.vision_reference_node import CcCKrea2VisionReference
from ccc_krea2.prompt_augmentation import PromptAugmentation


class FakeCLIP:
    """Fake CLIP text encoder tracking tokenize and encode calls."""

    def __init__(self):
        self.tokenize_calls = []
        self.encode_calls = []

    def tokenize(self, text, **kwargs):
        self.tokenize_calls.append((text, kwargs))
        return f"tokens_for_{text}"

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls.append(tokens)
        return [f"conditioning_for_{tokens}"]


class FakeModel:
    """Fake Model object for identity check."""

    pass


def test_t2i_returns_exact_input_model_and_no_patching():
    """1-3. Returns exact input MODEL object, without calling patch function or Krea2EditEngine."""
    node = CcCKrea2TextToImage()
    model = FakeModel()
    clip = FakeCLIP()

    out_model, pos, neg, latent, info = node.process(
        model=model,
        clip=clip,
        prompt="A serene beach at sunset",
    )

    assert out_model is model
    assert not hasattr(out_model, "patch_called")
    assert "=== Krea2 CcC Text to Image ===" in info
    assert "Vision References: 0" in info
    assert "Positive Conditioning: text-only" in info


def test_t2i_native_clip_tokenize_and_encode():
    """4-6. Positive and negative prompts use native tokenize and encode_from_tokens_scheduled independently."""
    node = CcCKrea2TextToImage()
    model = FakeModel()
    clip = FakeCLIP()

    out_model, pos, neg, latent, info = node.process(
        model=model,
        clip=clip,
        prompt="positive text",
        negative_prompt="negative text",
    )

    assert len(clip.tokenize_calls) == 2
    assert clip.tokenize_calls[0][0] == "positive text"
    assert clip.tokenize_calls[1][0] == "negative text"

    assert len(clip.encode_calls) == 2
    assert clip.encode_calls[0] == "tokens_for_positive text"
    assert clip.encode_calls[1] == "tokens_for_negative text"

    assert pos == ["conditioning_for_tokens_for_positive text"]
    assert neg == ["conditioning_for_tokens_for_negative text"]


def test_t2i_prompt_augmentation_preserves_and_orders():
    """7-10. Prompt augmentation ordering, empty augmentation handling, and non-mutation."""
    node = CcCKrea2TextToImage()
    model = FakeModel()
    clip = FakeCLIP()

    prompt_str = "original positive"
    neg_str = "original negative"
    aug = PromptAugmentation(
        positive_prepend=("pre_pos",),
        positive_append=("app_pos",),
        negative_prepend=("pre_neg",),
        negative_append=("app_neg",),
    )

    out_model, pos, neg, latent, info = node.process(
        model=model,
        clip=clip,
        prompt=prompt_str,
        negative_prompt=neg_str,
        prompt_augmentation=aug,
    )

    expected_pos = "pre_pos\n\noriginal positive\n\napp_pos"
    expected_neg = "pre_neg\n\noriginal negative\n\napp_neg"

    assert clip.tokenize_calls[0][0] == expected_pos
    assert clip.tokenize_calls[1][0] == expected_neg

    # Assert non-mutation of input strings and PromptAugmentation object
    assert prompt_str == "original positive"
    assert neg_str == "original negative"
    assert aug.positive_prepend == ("pre_pos",)
    assert aug.positive_append == ("app_pos",)


def test_t2i_empty_prompt_augmentation():
    """Empty prompt augmentation preserves original prompts."""
    node = CcCKrea2TextToImage()
    model = FakeModel()
    clip = FakeCLIP()

    node.process(
        model=model,
        clip=clip,
        prompt="original positive",
        negative_prompt="original negative",
        prompt_augmentation=None,
    )

    assert clip.tokenize_calls[0][0] == "original positive"
    assert clip.tokenize_calls[1][0] == "original negative"




def test_t2i_accepts_optional_vision_references():
    optional = CcCKrea2TextToImage.INPUT_TYPES()["optional"]
    assert optional["vision_references"][0] == "KREA2_VISION_REFERENCE_CHAIN"


def test_t2i_vision_reference_enriches_positive_conditioning(monkeypatch):
    node = CcCKrea2TextToImage()
    model = FakeModel()
    clip = FakeCLIP()
    image = torch.zeros((1, 64, 64, 3))
    vision_chain, vision_info = CcCKrea2VisionReference().process(
        image=image,
        mode="semantic",
        instruction="Use the scene idea.",
    )

    captured = {}

    fake_spec = SimpleNamespace(
        vision_reference_mode="semantic",
        semantic_extract="subject",
        semantic_strength=0.25,
        style_fidelity=1.0,
        style_processing="full",
    )
    fake_context = SimpleNamespace(
        pos_rows_before=100,
        pos_rows_after=100,
        physical_images=[image],
        physical_image_map=[
            {
                "logical_vision_slot": 1,
                "spec": fake_spec,
            }
        ],
        vision_row_spans=[(10, 50)],
        removed_row_indices=[],
        token_stream_key="qwen3vl_4b",
    )

    def fake_encode(clip, prompt, vision_references):
        captured["clip"] = clip
        captured["prompt"] = prompt
        captured["vision_references"] = vision_references
        return ["vision_positive"], fake_context, [fake_spec], "vision prompt"

    monkeypatch.setattr("ccc_krea2.t2i._encode_t2i_vision", fake_encode)

    out_model, positive, negative, latent, info = node.process(
        model=model,
        clip=clip,
        prompt="A futuristic city",
        negative_prompt="blurry",
        vision_references=vision_chain,
    )

    assert out_model is model
    assert positive == ["vision_positive"]
    assert captured["prompt"] == "A futuristic city"
    assert captured["vision_references"] is vision_chain
    assert clip.tokenize_calls == [("blurry", {})]
    assert negative == ["conditioning_for_tokens_for_blurry"]
    assert "samples" in latent
    assert "=== Krea2 CcC Vision Reference ===" in vision_info
    assert "Public Mode: semantic" in vision_info
    assert "Vision References: 1" in info
    assert "Vision Modes: semantic" in info
    assert "Positive Conditioning: multimodal Qwen/Vision" in info
    assert "T2I Vision Pipeline:" in info
    assert "Layout: text prompt first, Vision directive/image blocks after prompt" in info


def test_t2i_vision_layout_keeps_text_prompt_before_direct_reference(monkeypatch):
    image = torch.zeros((1, 64, 64, 3))
    vision_chain, _ = CcCKrea2VisionReference().process(
        image=image,
        mode="visual",
        instruction="Use the visible character design.",
    )
    captured = {}

    fake_context = SimpleNamespace(
        conditioning=["encoded"],
        pos_rows_before=100,
        pos_rows_after=100,
        physical_images=[image],
        physical_image_map=[],
        vision_row_spans=[],
        removed_row_indices=[],
        token_stream_key="qwen3vl_4b",
    )

    def fake_context_encoder(**kwargs):
        captured.update(kwargs)
        return fake_context

    monkeypatch.setattr("ccc_krea2.t2i.encode_krea2_qwen_context", fake_context_encoder)

    conditioning, _, specs, vision_prompt = _encode_t2i_vision(
        clip=FakeCLIP(),
        prompt="Only one manga girl facing forward.",
        vision_references=vision_chain,
    )

    assert conditioning == ["encoded"]
    assert len(specs) == 1
    assert vision_prompt.startswith("Only one manga girl facing forward.")
    assert VISION_PAD_TOKEN in vision_prompt
    assert vision_prompt.index("Only one manga girl facing forward.") < vision_prompt.index(VISION_PAD_TOKEN)
    assert captured["prompt"] == vision_prompt


def test_t2i_predefined_aspect_ratios_and_orientation():
    """11, 13, 14. Predefined aspect ratios produce correctly oriented aligned dimensions close to megapixels."""
    ratios = {
        "1:1": (992, 992),  # ~0.984 MP
        "4:3": (1152, 864),  # ~0.995 MP (W > H)
        "3:4": (864, 1152),  # ~0.995 MP (H > W)
        "16:9": (1328, 752),  # ~0.998 MP (W > H)
        "9:16": (752, 1328),  # ~0.998 MP (H > W)
        "3:2": (1232, 816),  # ~1.005 MP (W > H)
        "2:3": (816, 1232),  # ~1.005 MP (H > W)
    }

    for ratio_str, expected_wh in ratios.items():
        w, h = calculate_t2i_resolution(aspect_ratio=ratio_str, megapixels=1.0)
        assert (w, h) == expected_wh, f"Ratio {ratio_str} mismatch: got ({w}, {h}), expected {expected_wh}"
        assert w % 16 == 0 and h % 16 == 0
        assert w >= 128 and h >= 128
        area = w * h
        assert abs(area - 1_000_000) < 50_000


def test_t2i_custom_aspect_ratio():
    """12. Custom aspect ratio calculation."""
    w, h = calculate_t2i_resolution(
        aspect_ratio="custom",
        megapixels=1.0,
        custom_aspect_width=21,
        custom_aspect_height=9,
    )
    assert w > h
    assert w % 16 == 0 and h % 16 == 0
    assert w >= 128 and h >= 128
    assert abs(w * h - 1_000_000) < 50_000


def test_t2i_latent_contract_and_batch_size():
    """15-16. Generated latent follows EmptySD3LatentImage contract (16 channels, /8 downscale, batch_size)."""
    node = CcCKrea2TextToImage()
    model = FakeModel()
    clip = FakeCLIP()

    batch_size = 4
    _, _, _, latent, info = node.process(
        model=model,
        clip=clip,
        prompt="Test prompt",
        aspect_ratio="16:9",
        megapixels=1.0,
        batch_size=batch_size,
    )

    assert isinstance(latent, dict)
    assert "samples" in latent
    samples = latent["samples"]

    # 16 channels, downscale by 8
    # 16:9 at 1.0 MP gives w=1328, h=752 -> samples [4, 16, 94, 166]
    assert samples.shape[0] == batch_size
    assert samples.shape[1] == 16
    assert samples.shape[2] == 752 // 8
    assert samples.shape[3] == 1328 // 8
    assert samples.dtype == torch.float32


def test_t2i_invalid_clip_raises_value_error():
    """17. Invalid/None clip raises clear ValueError."""
    node = CcCKrea2TextToImage()
    model = FakeModel()

    with pytest.raises(ValueError, match="CLIP text encoder input cannot be None"):
        node.process(model=model, clip=None, prompt="Test prompt")


def test_t2i_determinism_and_no_cache_hooks():
    """18-19. Repeated calls are deterministic and class defines no cache-invalidating hooks."""
    node = CcCKrea2TextToImage()

    # Verify no IS_CHANGED or cache-invalidating attributes
    assert not hasattr(CcCKrea2TextToImage, "IS_CHANGED")
    assert not hasattr(node, "IS_CHANGED")
    assert not hasattr(node, "fingerprint_inputs")
    assert not hasattr(node, "not_idempotent")

    model = FakeModel()
    clip = FakeCLIP()

    _, pos1, neg1, lat1, info1 = node.process(model=model, clip=clip, prompt="det test", aspect_ratio="1:1", megapixels=1.0)
    _, pos2, neg2, lat2, info2 = node.process(model=model, clip=clip, prompt="det test", aspect_ratio="1:1", megapixels=1.0)

    assert pos1 == pos2
    assert neg1 == neg2
    assert torch.equal(lat1["samples"], lat2["samples"])
    assert info1 == info2
