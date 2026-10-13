"""Runtime behavior tests for CcCKrea2TextToImage node."""

import pytest
import torch

from ccc_krea2.t2i import (
    CcCKrea2TextToImage,
    _encode_t2i_vision,
    _generate_t2i_vision_guidance,
)
from ccc_krea2.krea_geometry import DEFAULT_KREA_PRESET_SIZE, KREA_PRESET_SIZES
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

    def fake_encode(clip, prompt, vision_references):
        captured["clip"] = clip
        captured["prompt"] = prompt
        captured["vision_references"] = vision_references
        return (
            ["vision_positive"],
            [
                {
                    "index": 1,
                    "mode": "semantic",
                    "runtime_mode": "semantic_only",
                    "processing": "full",
                    "instruction": "Use the scene idea.",
                    "guidance": "The same target subject is standing in a city street.",
                    "analysis_images": 1,
                    "generation_strategy": "blueprint-user-prompt",
                }
            ],
            "A futuristic city\n\nSemantic guidance: The same target subject is standing in a city street.",
            1,
        )

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
    assert "Mode: semantic" in vision_info
    assert "Consumer Processing: deferred to connected node" in vision_info
    assert "Vision References: 1" in info
    assert "Vision Modes: semantic" in info
    assert "Positive Conditioning: text-only with Qwen Vision-derived guidance" in info
    assert "T2I Vision Pipeline:" in info
    assert "Final Conditioning: text-only Krea2 encode" in info
    assert "Direct Vision Rows in Final Conditioning: 0" in info
    assert "The same target subject is standing in a city street." in info


def test_t2i_vision_guidance_is_encoded_as_text_only(monkeypatch):
    image = torch.zeros((1, 64, 64, 3))
    vision_chain, _ = CcCKrea2VisionReference().process(
        image=image,
        mode="visual",
        instruction="Use the visible character design.",
    )
    clip = FakeCLIP()

    def fake_generate_guidance(clip, entry, target_prompt):
        assert target_prompt == "Only one manga girl facing forward."
        assert entry.guidance_seed == 0
        return "Give the same target girl long black hair and a red jacket.", 1, "blueprint-user-prompt"

    monkeypatch.setattr(
        "ccc_krea2.t2i._generate_t2i_vision_guidance",
        fake_generate_guidance,
    )

    conditioning, guidance_items, conditioning_prompt, analysis_images = _encode_t2i_vision(
        clip=clip,
        prompt="Only one manga girl facing forward.",
        vision_references=vision_chain,
    )

    assert analysis_images == 1
    assert len(guidance_items) == 1
    assert guidance_items[0]["mode"] == "visual"
    assert "long black hair and a red jacket" in conditioning_prompt
    assert clip.tokenize_calls == [(conditioning_prompt, {})]
    assert conditioning == [f"conditioning_for_tokens_for_{conditioning_prompt}"]


class FakeGenerativeCLIP(FakeCLIP):
    def __init__(self, decoded_text):
        super().__init__()
        self.decoded_text = decoded_text
        self.generate_calls = []

    def generate(self, tokens, **kwargs):
        self.generate_calls.append((tokens, kwargs))
        return "generated_ids"

    def decode(self, generated_ids):
        assert generated_ids == "generated_ids"
        return self.decoded_text


class FakeSequentialGenerativeCLIP(FakeCLIP):
    def __init__(self, decoded_texts):
        super().__init__()
        self.decoded_texts = list(decoded_texts)
        self.generate_calls = []
        self.decode_calls = 0

    def generate(self, tokens, **kwargs):
        self.generate_calls.append((tokens, kwargs))
        return f"generated_ids_{len(self.generate_calls)}"

    def decode(self, generated_ids):
        text = self.decoded_texts[self.decode_calls]
        self.decode_calls += 1
        return text


def test_t2i_empty_generation_fails_without_retry():
    image = torch.zeros((1, 64, 64, 3))
    vision_chain, _ = CcCKrea2VisionReference().process(
        image=image,
        mode="semantic",
        instruction="",
    )
    clip = FakeSequentialGenerativeCLIP([""])

    with pytest.raises(RuntimeError, match="Qwen returned empty Vision guidance"):
        _generate_t2i_vision_guidance(
            clip=clip,
            entry=vision_chain.entries[0],
            target_prompt="A manga girl",
        )

    assert len(clip.generate_calls) == 1


def test_t2i_style_analysis_uses_mode_scoped_qwen_generation():
    image = torch.zeros((1, 64, 64, 3))
    vision_chain, _ = CcCKrea2VisionReference().process(
        image=image,
        mode="style",
        instruction="Use manga style and girl identity.",
        style_processing="full",
    )
    entry = vision_chain.entries[0]
    clip = FakeGenerativeCLIP(
        "Bold manga line work, flat cel shading, high-contrast black and white rendering."
    )

    guidance, image_count, strategy = _generate_t2i_vision_guidance(
        clip=clip,
        entry=entry,
        target_prompt="Only one manga girl facing forward.",
    )

    assert image_count == 1
    assert strategy == "blueprint-user-prompt-single-pass"
    assert guidance.startswith("Bold manga line work")
    analysis_text, kwargs = clip.tokenize_calls[0]
    assert "Analyze the supplied reference image or tiles only for visual STYLE." in analysis_text
    assert "TARGET TEXT-TO-IMAGE PROMPT" in analysis_text
    assert "VISION MODE: style" in analysis_text
    assert "Use manga style and girl identity." in analysis_text
    assert torch.equal(kwargs["image"], image)
    assert kwargs["skip_template"] is False
    assert kwargs["system_prompt"] == ""
    assert "thinking" not in kwargs
    assert clip.generate_calls[0][1]["seed"] == 0
    assert clip.generate_calls[0][1]["max_length"] == 512
    assert clip.generate_calls[0][1]["temperature"] == 0.25
    assert clip.generate_calls[0][1]["top_p"] == 0.90
    assert clip.generate_calls[0][1]["mtp"] is True


def test_guidance_controls_live_on_vision_reference_not_t2i(monkeypatch):
    t2i_required = CcCKrea2TextToImage.INPUT_TYPES()["required"]
    for name in (
        "guidance_seed_mode",
        "guidance_seed",
        "guidance_detail",
        "vision_guidance",
    ):
        assert name not in t2i_required

    ref_required = CcCKrea2VisionReference.INPUT_TYPES()["required"]
    assert ref_required["guidance_seed_mode"][0] == ("fixed", "incremental", "random")
    assert ref_required["guidance_seed_mode"][1]["default"] == "fixed"
    assert ref_required["guidance_seed"][1]["default"] == 0
    assert ref_required["guidance_detail"][0] == ("compact", "balanced", "detailed")
    assert ref_required["guidance_detail"][1]["default"] == "balanced"
    assert ref_required["vision_guidance"][0] == ("conservative", "balanced", "flexible")
    assert ref_required["vision_guidance"][1]["default"] == "balanced"

    node = CcCKrea2VisionReference()
    assert node._resolve_guidance_seed("fixed", 7) == 7
    assert node._resolve_guidance_seed("incremental", 10) == 10
    assert node._resolve_guidance_seed("incremental", 10) == 11
    assert node._resolve_guidance_seed("incremental", 10) == 12

    values = iter([1234, 5678])
    monkeypatch.setattr(
        "ccc_krea2.modular_nodes.vision_reference_node.random.SystemRandom.randint",
        lambda self, a, b: next(values),
    )
    assert node._resolve_guidance_seed("random", 0) == 1234
    assert node._resolve_guidance_seed("random", 0) == 5678


def test_vision_reference_guidance_seed_is_passed_to_single_qwen_attempt():
    image = torch.zeros((1, 64, 64, 3))
    vision_chain, _ = CcCKrea2VisionReference().process(
        image=image,
        mode="semantic",
        instruction="",
        guidance_seed=42,
    )
    clip = FakeSequentialGenerativeCLIP(
        ["A woman standing outdoors beside a stone wall."]
    )

    guidance, image_count, strategy = _generate_t2i_vision_guidance(
        clip=clip,
        entry=vision_chain.entries[0],
        target_prompt="A manga girl",
    )

    assert guidance
    assert image_count == 1
    assert strategy == "blueprint-user-prompt-single-pass"
    assert len(clip.generate_calls) == 1
    assert clip.generate_calls[0][1]["seed"] == 42


@pytest.mark.parametrize(
    ("guidance_detail", "expected_max_length"),
    [("compact", 256), ("balanced", 512), ("detailed", 1024)],
)
def test_vision_reference_guidance_detail_presets_only_change_token_limit(
    guidance_detail,
    expected_max_length,
):
    image = torch.zeros((1, 64, 64, 3))
    vision_chain, _ = CcCKrea2VisionReference().process(
        image=image,
        mode="semantic",
        guidance_detail=guidance_detail,
        vision_guidance="balanced",
    )
    clip = FakeGenerativeCLIP("A useful semantic description.")

    _generate_t2i_vision_guidance(
        clip=clip,
        entry=vision_chain.entries[0],
        target_prompt="A manga girl",
    )

    kwargs = clip.generate_calls[0][1]
    assert kwargs["max_length"] == expected_max_length
    assert kwargs["temperature"] == 0.25
    assert kwargs["top_p"] == 0.90
    assert kwargs["top_k"] == 64
    assert kwargs["min_p"] == 0.05
    assert kwargs["repetition_penalty"] == 1.05


@pytest.mark.parametrize(
    ("vision_guidance", "temperature", "top_p"),
    [
        ("conservative", 0.15, 0.85),
        ("balanced", 0.25, 0.90),
        ("flexible", 0.45, 0.95),
    ],
)
def test_vision_reference_sampling_presets(
    vision_guidance,
    temperature,
    top_p,
):
    image = torch.zeros((1, 64, 64, 3))
    vision_chain, _ = CcCKrea2VisionReference().process(
        image=image,
        mode="semantic",
        guidance_detail="balanced",
        vision_guidance=vision_guidance,
    )
    clip = FakeGenerativeCLIP("A useful semantic description.")

    _generate_t2i_vision_guidance(
        clip=clip,
        entry=vision_chain.entries[0],
        target_prompt="A manga girl",
    )

    kwargs = clip.generate_calls[0][1]
    assert kwargs["max_length"] == 512
    assert kwargs["temperature"] == temperature
    assert kwargs["top_p"] == top_p
    assert kwargs["top_k"] == 64
    assert kwargs["min_p"] == 0.05
    assert kwargs["repetition_penalty"] == 1.05


def test_t2i_uses_shared_krea_geometry_controls():
    required = CcCKrea2TextToImage.INPUT_TYPES()["required"]
    assert required["dimensions"][0] == ["preset", "fixed"]
    assert required["dimensions"][1]["default"] == "preset"
    assert required["preset_size"][0] == KREA_PRESET_SIZES
    assert required["preset_size"][1]["default"] == DEFAULT_KREA_PRESET_SIZE
    assert "aspect_ratio" not in required
    assert "megapixels" not in required
    assert "custom_aspect_width" not in required
    assert "custom_aspect_height" not in required


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
        dimensions="preset",
        preset_size="2048 x 1152 | 16:9 | ~2.36 MP",
        batch_size=batch_size,
    )

    assert isinstance(latent, dict)
    assert "samples" in latent
    samples = latent["samples"]

    # 16 channels, downscale by 8 from the shared 2048x1152 Krea preset.
    assert samples.shape[0] == batch_size
    assert samples.shape[1] == 16
    assert samples.shape[2] == 1152 // 8
    assert samples.shape[3] == 2048 // 8
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

    # T2I no longer owns Qwen generation state; Vision Reference does.
    assert not hasattr(CcCKrea2TextToImage, "IS_CHANGED")
    assert not hasattr(node, "fingerprint_inputs")
    assert not hasattr(node, "not_idempotent")

    model = FakeModel()
    clip = FakeCLIP()

    _, pos1, neg1, lat1, info1 = node.process(model=model, clip=clip, prompt="det test")
    _, pos2, neg2, lat2, info2 = node.process(model=model, clip=clip, prompt="det test")

    assert pos1 == pos2
    assert neg1 == neg2
    assert torch.equal(lat1["samples"], lat2["samples"])
    assert info1 == info2



def test_t2i_rejects_chat_role_only_guidance():
    clip = FakeGenerativeCLIP("user")
    image = torch.zeros((1, 64, 64, 3))
    vision_chain, _ = CcCKrea2VisionReference().process(image=image, mode="semantic")

    with pytest.raises(RuntimeError, match="chat-role token"):
        _generate_t2i_vision_guidance(
            clip=clip,
            entry=vision_chain.entries[0],
            target_prompt="A manga girl",
        )
