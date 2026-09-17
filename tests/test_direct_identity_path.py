"""Regression tests for the direct visual-only Identity Edit path."""

import torch

from ccc_krea2.constants import VISION_PAD_TOKEN
from ccc_krea2.identity_conditioning import encode_visual_identity_direct
from ccc_krea2.modular_nodes.edit_reference_types import VisualReferenceEntry


class _DummyClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, text, images=None, llama_template=None):
        self.calls.append((text, list(images or []), llama_template))
        return {"qwen3vl_4b": [[(1, 1.0)]]}

    def encode_from_tokens_scheduled(self, tokens):
        del tokens
        return [[torch.zeros((1, 4, 8)), {}]]


class _DummyVAE:
    def encode(self, image):
        return torch.zeros((1, 16, 1, image.shape[1] // 8, image.shape[2] // 8))


def _extras(conditioning):
    return conditioning[0][1]


def test_direct_identity_keeps_appearance_out_of_conditioning():
    clip = _DummyClip()
    vae = _DummyVAE()
    target = {"samples": torch.zeros((1, 16, 1, 22, 22))}  # 176x176 target pixels

    scene = VisualReferenceEntry(
        image=torch.zeros((1, 180, 120, 3)),
        boost=1.0,
        reference_fit="resize",
        semantic=True,
        semantic_resize=True,
        semantic_grounding_px=128,
        semantic_resize_method="lanczos",
        prompt_annotation="It is the scene image.",
    )
    subject = VisualReferenceEntry(
        image=torch.zeros((1, 205, 97, 3)),
        boost=4.0,
        reference_fit="resize",
        semantic=True,
        semantic_resize=True,
        semantic_grounding_px=128,
        semantic_resize_method="lanczos",
        prompt_annotation="It is the subject image.",
    )

    result = encode_visual_identity_direct(
        clip=clip,
        vae=vae,
        visual_entries=[scene, subject],
        target_latent=target,
        positive_prompt="Replace the woman.",
        negative_prompt="",
    )

    expected_positive = (
        VISION_PAD_TOKEN * 2
        + "Image 1: It is the scene image.\n"
        + "Image 2: It is the subject image.\n"
        + "Replace the woman."
    )
    expected_negative = VISION_PAD_TOKEN * 2
    assert result.positive_text == expected_positive
    assert result.negative_text == expected_negative
    assert "scene image" in result.positive_text
    assert "subject image" in result.positive_text

    assert len(clip.calls) == 2
    assert clip.calls[0][0] == expected_positive
    assert clip.calls[1][0] == expected_negative
    assert len(clip.calls[0][1]) == 2
    assert len(clip.calls[1][1]) == 2

    # Appearance state is owned by the MODEL wrapper, not positive/negative conditioning.
    assert "reference_latents" not in _extras(result.positive)
    assert "reference_latents" not in _extras(result.negative)
    assert "reference_boosts" not in _extras(result.positive)
    assert "reference_boosts" not in _extras(result.negative)

    assert len(result.reference_latents) == 2
    assert result.reference_boosts == (1.0, 4.0)
    assert result.reference_rope_positions == (
        "inside:center:center",
        "inside:center:center",
    )

    # Source order is preserved and each ref uses the selected public resize mode.
    assert result.geometries[0].source_size == (120, 180)
    assert result.geometries[1].source_size == (97, 205)
    assert result.reference_latent_shapes[0][1] == 16
    assert result.reference_latent_shapes[1][1] == 16
    assert result.reference_latent_shapes[0][2] == 1
    assert result.reference_latent_shapes[1][2] == 1
