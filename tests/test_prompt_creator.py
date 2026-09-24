"""Tests for Krea2 CcC Edit Prompt Creator."""

import pytest
import torch

from ccc_krea2.modular_nodes.edit_prompt_creator_node import (
    PROMPT_CREATOR_MODES,
    CcCKrea2EditPromptCreator,
)
from ccc_krea2.modular_nodes.edit_reference_types import VisualReferenceChain, VisualReferenceEntry


class _FakeClip:
    def __init__(self, output="Generated edit prompt."):
        self.output = output
        self.tokenize_prompt = None
        self.tokenize_kwargs = None
        self.generate_kwargs = None

    def tokenize(self, prompt, **kwargs):
        self.tokenize_prompt = prompt
        self.tokenize_kwargs = kwargs
        return {"fake": [[(1, 1.0)]]}

    def generate(self, tokens, **kwargs):
        self.generate_kwargs = kwargs
        return tokens

    def decode(self, generated_ids):
        return self.output


def _image(value):
    return torch.full((1, 32, 48, 3), float(value), dtype=torch.float32)


def _chain():
    return VisualReferenceChain(
        (
            VisualReferenceEntry(
                image=_image(0.2), semantic=True, prompt_annotation="This is the subject image."
            ),
            VisualReferenceEntry(
                image=_image(0.8), semantic=True, prompt_annotation="This is the outfit reference."
            ),
        )
    )


def test_prompt_creator_public_controls():
    required = CcCKrea2EditPromptCreator.INPUT_TYPES()["required"]
    optional = CcCKrea2EditPromptCreator.INPUT_TYPES()["optional"]
    assert required["mode"][0] == PROMPT_CREATOR_MODES
    assert required["max_tokens"][1]["default"] == 512
    assert required["temperature"][1]["default"] == 0.25
    assert required["top_p"][1]["default"] == 0.90
    assert required["seed"][1]["default"] == 0
    assert "reference_edit_image" in optional


def test_create_from_image_uses_same_clip_and_passthrough():
    clip = _FakeClip("The white dog from Image 1 is running through a river.")
    chain = _chain()

    created, passthrough, info = CcCKrea2EditPromptCreator().create(
        clip=clip,
        visual_references=chain,
        mode="create_from_image",
        user_prompt="Keep the subject from Image 1.",
        max_tokens=512,
        temperature=0.25,
        top_p=0.90,
        seed=1234,
        reference_edit_image=_image(0.5),
    )

    assert created == "The white dog from Image 1 is running through a river."
    assert passthrough is chain
    assert len(clip.tokenize_kwargs["images"]) == 3
    assert "Vision input 1 = Krea Image 1" in clip.tokenize_prompt
    assert "Vision input 2 = Krea Image 2" in clip.tokenize_prompt
    assert "INTERNAL REFERENCE EDIT IMAGE" in clip.tokenize_prompt
    assert clip.tokenize_kwargs["thinking"] is False
    assert clip.generate_kwargs["max_length"] == 512
    assert clip.generate_kwargs["temperature"] == 0.25
    assert clip.generate_kwargs["top_p"] == 0.90
    assert clip.generate_kwargs["seed"] == 1234
    assert "Mode: create_from_image" in info


def test_create_from_image_requires_reference_edit_image():
    with pytest.raises(ValueError, match="requires reference_edit_image"):
        CcCKrea2EditPromptCreator().create(
            clip=_FakeClip(), visual_references=_chain(), mode="create_from_image", user_prompt=""
        )


def test_enhance_and_create_from_theme_require_user_text():
    for mode in ("enhance", "create_from_theme"):
        with pytest.raises(ValueError, match="requires a non-empty user_prompt"):
            CcCKrea2EditPromptCreator().create(
                clip=_FakeClip(), visual_references=_chain(), mode=mode, user_prompt=""
            )


def test_appearance_only_reference_has_no_krea_image_number():
    clip = _FakeClip()
    chain = VisualReferenceChain(
        (
            VisualReferenceEntry(image=_image(0.2), semantic=False),
            VisualReferenceEntry(
                image=_image(0.8), semantic=True, prompt_annotation="This is the subject image."
            ),
        )
    )
    CcCKrea2EditPromptCreator().create(
        clip=clip,
        visual_references=chain,
        mode="enhance",
        user_prompt="Use the subject in an outdoor scene.",
    )
    assert "not shown to Krea Qwen as Image N" in clip.tokenize_prompt
    assert "Vision input 2 = Krea Image 1" in clip.tokenize_prompt
