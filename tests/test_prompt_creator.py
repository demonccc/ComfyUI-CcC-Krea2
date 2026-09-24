"""Tests for Krea2 CcC Edit Prompt Creator."""

import pytest
import torch

from ccc_krea2.modular_nodes.edit_prompt_creator_node import (
    PRESET_SYSTEM_PROMPTS,
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


def test_prompt_creator_public_controls_and_output_compatibility():
    required = CcCKrea2EditPromptCreator.INPUT_TYPES()["required"]
    optional = CcCKrea2EditPromptCreator.INPUT_TYPES()["optional"]

    assert required["mode"][0] == PROMPT_CREATOR_MODES
    assert PROMPT_CREATOR_MODES == ("enhance", "create_from_image", "create_from_theme", "custom")
    assert required["system_prompt"][1]["default"] == PRESET_SYSTEM_PROMPTS["enhance"]
    assert required["thinking"][1]["default"] is False
    assert required["max_tokens"][1]["default"] == 512
    assert required["temperature"][1]["default"] == 0.25
    assert required["top_p"][1]["default"] == 0.90
    assert required["seed"][1]["default"] == 0
    assert "reference_edit_image" in optional

    # Preserve original first three output slots; thinking remains appended.
    assert CcCKrea2EditPromptCreator.RETURN_NAMES == (
        "created_prompt",
        "visual_references",
        "creator_info",
        "thinking",
    )


def test_create_from_image_uses_generic_scene_preset_thinking_and_passthrough():
    # thinking=True pre-fills the assistant turn with <think>, so decode starts with reasoning text.
    clip = _FakeClip(
        "The internal image has a seated subject, another person nearby, and a white bed.</think>"
        "The woman from Krea Image 1 is seated on a white bed with her body angled slightly to the side, "
        "while another person remains beside the bed."
    )
    chain = _chain()

    created, passthrough, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip,
        visual_references=chain,
        mode="create_from_image",
        user_prompt="Use the subject from Image 1 in the situation shown by the reference edit image.",
        system_prompt="this must be ignored for preset modes",
        thinking=True,
        max_tokens=2048,
        temperature=0.25,
        top_p=0.90,
        seed=1234,
        reference_edit_image=_image(0.5),
    )

    assert created.startswith("The woman from Image 1 is seated on a white bed")
    assert thinking.startswith("The internal image has a seated subject")
    assert passthrough is chain
    assert len(clip.tokenize_kwargs["images"]) == 3
    assert "Vision input 1 = Image 1" in clip.tokenize_prompt
    assert "Vision input 2 = Image 2" in clip.tokenize_prompt
    assert "INTERNAL REFERENCE EDIT IMAGE" in clip.tokenize_prompt
    assert clip.tokenize_kwargs["thinking"] is True
    assert clip.tokenize_kwargs["system_prompt"] == PRESET_SYSTEM_PROMPTS["create_from_image"]
    assert clip.tokenize_kwargs["llama_template"].endswith("<think>")
    assert clip.tokenize_kwargs["llama_template"].count("<|image_pad|>") == 3
    assert "authoritative visual blueprint" in clip.tokenize_kwargs["system_prompt"]
    assert "direct description of the desired final image" in clip.tokenize_kwargs["system_prompt"]
    assert "other people in the scene" not in clip.tokenize_kwargs["system_prompt"]
    assert clip.generate_kwargs["max_length"] == 2048
    assert clip.generate_kwargs["seed"] == 1234
    assert "Thinking Output: present" in info
    assert "System Prompt Source: preset" in info
    assert "Thinking:" in info


def test_meta_instruction_leak_is_removed_and_reference_name_is_normalized():
    clip = _FakeClip(
        "You are a professional image editor. "
        "Your task is to create a new image based on the user's request, using the provided visual "
        "references and internal reference edit image as guides.\n\n"
        "The woman from Krea Image 1 is seated on a white bed."
    )

    created, _, _, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip,
        visual_references=_chain(),
        mode="enhance",
        user_prompt="Put the woman on a white bed.",
    )

    assert created == "The woman from Image 1 is seated on a white bed."
    assert thinking == ""


def test_meta_instruction_only_output_is_rejected():
    clip = _FakeClip(
        "You are a professional image editor. "
        "Your task is to create a new image based on the user's request, using the provided visual "
        "references and internal reference edit image as guides."
    )

    with pytest.raises(RuntimeError, match="meta-instructions"):
        CcCKrea2EditPromptCreator().create(
            clip=clip,
            visual_references=_chain(),
            mode="enhance",
            user_prompt="Improve this edit.",
        )


def test_custom_mode_uses_exact_system_prompt():
    clip = _FakeClip("The woman from Image 1 is standing under neon lights.")
    custom = "Build a cyberpunk situation around the referenced subject. Return only the edit prompt."

    created, _, info, _ = CcCKrea2EditPromptCreator().create(
        clip=clip,
        visual_references=_chain(),
        mode="custom",
        user_prompt="",
        system_prompt=custom,
    )

    assert created == "The woman from Image 1 is standing under neon lights."
    assert clip.tokenize_kwargs["system_prompt"] == custom
    assert "System Prompt Source: custom" in info


def test_custom_mode_requires_system_prompt():
    with pytest.raises(ValueError, match="requires a non-empty system_prompt"):
        CcCKrea2EditPromptCreator().create(
            clip=_FakeClip(),
            visual_references=_chain(),
            mode="custom",
            user_prompt="",
            system_prompt="",
        )


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


def test_appearance_only_reference_has_no_downstream_image_number():
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

    assert "has no downstream Image N label" in clip.tokenize_prompt
    assert "Vision input 2 = Image 1" in clip.tokenize_prompt


def test_thinking_truncation_without_final_prompt_is_reported():
    # With thinking prefilled, an unfinished generation contains reasoning text without a literal <think> prefix.
    clip = _FakeClip("I am still reasoning when max tokens end.")

    with pytest.raises(RuntimeError, match="ended inside the thinking block"):
        CcCKrea2EditPromptCreator().create(
            clip=clip,
            visual_references=_chain(),
            mode="enhance",
            user_prompt="Improve this edit.",
            thinking=True,
        )
