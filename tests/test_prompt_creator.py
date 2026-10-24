"""Tests for Krea2 CcC Edit Prompt Creator."""

import pytest
import torch

from ccc_krea2.modular_nodes.edit_prompt_creator_node import (
    _resolve_preset,
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
        references=chain,
        mode="create_from_image_i2i",
        instructions="Use the subject from Image 1 in the situation shown by the reference edit image.",
        thinking=True,
        max_tokens=2048,
        temperature=0.25,
        top_p=0.90,
        seed=1234,
        reference_edit_image=VisualReferenceChain((VisualReferenceEntry(image=_image(0.5), semantic=True),)),
    )

    assert created.startswith("The woman from Image 1 is seated on a white bed")
    assert thinking.startswith("The internal image has a seated subject")
    assert len(passthrough.entries) == len(chain.entries) + 1
    assert len(clip.tokenize_kwargs["images"]) == 3
    assert "Vision input 1 = Image 1" in clip.tokenize_prompt
    assert "Vision input 2 = Image 2" in clip.tokenize_prompt
    assert "KREA2 PROMPT GUIDELINES:" in clip.tokenize_prompt
    assert clip.tokenize_kwargs["thinking"] is True
    assert clip.tokenize_kwargs["system_prompt"] == ""
    assert clip.tokenize_kwargs["llama_template"].endswith("<think>")
    assert clip.tokenize_kwargs["llama_template"].count("<|image_pad|>") == 3
    assert clip.generate_kwargs["max_length"] == 2048
    assert clip.generate_kwargs["seed"] == 1234
    assert "Thinking Output: present" in info
    assert "Effective Qwen Prompt" in info
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
        references=_chain(),
        mode="enhance_t2i",
        instructions="Put the woman on a white bed.",
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
            references=_chain(),
            mode="enhance_t2i",
            instructions="Improve this edit.",
        )


def test_none_mode_uses_instructions_and_guidelines_only():
    clip = _FakeClip()
    CcCKrea2EditPromptCreator().create(clip=clip, mode="none", instructions="A new idea.")
    assert clip.tokenize_prompt.startswith("INSTRUCTIONS:\nA new idea.")
    assert "KREA2 PROMPT GUIDELINES:" in clip.tokenize_prompt
    assert clip.tokenize_kwargs["system_prompt"] == ""


def test_image_presets_accept_optional_image_inputs():
    for mode in ("create_from_image_t2i", "create_from_image_i2i"):
        clip = _FakeClip()
        CcCKrea2EditPromptCreator().create(
            clip=clip, mode=mode, instructions="Describe an imaginative scene."
        )
        assert clip.tokenize_kwargs["images"] == []
        assert "KREA2 PROMPT GUIDELINES:" in clip.tokenize_prompt
        assert _resolve_preset(mode) in clip.tokenize_prompt


def test_enhance_and_create_from_theme_require_user_text():
    for mode in ("enhance_t2i", "enhance_i2i", "create_from_theme_t2i", "create_from_theme_i2i"):
        with pytest.raises(ValueError, match="instructions must not be empty"):
            CcCKrea2EditPromptCreator().create(
                clip=_FakeClip(), references=_chain(), mode=mode, instructions=""
            )


def test_image_preset_receives_reference_mapping_without_automatic_roles():
    clip = _FakeClip()
    CcCKrea2EditPromptCreator().create(
        clip=clip, references=_chain(),
        mode="create_from_image_t2i", instructions="Describe Image 2.",
    )
    assert len(clip.tokenize_kwargs["images"]) == 2
    assert "Vision input 2 = Image 2." in clip.tokenize_prompt
    assert "Role:" not in clip.tokenize_prompt


def test_thinking_truncation_retries_without_thinking_and_preserves_reasoning():
    class SequenceClip(_FakeClip):
        def __init__(self):
            super().__init__()
            self.requests = []
            self.calls = 0

        def tokenize(self, prompt, **kwargs):
            self.requests.append(kwargs)
            return super().tokenize(prompt, **kwargs)

        def decode(self, generated_ids):
            self.calls += 1
            if self.calls == 1:
                return "I am still reasoning when max tokens end."
            return "A detailed image description of the final result."

    clip = SequenceClip()
    created, _, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, references=_chain(), mode="enhance_t2i",
        instructions="Improve this edit.", thinking=True,
    )
    assert created == "A detailed image description of the final result."
    assert thinking == "I am still reasoning when max tokens end."
    assert [request["thinking"] for request in clip.requests] == [True, False]
    assert "llama_template" not in clip.requests[1]
    assert "Thinking Fallback: retry without thinking" in info


def test_thinking_truncation_fails_if_retry_also_has_no_final_prompt():
    class NoFinalClip(_FakeClip):
        def decode(self, generated_ids):
            return ""

    with pytest.raises(RuntimeError, match="no final prompt"):
        CcCKrea2EditPromptCreator().create(
            clip=NoFinalClip(""), references=_chain(), mode="enhance_t2i",
            instructions="Improve this edit.", thinking=True,
        )


def test_high_token_budget_is_passed_unmodified_to_qwen():
    clip = _FakeClip("The woman from Image 1 is wearing a blue dress.")
    created, passthrough, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, references=_chain(), mode="enhance_t2i",
        instructions="Describe the dress in detail.",
        max_tokens=32768,
    )
    assert created
    assert passthrough is not None
    assert thinking == ""
    assert clip.generate_kwargs["max_length"] == 32768
    assert "Max Tokens: 32768" in info


def test_image_preset_allows_empty_instructions_when_images_exist():
    for mode in ("create_from_image_t2i", "create_from_image_i2i"):
        clip = _FakeClip()
        CcCKrea2EditPromptCreator().create(
            clip=clip, references=_chain(), mode=mode, instructions=""
        )
        assert len(clip.tokenize_kwargs["images"]) == 2
        assert "IMAGE MAPPING:" in clip.tokenize_prompt
        assert "INSTRUCTIONS:" not in clip.tokenize_prompt


def test_image_preset_requires_instructions_without_images():
    for mode in ("create_from_image_t2i", "create_from_image_i2i"):
        with pytest.raises(ValueError, match="instructions must not be empty"):
            CcCKrea2EditPromptCreator().create(
                clip=_FakeClip(), mode=mode, instructions=""
            )
