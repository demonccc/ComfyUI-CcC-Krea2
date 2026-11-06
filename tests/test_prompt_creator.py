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



def test_create_from_image_uses_structured_single_pass_and_passthrough():
    clip = _FakeClip(
        "ANALYSIS:\nImage 1 provides the dress and pose. "
        "Image 2 supplies the woman's facial identity.\n"
        "FINAL PROMPT:\nThe woman from Image 2 wears the detailed dress "
        "and reproduces the seated pose."
    )
    chain = _chain()
    created, passthrough, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, references=chain, mode="create_from_image_i2i",
        instructions="Use clothing from Image 1 and identity from Image 2.",
        thinking=True, max_tokens=2048, temperature=0.25, top_p=0.90, seed=1234,
    )
    assert created.startswith("The woman from Image 2 wears")
    assert "Image 1 provides the dress" in thinking
    assert "Image 2 supplies" in thinking
    assert passthrough is chain
    assert "ANALYSIS:" in clip.tokenize_prompt
    assert "FINAL PROMPT:" in clip.tokenize_prompt
    assert "thinking" not in clip.tokenize_kwargs
    assert "llama_template" not in clip.tokenize_kwargs
    assert clip.generate_kwargs["max_length"] == 2048
    assert clip.generate_kwargs["seed"] == 1234
    assert "Thinking Strategy: one-pass explicit" in info
    assert "Effective Qwen System Prompt" in info


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
    assert "Return only the final image description" in clip.tokenize_kwargs["system_prompt"]
    assert "Return only the final image description" in clip.tokenize_kwargs["system_prompt"]


def test_image_presets_accept_optional_image_inputs():
    for mode in ("create_from_image_t2i", "create_from_image_i2i"):
        clip = _FakeClip()
        CcCKrea2EditPromptCreator().create(
            clip=clip, mode=mode, instructions="Describe an imaginative scene."
        )
        assert clip.tokenize_kwargs["images"] == []
        assert "KREA2 PROMPT GUIDELINES:" in clip.tokenize_prompt
        assert _resolve_preset(mode) in clip.tokenize_kwargs["system_prompt"]


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


def test_thinking_missing_final_section_returns_partial_analysis_without_retry():
    class TrackingClip(_FakeClip):
        def __init__(self):
            super().__init__("ANALYSIS:\nImage 1 is a scene; Image 2 is an identity reference.")
            self.calls = 0

        def generate(self, tokens, **kwargs):
            self.calls += 1
            return tokens

    clip = TrackingClip()
    created, _, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, references=_chain(), mode="create_from_image_i2i",
        instructions="Preserve identity from Image 2.", thinking=True, max_tokens=32768,
    )
    assert created == ""
    assert clip.calls == 1
    assert "Image 1 is a scene" in thinking
    assert info.startswith("ERROR:")
    assert "FINAL PROMPT:" in info


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


def test_preset_is_sent_as_qwen_system_prompt_not_embedded_in_user_text():
    clip = _FakeClip()
    CcCKrea2EditPromptCreator().create(
        clip=clip, references=_chain(), mode="create_from_image_i2i",
        instructions="Describe the final image.", thinking=True,
    )
    assert _resolve_preset("create_from_image_i2i") in clip.tokenize_kwargs["system_prompt"]
    assert "Return exactly two sections" in clip.tokenize_kwargs["system_prompt"]
    assert _resolve_preset("create_from_image_i2i") not in clip.tokenize_prompt
    assert "INSTRUCTIONS:" in clip.tokenize_prompt


def test_thinking_single_pass_does_not_force_qwen_tags():
    clip = _FakeClip(
        "ANALYSIS:\nImage 1 is the outfit and scene, Image 2 is the identity.\n"
        "FINAL PROMPT:\nA woman wearing a green dress stands by a window."
    )
    created, _, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, references=_chain(), mode="create_from_image_i2i",
        instructions="Use Image 2 for identity.", thinking=True,
    )
    assert created == "A woman wearing a green dress stands by a window."
    assert "Image 2 is the identity" in thinking
    assert "<think>" not in clip.tokenize_prompt
    assert "llama_template" not in clip.tokenize_kwargs
    assert "Thinking Strategy: one-pass explicit" in info


def test_thinking_missing_analysis_section_is_an_error():
    clip = _FakeClip("FINAL PROMPT:\nA woman in a green dress.")
    created, _, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, mode="enhance_i2i", instructions="Dress.", thinking=True,
    )
    assert created == ""
    assert info.startswith("ERROR: Qwen did not produce the required ANALYSIS:")
    assert "A woman in a green dress" in thinking


def test_thinking_with_empty_final_section_is_an_error():
    clip = _FakeClip("ANALYSIS:\nDetailed reference comparison.\nFINAL PROMPT:")
    created, _, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, mode="enhance_i2i", instructions="Dress.", thinking=True,
    )
    assert created == ""
    assert info.startswith("ERROR: Qwen produced the FINAL PROMPT: heading but no final text")
    assert thinking == "Detailed reference comparison."


def test_unlabelled_answer_is_never_accepted_when_thinking_enabled():
    clip = _FakeClip("The woman wears a blue dress.")
    created, _, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, mode="enhance_i2i", instructions="Dress.", thinking=True,
    )
    assert created == ""
    assert info.startswith("ERROR:")
    assert thinking == "The woman wears a blue dress."


def test_non_thinking_behavior_remains_single_pass():
    clip = _FakeClip("The woman wears a blue dress.")
    created, _, _, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, mode="enhance_i2i", instructions="Dress.", thinking=False,
    )
    assert created == "The woman wears a blue dress."
    assert thinking == ""
