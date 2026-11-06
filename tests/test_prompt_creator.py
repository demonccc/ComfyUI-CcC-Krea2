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
    class StagedClip(_FakeClip):
        def __init__(self):
            super().__init__()
            self.stage = 0

        def generate(self, tokens, **kwargs):
            self.generate_kwargs = kwargs
            self.stage += 1
            return self.stage

        def decode(self, generated_ids):
            if generated_ids == 1:
                return "The internal image contains a seated subject and another person nearby."
            return (
                "The woman from Krea Image 1 is seated on a white bed with her body angled "
                "slightly to the side, while another person remains beside the bed."
            )

    clip = StagedClip()
    chain = _chain()
    created, passthrough, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, references=chain, mode="create_from_image_i2i",
        instructions="Use the subject from Image 1 in the situation shown by the reference edit image.",
        thinking=True, max_tokens=2048, temperature=0.25,
        top_p=0.90, seed=1234,
    )
    assert created.startswith("The woman from Image 1 is seated on a white bed")
    assert thinking.startswith("The internal image contains a seated subject")
    assert clip.stage == 2
    assert len(passthrough.entries) == len(chain.entries)
    assert len(clip.tokenize_kwargs["images"]) == 2
    assert "Vision input 1 = Image 1" in clip.tokenize_prompt
    assert "Vision input 2 = Image 2" in clip.tokenize_prompt
    assert "Return only the final image description" in clip.tokenize_kwargs["system_prompt"]
    assert "llama_template" not in clip.tokenize_kwargs
    assert clip.generate_kwargs["max_length"] == 2048
    assert clip.generate_kwargs["seed"] == 1234
    assert "Thinking Output: present (explicit Qwen analysis)" in info
    assert "Effective Qwen System Prompt" in info
    assert "Effective Qwen User Prompt" in info
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


def test_explicit_analysis_without_native_delimiters_produces_final_prompt():
    class TwoPassClip(_FakeClip):
        def __init__(self):
            super().__init__()
            self.calls = []

        def generate(self, tokens, **kwargs):
            self.calls.append(kwargs["max_length"])
            return len(self.calls)

        def decode(self, generated_ids):
            if generated_ids == 1:
                return "The first reference shows a red dress and a seated pose."
            return "The woman wears a red dress while seated."

    clip = TwoPassClip()
    created, _, info, reasoning = CcCKrea2EditPromptCreator().create(
        clip=clip, references=_chain(), mode="create_from_image_i2i",
        instructions="Preserve clothing from Image 1.", thinking=True,
        max_tokens=32768,
    )
    assert created == "The woman wears a red dress while seated."
    assert reasoning == "The first reference shows a red dress and a seated pose."
    assert clip.calls == [4096, 32768]
    assert "Generation Calls: 2" in info
    assert not info.startswith("ERROR:")
    assert "llama_template" not in clip.tokenize_kwargs


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
    assert "Return only the final image description" in clip.tokenize_kwargs["system_prompt"]
    assert _resolve_preset("create_from_image_i2i") not in clip.tokenize_prompt
    assert "INSTRUCTIONS:" in clip.tokenize_prompt


def test_thinking_requires_two_explicit_stages_with_no_forced_prefix():
    class TrackingClip(_FakeClip):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.tokenize_calls = []

        def tokenize(self, prompt, **kwargs):
            self.tokenize_calls.append((prompt, kwargs))
            return super().tokenize(prompt, **kwargs)

        def generate(self, tokens, **kwargs):
            self.calls += 1
            return self.calls

        def decode(self, ids):
            return ("The original scene contains a seated woman." if ids == 1
                    else "A woman is seated in a warmly lit room.")

    clip = TrackingClip()
    created, _, info, reasoning = CcCKrea2EditPromptCreator().create(
        clip=clip, references=_chain(), mode="create_from_image_i2i",
        instructions="Describe a new scene.", thinking=True,
    )
    assert created == "A woman is seated in a warmly lit room."
    assert reasoning == "The original scene contains a seated woman."
    assert clip.calls == 2
    assert "PREVIOUS VISUAL ANALYSIS:" in clip.tokenize_calls[1][0]
    assert "llama_template" not in clip.tokenize_calls[0][1]
    assert "llama_template" not in clip.tokenize_calls[1][1]
    assert "Thinking Strategy: explicit analysis then final generation" in info


def test_empty_analysis_returns_error_without_final_stage():
    class EmptyAnalysis(_FakeClip):
        def __init__(self):
            super().__init__("")
            self.calls = 0

        def generate(self, tokens, **kwargs):
            self.calls += 1
            return tokens

    clip = EmptyAnalysis()
    created, _, info, reasoning = CcCKrea2EditPromptCreator().create(
        clip=clip, mode="enhance_t2i", instructions="Describe the outfit.",
        thinking=True,
    )
    assert created == ""
    assert reasoning == ""
    assert clip.calls == 1
    assert info.startswith("ERROR: Qwen analysis stage produced no usable text")


def test_empty_final_returns_error_and_keeps_analysis():
    class EmptyFinal(_FakeClip):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate(self, tokens, **kwargs):
            self.calls += 1
            return self.calls

        def decode(self, ids):
            return "The woman is wearing a red coat." if ids == 1 else ""

    clip = EmptyFinal()
    created, _, info, reasoning = CcCKrea2EditPromptCreator().create(
        clip=clip, mode="enhance_t2i", instructions="Describe the outfit.",
        thinking=True,
    )
    assert created == ""
    assert reasoning == "The woman is wearing a red coat."
    assert clip.calls == 2
    assert info.startswith("ERROR: Qwen final stage returned no prompt")


def test_instruct_checkpoint_plain_text_with_thinking_is_accepted_in_second_stage():
    clip = _FakeClip("The woman wears a blue dress.")
    created, _, info, reasoning = CcCKrea2EditPromptCreator().create(
        clip=clip, mode="enhance_t2i", instructions="Describe the outfit.",
        thinking=True,
    )
    assert created == "The woman wears a blue dress."
    assert reasoning == "The woman wears a blue dress."
    assert "Generation Calls: 2" in info
