"""Regression tests for the public Prompt Creator wrapper."""

import torch

from ccc_krea2.modular_nodes.edit_prompt_creator_public_node import CcCKrea2EditPromptCreator
from ccc_krea2.modular_nodes.edit_reference_types import VisualReferenceChain, VisualReferenceEntry


class _FakeClip:
    def __init__(self, output="The subject from Image 1 is standing in the requested scene."):
        self.output = output
        self.tokenize_prompt = None
        self.tokenize_kwargs = None

    def tokenize(self, prompt, **kwargs):
        self.tokenize_prompt = prompt
        self.tokenize_kwargs = kwargs
        return {"fake": [[(1, 1.0)]]}

    def generate(self, tokens, **kwargs):
        return tokens

    def decode(self, generated_ids):
        return self.output


def _image(value):
    return torch.full((1, 24, 32, 3), float(value), dtype=torch.float32)


def _chain():
    return VisualReferenceChain(
        (
            VisualReferenceEntry(
                image=_image(0.2),
                semantic=True,
                prompt_annotation="This is the subject image.",
            ),
        )
    )


def test_create_from_image_analyzes_visual_references_and_reference_edit_image():
    clip = _FakeClip()
    chain = _chain()
    reference_edit = _image(0.9)

    created, passthrough, info, _ = CcCKrea2EditPromptCreator().create(
        clip=clip,
        visual_references=chain,
        mode="create_from_image",
        user_prompt="Use the subject from Image 1 in the situation shown by the reference edit image.",
        reference_edit_image=reference_edit,
    )

    assert created.startswith("The subject from Image 1")
    assert passthrough is chain
    assert len(clip.tokenize_kwargs["images"]) == 2
    assert torch.equal(clip.tokenize_kwargs["images"][0], chain.entries[0].image)
    assert torch.equal(clip.tokenize_kwargs["images"][1], reference_edit)
    assert "Vision input 1 = Image 1" in clip.tokenize_prompt
    assert "Vision input 2 = INTERNAL REFERENCE EDIT IMAGE" in clip.tokenize_prompt
    assert "Visual Reference Chain Entries: 1" in info
    assert "Images Analyzed: 2" in info


def test_create_from_theme_still_uses_visual_reference_pixels():
    clip = _FakeClip()
    chain = _chain()

    _, passthrough, _, _ = CcCKrea2EditPromptCreator().create(
        clip=clip,
        visual_references=chain,
        mode="create_from_theme",
        user_prompt="Put Image 1 in a neon city at night.",
    )

    assert passthrough is chain
    assert len(clip.tokenize_kwargs["images"]) == 1
    assert torch.equal(clip.tokenize_kwargs["images"][0], chain.entries[0].image)
    assert "Vision input 1 = Image 1" in clip.tokenize_prompt
