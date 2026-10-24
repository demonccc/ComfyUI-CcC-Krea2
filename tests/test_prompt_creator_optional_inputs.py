"""Functional Prompt Creator tests with a deterministic Qwen stand-in."""

from ccc_krea2.nodes import NODE_CLASS_MAPPINGS
from ccc_krea2.modular_nodes.edit_reference_types import VisualReferenceChain


class _FakeClip:
    def tokenize(self, prompt, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return {"fake": [[(1, 1.0)]]}

    def generate(self, tokens, **kwargs):
        return tokens

    def decode(self, generated_ids):
        return "A cinematic rainy city scene at night."



def test_create_from_theme_t2i_runs_without_references():
    cls = NODE_CLASS_MAPPINGS["CcCKrea2EditPromptCreator"]
    created, references, info, thinking = cls().create(
        clip=_FakeClip(),
        mode="create_from_theme_t2i",
        user_prompt="A dramatic rainy cyberpunk street at night.",
    )

    assert created == "A cinematic rainy city scene at night."
    assert isinstance(references, VisualReferenceChain)
    assert not references.entries
    assert "Mode: create_from_theme_t2i" in info
    assert thinking == ""



def test_prompt_creator_accepts_mixed_chain_and_passes_it_through():
    import torch
    from ccc_krea2.modular_nodes.edit_reference_types import (
        VisionReferenceEntry, VisualReferenceEntry,
    )
    image = torch.ones((1, 64, 64, 3))
    chain = (VisualReferenceChain()
             .append(VisualReferenceEntry(image=image, semantic=True))
             .append(VisionReferenceEntry(image=image, mode="context_indirect"))
             .append(VisualReferenceEntry(image=image, semantic=False)))
    cls = NODE_CLASS_MAPPINGS["CcCKrea2EditPromptCreator"]
    created, returned, info, thinking = cls().create(
        clip=_FakeClip(), references=chain,
        mode="create_from_theme_t2i", user_prompt="A new scene."
    )
    assert returned is chain
    assert "Vision input 1 = Image 1" in info
    assert "Vision input 2 = visual reference 2" in info
    assert "Vision input 3" not in info
    assert created
