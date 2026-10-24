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



def test_theme_t2i_ignores_images_and_forwards_references():
    import torch
    from ccc_krea2.modular_nodes.edit_reference_types import VisualReferenceEntry
    image = torch.ones((1, 64, 64, 3))
    chain = VisualReferenceChain((VisualReferenceEntry(image=image, semantic=True),))
    clip = _FakeClip()
    created, returned, info, _ = NODE_CLASS_MAPPINGS["CcCKrea2EditPromptCreator"]().create(
        clip=clip, references=chain,
        mode="create_from_theme_t2i", user_prompt="A new scene."
    )
    assert returned.entries == chain.entries
    assert clip.kwargs["images"] == []
    assert "KREA2 PROMPT GUIDELINES:" in clip.prompt
    assert clip.prompt.startswith("Create an original text-to-image prompt for Krea2")
    assert created


def test_theme_i2i_can_run_without_images():
    clip = _FakeClip()
    created, _, _, _ = NODE_CLASS_MAPPINGS["CcCKrea2EditPromptCreator"]().create(
        clip=clip, mode="create_from_theme_i2i", user_prompt="A whimsical village."
    )
    assert clip.kwargs["images"] == []
    assert "USER PROMPT:\\nA whimsical village." in clip.prompt
    assert "KREA2 PROMPT GUIDELINES:" in clip.prompt
    assert created
