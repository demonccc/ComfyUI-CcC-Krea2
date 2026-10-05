"""Public Prompt Creator input contract tests."""

from pathlib import Path

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


def test_prompt_creator_visual_references_are_optional():
    cls = NODE_CLASS_MAPPINGS["CcCKrea2EditPromptCreator"]
    schema = cls.INPUT_TYPES()

    assert "visual_references" not in schema["required"]
    assert schema["optional"]["visual_references"][0] == "KREA2_VISUAL_REFERENCE_CHAIN"
    assert schema["optional"]["reference_edit_image"][0] == "IMAGE"


def test_create_from_theme_runs_without_visual_references():
    cls = NODE_CLASS_MAPPINGS["CcCKrea2EditPromptCreator"]
    created, references, info, thinking = cls().create(
        clip=_FakeClip(),
        mode="create_from_theme",
        user_prompt="A dramatic rainy cyberpunk street at night.",
    )

    assert created == "A cinematic rainy city scene at night."
    assert isinstance(references, VisualReferenceChain)
    assert not references.entries
    assert "Mode: create_from_theme" in info
    assert thinking == ""


def test_prompt_creator_mode_title_extension_is_checked_in():
    js = Path("web/prompt_creator_mode_title.js").read_text(encoding="utf-8")
    assert 'node.comfyClass !== "CcCKrea2EditPromptCreator"' in js
    assert 'node.title = `Prompt Creator - ${mode}`' in js
    assert "modeWidget.callback" in js
