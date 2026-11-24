"""All Vision presets must retain their custom instruction in Edit conditioning."""
import torch
import pytest
from ccc_krea2.modular_nodes.edit_reference_types import VisionReferenceEntry
from ccc_krea2.vision_reference_runtime import build_vision_reference_spec


@pytest.mark.parametrize("mode", ["semantic_only", "style_direct", "style_indirect", "context_indirect"])
def test_custom_instruction_reaches_edit_spec(mode):
    entry = VisionReferenceEntry(
        image=torch.zeros((1, 32, 32, 3)),
        mode=mode,
        instruction="Keep the specific custom constraint.",
    )
    spec = build_vision_reference_spec(entry)
    assert "Keep the specific custom constraint." in spec.vision_instruction


@pytest.mark.parametrize("mode", ["semantic_only", "style_direct", "style_indirect", "context_indirect"])
def test_unmodified_base_preset_when_custom_instruction_is_empty(mode):
    spec = build_vision_reference_spec(
        VisionReferenceEntry(image=torch.zeros((1, 32, 32, 3)), mode=mode)
    )
    assert spec.vision_instruction
    assert "Additional " not in spec.vision_instruction


@pytest.mark.parametrize("mode", ["native", "guided", "semantic_only", "style_direct", "style_indirect", "context_indirect"])
def test_reference_annotation_is_independent_of_vision_instruction(mode):
    entry = VisionReferenceEntry(
        image=torch.zeros((1, 32, 32, 3)),
        mode=mode,
        instruction="Focus on the face.",
        reference_annotation="This is the subject image.",
    )
    spec = build_vision_reference_spec(entry)
    assert spec.alias == "This is the subject image."
    if mode == "native":
        assert spec.vision_instruction == ""
    else:
        assert "Focus on the face." in spec.vision_instruction


def test_fidelity_only_transforms_semantic_and_style():
    semantic = build_vision_reference_spec(
        VisionReferenceEntry(image=torch.zeros((1, 32, 32, 3)), mode="semantic_only", vision_fidelity=0.65)
    )
    style = build_vision_reference_spec(
        VisionReferenceEntry(image=torch.zeros((1, 32, 32, 3)), mode="style_indirect", vision_fidelity=0.35)
    )
    assert semantic.semantic_strength == pytest.approx(0.65)
    assert style.style_fidelity == pytest.approx(0.35)


def test_none_ignores_instruction_and_guided_uses_it():
    from ccc_krea2.modular_nodes.vision_reference_node import CcCKrea2VisionReference
    node = CcCKrea2VisionReference()
    base, _ = node.process(image=torch.zeros((1, 32, 32, 3)), mode="none", instruction="ignored")
    guided, _ = node.process(image=torch.zeros((1, 32, 32, 3)), mode="guided", instruction="Focus on the eyes.")
    assert base.entries[0].mode == "native"
    assert base.entries[0].instruction == ""
    assert guided.entries[0].mode == "guided"
    assert guided.entries[0].instruction == "Focus on the eyes."


def test_annotation_precedes_main_prompt():
    from ccc_krea2.identity_contract import build_grounded_positive_user_content
    spec = build_vision_reference_spec(VisionReferenceEntry(
        image=torch.zeros((1, 32, 32, 3)), mode="guided",
        reference_annotation="This is the subject image.",
        instruction="Focus on clothing.",
    ))
    prompt = build_grounded_positive_user_content([{"spec": spec}], "Change the outfit.")
    assert prompt.index("Image 1: This is the subject image.") < prompt.index("Image 1 guidance: Focus on clothing.")
    assert prompt.index("Image 1 guidance: Focus on clothing.") < prompt.index("Change the outfit.")
