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
