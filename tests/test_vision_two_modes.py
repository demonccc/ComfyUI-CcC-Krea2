"""Regression tests for the two-preset Qwen Vision contract."""

import torch
import pytest

from ccc_krea2.modular_nodes.vision_reference_node import CcCKrea2VisionReference
from ccc_krea2.vision_reference_runtime import build_vision_reference_spec
from ccc_krea2.reference_specs import ReferenceSpec
from ccc_krea2.identity_contract import build_grounded_positive_user_content
from ccc_krea2.conditioning import encode_krea2_qwen_context
from ccc_krea2 import conditioning as conditioning_module


def test_exactly_two_modes_and_optional_instruction():
    node = CcCKrea2VisionReference()
    assert node.INPUT_TYPES()["required"]["mode"][0] == ("visual", "semantic")
    image = torch.zeros((1, 8, 8, 3))
    for mode in ("visual", "semantic"):
        chain, info = node.process(image, mode=mode, instruction="  keep the pose  ")
        entry = chain.entries[0]
        assert entry.instruction == "keep the pose"
        spec = build_vision_reference_spec(entry)
        assert spec.vision_reference_mode == mode
        assert spec.vision_instruction == "keep the pose"
        assert spec.prepared_image.vision_image.shape == image.shape
        assert "keep the pose" in info
        no_instruction = node.process(image, mode=mode)[0].entries[0]
        assert not build_vision_reference_spec(no_instruction).vision_instruction
    with pytest.raises(ValueError, match="Unsupported"):
        node.process(image, mode="style")


def test_instruction_adds_text_only_when_nonempty():
    spec = ReferenceSpec(appearance_reference=False, vision_reference_mode="visual")
    ref = {"spec": spec, "expanded_aliases": (), "vae_reference_frame": None}
    assert "guidance" not in build_grounded_positive_user_content([ref], "hello")
    with_instruction = dict(spec.__dict__, vision_instruction="turn toward the window")
    with_instruction.pop("requested_vision_slot", None)
    # Build from dataclass constructor rather than inventing hidden preset text.
    from dataclasses import replace
    ref["spec"] = replace(spec, vision_instruction="turn toward the window")
    result = build_grounded_positive_user_content([ref], "hello")
    assert "Image 1 guidance: turn toward the window" in result
    assert result.endswith("hello")


class DummyQwen:
    is_test_dummy = True

    def tokenize(self, prompt, images=None, llama_template=None):
        return {"qwen3vl": [[0]]}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.arange(12, dtype=torch.float32).reshape(1, 6, 2), {}]]


@pytest.mark.parametrize("mode,expected_rows", [("visual", 6), ("semantic", 4)])
@pytest.mark.parametrize("positive", [True, False])
def test_semantic_removes_only_image_rows_after_multimodal_encode(monkeypatch, mode, expected_rows, positive):
    monkeypatch.setattr(conditioning_module, "extract_vision_spans_from_tokens",
                        lambda **kwargs: ([(1, 3)], [], "qwen3vl", 0))
    spec = ReferenceSpec(vision_reference_mode=mode, appearance_reference=False)
    context = encode_krea2_qwen_context(
        clip=DummyQwen(), prompt="the user's existing prompt",
        physical_images=[torch.zeros((1, 8, 8, 3))],
        physical_image_map=[{"spec": spec}], is_positive=positive,
    )
    assert context.conditioning[0][0].shape[1] == expected_rows
    if mode == "semantic":
        assert context.removed_row_indices == [1, 2]
        torch.testing.assert_close(
            context.conditioning[0][0][0, :, 0],
            torch.tensor([0.0, 6.0, 8.0, 10.0]),
        )
