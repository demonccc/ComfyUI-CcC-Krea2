"""Per-reference Qwen Vision token weighting without changing text or VAE/RoPE."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ccc_krea2.conditioning import apply_vision_weights_to_conditioning
from ccc_krea2.modular_nodes.vision_reference_node import CcCKrea2VisionReference
from ccc_krea2.modular_nodes.reference_node import CcCKrea2Reference


def _img(weight):
    return {"spec": SimpleNamespace(vision_weight=weight)}


def test_neutral_weight_returns_original_object_unmodified():
    conditioning = [[torch.arange(24, dtype=torch.float32).reshape(1, 6, 4), {"test": True}]]
    result = apply_vision_weights_to_conditioning(conditioning, [_img(1.0)], [(1, 3)])
    assert result is conditioning
    assert result[0][0] is conditioning[0][0]


def test_zero_disables_only_its_vision_rows():
    original = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) + 1
    result = apply_vision_weights_to_conditioning(
        [[original, {}]], [_img(0.0), _img(2.0)], [(1, 3), (4, 5)]
    )[0][0]
    assert torch.equal(result[:, 0], original[:, 0])
    assert torch.equal(result[:, 1:3], torch.zeros_like(original[:, 1:3]))
    assert torch.equal(result[:, 3], original[:, 3])
    assert torch.equal(result[:, 4], original[:, 4] * 2.0)
    assert torch.equal(result[:, 5], original[:, 5])
    assert torch.count_nonzero(original[:, 1:3]) > 0


def test_weight_after_style_compaction_uses_surviving_row_indices():
    original = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4) + 1
    # Original sequence had 8 rows: image1 was 1..3, image2 5..6.
    # Removed indices 2, 3 and 6; remaining indices map to 0,1,2,3,4.
    result = apply_vision_weights_to_conditioning(
        [[original, {}]], [_img(0.0), _img(2.0)],
        [(1, 4), (5, 7)], removed_row_indices=[2, 3, 6]
    )[0][0]
    assert torch.equal(result[:, 0], original[:, 0])
    assert torch.equal(result[:, 1], torch.zeros_like(original[:, 1]))
    assert torch.equal(result[:, 2], original[:, 2])
    assert torch.equal(result[:, 3], original[:, 3] * 2.0)
    assert torch.equal(result[:, 4], original[:, 4])


@pytest.mark.parametrize("weight", [-0.1, 2.1])
def test_out_of_range_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="vision_weight"):
        apply_vision_weights_to_conditioning(
            [[torch.ones(1, 3, 4), {}]], [_img(weight)], [(1, 2)]
        )


def test_reference_widget_schema_places_weight_below_enable_and_fidelity_below_style():
    required = CcCKrea2Reference.INPUT_TYPES()["required"]
    assert required["vision_weight"][1]["default"] == 1.0
    assert required["vision_weight"][1]["min"] == 0.0
    assert required["vision_weight"][1]["max"] == 2.0
    keys = list(required)
    assert keys.index("vision_weight") == keys.index("vision_enable") + 1
    assert keys.index("vision_fidelity") == keys.index("vision_style_processing") + 1
    assert "vision_vision_fidelity" not in required
    assert "vae_rope_attention_scope" not in required


def test_public_vision_node_preserves_weight_in_entry():
    img = torch.ones((1, 16, 16, 3))
    chain, _ = CcCKrea2VisionReference().process(
        img, mode="reference", weight=0.0
    )
    assert chain.entries[0].vision_weight == 0.0

def test_example_workflows_match_reference_widget_schema():
    fields = list(CcCKrea2Reference.INPUT_TYPES()["required"])
    widgets = [name for name in fields if name != "image"]
    examples = Path(__file__).resolve().parents[1] / "workflows"
    checked = 0
    for workflow_path in examples.glob("*.json"):
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        for node in workflow.get("nodes", []):
            if node.get("type") != "CcCKrea2Reference":
                continue
            values = node["widgets_values"]
            assert len(values) == len(widgets), workflow_path.name
            data = dict(zip(widgets, values))
            assert isinstance(data["vision_enable"], bool), workflow_path.name
            assert isinstance(data["vision_weight"], (int, float)), workflow_path.name
            assert 0.0 <= data["vision_weight"] <= 2.0, workflow_path.name
            assert data["vision_mode"] in ("reference", "semantic", "visual", "style", "context"), workflow_path.name
            assert data["vision_style_processing"] in ("full", "2x2", "4x4"), workflow_path.name
            assert isinstance(data["vision_fidelity"], (int, float)), workflow_path.name
            assert data["vision_guidance_seed_mode"] in ("fixed", "random", "incremental"), workflow_path.name
            assert isinstance(data["vision_guidance_seed"], int), workflow_path.name
            checked += 1
    assert checked > 0
