"""Reference tags identify inputs; Qwen instructions specify attention targets."""
import json
from pathlib import Path

import torch
from ccc_krea2.modular_nodes.reference_node import CcCKrea2Reference
from ccc_krea2.vision_reference_runtime import build_vision_reference_spec

def test_tag_does_not_leak_into_vision_instruction():
    instruction = "Focus only on the garment fabric and embroidery. Ignore the face."
    visuals, visions, _ = CcCKrea2Reference().process(
        image=torch.zeros((1, 64, 64, 3)), tag="outfit",
        vision_instruction=instruction, semantic_enabled=True, visual_enabled=True
    )
    assert visuals.entries[0].reference_tag == "outfit"
    assert visions.entries[0].reference_tag == "outfit"
    assert visuals.entries[0].prompt_annotation == instruction
    assert "@outfit" not in visuals.entries[0].prompt_annotation
    spec = build_vision_reference_spec(visions.entries[0])
    assert instruction in spec.vision_instruction
    assert "@outfit" not in spec.vision_instruction

def test_workflows_use_specific_vision_instructions():
    for path in Path("workflows").glob("*.json"):
        graph = json.loads(path.read_text(encoding="utf-8"))
        for node in graph["nodes"]:
            if node["type"] != "CcCKrea2Reference":
                continue
            tag, instruction = node["widgets_values"][0], node["widgets_values"][4]
            assert tag
            assert "Focus on" in instruction
            assert f"@{tag}" not in instruction
            assert "Preserve relevant details" not in instruction
