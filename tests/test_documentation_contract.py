"""Documentation matches the published unified-reference surface."""
from pathlib import Path
from ccc_krea2.nodes import NODE_DISPLAY_NAME_MAPPINGS

def test_documented_public_nodes():
    readme = Path("README.md").read_text(encoding="utf-8")
    nodes = Path("NODES.md").read_text(encoding="utf-8")
    for label in NODE_DISPLAY_NAME_MAPPINGS.values():
        assert label in readme and label in nodes
    assert "vision_attention_patch" in readme
    assert "visual_boost" in readme
    assert "vision_strength" in nodes

def test_seven_workflow_paths_are_documented():
    readme = Path("README.md").read_text(encoding="utf-8")
    for path in Path("workflows").glob("*.json"):
        assert str(path) in readme

def test_attention_region_is_not_a_public_node():
    nodes = Path("NODES.md").read_text(encoding="utf-8")
    assert "## Krea2 CcC Attention Region" not in nodes
    assert "## Krea2 CcC Visual Reference" not in nodes
    assert "## Krea2 CcC Vision Reference" not in nodes
