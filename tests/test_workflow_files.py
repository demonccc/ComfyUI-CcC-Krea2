"""Checked-in workflow contract for the split scene + subject Edit test."""

import json
from pathlib import Path


WORKFLOW = Path("workflows/01_scene_subject.json")


def _nodes_by_type(workflow, type_name):
    return [node for node in workflow["nodes"] if node["type"] == type_name]


def test_only_one_edit_workflow_is_checked_in():
    workflows = sorted(Path("workflows").rglob("*.json"))
    assert workflows == [WORKFLOW]


def test_scene_subject_workflow_uses_split_nodes():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))

    visual_nodes = _nodes_by_type(workflow, "CcCKrea2VisualReference")
    edit_nodes = _nodes_by_type(workflow, "CcCKrea2Edit")

    assert len(visual_nodes) == 2
    assert len(edit_nodes) == 1
    assert not _nodes_by_type(workflow, "CcCKrea2EasyEdit")
    assert not _nodes_by_type(workflow, "CcCKrea2EasyEditOstris")
    assert not _nodes_by_type(workflow, "CcCKrea2EditAdvanced")

    scene = next(node for node in visual_nodes if node.get("title") == "Scene Visual Reference")
    subject = next(node for node in visual_nodes if node.get("title") == "Subject Visual Reference")

    assert scene["widgets_values"][:4] == [1.0, "none", True, "scene image"]
    assert subject["widgets_values"][:4] == [4.0, "none", True, "subject image"]

    previous = next(inp for inp in subject["inputs"] if inp["name"] == "previous_references")
    assert previous["type"] == "KREA2_VISUAL_REFERENCE_CHAIN"
    assert previous["link"] is not None


def test_scene_subject_edit_geometry_sources_and_empty_target():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    edit = _nodes_by_type(workflow, "CcCKrea2Edit")[0]

    input_names = {item["name"] for item in edit["inputs"]}
    assert "target_image" not in input_names
    assert "grid_size_image" in input_names
    assert "grid_geometry_image" in input_names
    assert "visual_references" in input_names

    values = edit["widgets_values"]
    assert values[2] == "from source"
    assert values[3] == "from source"
    assert values[4] is False
    assert values[-1] is True


def test_workflow_reference_chain_link_types_are_consistent():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}

    for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
        source = nodes[source_id]["outputs"][source_slot]
        target = nodes[target_id]["inputs"][target_slot]
        assert source["type"] == link_type
        assert target["type"] == link_type
        assert link_id in (source.get("links") or [])
        assert target["link"] == link_id
