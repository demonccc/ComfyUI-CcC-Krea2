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
    resolver_nodes = _nodes_by_type(workflow, "CcCKrea2SizeResolver")
    latent_nodes = _nodes_by_type(workflow, "CcCKrea2Latent")
    edit_nodes = _nodes_by_type(workflow, "CcCKrea2Edit")

    assert len(visual_nodes) == 2
    assert len(resolver_nodes) == 1
    assert len(latent_nodes) == 1
    assert len(edit_nodes) == 1
    assert not _nodes_by_type(workflow, "CcCKrea2Geometry")

    scene = next(node for node in visual_nodes if node.get("title") == "Scene Visual Reference")
    subject = next(node for node in visual_nodes if node.get("title") == "Subject Visual Reference")

    assert scene["widgets_values"][:6] == [1.0, True, "inside", "center", "center", True]
    assert subject["widgets_values"][:6] == [4.0, True, "inside", "center", "center", True]

    previous = next(inp for inp in subject["inputs"] if inp["name"] == "previous_references")
    assert previous["type"] == "KREA2_VISUAL_REFERENCE_CHAIN"
    assert previous["link"] is not None


def test_scene_subject_size_resolver_feeds_fixed_latent_dimensions():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))

    resolver = _nodes_by_type(workflow, "CcCKrea2SizeResolver")[0]
    latent = _nodes_by_type(workflow, "CcCKrea2Latent")[0]
    edit = _nodes_by_type(workflow, "CcCKrea2Edit")[0]

    resolver_inputs = {item["name"]: item for item in resolver["inputs"]}
    latent_inputs = {item["name"]: item for item in latent["inputs"]}
    edit_inputs = {item["name"]: item for item in edit["inputs"]}

    assert resolver_inputs["long_edge_image"]["link"] is not None
    assert resolver_inputs["aspect_ratio_image"]["link"] is not None
    assert latent_inputs["width"]["link"] is not None
    assert latent_inputs["height"]["link"] is not None
    assert latent_inputs["width"]["type"] == "INT"
    assert latent_inputs["height"]["type"] == "INT"
    assert "latent" in edit_inputs

    latent_values = latent["widgets_values"]
    assert latent_values[0] == "fixed"
    assert "empty" in latent_values


def test_workflow_link_types_are_consistent():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}

    for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
        source = nodes[source_id]["outputs"][source_slot]
        target = nodes[target_id]["inputs"][target_slot]
        assert source["type"] == link_type
        assert target["type"] == link_type
        assert link_id in (source.get("links") or [])
        assert target["link"] == link_id
