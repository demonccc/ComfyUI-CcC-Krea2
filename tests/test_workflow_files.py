"""Checked-in workflow contracts for Krea2 CcC Edit and Paint tests."""

import json
from pathlib import Path


EDIT_WORKFLOW = Path("workflows/01_scene_subject.json")
PAINT_WORKFLOW = Path("workflows/02_anypaint_remove_people.json")


def _nodes_by_type(workflow, type_name):
    return [node for node in workflow["nodes"] if node["type"] == type_name]


def test_expected_workflows_are_checked_in():
    workflows = sorted(Path("workflows").rglob("*.json"))
    assert workflows == [EDIT_WORKFLOW, PAINT_WORKFLOW]


def test_scene_subject_workflow_uses_split_nodes():
    workflow = json.loads(EDIT_WORKFLOW.read_text(encoding="utf-8"))

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
    edit = edit_nodes[0]

    assert scene["widgets_values"][:6] == [1.0, "resize", "inside", "center", "center", "lanczos"]
    assert subject["widgets_values"][:6] == [4.0, "resize", "inside", "center", "center", "lanczos"]
    assert len(scene["widgets_values"]) == 11
    assert len(subject["widgets_values"]) == 11
    assert scene["widgets_values"][6:10] == [True, True, 768, "lanczos"]
    assert subject["widgets_values"][6:10] == [True, True, 768, "lanczos"]
    assert "scene image" in scene["widgets_values"][10]
    assert "subject image" in subject["widgets_values"][10]

    assert edit["widgets_values"] == [
        "Replace the person in the scene image with the person from the subject image.",
        True,
    ]

    previous = next(inp for inp in subject["inputs"] if inp["name"] == "previous_references")
    assert previous["type"] == "KREA2_VISUAL_REFERENCE_CHAIN"
    assert previous["link"] is not None


def test_scene_subject_size_resolver_feeds_fixed_latent_dimensions():
    workflow = json.loads(EDIT_WORKFLOW.read_text(encoding="utf-8"))

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
    assert latent_values[4] == "preserve_aspect_krea_bounds"
    assert latent_values[5] == "empty"
    assert latent_values[6] == "crop"


def test_paint_workflow_uses_anypaint_runtime_and_mask_controls():
    workflow = json.loads(PAINT_WORKFLOW.read_text(encoding="utf-8"))

    prepare_nodes = _nodes_by_type(workflow, "CcCKrea2PaintPrepare")
    paint_nodes = _nodes_by_type(workflow, "CcCKrea2Paint")
    lora_nodes = _nodes_by_type(workflow, "CcCKrea2LoRAStack")
    sampler_nodes = _nodes_by_type(workflow, "KSampler")

    assert len(prepare_nodes) == 1
    assert len(paint_nodes) == 1
    assert len(lora_nodes) == 1
    assert len(sampler_nodes) == 1

    prepare = prepare_nodes[0]
    paint = paint_nodes[0]
    lora = lora_nodes[0]
    sampler = sampler_nodes[0]

    assert prepare["widgets_values"] == [0, 0, 0, 0, 12, "gaussian_sigma", 4.0, "outside"]
    assert paint["widgets_values"][1:] == [True, True]
    assert lora["widgets_values"][3] == "krea2_anypaint_rank32.safetensors"
    assert sampler["widgets_values"][2:7] == [8, 1.0, "euler", "simple", 1.0]

    prepare_inputs = {item["name"]: item for item in prepare["inputs"]}
    paint_inputs = {item["name"]: item for item in paint["inputs"]}
    assert prepare_inputs["image"]["link"] is not None
    assert prepare_inputs["mask"]["link"] is not None
    assert paint_inputs["paint_context"]["type"] == "KREA2_PAINT_CONTEXT"
    assert paint_inputs["paint_context"]["link"] is not None


def test_all_workflow_link_types_are_consistent():
    for workflow_path in (EDIT_WORKFLOW, PAINT_WORKFLOW):
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}

        for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
            source = nodes[source_id]["outputs"][source_slot]
            target = nodes[target_id]["inputs"][target_slot]
            assert source["type"] == link_type
            assert target["type"] == link_type
            assert link_id in (source.get("links") or [])
            assert target["link"] == link_id
