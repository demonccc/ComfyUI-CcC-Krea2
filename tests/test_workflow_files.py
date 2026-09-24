"""Checked-in workflow contracts for Krea2 CcC Edit and Paint tests."""

import json
from pathlib import Path


EDIT_WORKFLOW = Path("workflows/01_scene_subject.json")
PAINT_WORKFLOW = Path("workflows/02_anypaint_remove_people.json")
REGIONAL_WORKFLOW = Path("workflows/03_regional_attention.json")
CHARACTER_SHEET_WORKFLOW = Path("workflows/04_character_sheet_identity.json")
PROMPT_CREATOR_WORKFLOW = Path("workflows/05_edit_prompt_creator.json")


def _nodes_by_type(workflow, type_name):
    return [node for node in workflow["nodes"] if node["type"] == type_name]


def test_expected_workflows_are_checked_in():
    workflows = sorted(Path("workflows").rglob("*.json"))
    assert workflows == [EDIT_WORKFLOW, PAINT_WORKFLOW, REGIONAL_WORKFLOW, CHARACTER_SHEET_WORKFLOW, PROMPT_CREATOR_WORKFLOW]


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
    assert len(scene["widgets_values"]) == 13
    assert len(subject["widgets_values"]) == 13
    assert scene["widgets_values"][-2:] == ["global", ""]
    assert subject["widgets_values"][-2:] == ["global", ""]
    assert scene["widgets_values"][6:10] == [True, True, 768, "lanczos"]
    assert subject["widgets_values"][6:10] == [True, True, 768, "lanczos"]
    assert scene["widgets_values"][10] == "This is the scene image."
    assert subject["widgets_values"][10] == "This is the subject image."

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


def test_paint_workflow_uses_combined_prepare_restore_and_anypaint_runtime():
    workflow = json.loads(PAINT_WORKFLOW.read_text(encoding="utf-8"))

    geometry_nodes = _nodes_by_type(workflow, "CcCKrea2PaintGeometry")
    prepare_nodes = _nodes_by_type(workflow, "CcCKrea2PaintPrepare")
    paint_nodes = _nodes_by_type(workflow, "CcCKrea2Paint")
    restore_nodes = _nodes_by_type(workflow, "CcCKrea2PaintRestore")
    lora_nodes = _nodes_by_type(workflow, "CcCKrea2LoRAStack")
    sampler_nodes = _nodes_by_type(workflow, "KSampler")

    assert len(geometry_nodes) == 0
    assert len(prepare_nodes) == 1
    assert len(paint_nodes) == 1
    assert len(restore_nodes) == 1
    assert len(lora_nodes) == 1
    assert len(sampler_nodes) == 1

    prepare = prepare_nodes[0]
    paint = paint_nodes[0]
    restore = restore_nodes[0]
    lora = lora_nodes[0]
    sampler = sampler_nodes[0]

    assert prepare["widgets_values"] == [
        "pad",
        "edge",
        "center",
        "center",
        0,
        0,
        0,
        0,
        False,
        12,
        "gaussian_sigma",
        4.0,
        "outside",
    ]
    assert paint["widgets_values"][1:] == [True, True]
    assert lora["widgets_values"][3] == "krea2_anypaint_rank32.safetensors"
    assert sampler["widgets_values"][2:7] == [8, 1.0, "euler", "simple", 1.0]

    prepare_inputs = {item["name"]: item for item in prepare["inputs"]}
    paint_inputs = {item["name"]: item for item in paint["inputs"]}
    restore_inputs = {item["name"]: item for item in restore["inputs"]}

    assert prepare_inputs["image"]["link"] is not None
    assert prepare_inputs["mask"]["link"] is not None
    assert prepare_inputs["vae"]["link"] is not None
    assert paint_inputs["paint_context"]["type"] == "KREA2_PAINT_CONTEXT"
    assert paint_inputs["paint_context"]["link"] is not None
    assert restore_inputs["paint_geometry"]["link"] is not None
    assert restore_inputs["generated_mask"]["link"] is not None

    outputs = {output["name"]: output for output in prepare["outputs"]}
    assert outputs["latent"]["type"] == "LATENT"
    assert outputs["paint_geometry"]["type"] == "KREA2_PAINT_GEOMETRY"


def test_regional_attention_workflow_wires_tags_through_latent():
    workflow = json.loads(REGIONAL_WORKFLOW.read_text(encoding="utf-8"))

    regions = _nodes_by_type(workflow, "CcCKrea2AttentionRegion")
    visuals = _nodes_by_type(workflow, "CcCKrea2VisualReference")
    latent = _nodes_by_type(workflow, "CcCKrea2Latent")[0]

    assert len(regions) == 2
    assert [node["widgets_values"][0] for node in regions] == ["woman", "man"]
    assert len(visuals) == 3

    woman = next(node for node in visuals if node.get("title") == "Woman Regional Reference")
    man = next(node for node in visuals if node.get("title") == "Man Regional Reference")
    assert woman["widgets_values"][10] == "This is the woman reference image."
    assert man["widgets_values"][10] == "This is the man reference image."
    assert woman["widgets_values"][-2:] == ["only in region", "woman"]
    assert man["widgets_values"][-2:] == ["only in region", "man"]

    latent_inputs = {item["name"]: item for item in latent["inputs"]}
    assert latent_inputs["attention_regions"]["type"] == "KREA2_ATTENTION_REGION_CHAIN"
    assert latent_inputs["attention_regions"]["link"] is not None
    assert latent["widgets_values"][0] == "from_image"
    assert latent["widgets_values"][5:8] == ["from_image", "contain", "lanczos"]


def test_character_sheet_workflow_builds_one_identity_reference_from_generic_views():
    workflow = json.loads(CHARACTER_SHEET_WORKFLOW.read_text(encoding="utf-8"))

    sheets = _nodes_by_type(workflow, "CcCKrea2CharacterSheet")
    visuals = _nodes_by_type(workflow, "CcCKrea2VisualReference")
    previews = _nodes_by_type(workflow, "PreviewImage")
    latent = _nodes_by_type(workflow, "CcCKrea2Latent")[0]

    assert len(sheets) == 1
    assert len(visuals) == 2
    assert len(previews) == 1

    sheet = sheets[0]
    assert sheet["widgets_values"] == [
        "4 portraits + 1 body",
        "1024 x 1024 | 1:1 | ~1.05 MP",
        "lanczos",
        8,
        8,
        "white",
    ]

    sheet_inputs = {item["name"]: item for item in sheet["inputs"]}
    assert set(sheet_inputs) == {
        "portrait_1",
        "portrait_2",
        "portrait_3",
        "portrait_4",
        "body_1",
    }
    assert all(item["link"] is not None for item in sheet_inputs.values())

    subject = next(
        node for node in visuals
        if node.get("title") == "Character Sheet Visual Reference"
    )
    assert subject["widgets_values"][:6] == [
        4.0,
        "native",
        "inside",
        "center",
        "center",
        "lanczos",
    ]
    assert subject["widgets_values"][6:10] == [True, True, 1024, "lanczos"]
    assert subject["widgets_values"][10] == "This is the subject image."
    assert subject["widgets_values"][-2:] == ["global", ""]

    subject_inputs = {item["name"]: item for item in subject["inputs"]}
    assert subject_inputs["image"]["link"] is not None
    assert subject_inputs["previous_references"]["link"] is not None

    latent_inputs = {item["name"]: item for item in latent["inputs"]}
    assert latent_inputs["dimensions_image"]["link"] is not None
    assert latent["widgets_values"][0] == "from_image"
    assert latent["widgets_values"][5] == "empty"


def test_prompt_creator_workflow_wires_optional_creator_and_disabled_semantic_branch():
    workflow = json.loads(PROMPT_CREATOR_WORKFLOW.read_text(encoding="utf-8"))

    creators = _nodes_by_type(workflow, "CcCKrea2EditPromptCreator")
    semantics = _nodes_by_type(workflow, "CcCKrea2SemanticReference")
    visuals = _nodes_by_type(workflow, "CcCKrea2VisualReference")
    edits = _nodes_by_type(workflow, "CcCKrea2Edit")

    assert len(creators) == 1
    assert len(semantics) == 1
    assert len(visuals) == 1
    assert len(edits) == 1

    creator = creators[0]
    semantic = semantics[0]
    edit = edits[0]

    assert creator["widgets_values"] == [
        "create_from_image",
        "Use the subject from Image 1 in the situation shown by the reference edit image.",
        512,
        0.25,
        0.9,
        0,
    ]
    creator_inputs = {item["name"]: item for item in creator["inputs"]}
    assert creator_inputs["clip"]["link"] is not None
    assert creator_inputs["visual_references"]["link"] is not None
    assert creator_inputs["reference_edit_image"]["link"] is not None

    latent = _nodes_by_type(workflow, "CcCKrea2Latent")[0]
    latent_inputs = {item["name"]: item for item in latent["inputs"]}
    assert latent["widgets_values"][0] == "from_image"
    assert latent["widgets_values"][5] == "empty"
    assert latent_inputs["dimensions_image"]["link"] is not None
    assert "content_image" not in latent_inputs

    edit_inputs = {item["name"]: item for item in edit["inputs"]}
    assert edit_inputs["positive_prompt"]["type"] == "STRING"
    assert edit_inputs["positive_prompt"]["link"] is not None
    assert edit_inputs["visual_references"]["link"] is not None
    assert edit_inputs["semantic_references"]["link"] is not None

    assert semantic["mode"] == 2
    semantic_inputs = {item["name"]: item for item in semantic["inputs"]}
    assert semantic_inputs["image"]["link"] is not None
    assert semantic["outputs"][0]["links"]


def test_all_workflow_link_types_are_consistent():
    for workflow_path in (EDIT_WORKFLOW, PAINT_WORKFLOW, REGIONAL_WORKFLOW, CHARACTER_SHEET_WORKFLOW, PROMPT_CREATOR_WORKFLOW):
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}

        for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
            source = nodes[source_id]["outputs"][source_slot]
            target = nodes[target_id]["inputs"][target_slot]
            assert source["type"] == link_type
            assert target["type"] == link_type
            assert link_id in (source.get("links") or [])
            assert target["link"] == link_id


def test_all_workflows_use_only_current_public_ccc_nodes():
    removed = {
        "CcCKrea2PaintGeometry",
        "CcCKrea2ReferenceCacheCreate",
        "CcCKrea2ReferenceCacheSave",
        "CcCKrea2ReferenceCacheLoad",
        "CcCKrea2CachedVisualReference",
    }

    from ccc_krea2.nodes import NODE_CLASS_MAPPINGS

    for workflow_path in (
        EDIT_WORKFLOW,
        PAINT_WORKFLOW,
        REGIONAL_WORKFLOW,
        CHARACTER_SHEET_WORKFLOW,
        PROMPT_CREATOR_WORKFLOW,
    ):
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        types = {node["type"] for node in workflow["nodes"]}
        assert not (types & removed)
        for type_name in types:
            if type_name.startswith("CcCKrea2"):
                assert type_name in NODE_CLASS_MAPPINGS


def test_character_sheet_workflow_uses_generic_reference_filenames():
    workflow = json.loads(CHARACTER_SHEET_WORKFLOW.read_text(encoding="utf-8"))
    loads = {
        node["id"]: node["widgets_values"][0]
        for node in _nodes_by_type(workflow, "LoadImage")
        if node["id"] in {6, 7, 8, 9, 10}
    }
    assert loads == {
        6: "character_portrait_1.jpg",
        7: "character_portrait_2.jpg",
        8: "character_portrait_3.jpg",
        9: "character_portrait_4.jpg",
        10: "character_body_1.jpg",
    }
