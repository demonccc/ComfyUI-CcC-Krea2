"""Structural contracts for the seven integrated example workflows.

These tests do not assert sampling quality: that needs Krea2/SAM/Qwen model
weights and a running ComfyUI instance.
"""

import json
from pathlib import Path

import pytest


WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / "workflows"
SCENARIOS = (
    "01_creative_generation.json",
    "02_outpainting_reframing.json",
    "03_local_image_editing.json",
    "04_identity_scene_transfer.json",
    "05_character_consistency.json",
    "06_outfit_transfer.json",
    "07_scene_composition.json",
)


@pytest.mark.parametrize("filename", SCENARIOS)
def test_showcase_graph_links_and_nodes(filename):
    graph = json.loads((WORKFLOWS_DIR / filename).read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}

    assert len(nodes) == len(graph["nodes"]), "Duplicate node IDs"
    assert len(links) == len(graph["links"]), "Duplicate link IDs"
    assert nodes and links
    assert graph["last_node_id"] >= max(nodes)
    assert graph["last_link_id"] >= max(links)

    for link_id, source_id, source_slot, dest_id, dest_slot, dtype in graph["links"]:
        assert isinstance(dest_slot, int), "Legacy named slot in links"
        source, dest = nodes[source_id], nodes[dest_id]
        output = source["outputs"][source_slot]
        input_ = dest["inputs"][dest_slot]
        assert input_["link"] == link_id
        assert link_id in output["links"]
        assert input_["type"] in (dtype, "*"), (filename, link_id)
        assert output["type"] == dtype

    for node in nodes.values():
        for input_ in node.get("inputs", []):
            assert input_.get("link") is None or input_["link"] in links
        for output in node.get("outputs", []):
            for link_id in output.get("links") or []:
                assert link_id in links


@pytest.mark.parametrize("filename", SCENARIOS)
def test_showcase_uses_current_custom_controls(filename):
    graph = json.loads((WORKFLOWS_DIR / filename).read_text(encoding="utf-8"))
    nodes = graph["nodes"]
    for node in nodes:
        if node["type"] == "CcCKrea2Reference":
            values = node["widgets_values"]
            assert len(values) == 16
            assert -5.0 <= values[1] <= 5.0
            assert values[14] in ("visual", "semantic")
        if node["type"] == "CcCKrea2Mask":
            values = node["widgets_values"]
            assert len(values) == 7
            assert -5.0 <= values[2] <= 5.0
            assert isinstance(values[6], bool)  # invert (appended for backward compatibility)
            assert node["outputs"][3]["name"] == "composite_image"
        if node["type"] == "CcCKrea2Latent":
            values = node["widgets_values"]
            assert len(values) == 14
            assert values[8] in (True, False)  # latent_vision_resize
            assert values[11] in ("visual", "semantic")


def test_studio_workflow_paints_background_not_subject():
    graph = json.loads(
        (WORKFLOWS_DIR / SCENARIOS[2]).read_text(encoding="utf-8")
    )
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes[62]["type"] == "InvertMask"
    paint_prepare = next(n for n in nodes.values() if n["type"] == "CcCKrea2PaintPrepare")
    mask_input = next(i for i in paint_prepare["inputs"] if i["name"] == "mask")
    edge = next(l for l in graph["links"] if l[0] == mask_input["link"])
    assert edge[1] == 62


def test_creative_t2i_uses_native_prompt_input():
    graph = json.loads(
        (WORKFLOWS_DIR / SCENARIOS[0]).read_text(encoding="utf-8")
    )
    creator = next(n for n in graph["nodes"] if n["type"] == "CcCKrea2EditPromptCreator")
    t2i = next(n for n in graph["nodes"] if n["type"] == "CcCKrea2TextToImage")
    prompt_input = next(i for i in t2i["inputs"] if i["name"] == "prompt")
    edge = next(l for l in graph["links"] if l[0] == prompt_input["link"])
    assert edge[1] == creator["id"] and edge[2] == 0


def test_character_identity_mask_has_final_priority():
    graph = json.loads(
        (WORKFLOWS_DIR / SCENARIOS[4]).read_text(encoding="utf-8")
    )
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes[71]["widgets_values"][1] == "body"
    assert nodes[72]["widgets_values"][1] == "face,hair"
    assert nodes[72]["widgets_values"][2] > nodes[71]["widgets_values"][2]
