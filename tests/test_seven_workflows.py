"""Validation for the seven checked-in ComfyUI test graphs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "workflows"
EXPECTED = {
    "01_t2i_one_reference.json",
    "02_t2i_prompt_edit.json",
    "03_i2i_one_reference.json",
    "04_i2i_two_references.json",
    "05_i2i_three_references.json",
    "06_i2i_prompt_edit.json",
    "07_i2i_paint.json",
}

def test_only_seven_workflows():
    assert {p.name for p in WORKFLOW_DIR.glob("*.json")} == EXPECTED

def test_all_references_use_unified_public_node():
    for path in WORKFLOW_DIR.glob("*.json"):
        graph = json.loads(path.read_text(encoding="utf-8"))
        types = {n["type"] for n in graph["nodes"]}
        assert "CcCKrea2VisualReference" not in types, path.name
        assert "CcCKrea2VisionReference" not in types, path.name
        assert "CcCKrea2AttentionRegion" not in types, path.name

def test_graph_links_match_sockets():
    for path in WORKFLOW_DIR.glob("*.json"):
        graph = json.loads(path.read_text(encoding="utf-8"))
        nodes = {n["id"]: n for n in graph["nodes"]}
        links = {link[0]: link for link in graph["links"]}
        assert len(links) == len(graph["links"]), path.name
        for link_id, src_id, out_idx, dst_id, in_idx, kind in graph["links"]:
            assert src_id in nodes and dst_id in nodes, (path.name, link_id)
            src, dst = nodes[src_id], nodes[dst_id]
            assert src["outputs"][out_idx]["type"] == kind, (path.name, link_id)
            assert dst["inputs"][in_idx]["type"] == kind, (path.name, link_id)
            assert dst["inputs"][in_idx]["link"] == link_id, (path.name, link_id)
        for node in nodes.values():
            for inp in node.get("inputs", []):
                assert inp["link"] is None or inp["link"] in links, (path.name, node["id"])
            for out in node.get("outputs", []):
                for link_id in out.get("links", []) or []:
                    assert link_id in links, (path.name, node["id"])
