"""Static checks for checked-in ComfyUI workflows.

No GPU/ComfyUI runtime is needed to catch stale node IDs, broken graph links
or deprecated ten-slot LoRA serialization.
"""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1] / "workflows"
FILES = sorted(ROOT.glob("*.json"))
RETIRED_TYPES = {
    "CcCKrea2VisualReference",
    "CcCKrea2VisionReference",
    "CcCKrea2AttentionRegion",
    "CcCKrea2ReferenceCache",
}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_workflow_graph_contract(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in data["nodes"]}
    links = {link[0]: link for link in data["links"]}

    for node in nodes.values():
        assert node["type"] not in RETIRED_TYPES, (path, node["type"])
        if node["type"] in {"CcCKrea2LoRAStack", "CcCKrea2LoRALoader"}:
            assert len(node.get("widgets_values", ())) == 15, (path, node["id"])
        if node["type"] == "CcCKrea2Reference":
            assert len(node.get("widgets_values", ())) == 20, (path, node["id"])
        for index, socket in enumerate(node.get("inputs", ())):
            link_id = socket.get("link")
            if link_id is not None:
                assert link_id in links, (path, node["id"], socket)
                assert links[link_id][3:5] == [node["id"], index], (path, node["id"], socket)
        for index, socket in enumerate(node.get("outputs", ())):
            for link_id in socket.get("links") or ():
                assert link_id in links, (path, node["id"], link_id)
                assert links[link_id][1:3] == [node["id"], index], (path, node["id"], link_id)
    for link in links.values():
        _, src, output, dst, input_index, _ = link
        assert src in nodes and dst in nodes, (path, link)
        assert output < len(nodes[src].get("outputs", ())), (path, link)
        assert input_index < len(nodes[dst].get("inputs", ())), (path, link)
