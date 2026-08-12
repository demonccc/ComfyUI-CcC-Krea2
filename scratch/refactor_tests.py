import sys

def run():
    with open("tests/test_workflow_files.py", "r", encoding="utf-8") as f:
        content = f.read()
        
    # We want to replace test_strict_workflow_schema up to the start of test_easy_workflows_sockets_and_presets
    import re
    
    # Extract test_strict_workflow_schema body
    start_idx = content.find("def test_strict_workflow_schema():")
    end_idx = content.find("def test_easy_workflows_sockets_and_presets():")
    
    new_validator = """
def validate_workflow_schema(wf, filename="<workflow>"):
    nodes_by_id = {n["id"]: n for n in wf.get("nodes", [])}
    links_by_id = {}
    
    for link in wf.get("links", []):
        assert len(link) == 6, f"Invalid link format in {filename}: {link}"
        link_id, from_id, from_slot, to_id, to_slot, link_type = link
        links_by_id[link_id] = {
            "source_node_id": from_id,
            "source_slot": from_slot,
            "target_node_id": to_id,
            "target_slot": to_slot,
            "type": link_type,
        }
        assert from_id in nodes_by_id, f"Link source node {from_id} not found in {filename}"
        assert to_id in nodes_by_id, f"Link target node {to_id} not found in {filename}"

        from_node = nodes_by_id[from_id]
        to_node = nodes_by_id[to_id]

        assert from_slot < len(from_node.get("outputs", [])), f"Link source slot {from_slot} out of bounds on node {from_id} in {filename}"
        assert to_slot < len(to_node.get("inputs", [])), f"Link target slot {to_slot} out of bounds on node {to_id} in {filename}"

        out_type = from_node["outputs"][from_slot]["type"]
        in_type = to_node["inputs"][to_slot]["type"]

        # Strict independent assertions
        if out_type != "*":
            assert link_type == out_type, f"Link type mismatch on source in {filename}: {link_type} != {out_type} on {from_id}->{to_id}"
        if in_type != "*":
            assert link_type == in_type, f"Link type mismatch on destination in {filename}: {link_type} != {in_type} on {from_id}->{to_id}"

        # Record that the socket is connected for required checking
        to_node.setdefault("_connected_inputs", set()).add(to_node["inputs"][to_slot]["name"])

    # Check for unique IDs and maximum match
    link_ids = [link[0] for link in wf.get("links", [])]
    assert len(link_ids) == len(set(link_ids)), f"Duplicate link IDs in {filename}"
    if link_ids:
        assert max(link_ids) == wf.get("last_link_id"), f"last_link_id mismatch in {filename}"

    node_ids = [n["id"] for n in wf.get("nodes", [])]
    assert len(node_ids) == len(set(node_ids)), f"Duplicate node IDs in {filename}"
    if node_ids:
        assert max(node_ids) == wf.get("last_node_id"), f"last_node_id mismatch in {filename}"

    used_link_ids = set()
    target_side_link_ids = set()
    source_side_link_ids = set()
    
    for node in wf.get("nodes", []):
        for i, inp in enumerate(node.get("inputs", [])):
            link_id = inp.get("link")
            if link_id is not None:
                assert link_id in links_by_id, f"Node {node['id']} refers to missing link {link_id}"
                assert links_by_id[link_id]["target_node_id"] == node["id"]
                assert links_by_id[link_id]["target_slot"] == i
                used_link_ids.add(link_id)
                target_side_link_ids.add(link_id)

        for i, out in enumerate(node.get("outputs", [])):
            for link_id in out.get("links") or []:
                assert link_id in links_by_id, f"Node {node['id']} refers to missing link {link_id}"
                assert links_by_id[link_id]["source_node_id"] == node["id"]
                assert links_by_id[link_id]["source_slot"] == i
                used_link_ids.add(link_id)
                source_side_link_ids.add(link_id)

        ntype = node.get("type")
        if not ntype:
            continue

        cls = get_node_class(ntype)
        if not cls:
            continue

        if ntype.startswith("CcCKrea2"):
            assert node.get("properties", {}).get("Node name for S&R") == ntype, f"Missing S&R name on {ntype}"

        in_types = cls.INPUT_TYPES()
        req = in_types.get("required", {})
        opt = in_types.get("optional", {})
        all_in = {**req, **opt}

        # Enforce required non-widget sockets
        for name, schema_val in req.items():
            val_type = schema_val[0]
            is_widget = isinstance(val_type, list) or isinstance(val_type, tuple) or val_type in ["STRING", "INT", "FLOAT", "BOOLEAN"]
            if not is_widget:
                assert name in node.get("_connected_inputs", set()), f"Required socket '{name}' is missing or unconnected on node {ntype} in {filename}"

        # Check linked input types match the schema socket types (they shouldn't be widgets)
        for inp in node.get("inputs", []):
            name = inp.get("name")
            assert name in all_in, f"Invalid input '{name}' on node {ntype} in {filename}. Allowed: {list(all_in.keys())}"
            expected_type = all_in[name][0]
            assert not (isinstance(expected_type, list) or expected_type in ["STRING", "INT", "FLOAT", "BOOLEAN"]), f"Input '{name}' is serialized as a linked input but the schema defines it as a widget on {ntype}"
            assert inp.get("type") == expected_type, f"Input '{name}' type mismatch on {ntype}. Expected {expected_type}, got {inp.get('type')}"
            # Assert link exists in link table
            assert inp.get("link") in link_ids, f"Input link {inp.get('link')} not found in links table in {filename}"

        for out in node.get("outputs", []):
            for link_id in out.get("links", []):
                assert link_id in link_ids, f"Output link {link_id} not found in links table in {filename}"

        # Validate widget count matches schema exactly
        widget_count_schema = sum(1 for name, schema_val in all_in.items() if isinstance(schema_val[0], list) or isinstance(schema_val[0], tuple) or schema_val[0] in ["STRING", "INT", "FLOAT", "BOOLEAN"])

        # Account for frontend-only injected widgets
        frontend_extras = 0
        if ntype == "KSampler":
            frontend_extras = 1
        widget_count_schema += frontend_extras

        widgets = node.get("widgets_values", [])
        assert len(widgets) == widget_count_schema, f"Widget count mismatch on {ntype}. Expected {widget_count_schema}, got {len(widgets)}"

        # Explicit frontend serialization tests
        if ntype == "KSampler":
            assert widgets == [0, "randomize", 20, 1.0, "euler", "normal", 1.0], f"KSampler widgets do not match frontend spec in {filename}: {widgets}"
        elif ntype == "LoadImage":
            assert len(widgets) == 1, f"LoadImage must have exactly 1 widget, got {widgets} in {filename}"
            assert isinstance(widgets[0], str), "LoadImage widget must be a string filename"
        elif ntype == "CLIPLoader":
            assert widgets[1] == "krea2", f"CLIPLoader type must be 'krea2', got {widgets} in {filename}"

        out_types = getattr(cls, "RETURN_TYPES", tuple())
        out_names = getattr(cls, "RETURN_NAMES", out_types)

        outputs = node.get("outputs", [])
        assert len(outputs) == len(out_types), f"Output count mismatch on {ntype}"

        for i, out in enumerate(outputs):
            name = out_names[i] if i < len(out_names) else out_types[i]
            assert out.get("name") == name, f"Output name mismatch on {ntype}, expected {name}, got {out.get('name')}"
            assert out.get("type") == out_types[i], f"Output type mismatch on {ntype}, expected {out_types[i]}, got {out.get('type')}"

    # EVERY GLOBAL LINK MUST EXIST ON BOTH SIDES
    assert set(links_by_id) == target_side_link_ids, f"Not all global links have target endpoints in {filename}"
    assert set(links_by_id) == source_side_link_ids, f"Not all global links have source endpoints in {filename}"


def test_strict_workflow_schema():
    for filename in MODERN_CANONICAL:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)
        validate_workflow_schema(wf, filename)

"""
    content = content[:start_idx] + new_validator + content[end_idx:]
    
    # Now replace test_orphan_link_rejection
    start_idx = content.find("def test_orphan_link_rejection():")
    end_idx = content.find("def test_generator_matches_checked_in_canonical_workflows(tmp_path):")
    
    new_corruption = """
def test_orphan_link_rejection():
    import json
    import copy
    import pytest
    from test_workflow_files import MODERN_CANONICAL
    
    with open(f"workflows/{MODERN_CANONICAL[0]}", "r", encoding="utf-8") as f:
        wf = json.load(f)
        
    # Case A: Change one input["link"] to another existing but WRONG link ID.
    wf_a = copy.deepcopy(wf)
    link_ids = [link[0] for link in wf_a["links"]]
    for node in wf_a["nodes"]:
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                other_link = next(lnk for lnk in link_ids if lnk != inp["link"])
                inp["link"] = other_link
                break
        else:
            continue
        break
        
    with pytest.raises(AssertionError):
        validate_workflow_schema(wf_a, "Case A")
        
    # Case B: Add an unused link to the global links table
    wf_b = copy.deepcopy(wf)
    wf_b["links"].append([9999, 1, 0, 2, 0, "TEST"])
    with pytest.raises(AssertionError):
        validate_workflow_schema(wf_b, "Case B")
    
    # Case C: Put the same destination input behind two global links.
    wf_c = copy.deepcopy(wf)
    valid_link = wf_c["links"][0]
    new_link = list(valid_link)
    new_link[0] = 9999
    wf_c["links"].append(new_link)
    with pytest.raises(AssertionError):
        validate_workflow_schema(wf_c, "Case C")

    # Case D: MISSING SOURCE DECLARATION
    wf_d = copy.deepcopy(wf)
    valid_link = wf_d["links"][0]
    link_id = valid_link[0]
    source_node_id = valid_link[1]
    source_slot = valid_link[2]
    for node in wf_d["nodes"]:
        if node["id"] == source_node_id:
            out_links = node["outputs"][source_slot].get("links", [])
            if link_id in out_links:
                out_links.remove(link_id)
    with pytest.raises(AssertionError):
        validate_workflow_schema(wf_d, "Case D")

    # Case E: MISSING TARGET DECLARATION
    wf_e = copy.deepcopy(wf)
    valid_link = wf_e["links"][0]
    link_id = valid_link[0]
    target_node_id = valid_link[3]
    target_slot = valid_link[4]
    for node in wf_e["nodes"]:
        if node["id"] == target_node_id:
            node["inputs"][target_slot]["link"] = None
    with pytest.raises(AssertionError):
        validate_workflow_schema(wf_e, "Case E")

def test_validator_regression_all_nodes():
    import json
    import copy
    import pytest
    from test_workflow_files import MODERN_CANONICAL
    
    with open(f"workflows/{MODERN_CANONICAL[0]}", "r", encoding="utf-8") as f:
        wf = json.load(f)
        
    wf_mutated = copy.deepcopy(wf)
    
    # Mutate a middle node (not the last one)
    # The last node is SaveImage, we'll mutate LoadImage or KSampler
    for node in wf_mutated["nodes"]:
        if node["type"] == "LoadImage":
            node["outputs"][0]["type"] = "INVALID_TYPE"
            break
            
    with pytest.raises(AssertionError):
        validate_workflow_schema(wf_mutated, "All nodes regression")

"""
    content = content[:start_idx] + new_corruption + content[end_idx:]
    
    with open("tests/test_workflow_files.py", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    run()
