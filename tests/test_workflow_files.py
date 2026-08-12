import json
from pathlib import Path
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS


class MockUNETLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (["krea2_model.safetensors"],), "weight_dtype": (["default"],)}}

    RETURN_TYPES = ("MODEL",)


class MockCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (["qwen3_vl.safetensors"],),
                "type": (["krea2", "sdxl", "sd3"],),
                "device": (["default", "cpu"],),
            }
        }

    RETURN_TYPES = ("CLIP",)


class MockVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_name": (["ae.safetensors"],)}}

    RETURN_TYPES = ("VAE",)


class MockLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": (["subject.jpg", "scene.jpg", "outfit.jpg", "style.jpg"],)}}

    RETURN_TYPES = ("IMAGE", "MASK")


class MockKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 1.0}),
                "sampler_name": (["euler", "euler_ancestral"],),
                "scheduler": (["normal", "karras"],),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0}),
            }
        }

    RETURN_TYPES = ("LATENT",)


class MockVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE",)


class MockSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "CcCKrea2"})}}

    RETURN_TYPES = tuple()


try:
    import nodes as comfy_nodes

    CORE_MAPPINGS = comfy_nodes.NODE_CLASS_MAPPINGS
except ImportError:
    CORE_MAPPINGS = {
        "UNETLoader": MockUNETLoader,
        "CLIPLoader": MockCLIPLoader,
        "VAELoader": MockVAELoader,
        "LoadImage": MockLoadImage,
        "KSampler": MockKSampler,
        "VAEDecode": MockVAEDecode,
        "SaveImage": MockSaveImage,
    }


def get_node_class(node_type):
    if node_type in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS[node_type]
    if node_type in CORE_MAPPINGS:
        return CORE_MAPPINGS[node_type]
    return None


def get_serialized_widget_values_by_name(node, cls):
    in_types = cls.INPUT_TYPES()
    req = in_types.get("required", {})
    opt = in_types.get("optional", {})
    all_in = {**req, **opt}

    widgets = node.get("widgets_values", [])

    values_by_name = {}
    widget_idx = 0
    for name, schema_val in all_in.items():
        val_type = schema_val[0]
        is_widget = (
            isinstance(val_type, list)
            or isinstance(val_type, tuple)
            or val_type in ["STRING", "INT", "FLOAT", "BOOLEAN"]
        )
        if is_widget:
            if widget_idx < len(widgets):
                values_by_name[name] = widgets[widget_idx]
                widget_idx += 1
    return values_by_name


MODERN_CANONICAL = [
    "01_easy_subject.json",
    "02_easy_subject_scene.json",
    "03_easy_subject_outfit.json",
    "04_easy_subject_scene_outfit.json",
    "05_easy_outfit_from_scene.json",
    "06_easy_style_transfer.json",
    "07_easy_ostris.json",
    "08_advanced_krea2_edit.json",
    "09_advanced_native.json",
    "10_advanced_ostris.json",
]


def test_canonical_filenames():
    workflows_dir = Path("workflows")
    assert workflows_dir.exists()

    found_files = set(str(p.name) for p in workflows_dir.glob("*.json") if p.name.startswith(("0", "1")))
    expected_files = set(MODERN_CANONICAL)

    assert expected_files == found_files, f"Filename mismatch. Found: {found_files}, Expected: {expected_files}"


def _get_nodes_by_type(workflow, type_name):
    return [n for n in workflow.get("nodes", []) if n.get("type") == type_name]


def _is_connected(workflow, node, input_name):
    inputs = node.get("inputs", [])
    for inp in inputs:
        if inp.get("name") == input_name and inp.get("link") is not None:
            return True
    return False


def validate_workflow_schema(wf, filename="<workflow>"):
    nodes_by_id = {n["id"]: n for n in wf.get("nodes", [])}
    links_by_id = {}
    connected_inputs_by_node = {}

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

        assert from_slot < len(from_node.get("outputs", [])), (
            f"Link source slot {from_slot} out of bounds on node {from_id} in {filename}"
        )
        assert to_slot < len(to_node.get("inputs", [])), (
            f"Link target slot {to_slot} out of bounds on node {to_id} in {filename}"
        )

        out_type = from_node["outputs"][from_slot]["type"]
        in_type = to_node["inputs"][to_slot]["type"]

        # Strict independent assertions
        if out_type != "*":
            assert link_type == out_type, (
                f"Link type mismatch on source in {filename}: {link_type} != {out_type} on {from_id}->{to_id}"
            )
        if in_type != "*":
            assert link_type == in_type, (
                f"Link type mismatch on destination in {filename}: {link_type} != {in_type} on {from_id}->{to_id}"
            )

        # Record that the socket is connected for required checking
        connected_inputs_by_node.setdefault(to_id, set()).add(to_node["inputs"][to_slot]["name"])

    # Check for unique IDs and maximum match
    link_ids = [link[0] for link in wf.get("links", [])]
    assert len(link_ids) == len(set(link_ids)), f"Duplicate link IDs in {filename}"
    if link_ids:
        assert max(link_ids) == wf.get("last_link_id"), f"last_link_id mismatch in {filename}"

    node_ids = [n["id"] for n in wf.get("nodes", [])]
    assert len(node_ids) == len(set(node_ids)), f"Duplicate node IDs in {filename}"
    if node_ids:
        assert max(node_ids) == wf.get("last_node_id"), f"last_node_id mismatch in {filename}"

    target_side_link_ids = set()
    source_side_link_ids = set()

    for node in wf.get("nodes", []):
        for i, inp in enumerate(node.get("inputs", [])):
            link_id = inp.get("link")
            if link_id is not None:
                assert link_id in links_by_id, f"Node {node['id']} refers to missing link {link_id}"
                assert links_by_id[link_id]["target_node_id"] == node["id"]
                assert links_by_id[link_id]["target_slot"] == i
                target_side_link_ids.add(link_id)

        for i, out in enumerate(node.get("outputs", [])):
            for link_id in out.get("links") or []:
                assert link_id in links_by_id, f"Node {node['id']} refers to missing link {link_id}"
                assert links_by_id[link_id]["source_node_id"] == node["id"]
                assert links_by_id[link_id]["source_slot"] == i
                source_side_link_ids.add(link_id)

        ntype = node.get("type")
        assert ntype, f"Node {node['id']} is missing a type in {filename}"

        cls = get_node_class(ntype)
        assert cls is not None, f"Unknown node type '{ntype}' in {filename}"

        if ntype.startswith("CcCKrea2"):
            assert node.get("properties", {}).get("Node name for S&R") == ntype, f"Missing S&R name on {ntype}"

        in_types = cls.INPUT_TYPES()
        req = in_types.get("required", {})
        opt = in_types.get("optional", {})
        all_in = {**req, **opt}

        # Enforce required non-widget sockets
        for name, schema_val in req.items():
            val_type = schema_val[0]
            is_widget = (
                isinstance(val_type, list)
                or isinstance(val_type, tuple)
                or val_type in ["STRING", "INT", "FLOAT", "BOOLEAN"]
            )
            if not is_widget:
                assert name in connected_inputs_by_node.get(node["id"], set()), (
                    f"Required socket '{name}' is missing or unconnected on node {ntype} in {filename}"
                )

        # Check linked input types match the schema socket types (they shouldn't be widgets)
        for inp in node.get("inputs", []):
            name = inp.get("name")
            assert name in all_in, (
                f"Invalid input '{name}' on node {ntype} in {filename}. Allowed: {list(all_in.keys())}"
            )
            expected_type = all_in[name][0]
            assert not (isinstance(expected_type, list) or expected_type in ["STRING", "INT", "FLOAT", "BOOLEAN"]), (
                f"Input '{name}' is serialized as a linked input but the schema defines it as a widget on {ntype}"
            )
            assert inp.get("type") == expected_type, (
                f"Input '{name}' type mismatch on {ntype}. Expected {expected_type}, got {inp.get('type')}"
            )
            # Assert link exists in link table
            assert inp.get("link") in link_ids, f"Input link {inp.get('link')} not found in links table in {filename}"

        for out in node.get("outputs", []):
            for link_id in out.get("links") or []:
                assert link_id in link_ids, f"Output link {link_id} not found in links table in {filename}"

        # Validate widget count matches schema exactly
        widget_count_schema = sum(
            1
            for name, schema_val in all_in.items()
            if isinstance(schema_val[0], list)
            or isinstance(schema_val[0], tuple)
            or schema_val[0] in ["STRING", "INT", "FLOAT", "BOOLEAN"]
        )

        # Account for frontend-only injected widgets
        frontend_extras = 0
        if ntype == "KSampler":
            frontend_extras = 1
        widget_count_schema += frontend_extras

        widgets = node.get("widgets_values", [])
        assert len(widgets) == widget_count_schema, (
            f"Widget count mismatch on {ntype}. Expected {widget_count_schema}, got {len(widgets)}"
        )

        # Explicit frontend serialization tests
        if ntype == "KSampler":
            assert widgets == [0, "randomize", 20, 1.0, "euler", "normal", 1.0], (
                f"KSampler widgets do not match frontend spec in {filename}: {widgets}"
            )
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
            assert out.get("type") == out_types[i], (
                f"Output type mismatch on {ntype}, expected {out_types[i]}, got {out.get('type')}"
            )

    # EVERY GLOBAL LINK MUST EXIST ON BOTH SIDES
    assert set(links_by_id) == target_side_link_ids, f"Not all global links have target endpoints in {filename}"
    assert set(links_by_id) == source_side_link_ids, f"Not all global links have source endpoints in {filename}"


def test_strict_workflow_schema():
    for filename in MODERN_CANONICAL:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)
        validate_workflow_schema(wf, filename)


def test_easy_workflows_sockets_and_presets():
    for filename in MODERN_CANONICAL[:7]:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)

        is_ostris = "ostris" in filename
        edit_type = "CcCKrea2EasyEditOstris" if is_ostris else "CcCKrea2EasyEdit"
        edit_nodes = _get_nodes_by_type(wf, edit_type)
        assert len(edit_nodes) == 1, f"Missing {edit_type} in {filename}"
        edit = edit_nodes[0]

        has_s = _is_connected(wf, edit, "subject")
        has_sc = _is_connected(wf, edit, "scene")
        has_o = _is_connected(wf, edit, "outfit")
        has_st = _is_connected(wf, edit, "style")

        # Explicitly test that legacy sockets are NOT present
        assert not _is_connected(wf, edit, "subject_image"), f"Legacy socket subject_image present in {filename}"
        assert not _is_connected(wf, edit, "scene_image"), f"Legacy socket scene_image present in {filename}"
        assert not _is_connected(wf, edit, "outfit_image"), f"Legacy socket outfit_image present in {filename}"
        assert not _is_connected(wf, edit, "style_image"), f"Legacy socket style_image present in {filename}"
        assert not _is_connected(wf, edit, "prompt_augmentation"), f"Invalid prompt_augmentation present in {filename}"

        values = get_serialized_widget_values_by_name(edit, NODE_CLASS_MAPPINGS[edit_type])
        preset = values.get("preset")
        outfit_source = values.get("outfit_source")

        if filename == "01_easy_subject.json":
            assert has_s and not has_sc and not has_o and not has_st
        elif filename == "02_easy_subject_scene.json":
            assert has_s and has_sc and not has_o and not has_st
        elif filename == "03_easy_subject_outfit.json":
            assert has_s and not has_sc and has_o and not has_st
        elif filename == "04_easy_subject_scene_outfit.json":
            assert has_s and has_sc and has_o and not has_st
        elif filename == "05_easy_outfit_from_scene.json":
            assert has_s and has_sc and not has_o and not has_st
            assert preset == "outfit_transfer"
            assert outfit_source == "scene image"
        elif filename == "06_easy_style_transfer.json":
            assert has_s and not has_sc and not has_o and has_st
            assert preset == "style_transfer"
        elif filename == "07_easy_ostris.json":
            assert has_s and has_sc
            assert values.get("apply_ostris_edit_patch") is True
            assert values.get("ostris_kv_cache") is False


def test_advanced_workflows():
    for filename in MODERN_CANONICAL[7:10]:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)

        edit_nodes = _get_nodes_by_type(wf, "CcCKrea2Edit")
        assert len(edit_nodes) == 1
        edit = edit_nodes[0]

        values = get_serialized_widget_values_by_name(edit, NODE_CLASS_MAPPINGS["CcCKrea2Edit"])
        ref_method = values.get("reference_method")

        assert "apply_patch" not in values, "apply_patch must be removed from CcCKrea2Edit widgets"

        if "krea2_edit" in filename:
            assert ref_method == "krea2_edit"
        elif "native" in filename:
            assert ref_method == "native"
        elif "ostris" in filename:
            assert ref_method == "ostris_edit"

        gen_refs = _get_nodes_by_type(wf, "CcCKrea2ReferenceImage")
        assert len(gen_refs) >= 2

        for ntype in ["CcCKrea2SubjectImage", "CcCKrea2SceneImage", "CcCKrea2OutfitImage", "CcCKrea2StyleImage"]:
            assert len(_get_nodes_by_type(wf, ntype)) == 0, f"No legacy {ntype} allowed in canonical advanced"


def test_lora_serialization():
    for filename in MODERN_CANONICAL:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)

        loras = _get_nodes_by_type(wf, "CcCKrea2LoRAStack")

        if "09_advanced_native" in filename:
            assert not loras, f"Native workflow {filename} must not contain CcCKrea2LoRAStack"
            continue

        if not loras:
            continue
        lora = loras[0]
        values = get_serialized_widget_values_by_name(lora, NODE_CLASS_MAPPINGS["CcCKrea2LoRAStack"])
        assert isinstance(values.get("enabled"), bool), "enabled must be bool"
        assert isinstance(values.get("global_strength"), float), "global_strength must be float"
        assert isinstance(values.get("lora_1_enabled"), bool), "lora_1_enabled must be bool"
        assert isinstance(values.get("lora_1_name"), str), "lora_1_name must be str"
        assert isinstance(values.get("lora_1_strength"), float), "lora_1_strength must be float"


def test_generator_idempotency(tmp_path):
    import sys
    import os

    # Import the workflow generator
    sys.path.insert(0, str(Path(__file__).parent.parent / "scratch"))
    import build_canonical_workflows

    # Run generator for run1
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        build_canonical_workflows.main()
    finally:
        os.chdir(original_cwd)

    gen_dir = tmp_path / "workflows"
    run1_contents = {}
    for p in gen_dir.glob("*.json"):
        with open(p, "r") as f:
            run1_contents[p.name] = json.load(f)

    assert set(run1_contents.keys()) == set(MODERN_CANONICAL), "Run 1 produced unexpected extra or missing files"

    # Run generator for run2
    os.chdir(tmp_path)
    try:
        build_canonical_workflows.main()
    finally:
        os.chdir(original_cwd)

    run2_contents = {}
    for p in gen_dir.glob("*.json"):
        with open(p, "r") as f:
            run2_contents[p.name] = json.load(f)

    assert set(run1_contents.keys()) == set(run2_contents.keys()), "Run 2 file set differs from Run 1"

    for filename in MODERN_CANONICAL:
        assert run1_contents[filename] == run2_contents[filename], (
            f"Generator output for {filename} is not idempotent (Run 1 != Run 2)!"
        )


def test_builder_validation():
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scratch"))
    import build_canonical_workflows

    builder = build_canonical_workflows.WorkflowBuilder()

    # Test unknown widget rejection
    import pytest

    with pytest.raises(ValueError, match="Unknown widget values"):
        builder.add_node("CcCKrea2Edit", pos=[0, 0], size=[200, 200], widgets_values={"apply_patch": True})

    with pytest.raises(ValueError, match="Unknown widget values"):
        builder.add_node("CcCKrea2EasyEdit", pos=[0, 0], size=[200, 200], widgets_values={"some_fake_widget": 123})

    # Test strict combo validation bypass prevention
    with pytest.raises(ValueError, match="not in choices"):
        builder.add_node(
            "CcCKrea2Edit", pos=[0, 0], size=[200, 200], widgets_values={"reference_method": "invalid.safetensors"}
        )


def test_builder_duplicate_link_rejection():
    import copy
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scratch"))
    import build_canonical_workflows

    builder = build_canonical_workflows.WorkflowBuilder()
    n1 = builder.add_node("CcCKrea2TargetLatent", pos=[0, 0], size=[1, 1])
    n2 = builder.add_node("CcCKrea2Edit", pos=[0, 0], size=[1, 1], inputs=["target_latent"])

    # First link should succeed
    builder.link(n1, "target_latent", n2, "target_latent")

    links_before = copy.deepcopy(builder.links)
    source_links_before = copy.deepcopy(n1["outputs"][0]["links"])
    destination_link_before = n2["inputs"][0]["link"]

    # Second link to the same input should fail
    import pytest

    with pytest.raises(ValueError, match="already has a link"):
        builder.link(n1, "target_latent", n2, "target_latent")

    assert builder.links == links_before
    assert n1["outputs"][0]["links"] == source_links_before
    assert n2["inputs"][0]["link"] == destination_link_before


def test_orphan_link_rejection():
    import copy
    import pytest

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
    target_node_id = valid_link[3]
    target_slot = valid_link[4]
    for node in wf_e["nodes"]:
        if node["id"] == target_node_id:
            node["inputs"][target_slot]["link"] = None
    with pytest.raises(AssertionError):
        validate_workflow_schema(wf_e, "Case E")


def test_validator_regression_all_nodes():
    import copy
    import pytest

    with open(f"workflows/{MODERN_CANONICAL[0]}", "r", encoding="utf-8") as f:
        wf = json.load(f)

    # Case A: Unconnected output contract
    wf_a = copy.deepcopy(wf)
    for node in wf_a["nodes"]:
        if node["type"] == "CcCKrea2LoRAStack":
            for out in node["outputs"]:
                if out["name"] == "prompt_augmentation":
                    out["type"] = "INVALID_TYPE"
                    break
            break
    with pytest.raises(AssertionError):
        validate_workflow_schema(wf_a, "Case A")

    # Case B: Widget contract
    wf_b = copy.deepcopy(wf)
    for node in wf_b["nodes"]:
        if node["type"] == "KSampler":
            node["widgets_values"].pop()
            break
    with pytest.raises(AssertionError):
        validate_workflow_schema(wf_b, "Case B")

    # Case C: Unknown node class
    wf_c = copy.deepcopy(wf)
    for node in wf_c["nodes"]:
        if node["type"] == "LoadImage":
            node["type"] = "UnknownNode"
            break
    with pytest.raises(AssertionError):
        validate_workflow_schema(wf_c, "Case C")


def test_validate_workflow_schema_does_not_mutate_workflow():
    import copy

    with open(f"workflows/{MODERN_CANONICAL[0]}", "r", encoding="utf-8") as f:
        wf = json.load(f)

    before = copy.deepcopy(wf)

    validate_workflow_schema(wf, MODERN_CANONICAL[0])
    assert wf == before, "Workflow was mutated by validation on first run!"

    validate_workflow_schema(wf, MODERN_CANONICAL[0])
    assert wf == before, "Workflow was mutated by validation on second run!"


def test_validator_handles_none_links():
    import copy

    with open(f"workflows/{MODERN_CANONICAL[0]}", "r", encoding="utf-8") as f:
        wf = json.load(f)

    wf_mutated = copy.deepcopy(wf)
    found_unconnected = False
    for node in wf_mutated["nodes"]:
        for out in node.get("outputs", []):
            if not out.get("links"):
                out["links"] = None
                found_unconnected = True
                break
        if found_unconnected:
            break

    assert found_unconnected, "No unconnected output found to test None links!"
    # Should not raise exception
    validate_workflow_schema(wf_mutated, "None links test")


def test_generator_matches_checked_in_canonical_workflows(tmp_path):
    import sys
    import os

    sys.path.insert(0, str(Path(__file__).parent.parent / "scratch"))
    import build_canonical_workflows

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        build_canonical_workflows.main()
    finally:
        os.chdir(original_cwd)

    gen_dir = tmp_path / "workflows"
    for filename in MODERN_CANONICAL:
        with open(gen_dir / filename, "r", encoding="utf-8") as f:
            generated = json.load(f)
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            checked_in = json.load(f)

        assert generated == checked_in, (
            f"Generated {filename} does not match checked-in version! Please commit the newly generated workflows."
        )
