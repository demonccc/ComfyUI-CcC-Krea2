import json
from pathlib import Path
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS

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

    assert expected_files.issubset(found_files), f"Missing canonical files. Found: {found_files}, Expected: {expected_files}"

def _get_nodes_by_type(workflow, type_name):
    return [n for n in workflow.get("nodes", []) if n.get("type") == type_name]

def _is_connected(workflow, node, input_name):
    inputs = node.get("inputs", [])
    for inp in inputs:
        if inp.get("name") == input_name and inp.get("link") is not None:
            return True
    return False

def test_strict_workflow_schema():
    for filename in MODERN_CANONICAL:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)

        for node in wf.get("nodes", []):
            ntype = node.get("type")
            if not ntype or not ntype.startswith("CcCKrea2"):
                continue

            assert ntype in NODE_CLASS_MAPPINGS, f"Unknown CcCKrea2 node type: {ntype}"
            cls = NODE_CLASS_MAPPINGS[ntype]

            assert node.get("properties", {}).get("Node name for S&R") == ntype, f"Missing S&R name on {ntype}"

            in_types = cls.INPUT_TYPES()
            req = in_types.get("required", {})
            opt = in_types.get("optional", {})
            all_in = {**req, **opt}

            for inp in node.get("inputs", []):
                name = inp.get("name")
                assert name in all_in, f"Invalid input '{name}' on node {ntype} in {filename}. Allowed: {list(all_in.keys())}"

            out_types = getattr(cls, "RETURN_TYPES", tuple())
            out_names = getattr(cls, "RETURN_NAMES", out_types)

            outputs = node.get("outputs", [])
            assert len(outputs) == len(out_types), f"Output count mismatch on {ntype}"

            for i, out in enumerate(outputs):
                name = out_names[i] if i < len(out_names) else out_types[i]
                assert out.get("name") == name, f"Output name mismatch on {ntype}, expected {name}, got {out.get('name')}"
                assert out.get("type") == out_types[i], f"Output type mismatch on {ntype}, expected {out_types[i]}, got {out.get('type')}"

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

        widgets = edit.get("widgets_values", [])
        preset = widgets[0]
        outfit_source = widgets[1]

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
            assert widgets[3] is True # apply_ostris_edit_patch
            assert widgets[4] is False # ostris_kv_cache

def test_advanced_workflows():
    for filename in MODERN_CANONICAL[7:10]:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)

        edit_nodes = _get_nodes_by_type(wf, "CcCKrea2Edit")
        assert len(edit_nodes) == 1
        edit = edit_nodes[0]

        ref_method = edit.get("widgets_values", [])[0]
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
        if not loras:
            continue
        lora = loras[0]
        widgets = lora.get("widgets_values", [])
        assert isinstance(widgets[0], bool), "enabled must be bool"
        assert isinstance(widgets[1], float), "global_strength must be float"
        assert isinstance(widgets[2], bool), "lora_1_enabled must be bool"
        assert isinstance(widgets[3], str), "lora_1_name must be str"
        assert isinstance(widgets[4], float), "lora_1_strength must be float"
