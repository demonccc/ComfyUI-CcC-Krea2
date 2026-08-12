import json
import pytest
from pathlib import Path
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS

LEGACY_WORKFLOW_FILES = [
    "workflows/legacy/subject_edit.json",
    "workflows/legacy/subject_scene.json",
    "workflows/legacy/subject_outfit.json",
    "workflows/legacy/subject_outfit_scene.json",
    "workflows/legacy/subject_scene_qwen_simple.json",
    "workflows/legacy/subject_scene_outfit_qwen_simple.json",
    "workflows/legacy/text_to_image.json",
]

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
    
    found_files = set(str(p.name) for p in workflows_dir.glob("*.json"))
    expected_files = set(MODERN_CANONICAL)
    
    assert found_files == expected_files, f"Mismatch in canonical files. Found: {found_files}, Expected: {expected_files}"

def _get_nodes_by_type(workflow, type_name):
    return [n for n in workflow.get("nodes", []) if n.get("type") == type_name]

def _is_connected(workflow, node, input_name):
    inputs = node.get("inputs", [])
    for inp in inputs:
        if inp.get("name") == input_name and inp.get("link") is not None:
            return True
    return False

def test_easy_workflows_sockets_and_presets():
    for filename in MODERN_CANONICAL[:7]:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)
            
        is_ostris = "ostris" in filename
        edit_type = "CcCKrea2EasyEditOstris" if is_ostris else "CcCKrea2EasyEdit"
        edit_nodes = _get_nodes_by_type(wf, edit_type)
        assert len(edit_nodes) == 1, f"Missing {edit_type} in {filename}"
        edit = edit_nodes[0]
        
        assert edit.get("properties", {}).get("Node name for S&R") == edit_type, f"Wrong S&R name in {filename}"
        
        has_s = _is_connected(wf, edit, "subject_image")
        has_sc = _is_connected(wf, edit, "scene_image")
        has_o = _is_connected(wf, edit, "outfit_image")
        has_st = _is_connected(wf, edit, "style_image")
        
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
            assert widgets[5] is False # ostris_kv_cache

def test_advanced_workflows():
    for filename in MODERN_CANONICAL[7:10]:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)
            
        edit_nodes = _get_nodes_by_type(wf, "CcCKrea2Edit")
        assert len(edit_nodes) == 1
        edit = edit_nodes[0]
        
        assert edit.get("properties", {}).get("Node name for S&R") == "CcCKrea2Edit"
        
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

def test_clip_loader_and_lora():
    for filename in MODERN_CANONICAL:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)
            
        clips = _get_nodes_by_type(wf, "CLIPLoader")
        assert len(clips) == 1
        assert clips[0].get("widgets_values", [])[1] == "krea2"
        
        if "native" not in filename:
            loras = _get_nodes_by_type(wf, "CcCKrea2LoRAStack")
            assert len(loras) == 1
            assert loras[0].get("widgets_values", [])[0] is True # enabled

def test_groups():
    for filename in MODERN_CANONICAL:
        with open(f"workflows/{filename}", "r", encoding="utf-8") as f:
            wf = json.load(f)
        groups = wf.get("groups", [])
        assert len(groups) >= 3, f"Must have meaningful groups in {filename}"
        title_count = sum("CcC Krea2 Edit Pipeline" == g.get("title") for g in groups)
        assert title_count < len(groups), f"Must use meaningful names, not just CcC Krea2 Edit Pipeline for all"

