"""Structural validation tests for example workflow files."""

import json
from pathlib import Path

EXPECTED_WORKFLOW_FILES = [
    "workflows/subject_edit.json",
    "workflows/subject_scene.json",
    "workflows/subject_outfit.json",
    "workflows/subject_outfit_scene.json",
    "workflows/subject_scene_qwen_simple.json",
    "workflows/subject_scene_outfit_qwen_simple.json",
]

CCC_MAIN_NODE_TYPES = {
    "CcCKrea2Subject",
    "CcCKrea2SubjectOutfit",
    "CcCKrea2SubjectScene",
    "CcCKrea2SubjectSceneOutfit",
    "CcCKrea2Inpaint",
    "CcCKrea2InpaintSubjectOutfit",
    "CcCKrea2InpaintSubjectScene",
}


def test_workflow_files_exist_and_parse():
    """1. The six workflow files exist and parse as valid JSON."""
    for rel_path in EXPECTED_WORKFLOW_FILES:
        path = Path(rel_path)
        assert path.exists(), f"Missing workflow file: {rel_path}"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict), f"Invalid JSON structure in {rel_path}"
        assert "nodes" in data, f"Workflow {rel_path} missing 'nodes' array"
        assert "groups" in data, f"Workflow {rel_path} missing 'groups' array"


def test_workflow_contains_ccc_main_node():
    """2. Each workflow contains at least one CcC Krea2 main node."""
    for rel_path in EXPECTED_WORKFLOW_FILES:
        with open(rel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        node_types = {node.get("type") for node in data.get("nodes", [])}
        main_nodes_found = node_types.intersection(CCC_MAIN_NODE_TYPES)
        assert len(main_nodes_found) > 0, f"No CcC Krea2 main node found in {rel_path}"


def test_workflow_contains_advanced_settings_group():
    """3. Each workflow contains an Advanced Settings group."""
    for rel_path in EXPECTED_WORKFLOW_FILES:
        with open(rel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        group_titles = [g.get("title", "") for g in data.get("groups", [])]
        has_adv_group = any("Advanced Settings" in title for title in group_titles)
        assert has_adv_group, f"Advanced Settings group missing in {rel_path} (found: {group_titles})"


def test_qwen_workflows_contain_qwen_vlm_node():
    """4. The two Qwen workflows contain a node indicating Simple Qwen-VL Vision Language Model."""
    qwen_files = [
        "workflows/subject_scene_qwen_simple.json",
        "workflows/subject_scene_outfit_qwen_simple.json",
    ]

    for rel_path in qwen_files:
        with open(rel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        found_qwen = False
        for node in data.get("nodes", []):
            node_type = str(node.get("type", ""))
            node_title = str(node.get("properties", {}).get("Node name for S&R", "")) or str(node.get("title", ""))
            if "Simple Qwen-VL" in node_type or "Simple Qwen-VL" in node_title or "SimpleQwenVL" in node_type:
                found_qwen = True
                break

        assert found_qwen, f"Qwen VLM node missing in {rel_path}"


def test_expected_filenames_match_directory_contents():
    """5. The workflow directory contains exactly the six expected workflow files."""
    workflows_dir = Path("workflows")
    assert workflows_dir.exists(), "workflows directory does not exist"

    found_files = set(str(p).replace("\\", "/") for p in workflows_dir.glob("*.json"))
    expected_files = set(EXPECTED_WORKFLOW_FILES)

    assert found_files == expected_files, f"Mismatch in workflow files: found {found_files}, expected {expected_files}"
