"""Structural validation tests for example workflow JSON files."""

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

EXPECTED_ROLES_PER_WORKFLOW = {
    "workflows/subject_edit.json": (["subject"], "CcCKrea2Subject"),
    "workflows/subject_scene.json": (["scene", "subject"], "CcCKrea2SubjectScene"),
    "workflows/subject_outfit.json": (["outfit", "subject"], "CcCKrea2SubjectOutfit"),
    "workflows/subject_outfit_scene.json": (["scene", "outfit", "subject"], "CcCKrea2SubjectSceneOutfit"),
    "workflows/subject_scene_qwen_simple.json": (["scene", "subject"], "CcCKrea2SubjectScene"),
    "workflows/subject_scene_outfit_qwen_simple.json": (["scene", "outfit", "subject"], "CcCKrea2SubjectSceneOutfit"),
}

MAIN_NODE_TYPES = (
    "CcCKrea2Subject",
    "CcCKrea2SubjectScene",
    "CcCKrea2SubjectOutfit",
    "CcCKrea2SubjectSceneOutfit",
)


def test_expected_filenames_and_directory_contents():
    """1. Exactly the six expected workflow files exist."""
    workflows_dir = Path("workflows")
    assert workflows_dir.exists(), "workflows directory does not exist"

    found_files = set(str(p).replace("\\", "/") for p in workflows_dir.glob("*.json"))
    expected_files = set(EXPECTED_WORKFLOW_FILES)
    assert found_files == expected_files, f"Mismatch in workflow files: found {found_files}, expected {expected_files}"


def test_workflow_json_parsing_and_loaders():
    """2-7. JSON parses, uses Krea2 loader stack, and does not use legacy LoRA loaders."""
    for rel_path in EXPECTED_WORKFLOW_FILES:
        with open(rel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict), f"Invalid JSON structure in {rel_path}"

        node_types = [n.get("type") for n in data.get("nodes", [])]
        assert "UNETLoader" in node_types, f"Missing UNETLoader in {rel_path}"
        assert "CLIPLoader" in node_types, f"Missing CLIPLoader in {rel_path}"
        assert "VAELoader" in node_types, f"Missing VAELoader in {rel_path}"
        assert "CcCKrea2LoRAStack" in node_types, f"Missing CcCKrea2LoRAStack in {rel_path}"
        assert "CcCKrea2LoRAPromptSettings" in node_types, f"Missing CcCKrea2LoRAPromptSettings in {rel_path}"

        assert "LoraLoaderModelOnly" not in node_types, f"LoraLoaderModelOnly found in {rel_path}"
        assert "CheckpointLoaderSimple" not in node_types, f"CheckpointLoaderSimple found in {rel_path}"
        assert "LoraLoader" not in node_types, f"Standard LoraLoader found in {rel_path}"

        raw_json_str = json.dumps(data)
        assert "flux1-dev.safetensors" not in raw_json_str, f"flux1-dev.safetensors found in {rel_path}"


def test_workflow_main_nodes_and_groups():
    """8-10. Main nodes, individual image groups, and required standard groups."""
    for rel_path, (roles, expected_main_node) in EXPECTED_ROLES_PER_WORKFLOW.items():
        with open(rel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        node_types = [n.get("type") for n in nodes]
        assert expected_main_node in node_types, f"Expected {expected_main_node} in {rel_path}"

        group_titles = [g.get("title", "") for g in data.get("groups", [])]

        # Check individual image groups
        for role in roles:
            expected_title = f"{role.capitalize()} Image"
            assert expected_title in group_titles, f"Missing group '{expected_title}' in {rel_path}"

        # Standard groups
        assert "Models" in group_titles, f"Missing 'Models' group in {rel_path}"
        assert "Advanced Settings (disabled by default)" in group_titles, f"Missing Advanced Settings group in {rel_path}"
        assert "LoRA Stack + Edit + KSampler" in group_titles, f"Missing 'LoRA Stack + Edit + KSampler' group in {rel_path}"
        assert "Output" in group_titles, f"Missing 'Output' group in {rel_path}"


def test_lora_stack_and_prompt_settings_wiring():
    """Verify wiring between CcCKrea2LoRAPromptSettings, CcCKrea2LoRAStack, and main node."""
    for rel_path in EXPECTED_WORKFLOW_FILES:
        with open(rel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        node_by_type = {n["type"]: n for n in nodes if "type" in n}

        stack_node = node_by_type["CcCKrea2LoRAStack"]
        ps_node = node_by_type["CcCKrea2LoRAPromptSettings"]
        main_node = next(n for n in nodes if n.get("type") in MAIN_NODE_TYPES)

        link_map = {l[0]: l for l in data.get("links", [])}

        # 1. Prompt settings output -> LoRA stack input
        ps_out_links = ps_node["outputs"][0]["links"]
        assert ps_out_links is not None and len(ps_out_links) == 1
        ps_link = link_map[ps_out_links[0]]
        assert ps_link[3] == stack_node["id"]
        assert ps_link[5] == "CCC_KREA2_LORA_PROMPT_SETTINGS"

        # 2. LoRA stack prompt_augmentation output -> main node input
        stack_aug_out_links = stack_node["outputs"][1]["links"]
        assert stack_aug_out_links is not None and len(stack_aug_out_links) == 1
        aug_link = link_map[stack_aug_out_links[0]]
        assert aug_link[3] == main_node["id"]
        assert aug_link[5] == "CCC_KREA2_PROMPT_AUGMENTATION"

        # 3. Verify slot default values in stack (disabled slots)
        stack_widgets = stack_node.get("widgets_values", [])
        assert stack_widgets[0] is True  # stack enabled
        assert stack_widgets[1] == 1.0  # global strength
        assert stack_widgets[2] is True  # lora 1 enabled
        # slots 2..4 disabled
        assert stack_widgets[5] is False
        assert stack_widgets[8] is False
        assert stack_widgets[11] is False


def test_advanced_settings_chaining_and_bypass():
    """11-14. Mode == 2, main node adv inputs disconnected, role count & chain order match."""
    for rel_path, (expected_chain, expected_main_node) in EXPECTED_ROLES_PER_WORKFLOW.items():
        with open(rel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        main_node = next(n for n in nodes if n.get("type") == expected_main_node)

        # Check main node sockets disconnected
        for inp in main_node.get("inputs", []):
            if inp.get("name") in ("image_advanced_settings", "edit_advanced_settings"):
                assert inp.get("link") is None, f"{inp['name']} should be disconnected in {rel_path}"

        # Check advanced setting nodes mode == 2
        img_adv_nodes = [n for n in nodes if n.get("type") == "CcCKrea2ImageAdvancedSettings"]
        edit_adv_nodes = [n for n in nodes if n.get("type") == "CcCKrea2EditAdvancedSettings"]

        assert len(img_adv_nodes) == len(expected_chain), f"Expected {len(expected_chain)} image adv nodes in {rel_path}"
        assert len(edit_adv_nodes) == 1, f"Expected 1 edit adv node in {rel_path}"

        for n in img_adv_nodes + edit_adv_nodes:
            assert n.get("mode") == 2, f"Node {n['type']} in {rel_path} should have mode=2 (bypassed)"

        # Check chained order
        link_map = {link_item[0]: link_item for link_item in data.get("links", [])}
        node_by_id = {n["id"]: n for n in nodes}

        # Find head of chain (no incoming link)
        head_node = next(n for n in img_adv_nodes if n.get("inputs", [{}])[0].get("link") is None)

        chain_roles = []
        curr = head_node
        while curr:
            role = curr.get("widgets_values", [""])[0]
            chain_roles.append(role)

            out_links = curr.get("outputs", [{}])[0].get("links")
            if out_links:
                next_link_id = out_links[0]
                link_tuple = link_map[next_link_id]
                target_node_id = link_tuple[3]
                curr = node_by_id.get(target_node_id)
                if curr and curr.get("type") != "CcCKrea2ImageAdvancedSettings":
                    curr = None
            else:
                curr = None

        assert chain_roles == expected_chain, f"Chain order mismatch in {rel_path}: got {chain_roles}, expected {expected_chain}"


def test_qwen_workflows_and_non_qwen_isolation():
    """15-20. Upstream Qwen3VL node types, config chaining, scene img wiring, prompt wiring."""
    qwen_files = [
        "workflows/subject_scene_qwen_simple.json",
        "workflows/subject_scene_outfit_qwen_simple.json",
    ]
    non_qwen_files = [f for f in EXPECTED_WORKFLOW_FILES if f not in qwen_files]

    for rel_path in non_qwen_files:
        with open(rel_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_json_str = json.dumps(data)
        assert "QWEN3VL_CONFIG" not in raw_json_str, f"QWEN3VL_CONFIG found in {rel_path}"
        node_types = [n.get("type") for n in data.get("nodes", [])]
        assert not any("Qwen" in t for t in node_types), f"Qwen node found in non-Qwen workflow {rel_path}"

    for rel_path in qwen_files:
        with open(rel_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Failed json.load for {rel_path}"

        raw_json_str = json.dumps(data)
        assert "QWEN3VL_CONFIG" not in raw_json_str, f"QWEN3VL_CONFIG string found in {rel_path}"

        group_titles = [g.get("title", "") for g in data.get("groups", [])]
        assert "Qwen Prompt Builder" in group_titles, f"Missing 'Qwen Prompt Builder' group in {rel_path}"

        nodes = data.get("nodes", [])
        node_by_type = {n.get("type"): n for n in nodes}

        assert "SimpleQwenVLggufV2" in node_by_type, f"Missing SimpleQwenVLggufV2 in {rel_path}"
        assert "Qwen3VL_ModelConfig" in node_by_type, f"Missing Qwen3VL_ModelConfig in {rel_path}"
        assert "Qwen3VL_SamplingConfig" in node_by_type, f"Missing Qwen3VL_SamplingConfig in {rel_path}"

        # Socket types check: STRING
        m_cfg = node_by_type["Qwen3VL_ModelConfig"]
        s_cfg = node_by_type["Qwen3VL_SamplingConfig"]
        qwen_node = node_by_type["SimpleQwenVLggufV2"]

        assert m_cfg["outputs"][0]["type"] == "STRING", f"ModelConfig output not STRING in {rel_path}"
        assert s_cfg["inputs"][0]["type"] == "STRING", f"SamplingConfig input not STRING in {rel_path}"
        assert s_cfg["outputs"][0]["type"] == "STRING", f"SamplingConfig output not STRING in {rel_path}"

        cfg_override_inp = next(i for i in qwen_node["inputs"] if i["name"] == "config_override")
        assert cfg_override_inp["type"] == "STRING", f"SimpleQwenVLggufV2 config_override input not STRING in {rel_path}"

        # Verify video input type is "*"
        video_inp = next(i for i in qwen_node["inputs"] if i["name"] == "video")
        assert video_inp["type"] == "*", f"SimpleQwenVLggufV2 video input type must be '*' in {rel_path}"

        link_map = {link_item[0]: link_item for link_item in data.get("links", [])}

        # Link tuples type check
        m_out_link_id = m_cfg["outputs"][0]["links"][0]
        m_out_link = link_map[m_out_link_id]
        assert m_out_link[5] == "STRING", f"ModelConfig link type not STRING in {rel_path}"
        assert m_out_link[3] == s_cfg["id"], f"ModelConfig link target mismatch in {rel_path}"

        s_out_link_id = s_cfg["outputs"][0]["links"][0]
        s_out_link = link_map[s_out_link_id]
        assert s_out_link[5] == "STRING", f"SamplingConfig link type not STRING in {rel_path}"
        assert s_out_link[3] == qwen_node["id"], f"SamplingConfig link target mismatch in {rel_path}"
        assert s_out_link[4] == 6, f"SamplingConfig link target slot should be 6 (config_override) in {rel_path}"

        # ModelConfig widget count and specific values
        m_widgets = m_cfg.get("widgets_values", [])
        assert len(m_widgets) == 19, f"ModelConfig widget count mismatch: got {len(m_widgets)}, expected 19 in {rel_path}"
        assert m_widgets[11] == "qwen25", f"chat_handler must be 'qwen25' in {rel_path}"
        assert m_widgets[12] == "none", f"chat_format must be 'none' in {rel_path}"
        assert m_widgets[17] == "F16", f"type_k must be 'F16' in {rel_path}"
        assert m_widgets[18] == "F16", f"type_v must be 'F16' in {rel_path}"
        assert "auto" not in (m_widgets[11], m_widgets[12]), f"auto found in chat handler/format in {rel_path}"
        assert "default" not in (m_widgets[17], m_widgets[18]), f"default found in type_k/type_v in {rel_path}"

        # Qwen3VL_SamplingConfig widgets (10 items)
        s_widgets = s_cfg.get("widgets_values", [])
        assert len(s_widgets) == 10, f"Qwen3VL_SamplingConfig must have exactly 10 widget values in {rel_path}"

        # SimpleQwenVLggufV2 widgets (7 items)
        q_widgets = qwen_node.get("widgets_values", [])
        assert len(q_widgets) == 7, f"SimpleQwenVLggufV2 widget count mismatch: got {len(q_widgets)}, expected 7 in {rel_path}"
        assert q_widgets[0] == "None", f"model_preset must be 'None' in {rel_path}"
        assert q_widgets[1] == "None", f"system_preset must be 'None' in {rel_path}"
        user_prompt_text = q_widgets[2]
        assert isinstance(user_prompt_text, str)
        assert isinstance(q_widgets[3], int), f"seed must be integer in {rel_path}"
        assert q_widgets[4] == "fixed", f"control_after_generate must be 'fixed' in {rel_path}"
        assert q_widgets[5] is False, f"unload_all_models must be false in {rel_path}"
        assert q_widgets[6] == "subprocess", f"mode must be 'subprocess' in {rel_path}"
        assert q_widgets[6] != "full", f"mode must not be 'full' in {rel_path}"

        # Canonical image role terms and escaped newline in user prompt
        assert "subject image" in user_prompt_text
        assert "scene image" in user_prompt_text
        if "outfit" in rel_path:
            assert "outfit image" in user_prompt_text
        assert "Return only the final transformation prompt.\nDo not explain" in user_prompt_text, (
            f"user_prompt missing required escaped newline segment in {rel_path}"
        )

        # Outputs check (4 outputs serialized, only text output linked)
        outputs = qwen_node.get("outputs", [])
        assert len(outputs) == 4, f"SimpleQwenVLggufV2 must serialize 4 outputs in {rel_path}"
        output_names = [o["name"] for o in outputs]
        assert output_names == ["text", "conditioning", "system_prompt", "user_prompt"]

        text_out = next(o for o in outputs if o["name"] == "text")
        assert text_out["links"] is not None and len(text_out["links"]) == 1

        for unlinked_name in ("conditioning", "system_prompt", "user_prompt"):
            out_obj = next(o for o in outputs if o["name"] == unlinked_name)
            assert out_obj["links"] is None, f"{unlinked_name} output should not be linked in {rel_path}"

        # Scene image connected to real `image` socket (slot index 0)
        qwen_img_link_id = qwen_node["inputs"][0]["link"]  # slot 0 is "image"
        qwen_img_link = link_map[qwen_img_link_id]
        scene_load_img_node = next(n for n in nodes if n["type"] == "LoadImage" and n.get("widgets_values", [""])[0] == "scene.jpg")
        assert qwen_img_link[1] == scene_load_img_node["id"], f"Scene LoadImage not connected to Qwen image socket in {rel_path}"

        # Qwen text output connected to CcC main node prompt input
        qwen_text_link_id = text_out["links"][0]
        main_node = next(n for n in nodes if n["type"] in MAIN_NODE_TYPES)
        prompt_input = next(i for i in main_node["inputs"] if i["name"] == "prompt")
        assert prompt_input["link"] == qwen_text_link_id, f"Qwen text output not connected to main node prompt in {rel_path}"


def test_metadata_and_graph_integrity():
    """21-22. Maxima match, unique IDs, valid links."""
    for rel_path in EXPECTED_WORKFLOW_FILES:
        with open(rel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        links = data.get("links", [])

        node_ids = [n["id"] for n in nodes]
        link_ids = [link_item[0] for link_item in links]

        assert len(node_ids) == len(set(node_ids)), f"Duplicate node IDs in {rel_path}"
        assert len(link_ids) == len(set(link_ids)), f"Duplicate link IDs in {rel_path}"

        max_node_id = max(node_ids)
        max_link_id = max(link_ids)

        assert data.get("last_node_id") == max_node_id, f"last_node_id mismatch in {rel_path}: got {data.get('last_node_id')}, expected {max_node_id}"
        assert data.get("last_link_id") == max_link_id, f"last_link_id mismatch in {rel_path}: got {data.get('last_link_id')}, expected {max_link_id}"

        node_id_set = set(node_ids)
        link_id_set = set(link_ids)

        for link_item in links:
            lid, src_id, src_slot, dst_id, dst_slot, ltype = link_item
            assert src_id in node_id_set, f"Link {lid} references non-existent src node {src_id} in {rel_path}"
            assert dst_id in node_id_set, f"Link {lid} references non-existent dst node {dst_id} in {rel_path}"

        for n in nodes:
            for inp in n.get("inputs", []):
                link_id = inp.get("link")
                if link_id is not None:
                    assert link_id in link_id_set, f"Node {n['id']} input references non-existent link {link_id} in {rel_path}"

            for out in n.get("outputs", []):
                out_links = out.get("links")
                if out_links:
                    for lid in out_links:
                        assert lid in link_id_set, f"Node {n['id']} output references non-existent link {lid} in {rel_path}"
