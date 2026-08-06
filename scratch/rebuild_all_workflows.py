import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS


def get_max_link_id(workflow):
    max_id = 0
    for link in workflow.get("links", []):
        if link[0] > max_id:
            max_id = link[0]
    return max_id


def ensure_link(workflow, from_node_id, from_slot, to_node_id, to_input_name, type_name):
    nodes = {n["id"]: n for n in workflow["nodes"]}
    target_node = nodes[to_node_id]

    # Find or add input slot in target node
    target_input = None
    for inp in target_node.get("inputs", []):
        if inp["name"] == to_input_name:
            target_input = inp
            break

    if target_input is None:
        target_input = {"name": to_input_name, "type": type_name, "link": None}
        target_node.setdefault("inputs", []).append(target_input)

    # Check if link already exists
    existing_link_id = target_input.get("link")
    if existing_link_id is not None:
        # Check if existing link matches
        for link in workflow.get("links", []):
            if link[0] == existing_link_id:
                link[1] = from_node_id
                link[2] = from_slot
                link[3] = to_node_id
                link[4] = next((idx for idx, inp in enumerate(target_node["inputs"]) if inp["name"] == to_input_name), 0)
                link[5] = type_name
                return existing_link_id

    # Create new link
    new_link_id = get_max_link_id(workflow) + 1
    to_slot_index = next((idx for idx, inp in enumerate(target_node["inputs"]) if inp["name"] == to_input_name), 0)

    target_input["link"] = new_link_id
    workflow.setdefault("links", []).append([
        new_link_id,
        from_node_id,
        from_slot,
        to_node_id,
        to_slot_index,
        type_name
    ])

    # Also add link_id to outputs of source node if present
    source_node = nodes[from_node_id]
    if "outputs" in source_node and from_slot < len(source_node["outputs"]):
        out = source_node["outputs"][from_slot]
        out.setdefault("links", [])
        if new_link_id not in out["links"]:
            out["links"].append(new_link_id)

    return new_link_id


def sanitize_node_sockets_and_outputs(workflow):
    nodes = workflow.get("nodes", [])
    for n in nodes:
        ntype = n.get("type")
        if ntype in NODE_CLASS_MAPPINGS:
            cls = NODE_CLASS_MAPPINGS[ntype]
            inp_spec = cls.INPUT_TYPES()
            req = list(inp_spec.get("required", {}).keys())
            opt = list(inp_spec.get("optional", {}).keys())
            allowed_inputs = set(req + opt)

            # Filter invalid input sockets
            n["inputs"] = [i for i in n.get("inputs", []) if i["name"] in allowed_inputs]

            # Fix output names and types
            ret_names = getattr(cls, "RETURN_NAMES", ())
            ret_types = getattr(cls, "RETURN_TYPES", ())
            for idx, out in enumerate(n.get("outputs", [])):
                if idx < len(ret_names):
                    out["name"] = ret_names[idx]
                if idx < len(ret_types):
                    out["type"] = ret_types[idx]


def update_target_latent_widgets(node, target_content, geometry_mode):
    wvals = node.get("widgets_values", [])
    target_mp = float(wvals[2]) if len(wvals) > 2 and isinstance(wvals[2], (int, float)) else 2.0
    fixed_mp = float(wvals[3]) if len(wvals) > 3 and isinstance(wvals[3], (int, float)) else 2.0
    aspect = wvals[4] if len(wvals) > 4 and isinstance(wvals[4], str) else "1:1"
    batch = int(wvals[5]) if len(wvals) > 5 and isinstance(wvals[5], (int, float)) else 1
    node["widgets_values"] = [target_content, geometry_mode, target_mp, fixed_mp, aspect, batch]


def wire_reference_chain(data, role_node_ids, edit_node_id):
    nodes = {n["id"]: n for n in data["nodes"]}
    # First node in chain has no reference_chain input link
    first_node = nodes[role_node_ids[0]]
    for inp in first_node.get("inputs", []):
        if inp["name"] == "reference_chain":
            inp["link"] = None

    # Wire chain sequentially
    for i in range(len(role_node_ids) - 1):
        src_id = role_node_ids[i]
        dst_id = role_node_ids[i + 1]
        ensure_link(data, src_id, 0, dst_id, "reference_chain", "REFERENCE_CHAIN")

    # Wire last node to edit
    last_id = role_node_ids[-1]
    ensure_link(data, last_id, 0, edit_node_id, "references", "REFERENCE_CHAIN")


def process_workflow_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = os.path.basename(filepath)
    nodes_by_type = {}
    nodes_by_id = {n["id"]: n for n in data.get("nodes", [])}

    for n in data.get("nodes", []):
        nodes_by_type.setdefault(n["type"], []).append(n)

    target_latent = nodes_by_type.get("CcCKrea2TargetLatent", [None])[0]
    vae_loader = nodes_by_type.get("VAELoader", [None])[0]

    if filename == "03_subject_edit.json":
        if target_latent:
            update_target_latent_widgets(target_latent, "subject", "favor_subject")
            subj_prep = nodes_by_id.get(6)
            if subj_prep and vae_loader:
                ensure_link(data, vae_loader["id"], 0, target_latent["id"], "vae", "VAE")
                ensure_link(data, subj_prep["id"], 0, target_latent["id"], "subject_image", "PREPARED_VISION_IMAGE")

            wire_reference_chain(data, [7], 8)

    elif filename == "04_subject_scene_edit.json":
        if target_latent:
            update_target_latent_widgets(target_latent, "scene", "favor_scene")
            scene_prep = nodes_by_id.get(9)
            if scene_prep and vae_loader:
                ensure_link(data, vae_loader["id"], 0, target_latent["id"], "vae", "VAE")
                ensure_link(data, scene_prep["id"], 0, target_latent["id"], "scene_image", "PREPARED_VISION_IMAGE")

            scene_prep = nodes_by_id.get(9)
            scene_node = nodes_by_id.get(10)
            if scene_prep and scene_node:
                ensure_link(data, scene_prep["id"], 0, scene_node["id"], "prepared_image", "PREPARED_VISION_IMAGE")

            # Reference chain order: Scene (10) -> Subject (7) -> Edit (11)
            wire_reference_chain(data, [10, 7], 11)

    elif filename == "05_subject_outfit_edit.json":
        if target_latent:
            update_target_latent_widgets(target_latent, "empty", "favor_subject")
            subj_prep = nodes_by_id.get(6)
            if subj_prep:
                ensure_link(data, subj_prep["id"], 0, target_latent["id"], "subject_image", "PREPARED_VISION_IMAGE")

            wire_reference_chain(data, [7, 10], 11)

    elif filename == "06_subject_scene_outfit_edit.json":
        if target_latent:
            update_target_latent_widgets(target_latent, "empty", "favor_subject")
            subj_prep = nodes_by_id.get(6)
            if subj_prep:
                ensure_link(data, subj_prep["id"], 0, target_latent["id"], "subject_image", "PREPARED_VISION_IMAGE")

            # Reference chain order: Scene (10) -> Subject (7) -> Outfit (13) -> Edit (14)
            wire_reference_chain(data, [10, 7, 13], 14)

    elif filename == "07_style_moodboard_edit.json":
        if target_latent:
            update_target_latent_widgets(target_latent, "empty", "fixed")

        clip_loader = nodes_by_type.get("CLIPLoader", [None])[0]
        subj_ref = nodes_by_id.get(14)

        if not subj_ref:
            subj_load = {
                "id": 12,
                "type": "LoadImage",
                "pos": [100, 100],
                "size": [315, 314],
                "flags": {},
                "order": 4,
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}, {"name": "MASK", "type": "MASK", "links": []}],
                "widgets_values": ["subject.jpg", "image"]
            }
            subj_prep = {
                "id": 13,
                "type": "CcCKrea2QwenVisionImagePrep",
                "pos": [450, 100],
                "size": [315, 200],
                "flags": {},
                "order": 5,
                "mode": 0,
                "inputs": [
                    {"name": "clip", "type": "CLIP", "link": None},
                    {"name": "image", "type": "IMAGE", "link": None}
                ],
                "outputs": [
                    {"name": "prepared_image", "type": "PREPARED_VISION_IMAGE", "links": []},
                    {"name": "vision_image", "type": "IMAGE", "links": []},
                    {"name": "vision_info", "type": "STRING", "links": []}
                ],
                "widgets_values": ["native", 0.0, 1.0, 1.0, "auto", "auto"]
            }
            subj_ref = {
                "id": 14,
                "type": "CcCKrea2SubjectImage",
                "pos": [800, 100],
                "size": [315, 250],
                "flags": {},
                "order": 6,
                "mode": 0,
                "inputs": [
                    {"name": "prepared_image", "type": "PREPARED_VISION_IMAGE", "link": None},
                    {"name": "attention_mask", "type": "MASK", "link": None},
                    {"name": "reference_chain", "type": "REFERENCE_CHAIN", "link": None}
                ],
                "outputs": [{"name": "reference_chain", "type": "REFERENCE_CHAIN", "links": []}],
                "widgets_values": ["auto", 1.0, 1.0, 0.0, 0.0, 0.0, "", 0, ""]
            }
            data["nodes"].extend([subj_load, subj_prep, subj_ref])

        if clip_loader:
            ensure_link(data, clip_loader["id"], 0, 13, "clip", "CLIP")
        ensure_link(data, 12, 0, 13, "image", "IMAGE")
        ensure_link(data, 13, 0, 14, "prepared_image", "PREPARED_VISION_IMAGE")

        wire_reference_chain(data, [14, 7], 8)

    elif filename == "08_inpaint_subject_edit.json":
        if target_latent:
            update_target_latent_widgets(target_latent, "subject", "fixed")
            subj_prep = nodes_by_id.get(6)
            if subj_prep and vae_loader:
                ensure_link(data, vae_loader["id"], 0, target_latent["id"], "vae", "VAE")
                ensure_link(data, subj_prep["id"], 0, target_latent["id"], "subject_image", "PREPARED_VISION_IMAGE")

            wire_reference_chain(data, [7], 8)

    elif filename == "09_inpaint_scene_edit.json":
        if target_latent:
            update_target_latent_widgets(target_latent, "scene", "fixed")
            scene_prep = nodes_by_id.get(6)
            if scene_prep and vae_loader:
                ensure_link(data, vae_loader["id"], 0, target_latent["id"], "vae", "VAE")
                ensure_link(data, scene_prep["id"], 0, target_latent["id"], "scene_image", "PREPARED_VISION_IMAGE")

            wire_reference_chain(data, [7], 8)

    elif filename == "10_multi_subject_chasing_slots.json":
        if target_latent:
            update_target_latent_widgets(target_latent, "empty", "favor_subject")
            subj1_prep = nodes_by_id.get(6)
            if subj1_prep:
                ensure_link(data, subj1_prep["id"], 0, target_latent["id"], "subject_image", "PREPARED_VISION_IMAGE")

            wire_reference_chain(data, [7, 10], 11)

    elif filename == "11_advanced_directives_fit_modes.json":
        if target_latent:
            update_target_latent_widgets(target_latent, "empty", "favor_subject")
            subj_prep = nodes_by_id.get(6)
            if subj_prep:
                ensure_link(data, subj_prep["id"], 0, target_latent["id"], "subject_image", "PREPARED_VISION_IMAGE")

            wire_reference_chain(data, [7], 8)

    elif filename == "12_full_pipeline_composition.json":
        if target_latent:
            update_target_latent_widgets(target_latent, "empty", "favor_subject")
            subj_prep = nodes_by_id.get(6)
            if subj_prep:
                ensure_link(data, subj_prep["id"], 0, target_latent["id"], "subject_image", "PREPARED_VISION_IMAGE")

            wire_reference_chain(data, [10, 7, 13, 16], 17)

    # Sanitize socket names and output specifications
    sanitize_node_sockets_and_outputs(data)

    # Clean up output link lists to ensure consistency
    links_by_id = {lnk[0]: lnk for lnk in data.get("links", [])}
    for n in data.get("nodes", []):
        for out in n.get("outputs", []):
            out["links"] = [lid for lid in (out.get("links") or []) if lid in links_by_id]

    # Update last_node_id and last_link_id
    if data.get("nodes"):
        data["last_node_id"] = max(n["id"] for n in data["nodes"])
    if data.get("links"):
        data["last_link_id"] = max(lnk[0] for lnk in data["links"])

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"Updated {filepath}")


if __name__ == "__main__":
    import glob
    for p in sorted(glob.glob("workflows/*.json")):
        process_workflow_file(p)
