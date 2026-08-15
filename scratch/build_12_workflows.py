"""Complete script to write 12 curated modular workflow JSON files to workflows/."""

import json
from pathlib import Path

WORKFLOWS_DIR = Path("workflows")
WORKFLOWS_DIR.mkdir(exist_ok=True)


def make_wf(nodes, links, groups=None):
    max_node_id = max(n["id"] for n in nodes) if nodes else 0
    max_link_id = max(link_item[0] for link_item in links) if links else 0
    return {
        "last_node_id": max_node_id,
        "last_link_id": max_link_id,
        "nodes": nodes,
        "links": links,
        "groups": groups or [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def base_loaders():
    return [
        {
            "id": 1,
            "type": "UNETLoader",
            "pos": [100, 100],
            "size": [315, 82],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0}],
            "properties": {"Node name for S&R": "UNETLoader"},
            "widgets_values": ["krea2_unet.safetensors", "default"],
        },
        {
            "id": 2,
            "type": "CLIPLoader",
            "pos": [100, 220],
            "size": [315, 82],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "CLIP", "type": "CLIP", "links": [2], "slot_index": 0}],
            "properties": {"Node name for S&R": "CLIPLoader"},
            "widgets_values": ["qwen2_5_vl.safetensors", "qwen2_5_vl", "default"],
        },
        {
            "id": 3,
            "type": "VAELoader",
            "pos": [100, 340],
            "size": [315, 82],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "VAE", "type": "VAE", "links": [3], "slot_index": 0}],
            "properties": {"Node name for S&R": "VAELoader"},
            "widgets_values": ["krea2_vae.safetensors"],
        },
    ]


def build_modular_edit_pipeline(
    role_specs,  # list of tuples: (role_type, role_node_type, default_aliases, extra_directive, specific_widgets)
    target_content="empty",
    target_geometry="fixed",
    has_lora=False,
    inpaint_mask=False,
):
    nodes = base_loaders()
    links = []

    # Model/Clip/VAE link IDs
    next_node_id = 4
    next_link_id = 4

    # 1. Target Latent Node
    target_latent_id = next_node_id
    next_node_id += 1
    t_lat_link = next_link_id
    next_link_id += 1

    target_latent_node = {
        "id": target_latent_id,
        "type": "CcCKrea2TargetLatent",
        "pos": [460, 100],
        "size": [315, 220],
        "flags": {},
        "order": 3,
        "mode": 0,
        "inputs": [
            {"name": "vae", "type": "VAE", "link": 3},
            {"name": "subject_image", "type": "CCC_KREA2_PREPARED_IMAGE", "link": None},
            {"name": "scene_image", "type": "CCC_KREA2_PREPARED_IMAGE", "link": None},
        ],
        "outputs": [
            {"name": "target_latent", "type": "LATENT", "links": [t_lat_link], "slot_index": 0},
            {"name": "latent_info", "type": "STRING", "links": None, "slot_index": 1},
        ],
        "properties": {"Node name for S&R": "CcCKrea2TargetLatent"},
        "widgets_values": [target_content, target_geometry, 2.0, 1.0, "1:1", 1, 1, 1],
    }
    nodes.append(target_latent_node)

    # 2. Vision Prep and Declarative Role Nodes
    curr_chain_link = None
    role_pos_y = 350

    for idx, (role_name, role_node_type, aliases, directive, extra_widgets) in enumerate(role_specs):
        # Load Image Node
        load_img_id = next_node_id
        next_node_id += 1
        img_out_link = next_link_id
        next_link_id += 1

        mask_out_link = None
        if inpaint_mask:
            mask_out_link = next_link_id
            next_link_id += 1

        load_img_node = {
            "id": load_img_id,
            "type": "LoadImage",
            "pos": [100, role_pos_y],
            "size": [315, 314],
            "flags": {},
            "order": 4 + idx * 3,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": [img_out_link], "slot_index": 0},
                {"name": "MASK", "type": "MASK", "links": [mask_out_link] if mask_out_link else None, "slot_index": 1},
            ],
            "properties": {"Node name for S&R": "LoadImage"},
            "widgets_values": [f"{role_name}.jpg", "image"],
        }
        nodes.append(load_img_node)

        # Vision Prep Node
        prep_id = next_node_id
        next_node_id += 1
        prep_out_link = next_link_id
        next_link_id += 1

        prep_node = {
            "id": prep_id,
            "type": "CcCKrea2QwenVisionImagePrep",
            "pos": [460, role_pos_y],
            "size": [315, 220],
            "flags": {},
            "order": 5 + idx * 3,
            "mode": 0,
            "inputs": [
                {"name": "clip", "type": "CLIP", "link": 2},
                {"name": "image", "type": "IMAGE", "link": img_out_link},
            ],
            "outputs": [
                {
                    "name": "prepared_image",
                    "type": "CCC_KREA2_PREPARED_IMAGE",
                    "links": [prep_out_link],
                    "slot_index": 0,
                },
                {"name": "vision_image", "type": "IMAGE", "links": None, "slot_index": 1},
                {"name": "vision_info", "type": "STRING", "links": None, "slot_index": 2},
            ],
            "properties": {"Node name for S&R": "CcCKrea2QwenVisionImagePrep"},
            "widgets_values": ["native", 0.0, 1.0, 1.0, "auto", "auto"],
        }
        nodes.append(prep_node)

        links.append([img_out_link, load_img_id, 0, prep_id, 1, "IMAGE"])

        # Role Node
        role_id = next_node_id
        next_node_id += 1
        role_out_link = next_link_id
        next_link_id += 1

        role_inputs = [
            {"name": "prepared_image", "type": "CCC_KREA2_PREPARED_IMAGE", "link": prep_out_link},
            {"name": "attention_mask", "type": "MASK", "link": mask_out_link if (inpaint_mask and idx == 0) else None},
            {"name": "previous_references", "type": "CCC_KREA2_REFERENCE_CHAIN", "link": curr_chain_link},
        ]

        # Widgets values according to node type
        w_vals = ["auto", aliases, directive] + extra_widgets

        role_node = {
            "id": role_id,
            "type": role_node_type,
            "pos": [820, role_pos_y],
            "size": [315, 260],
            "flags": {},
            "order": 6 + idx * 3,
            "mode": 0,
            "inputs": role_inputs,
            "outputs": [
                {"name": "references", "type": "CCC_KREA2_REFERENCE_CHAIN", "links": [role_out_link], "slot_index": 0}
            ],
            "properties": {"Node name for S&R": role_node_type},
            "widgets_values": w_vals,
        }
        nodes.append(role_node)

        links.append([prep_out_link, prep_id, 0, role_id, 0, "CCC_KREA2_PREPARED_IMAGE"])
        if mask_out_link and idx == 0:
            links.append([mask_out_link, load_img_id, 1, role_id, 1, "MASK"])

        if curr_chain_link is not None:
            # Connect previous role node output to this role node's previous_references input
            prev_role_node = nodes[-2]  # previous role node
            links.append([curr_chain_link, prev_role_node["id"], 0, role_id, 2, "CCC_KREA2_REFERENCE_CHAIN"])

        curr_chain_link = role_out_link
        role_pos_y += 350

    # 3. Edit Orchestrator Node
    edit_id = next_node_id
    next_node_id += 1
    edit_model_link = next_link_id
    next_link_id += 1
    edit_pos_link = next_link_id
    next_link_id += 1
    edit_neg_link = next_link_id
    next_link_id += 1
    edit_lat_link = next_link_id
    next_link_id += 1

    # Base loader links
    links.extend([[1, 1, 0, edit_id, 0, "MODEL"], [2, 2, 0, edit_id, 1, "CLIP"], [3, 3, 0, target_latent_id, 0, "VAE"]])

    edit_node = {
        "id": edit_id,
        "type": "CcCKrea2Edit",
        "pos": [1180, 100],
        "size": [400, 300],
        "flags": {},
        "order": 50,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": 1},
            {"name": "clip", "type": "CLIP", "link": 2},
            {"name": "vae", "type": "VAE", "link": 3},
            {"name": "references", "type": "CCC_KREA2_REFERENCE_CHAIN", "link": curr_chain_link},
            {"name": "target_latent", "type": "LATENT", "link": t_lat_link},
            {"name": "prompt_augmentation", "type": "CCC_KREA2_PROMPT_AUGMENTATION", "link": None},
            {"name": "global_vision_directive", "type": "STRING", "link": None},
        ],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": [edit_model_link], "slot_index": 0},
            {"name": "positive", "type": "CONDITIONING", "links": [edit_pos_link], "slot_index": 1},
            {"name": "negative", "type": "CONDITIONING", "links": [edit_neg_link], "slot_index": 2},
            {"name": "latent", "type": "LATENT", "links": [edit_lat_link], "slot_index": 3},
            {"name": "edit_info", "type": "STRING", "links": None, "slot_index": 4},
        ],
        "properties": {"Node name for S&R": "CcCKrea2Edit"},
        "widgets_values": ["Edit prompt describing the transformation.", "blurry, distorted", ""],
    }
    nodes.append(edit_node)

    links.append([t_lat_link, target_latent_id, 0, edit_id, 4, "LATENT"])
    if curr_chain_link is not None:
        last_role_node = nodes[-2]
        links.append([curr_chain_link, last_role_node["id"], 0, edit_id, 3, "CCC_KREA2_REFERENCE_CHAIN"])

    # 4. KSampler Node
    ksampler_id = next_node_id
    next_node_id += 1
    ksamp_out_link = next_link_id
    next_link_id += 1

    ksampler_node = {
        "id": ksampler_id,
        "type": "KSampler",
        "pos": [1620, 100],
        "size": [315, 262],
        "flags": {},
        "order": 51,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": edit_model_link},
            {"name": "positive", "type": "CONDITIONING", "link": edit_pos_link},
            {"name": "negative", "type": "CONDITIONING", "link": edit_neg_link},
            {"name": "latent_image", "type": "LATENT", "link": edit_lat_link},
        ],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": [ksamp_out_link], "slot_index": 0}],
        "properties": {"Node name for S&R": "KSampler"},
        "widgets_values": [12345, "fixed", 20, 4.5, "euler", "simple", 1.0],
    }
    nodes.append(ksampler_node)

    links.extend(
        [
            [edit_model_link, edit_id, 0, ksampler_id, 0, "MODEL"],
            [edit_pos_link, edit_id, 1, ksampler_id, 1, "CONDITIONING"],
            [edit_neg_link, edit_id, 2, ksampler_id, 2, "CONDITIONING"],
            [edit_lat_link, edit_id, 3, ksampler_id, 3, "LATENT"],
        ]
    )

    # 5. VAE Decode Node
    decode_id = next_node_id
    next_node_id += 1
    decode_out_link = next_link_id
    next_link_id += 1
    vae_decode_link = next_link_id
    next_link_id += 1

    # Add vae_decode_link to VAELoader node (node id 3) output links
    nodes[2]["outputs"][0]["links"].append(vae_decode_link)

    decode_node = {
        "id": decode_id,
        "type": "VAEDecode",
        "pos": [1970, 100],
        "size": [210, 46],
        "flags": {},
        "order": 52,
        "mode": 0,
        "inputs": [
            {"name": "samples", "type": "LATENT", "link": ksamp_out_link},
            {"name": "vae", "type": "VAE", "link": vae_decode_link},
        ],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [decode_out_link], "slot_index": 0}],
        "properties": {"Node name for S&R": "VAEDecode"},
        "widgets_values": [],
    }
    nodes.append(decode_node)

    links.extend(
        [[ksamp_out_link, ksampler_id, 0, decode_id, 0, "LATENT"], [vae_decode_link, 3, 0, decode_id, 1, "VAE"]]
    )

    # 6. Save Image Node
    save_id = next_node_id
    next_node_id += 1

    save_node = {
        "id": save_id,
        "type": "SaveImage",
        "pos": [2220, 100],
        "size": [315, 270],
        "flags": {},
        "order": 53,
        "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": decode_out_link}],
        "outputs": [],
        "properties": {"Node name for S&R": "SaveImage"},
        "widgets_values": ["Krea2_Edit"],
    }
    nodes.append(save_node)

    links.append([decode_out_link, decode_id, 0, save_id, 0, "IMAGE"])

    groups = [
        {"title": "Models", "bounding": [80, 50, 350, 400], "color": "#3f51b5"},
        {"title": "Target Latent", "bounding": [440, 50, 350, 280], "color": "#009688"},
        {"title": "Modular Edit Pipeline", "bounding": [1160, 50, 1400, 450], "color": "#ff9800"},
    ]

    return make_wf(nodes, links, groups)


# 01_t2i_basic.json
def wf_01():
    nodes = base_loaders() + [
        {
            "id": 4,
            "type": "CcCKrea2TextToImage",
            "pos": [460, 100],
            "size": [400, 260],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 1},
                {"name": "clip", "type": "CLIP", "link": 2},
                {"name": "prompt_augmentation", "type": "CCC_KREA2_PROMPT_AUGMENTATION", "link": None},
            ],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": [4], "slot_index": 0},
                {"name": "positive", "type": "CONDITIONING", "links": [5], "slot_index": 1},
                {"name": "negative", "type": "CONDITIONING", "links": [6], "slot_index": 2},
                {"name": "latent", "type": "LATENT", "links": [7], "slot_index": 3},
            ],
            "properties": {"Node name for S&R": "CcCKrea2TextToImage"},
            "widgets_values": ["A majestic mountain landscape at sunset.", "", 1024, 1024, 1.0, 1],
        },
        {
            "id": 5,
            "type": "KSampler",
            "pos": [900, 100],
            "size": [315, 262],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 4},
                {"name": "positive", "type": "CONDITIONING", "link": 5},
                {"name": "negative", "type": "CONDITIONING", "link": 6},
                {"name": "latent_image", "type": "LATENT", "link": 7},
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [8], "slot_index": 0}],
            "properties": {"Node name for S&R": "KSampler"},
            "widgets_values": [12345, "fixed", 20, 4.5, "euler", "simple", 1.0],
        },
        {
            "id": 6,
            "type": "VAEDecode",
            "pos": [1250, 100],
            "size": [210, 46],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [{"name": "samples", "type": "LATENT", "link": 8}, {"name": "vae", "type": "VAE", "link": 3}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [9], "slot_index": 0}],
            "properties": {"Node name for S&R": "VAEDecode"},
            "widgets_values": [],
        },
        {
            "id": 7,
            "type": "SaveImage",
            "pos": [1500, 100],
            "size": [315, 270],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": 9}],
            "outputs": [],
            "properties": {"Node name for S&R": "SaveImage"},
            "widgets_values": ["Krea2_T2I"],
        },
    ]
    links = [
        [1, 1, 0, 4, 0, "MODEL"],
        [2, 2, 0, 4, 1, "CLIP"],
        [3, 3, 0, 6, 1, "VAE"],
        [4, 4, 0, 5, 0, "MODEL"],
        [5, 4, 1, 5, 1, "CONDITIONING"],
        [6, 4, 2, 5, 2, "CONDITIONING"],
        [7, 4, 3, 5, 3, "LATENT"],
        [8, 5, 0, 6, 0, "LATENT"],
        [9, 6, 0, 7, 0, "IMAGE"],
    ]
    groups = [
        {"title": "Models", "bounding": [80, 50, 350, 400], "color": "#3f51b5"},
        {"title": "Text to Image Generation", "bounding": [440, 50, 1400, 400], "color": "#2196f3"},
    ]
    return make_wf(nodes, links, groups)


# 02_t2i_lora_stack.json
def wf_02():
    nodes = base_loaders() + [
        {
            "id": 4,
            "type": "CcCKrea2LoRAPromptSettings",
            "pos": [460, 400],
            "size": [315, 300],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {
                    "name": "lora_prompt_settings",
                    "type": "CCC_KREA2_LORA_PROMPT_SETTINGS",
                    "links": [4],
                    "slot_index": 0,
                }
            ],
            "properties": {"Node name for S&R": "CcCKrea2LoRAPromptSettings"},
            "widgets_values": [
                True,
                True,
                "style",
                " cinematic lighting",
                "",
                False,
                "outfit",
                "",
                "",
                False,
                "pose",
                "",
                "",
                False,
                "identity",
                "",
                "",
            ],
        },
        {
            "id": 5,
            "type": "CcCKrea2LoRAStack",
            "pos": [460, 100],
            "size": [315, 260],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 1},
                {"name": "lora_prompt_settings", "type": "CCC_KREA2_LORA_PROMPT_SETTINGS", "link": 4},
            ],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": [5], "slot_index": 0},
                {"name": "prompt_augmentation", "type": "CCC_KREA2_PROMPT_AUGMENTATION", "links": [6], "slot_index": 1},
            ],
            "properties": {"Node name for S&R": "CcCKrea2LoRAStack"},
            "widgets_values": [
                True,
                1.0,
                True,
                "krea2_style_lora.safetensors",
                0.8,
                False,
                "",
                1.0,
                False,
                "",
                1.0,
                False,
                "",
                1.0,
            ],
        },
        {
            "id": 6,
            "type": "CcCKrea2TextToImage",
            "pos": [820, 100],
            "size": [400, 260],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 5},
                {"name": "clip", "type": "CLIP", "link": 2},
                {"name": "prompt_augmentation", "type": "CCC_KREA2_PROMPT_AUGMENTATION", "link": 6},
            ],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": [7], "slot_index": 0},
                {"name": "positive", "type": "CONDITIONING", "links": [8], "slot_index": 1},
                {"name": "negative", "type": "CONDITIONING", "links": [9], "slot_index": 2},
                {"name": "latent", "type": "LATENT", "links": [10], "slot_index": 3},
            ],
            "properties": {"Node name for S&R": "CcCKrea2TextToImage"},
            "widgets_values": ["A futuristic city skyline at night.", "blurry, low quality", 1024, 1024, 1.0, 1],
        },
        {
            "id": 7,
            "type": "KSampler",
            "pos": [1260, 100],
            "size": [315, 262],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 7},
                {"name": "positive", "type": "CONDITIONING", "link": 8},
                {"name": "negative", "type": "CONDITIONING", "link": 9},
                {"name": "latent_image", "type": "LATENT", "link": 10},
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [11], "slot_index": 0}],
            "properties": {"Node name for S&R": "KSampler"},
            "widgets_values": [67890, "fixed", 25, 5.0, "euler", "simple", 1.0],
        },
        {
            "id": 8,
            "type": "VAEDecode",
            "pos": [1610, 100],
            "size": [210, 46],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [{"name": "samples", "type": "LATENT", "link": 11}, {"name": "vae", "type": "VAE", "link": 3}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [12], "slot_index": 0}],
            "properties": {"Node name for S&R": "VAEDecode"},
            "widgets_values": [],
        },
        {
            "id": 9,
            "type": "SaveImage",
            "pos": [1850, 100],
            "size": [315, 270],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": 12}],
            "outputs": [],
            "properties": {"Node name for S&R": "SaveImage"},
            "widgets_values": ["Krea2_T2I_LoRA"],
        },
    ]
    links = [
        [1, 1, 0, 5, 0, "MODEL"],
        [2, 2, 0, 6, 1, "CLIP"],
        [3, 3, 0, 8, 1, "VAE"],
        [4, 4, 0, 5, 1, "CCC_KREA2_LORA_PROMPT_SETTINGS"],
        [5, 5, 0, 6, 0, "MODEL"],
        [6, 5, 1, 6, 2, "CCC_KREA2_PROMPT_AUGMENTATION"],
        [7, 6, 0, 7, 0, "MODEL"],
        [8, 6, 1, 7, 1, "CONDITIONING"],
        [9, 6, 2, 7, 2, "CONDITIONING"],
        [10, 6, 3, 7, 3, "LATENT"],
        [11, 7, 0, 8, 0, "LATENT"],
        [12, 8, 0, 9, 0, "IMAGE"],
    ]
    groups = [
        {"title": "Models", "bounding": [80, 50, 350, 400], "color": "#3f51b5"},
        {"title": "LoRA Pipeline", "bounding": [440, 50, 1750, 700], "color": "#9c27b0"},
    ]
    return make_wf(nodes, links, groups)


def write_all_12_workflows():
    (WORKFLOWS_DIR / "01_t2i_basic.json").write_text(json.dumps(wf_01(), indent=2), encoding="utf-8")
    (WORKFLOWS_DIR / "02_t2i_lora_stack.json").write_text(json.dumps(wf_02(), indent=2), encoding="utf-8")

    # 03_subject_edit.json
    wf03 = build_modular_edit_pipeline(
        [("subject", "CcCKrea2SubjectImage", "subject_image, Image {slot}", "", [1.0, 0.0, 0.0, 1.0, 1.0, "auto"])]
    )
    (WORKFLOWS_DIR / "03_subject_edit.json").write_text(json.dumps(wf03, indent=2), encoding="utf-8")

    # 04_subject_scene_edit.json
    wf04 = build_modular_edit_pipeline(
        [
            ("subject", "CcCKrea2SubjectImage", "subject_image, Image {slot}", "", [1.0, 0.0, 0.0, 1.0, 1.0, "auto"]),
            ("scene", "CcCKrea2SceneImage", "scene_image, Image {slot}", "", [1.0, 0.0, 1.0, 1.0, "auto"]),
        ]
    )
    (WORKFLOWS_DIR / "04_subject_scene_edit.json").write_text(json.dumps(wf04, indent=2), encoding="utf-8")

    # 05_subject_outfit_edit.json
    wf05 = build_modular_edit_pipeline(
        [
            ("subject", "CcCKrea2SubjectImage", "subject_image, Image {slot}", "", [1.0, 0.0, 0.0, 1.0, 1.0, "auto"]),
            ("outfit", "CcCKrea2OutfitImage", "outfit_image, Image {slot}", "", [1.0, 0.0, 1.0, "auto"]),
        ]
    )
    (WORKFLOWS_DIR / "05_subject_outfit_edit.json").write_text(json.dumps(wf05, indent=2), encoding="utf-8")

    # 06_subject_scene_outfit_edit.json
    wf06 = build_modular_edit_pipeline(
        [
            ("subject", "CcCKrea2SubjectImage", "subject_image, Image {slot}", "", [1.0, 0.0, 0.0, 1.0, 1.0, "auto"]),
            ("scene", "CcCKrea2SceneImage", "scene_image, Image {slot}", "", [1.0, 0.0, 1.0, 1.0, "auto"]),
            ("outfit", "CcCKrea2OutfitImage", "outfit_image, Image {slot}", "", [1.0, 0.0, 1.0, "auto"]),
        ]
    )
    (WORKFLOWS_DIR / "06_subject_scene_outfit_edit.json").write_text(json.dumps(wf06, indent=2), encoding="utf-8")

    # 07_style_moodboard_edit.json
    wf07 = build_modular_edit_pipeline([("style", "CcCKrea2StyleImage", "style_image", "", [0.5, "2x2", True, True])])
    (WORKFLOWS_DIR / "07_style_moodboard_edit.json").write_text(json.dumps(wf07, indent=2), encoding="utf-8")

    # 08_inpaint_subject_edit.json
    wf08 = build_modular_edit_pipeline(
        [("subject", "CcCKrea2SubjectImage", "subject_image, Image {slot}", "", [1.0, 0.0, 0.0, 1.0, 1.0, "auto"])],
        target_content="subject",
        inpaint_mask=True,
    )
    (WORKFLOWS_DIR / "08_inpaint_subject_edit.json").write_text(json.dumps(wf08, indent=2), encoding="utf-8")

    # 09_inpaint_scene_edit.json
    wf09 = build_modular_edit_pipeline(
        [("scene", "CcCKrea2SceneImage", "scene_image, Image {slot}", "", [1.0, 0.0, 1.0, 1.0, "auto"])],
        target_content="scene",
    )
    (WORKFLOWS_DIR / "09_inpaint_scene_edit.json").write_text(json.dumps(wf09, indent=2), encoding="utf-8")

    # 10_multi_subject_chasing_slots.json
    wf10 = build_modular_edit_pipeline(
        [
            ("subject1", "CcCKrea2SubjectImage", "subject_1, Image 1", "", [1.0, 0.0, 0.0, 1.0, 1.0, "auto"]),
            ("subject2", "CcCKrea2SubjectImage", "subject_2, Image 2", "", [1.0, 0.0, 0.0, 1.0, 1.0, "auto"]),
        ]
    )
    (WORKFLOWS_DIR / "10_multi_subject_chasing_slots.json").write_text(json.dumps(wf10, indent=2), encoding="utf-8")

    # 11_advanced_directives_fit_modes.json
    wf11 = build_modular_edit_pipeline(
        [
            (
                "subject",
                "CcCKrea2SubjectImage",
                "subject_image",
                "Apply high contrast lighting.",
                [1.5, 0.2, 0.1, 1.2, 1.0, "crop"],
            )
        ]
    )
    (WORKFLOWS_DIR / "11_advanced_directives_fit_modes.json").write_text(json.dumps(wf11, indent=2), encoding="utf-8")

    # 12_full_pipeline_composition.json
    wf12 = build_modular_edit_pipeline(
        [
            ("subject", "CcCKrea2SubjectImage", "subject_image, Image {slot}", "", [1.0, 0.0, 0.0, 1.0, 1.0, "auto"]),
            ("scene", "CcCKrea2SceneImage", "scene_image, Image {slot}", "", [1.0, 0.0, 1.0, 1.0, "auto"]),
            ("outfit", "CcCKrea2OutfitImage", "outfit_image, Image {slot}", "", [1.0, 0.0, 1.0, "auto"]),
            ("style", "CcCKrea2StyleImage", "style_image", "", [0.5, "2x2", True, True]),
        ]
    )
    (WORKFLOWS_DIR / "12_full_pipeline_composition.json").write_text(json.dumps(wf12, indent=2), encoding="utf-8")

    print("All 12 curated workflows generated successfully.")


if __name__ == "__main__":
    write_all_12_workflows()
