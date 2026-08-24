import json
import os
import sys

# Ensure ccc_krea2 is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS


# Recommended model dependency definitions
MODEL_UNET = {
    "name": "Neutrino_v2_base_nvfp4_svd.safetensors",
    "url": "https://huggingface.co/Crowlley/Krea2Neutrino/resolve/main/Neutrino_v2_base_nvfp4_svd.safetensors",
    "directory": "diffusion_models",
    "hash": "6844374f31e7de278c3408b6333b8fbb7bf5cdd1321646d5480bf1aeda683e1a",
    "hash_type": "SHA256",
}

MODEL_CLIP = {
    "name": "Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors",
    "url": "https://huggingface.co/brewbadgertim/Huihui-Qwen3-VL-4B-Instruct-abliterated-Quants/resolve/main/Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors",
    "directory": "text_encoders",
}

MODEL_VAE = {
    "name": "qwen_image_vae.safetensors",
    "url": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors",
    "directory": "vae",
    "hash": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
    "hash_type": "SHA256",
}

MODEL_LORA_IDENTITY_EDIT = {
    "name": "krea2_identity_edit_v1_2.safetensors",
    "url": "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/krea2_identity_edit_v1_2.safetensors",
    "directory": "loras",
    "hash": "6adf9a69cc9502d286db7b69964d37da7e9cfe4b05b4d004bc275f087d3fd3cf",
    "hash_type": "SHA256",
}

MODEL_LORA_BODY_SWAP = {
    "name": "bfs_body_swap_v1_krea2.safetensors",
    "url": "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_body_swap_v1_krea2.safetensors",
    "directory": "loras",
    "hash": "0b3d043714c912c55c525ac68a53f50dbfaa6a024d735c28dfed12cd214a0d79",
    "hash_type": "SHA256",
}


# Explicit schema fixture ONLY for the standard nodes used by these canonical workflows
class MockUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (["Neutrino_v2_base_nvfp4_svd.safetensors", "krea2_model.safetensors"],),
                "weight_dtype": (["default"],),
            }
        }

    RETURN_TYPES = ("MODEL",)


class MockCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (
                    [
                        "Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors",
                        "qwen3_vl.safetensors",
                    ],
                ),
                "type": (["krea2", "sdxl", "sd3"],),
                "device": (["default", "cpu"],),
            }
        }

    RETURN_TYPES = ("CLIP",)


class MockVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae_name": (["qwen_image_vae.safetensors", "ae.safetensors"],)}}

    RETURN_TYPES = ("VAE",)


class MockLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": (["subject.jpg", "scene.jpg", "outfit.jpg", "style.jpg"],)}}

    RETURN_TYPES = ("IMAGE", "MASK")


class MockKSampler:
    @classmethod
    # NOTE: 'control_after_generate' is a FRONTEND ONLY widget in ComfyUI and is intentionally omitted from the Python INPUT_TYPES.
    def INPUT_TYPES(s):
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
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    RETURN_TYPES = ("IMAGE",)


class MockSaveImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "CcCKrea2"})}}

    RETURN_TYPES = tuple()


class MockModelSamplingAuraFlow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",), "shift": ("FLOAT", {"default": 1.15})}}

    RETURN_TYPES = ("MODEL",)


class MockBlackwellAttentionFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",), "mode": (["pytorch", "sdpa", "flash_attn"],)}}

    RETURN_TYPES = ("MODEL",)


class MockNote:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = tuple()


CORE_MAPPINGS = {
    "UNETLoader": MockUNETLoader,
    "CLIPLoader": MockCLIPLoader,
    "VAELoader": MockVAELoader,
    "LoadImage": MockLoadImage,
    "KSampler": MockKSampler,
    "VAEDecode": MockVAEDecode,
    "SaveImage": MockSaveImage,
    "ModelSamplingAuraFlow": MockModelSamplingAuraFlow,
    "BlackwellAttentionFix": MockBlackwellAttentionFix,
    "Note": MockNote,
}


def get_node_class(node_type):
    if node_type in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS[node_type]
    if node_type in CORE_MAPPINGS:
        return CORE_MAPPINGS[node_type]
    return None


# List of widgets that are permitted to bypass strict combo enum validation because they represent dynamic file lists.
# Now strict tuples of (node_type, widget_name)
DYNAMIC_FILE_SELECTORS = {
    ("UNETLoader", "unet_name"),
    ("CLIPLoader", "clip_name"),
    ("VAELoader", "vae_name"),
    ("LoadImage", "image"),
    ("CcCKrea2LoRAStack", "lora_1_name"),
    ("CcCKrea2LoRAStack", "lora_2_name"),
    ("CcCKrea2LoRAStack", "lora_3_name"),
    ("CcCKrea2LoRAStack", "lora_4_name"),
}

# Frontend-only widgets mapping: node_type -> list of (widget_name, injection_index, default_value)
FRONTEND_WIDGETS_MAP = {"KSampler": [("control_after_generate", 1, "randomize")]}

CANONICAL_NAMES = [
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
    "11_easy_scene_reinterpretation.json",
]


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


class WorkflowBuilder:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.groups = []
        self.node_id = 1
        self.link_id = 1

    def add_node(self, node_type, pos, size, inputs=None, widgets_values=None, title=None, model_dependencies=None):
        inputs = inputs or {}
        values_by_name = widgets_values or {}

        cls = get_node_class(node_type)
        if not cls:
            raise ValueError(f"Unknown node type: {node_type}")

        in_types = cls.INPUT_TYPES()
        req = in_types.get("required", {})
        opt = in_types.get("optional", {})
        all_in = {**req, **opt}

        # Format inputs
        formatted_inputs = []
        for name in inputs:
            if name not in all_in:
                raise ValueError(f"Invalid input '{name}' for node '{node_type}'. Valid: {list(all_in.keys())}")
            expected_type = all_in[name][0]
            if isinstance(expected_type, list) or expected_type in ["STRING", "INT", "FLOAT", "BOOLEAN"]:
                raise ValueError(f"'{name}' is a widget, not a socket on {node_type}")
            formatted_inputs.append({"name": name, "type": expected_type})

        # Format outputs
        formatted_outputs = []
        if hasattr(cls, "RETURN_TYPES"):
            for i, t in enumerate(cls.RETURN_TYPES):
                name = getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)[i]
                formatted_outputs.append({"name": name, "type": t, "links": []})

        # Validate unknown widgets
        recognized_widget_names = {
            name
            for name, schema_val in all_in.items()
            if isinstance(schema_val[0], (list, tuple)) or schema_val[0] in ["STRING", "INT", "FLOAT", "BOOLEAN"]
        }

        frontend_configs = FRONTEND_WIDGETS_MAP.get(node_type, [])
        allowed_frontend = {fw[0] for fw in frontend_configs}

        unknown = set(values_by_name.keys()) - recognized_widget_names - allowed_frontend
        if unknown:
            raise ValueError(f"Unknown widget values supplied for {node_type}: {unknown}")

        # Serialize widgets
        serialized_widgets = []
        for name, schema_val in all_in.items():
            val_type = schema_val[0]
            is_widget = (
                isinstance(val_type, list)
                or isinstance(val_type, tuple)
                or val_type in ["STRING", "INT", "FLOAT", "BOOLEAN"]
            )
            if not is_widget:
                continue

            if name in values_by_name:
                val = values_by_name[name]
                if isinstance(val_type, list) or isinstance(val_type, tuple):
                    if val not in val_type:
                        if (node_type, name) not in DYNAMIC_FILE_SELECTORS:
                            raise ValueError(f"Value '{val}' not in choices {val_type} for '{name}' on '{node_type}'")
                elif val_type == "BOOLEAN":
                    if not isinstance(val, bool):
                        raise ValueError(f"Expected bool for '{name}', got {type(val)}")
                elif val_type == "STRING":
                    if not isinstance(val, str):
                        raise ValueError(f"Expected str for '{name}', got {type(val)}")
                elif val_type == "INT":
                    if not isinstance(val, int):
                        raise ValueError(f"Expected int for '{name}', got {type(val)}")
                elif val_type == "FLOAT":
                    if not isinstance(val, (int, float)):
                        raise ValueError(f"Expected float for '{name}', got {type(val)}")

                serialized_widgets.append(val)
            elif len(schema_val) > 1 and "default" in schema_val[1]:
                serialized_widgets.append(schema_val[1]["default"])
            else:
                raise ValueError(f"Missing required widget '{name}' for node '{node_type}'")

        # Inject frontend-only widgets
        for fw_name, fw_idx, fw_default in frontend_configs:
            val = values_by_name.get(fw_name, fw_default)
            if fw_idx <= len(serialized_widgets):
                serialized_widgets.insert(fw_idx, val)
            else:
                serialized_widgets.append(val)

        node = {
            "id": self.node_id,
            "type": node_type,
            "pos": pos,
            "size": size,
            "flags": {},
            "order": self.node_id,
            "mode": 0,
            "inputs": formatted_inputs,
            "outputs": formatted_outputs,
            "properties": {"Node name for S&R": node_type},
            "widgets_values": serialized_widgets,
        }
        if model_dependencies:
            for dep in model_dependencies:
                if not isinstance(dep, dict):
                    raise ValueError(f"model_dependency entry must be dict, got {type(dep)}")
                if not all(k in dep for k in ("name", "url", "directory")):
                    raise ValueError(f"model_dependency missing required keys: {dep}")
                if "/resolve/main/" not in dep["url"]:
                    raise ValueError(f"model_dependency URL must contain /resolve/main/: {dep['url']}")
            node["properties"]["models"] = [dict(d) for d in model_dependencies]
        if title:
            node["title"] = title

        self.nodes.append(node)
        self.node_id += 1
        return node

    def link(self, from_node, from_out_name, to_node, to_in_name):
        from_idx = next((i for i, o in enumerate(from_node["outputs"]) if o["name"] == from_out_name), None)
        to_idx = next((i for i, inp in enumerate(to_node["inputs"]) if inp["name"] == to_in_name), None)

        if from_idx is None:
            raise ValueError(f"Output {from_out_name} not found on node {from_node['type']}")
        if to_idx is None:
            raise ValueError(f"Input {to_in_name} not found on node {to_node['type']}")

        if to_node["inputs"][to_idx].get("link") is not None:
            raise ValueError(f"Input {to_in_name} on node {to_node['type']} already has a link")

        out_type = from_node["outputs"][from_idx]["type"]
        in_type = to_node["inputs"][to_idx]["type"]

        if out_type != in_type and in_type != "*" and out_type != "*":
            raise ValueError(
                f"Link type mismatch: {from_node['type']}.{from_out_name} ({out_type}) -> {to_node['type']}.{to_in_name} ({in_type})"
            )

        lnk = [self.link_id, from_node["id"], from_idx, to_node["id"], to_idx, out_type]
        self.links.append(lnk)

        to_node["inputs"][to_idx]["link"] = self.link_id
        from_node["outputs"][from_idx]["links"].append(self.link_id)

        self.link_id += 1
        return self.link_id

    def add_group(self, title, bounding, color="#3f789e"):
        self.groups.append({"title": title, "bounding": bounding, "color": color})

    def build(self):
        return {
            "last_node_id": self.node_id - 1,
            "last_link_id": self.link_id - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "version": 0.4,
        }


def build_base_graph(
    b: WorkflowBuilder,
    is_ostris=False,
    is_native=False,
    lora_2_name="None",
    lora_2_strength=1.0,
    note_text=None,
):
    unet = b.add_node(
        "UNETLoader",
        [0, 0],
        [300, 100],
        widgets_values={"unet_name": "Neutrino_v2_base_nvfp4_svd.safetensors", "weight_dtype": "default"},
        model_dependencies=[MODEL_UNET],
    )
    clip = b.add_node(
        "CLIPLoader",
        [0, 150],
        [300, 100],
        widgets_values={
            "clip_name": "Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors",
            "type": "krea2",
            "device": "default",
        },
        model_dependencies=[MODEL_CLIP],
    )
    vae = b.add_node(
        "VAELoader",
        [0, 300],
        [300, 100],
        widgets_values={"vae_name": "qwen_image_vae.safetensors"},
        model_dependencies=[MODEL_VAE],
    )

    lora = None
    if not is_native:
        lora_name = "krea2_ostris_edit_lora.safetensors" if is_ostris else "krea2_identity_edit_v1_2.safetensors"
        lora_deps = []
        if not is_ostris:
            lora_deps.append(MODEL_LORA_IDENTITY_EDIT)

        widgets = {
            "enabled": True,
            "global_strength": 1.0,
            "lora_1_enabled": True,
            "lora_1_name": lora_name,
            "lora_1_strength": 1.0,
        }
        if lora_2_name and lora_2_name != "None":
            widgets["lora_2_enabled"] = True
            widgets["lora_2_name"] = lora_2_name
            widgets["lora_2_strength"] = lora_2_strength
            if lora_2_name == "bfs_body_swap_v1_krea2.safetensors":
                lora_deps.append(MODEL_LORA_BODY_SWAP)

        lora = b.add_node(
            "CcCKrea2LoRAStack",
            [350, 0],
            [300, 250],
            inputs={"model": None},
            widgets_values=widgets,
            model_dependencies=lora_deps if lora_deps else None,
        )

        b.link(unet, "MODEL", lora, "model")

    auraflow = b.add_node(
        "ModelSamplingAuraFlow",
        [1500, 0],
        [300, 100],
        inputs={"model": None},
        widgets_values={"shift": 1.15},
    )

    blackwell = b.add_node(
        "BlackwellAttentionFix",
        [1850, 0],
        [300, 100],
        inputs={"model": None},
        widgets_values={"mode": "pytorch"},
    )

    default_note = (
        "Note on Blackwell GPUs:\n"
        "`BlackwellAttentionFix` is recommended for NVIDIA Blackwell GPUs (RTX 5090, 5080, 5070 family) "
        "to prevent NaN/black image sampling issues. On non-Blackwell systems, this node acts as a pass-through "
        "or can be muted/bypassed."
    )
    b.add_node(
        "Note",
        [1500, 150],
        [300, 150],
        widgets_values={"text": note_text if note_text else default_note},
    )

    sampler = b.add_node(
        "KSampler",
        [2200, 0],
        [300, 200],
        inputs={"model": None, "positive": None, "negative": None, "latent_image": None},
        widgets_values={
            "seed": 0,
            "control_after_generate": "randomize",
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
    )

    decode = b.add_node("VAEDecode", [2550, 0], [300, 100], inputs={"samples": None, "vae": None})
    save = b.add_node(
        "SaveImage", [2900, 0], [300, 200], inputs={"images": None}, widgets_values={"filename_prefix": "CcCKrea2"}
    )

    b.link(auraflow, "MODEL", blackwell, "model")
    b.link(blackwell, "MODEL", sampler, "model")
    b.link(sampler, "LATENT", decode, "samples")
    b.link(vae, "VAE", decode, "vae")
    b.link(decode, "IMAGE", save, "images")

    return unet, clip, vae, lora, auraflow, sampler


def build_easy_workflow(
    filename,
    preset,
    outfit_source,
    style_source,
    is_ostris=False,
    has_subj=True,
    has_scene=False,
    has_outfit=False,
    has_style=False,
    lora_2_name="None",
    lora_2_strength=1.0,
    note_text=None,
):
    b = WorkflowBuilder()
    b.add_group("Loaders", [0, -50, 700, 500])
    b.add_group("Easy Edit", [750, -50, 700, 500])
    b.add_group("Sampling", [1480, -50, 1050, 500])
    b.add_group("Output", [2540, -50, 680, 500])

    unet, clip, vae, lora, auraflow, sampler = build_base_graph(
        b,
        is_ostris=is_ostris,
        lora_2_name=lora_2_name,
        lora_2_strength=lora_2_strength,
        note_text=note_text,
    )

    easy_type = "CcCKrea2EasyEditOstris" if is_ostris else "CcCKrea2EasyEdit"

    easy_inputs = {"model": None, "clip": None, "vae": None}
    if has_subj:
        easy_inputs["subject"] = None
    if has_scene:
        easy_inputs["scene"] = None
    if has_outfit:
        easy_inputs["outfit"] = None
    if has_style:
        easy_inputs["style"] = None

    easy_widgets = {
        "positive_prompt": "A photo of a person",
        "use_default_prompt": True,
        "preset": preset,
        "reference_subject": "main subject",
        "subject_description": "main subject",
        "outfit_source": outfit_source,
        "style_source": style_source,
        "negative_prompt": "bad quality",
    }

    if is_ostris:
        easy_widgets["apply_ostris_edit_patch"] = True
        easy_widgets["ostris_kv_cache"] = False
    else:
        easy_widgets["apply_krea2_edit_patch"] = True

    easy = b.add_node(easy_type, [800, 0], [400, 400], inputs=easy_inputs, widgets_values=easy_widgets)

    if lora:
        b.link(lora, "model", easy, "model")
    else:
        b.link(unet, "MODEL", easy, "model")

    b.link(clip, "CLIP", easy, "clip")
    b.link(vae, "VAE", easy, "vae")

    y = 500
    img_count = sum([has_subj, has_scene, has_outfit, has_style])
    if img_count > 0:
        b.add_group("Load Images", [0, 470, 350, img_count * 250 + 40])

    if has_subj:
        sub = b.add_node(
            "LoadImage", [20, y], [300, 200], widgets_values={"image": "subject.jpg"}, title="Subject Image"
        )
        b.link(sub, "IMAGE", easy, "subject")
        y += 250
    if has_scene:
        scn = b.add_node("LoadImage", [20, y], [300, 200], widgets_values={"image": "scene.jpg"}, title="Scene Image")
        b.link(scn, "IMAGE", easy, "scene")
        y += 250
    if has_outfit:
        outf = b.add_node(
            "LoadImage", [20, y], [300, 200], widgets_values={"image": "outfit.jpg"}, title="Outfit Image"
        )
        b.link(outf, "IMAGE", easy, "outfit")
        y += 250
    if has_style:
        sty = b.add_node("LoadImage", [20, y], [300, 200], widgets_values={"image": "style.jpg"}, title="Style Image")
        b.link(sty, "IMAGE", easy, "style")

    b.link(easy, "patched_model", auraflow, "model")
    b.link(easy, "positive", sampler, "positive")
    b.link(easy, "negative", sampler, "negative")
    b.link(easy, "latent", sampler, "latent_image")

    save_json(b.build(), f"workflows/{filename}")


def build_advanced_workflow(filename, is_ostris=False, is_native=False):
    b = WorkflowBuilder()
    b.add_group("Loaders", [0, -50, 700, 500])
    b.add_group("Load Images", [0, 500, 320, 540])
    b.add_group("References", [340, 500, 1050, 600])
    b.add_group("Edit Orchestrator", [1450, -50, 500, 700])
    b.add_group("Sampling", [2000, -50, 1050, 500])
    b.add_group("Output", [3060, -50, 680, 500])

    if is_native:
        b.add_group("Native / compatible reference runtime required", [320, 0, 30, 30])

    unet, clip, vae, lora, auraflow, sampler = build_base_graph(b, is_ostris, is_native)

    # Subject ref
    sub_img = b.add_node(
        "LoadImage", [20, 550], [280, 200], widgets_values={"image": "subject.jpg"}, title="Subject Image"
    )
    sub_prep = b.add_node(
        "CcCKrea2QwenVisionImagePrep",
        [360, 550],
        [300, 200],
        inputs={"clip": None, "image": None},
        widgets_values={
            "mode": "native",
            "min_mp": 0.0,
            "max_mp": 1.0,
            "fixed_mp": 1.0,
            "downscale_method": "auto",
            "upscale_method": "auto",
        },
    )
    sub_ref = b.add_node(
        "CcCKrea2ReferenceImage",
        [700, 550],
        [300, 200],
        inputs={"prepared_image": None},
        widgets_values={
            "reference_path": "edit",
            "vision_slot": "auto",
            "alias": "subject",
            "vision_instruction": "Use this reference for subject identity, facial features, hair, anatomy, body shape and body proportions.",
            "attention_boost": 1.0,
            "masked_attention_boost": 1.0,
            "visual_reference_fit": "auto",
            "style_fidelity": 0.5,
            "style_processing": "2x2",
            "indirect_style_transfer": True,
        },
    )

    b.link(clip, "CLIP", sub_prep, "clip")
    b.link(sub_img, "IMAGE", sub_prep, "image")
    b.link(sub_prep, "prepared_image", sub_ref, "prepared_image")

    # Outfit ref
    outf_img = b.add_node(
        "LoadImage", [20, 800], [280, 200], widgets_values={"image": "outfit.jpg"}, title="Outfit Image"
    )
    outf_prep = b.add_node(
        "CcCKrea2QwenVisionImagePrep",
        [360, 800],
        [300, 200],
        inputs={"clip": None, "image": None},
        widgets_values={
            "mode": "native",
            "min_mp": 0.0,
            "max_mp": 1.0,
            "fixed_mp": 1.0,
            "downscale_method": "auto",
            "upscale_method": "auto",
        },
    )
    outf_ref = b.add_node(
        "CcCKrea2ReferenceImage",
        [700, 800],
        [300, 200],
        inputs={"prepared_image": None, "previous_references": None},
        widgets_values={
            "reference_path": "edit",
            "vision_slot": "auto",
            "alias": "outfit",
            "vision_instruction": "Use this reference for clothing, garments and accessories.\nDo not use the wearer's identity as the subject identity.",
            "attention_boost": 1.0,
            "masked_attention_boost": 1.0,
            "visual_reference_fit": "auto",
            "style_fidelity": 0.5,
            "style_processing": "2x2",
            "indirect_style_transfer": True,
        },
    )

    b.link(clip, "CLIP", outf_prep, "clip")
    b.link(outf_img, "IMAGE", outf_prep, "image")
    b.link(outf_prep, "prepared_image", outf_ref, "prepared_image")
    b.link(sub_ref, "reference_chain", outf_ref, "previous_references")

    # Target Latent
    t_latent = b.add_node(
        "CcCKrea2TargetLatent",
        [1050, 550],
        [300, 250],
        inputs={"vae": None, "geometry_image": None},
        widgets_values={
            "target_content": "empty",
            "geometry_mode": "favor_image",
            "target_megapixels": 2.0,
            "fixed_megapixels": 2.0,
            "aspect_ratio": "1:1",
            "batch_size": 1,
            "include_in_vision": "auto",
            "target_vision_slot": "auto",
            "target_alias": "",
            "target_vision_instruction": "",
        },
    )
    b.link(vae, "VAE", t_latent, "vae")
    b.link(sub_prep, "prepared_image", t_latent, "geometry_image")

    # Edit Orchestrator
    edit_inputs = {"model": None, "clip": None, "vae": None, "references": None, "target_latent": None}

    ref_method = "krea2_edit"
    if is_native:
        ref_method = "native"
    elif is_ostris:
        ref_method = "ostris_edit"

    edit_widgets = {
        "positive_prompt": "A photo of a person",
        "negative_prompt": "bad quality",
        "global_vision_directive": "",
        "reference_method": ref_method,
        "ostris_kv_cache": False,
    }
    edit = b.add_node("CcCKrea2Edit", [1500, 0], [400, 400], inputs=edit_inputs, widgets_values=edit_widgets)

    if is_native:
        b.link(unet, "MODEL", edit, "model")
    else:
        b.link(lora, "model", edit, "model")

    b.link(clip, "CLIP", edit, "clip")
    b.link(vae, "VAE", edit, "vae")
    b.link(outf_ref, "reference_chain", edit, "references")
    b.link(t_latent, "target_latent", edit, "target_latent")

    b.link(edit, "patched_model", auraflow, "model")
    b.link(edit, "positive", sampler, "positive")
    b.link(edit, "negative", sampler, "negative")
    b.link(edit, "latent", sampler, "latent_image")

    save_json(b.build(), f"workflows/{filename}")


def main():
    os.makedirs("workflows", exist_ok=True)

    # 01
    build_easy_workflow("01_easy_subject.json", "balanced", "outfit image", "style image", has_subj=True)
    subject_transfer_note = (
        "Note on Easy Transfer Presets, Helper LoRAs & Model Downloads:\n"
        "• Identity Transfer: Uses Scene as target content/latent and geometry anchor, with Scene and Subject appearance boosts of 2.0. Outfit is disabled, Style is automatic Scene.\n"
        "• Subject Transfer 1 / 2: Current full-Subject transfer test candidates. Both use an empty target latent with Scene geometry and Subject appearance boosts of 5.0 / 6.0.\n"
        "• Identity Transfer Test 5: Restored calibration preset using Scene target content/geometry and only Subject appearance at 7.0. Outfit and Style are disabled.\n"
        "• Easy Visual References: Easy appearance references preserve the complete source image and fit them into the target latent geometry without destructive cropping.\n"
        "• Krea 2 Identity Edit LoRA (krea2_identity_edit_v1_2.safetensors at 1.0) recovers subject facial identity.\n"
        "• BFS Body Swap LoRA (bfs_body_swap_v1_krea2.safetensors at 0.35) is an experimental full-person replacement model where Scene acts as base image and Subject as reference person. Exact pose transfer is not guaranteed.\n"
        "• Upstream BFS Trigger: 'body_swap: replace the person with the reference person.'\n"
        "• ModelSamplingAuraFlow shift is set to 1.15.\n"
        "• BlackwellAttentionFix is included for NVIDIA Blackwell GPUs (RTX 5090/5080/5070).\n\n"
        "Model downloads:\n"
        "• Krea/Neutrino: https://huggingface.co/Crowlley/Krea2Neutrino\n"
        "• Qwen3-VL: https://huggingface.co/brewbadgertim/Huihui-Qwen3-VL-4B-Instruct-abliterated-Quants/tree/main\n"
        "• Krea/ComfyUI alternatives: https://huggingface.co/Comfy-Org/Krea-2/\n"
        "• Identity Edit: https://huggingface.co/conradlocke/krea2-identity-edit\n"
        "• BFS Body Swap: https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap"
    )
    # 02
    build_easy_workflow(
        "02_easy_subject_scene.json",
        "subject_transfer_1",
        "outfit image",
        "style image",
        has_subj=True,
        has_scene=True,
        lora_2_name="bfs_body_swap_v1_krea2.safetensors",
        lora_2_strength=0.35,
        note_text=subject_transfer_note,
    )
    # 03
    build_easy_workflow(
        "03_easy_subject_outfit.json", "balanced", "outfit image", "style image", has_subj=True, has_outfit=True
    )
    # 04
    build_easy_workflow(
        "04_easy_subject_scene_outfit.json",
        "balanced",
        "outfit image",
        "style image",
        has_subj=True,
        has_scene=True,
        has_outfit=True,
    )
    # 05
    build_easy_workflow(
        "05_easy_outfit_from_scene.json", "outfit_transfer", "scene image", "style image", has_subj=True, has_scene=True
    )
    # 06
    build_easy_workflow(
        "06_easy_style_transfer.json", "style_transfer", "outfit image", "style image", has_subj=True, has_style=True
    )
    # 07
    build_easy_workflow(
        "07_easy_ostris.json", "balanced", "outfit image", "style image", is_ostris=True, has_subj=True, has_scene=True
    )
    # 08
    build_advanced_workflow("08_advanced_krea2_edit.json", is_ostris=False)
    # 09
    build_advanced_workflow("09_advanced_native.json", is_native=True)
    # 10
    build_advanced_workflow("10_advanced_ostris.json", is_ostris=True)
    # 11
    build_easy_workflow(
        "11_easy_scene_reinterpretation.json",
        "scene_reinterpretation",
        "none",
        "scene image",
        has_subj=True,
        has_scene=True,
    )


if __name__ == "__main__":
    main()
