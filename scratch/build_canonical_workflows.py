import json
import os
import sys

# Ensure ccc_krea2 is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS

try:
    import nodes as comfy_nodes
    CORE_MAPPINGS = comfy_nodes.NODE_CLASS_MAPPINGS
except ImportError:
    # Explicit schema fixture ONLY for the standard nodes used by these ten workflows
    class MockUNETLoader:
        @classmethod
        def INPUT_TYPES(s): return {"required": {"unet_name": (["krea2_model.safetensors"],), "weight_dtype": (["default"],)}}
        RETURN_TYPES = ("MODEL",)
    class MockCLIPLoader:
        @classmethod
        def INPUT_TYPES(s): return {"required": {"clip_name": (["qwen3_vl.safetensors"],), "type": (["krea2", "sdxl", "sd3"],), "device": (["default", "cpu"],)}}
        RETURN_TYPES = ("CLIP",)
    class MockVAELoader:
        @classmethod
        def INPUT_TYPES(s): return {"required": {"vae_name": (["ae.safetensors"],)}}
        RETURN_TYPES = ("VAE",)
    class MockLoadImage:
        @classmethod
        def INPUT_TYPES(s): return {"required": {"image": (["subject.jpg", "scene.jpg", "outfit.jpg", "style.jpg"],)}}
        RETURN_TYPES = ("IMAGE", "MASK")
    class MockKSampler:
        @classmethod
        def INPUT_TYPES(s): return {"required": {"model": ("MODEL",), "seed": ("INT", {"default": 0}), "control_after_generate": (["randomize", "fixed", "increment", "decrement"],), "steps": ("INT", {"default": 20}), "cfg": ("FLOAT", {"default": 1.0}), "sampler_name": (["euler", "euler_ancestral"],), "scheduler": (["normal", "karras"],), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",), "denoise": ("FLOAT", {"default": 1.0})}}
        RETURN_TYPES = ("LATENT",)
    class MockVAEDecode:
        @classmethod
        def INPUT_TYPES(s): return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}
        RETURN_TYPES = ("IMAGE",)
    class MockSaveImage:
        @classmethod
        def INPUT_TYPES(s): return {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "CcCKrea2"})}}
        RETURN_TYPES = tuple()

    CORE_MAPPINGS = {
        "UNETLoader": MockUNETLoader,
        "CLIPLoader": MockCLIPLoader,
        "VAELoader": MockVAELoader,
        "LoadImage": MockLoadImage,
        "KSampler": MockKSampler,
        "VAEDecode": MockVAEDecode,
        "SaveImage": MockSaveImage
    }

def get_node_class(node_type):
    if node_type in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS[node_type]
    if node_type in CORE_MAPPINGS:
        return CORE_MAPPINGS[node_type]
    return None

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

    def add_node(self, node_type, pos, size, inputs=None, widgets_values=None):
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
            # Derive type
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

        # Serialize widgets
        serialized_widgets = []
        for name, schema_val in all_in.items():
            val_type = schema_val[0]
            is_widget = isinstance(val_type, list) or isinstance(val_type, tuple) or val_type in ["STRING", "INT", "FLOAT", "BOOLEAN"]
            if not is_widget:
                continue

            if name in values_by_name:
                val = values_by_name[name]
                # Validate primitive type/combo membership
                if isinstance(val_type, list) or isinstance(val_type, tuple):
                    if val not in val_type and not (isinstance(val, str) and val.endswith(".safetensors")):
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
            "widgets_values": serialized_widgets
        }
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

        out_type = from_node["outputs"][from_idx]["type"]
        in_type = to_node["inputs"][to_idx]["type"]

        if out_type != in_type and in_type != "*" and out_type != "*":
            # Just ignore if any is "*" (ComfyUI generic). For strict checking, exact match:
            if out_type != in_type:
                raise ValueError(f"Link type mismatch: {from_node['type']}.{from_out_name} ({out_type}) -> {to_node['type']}.{to_in_name} ({in_type})")

        lnk = [self.link_id, from_node["id"], from_idx, to_node["id"], to_idx, out_type]
        self.links.append(lnk)

        to_node["inputs"][to_idx]["link"] = self.link_id
        from_node["outputs"][from_idx]["links"].append(self.link_id)

        self.link_id += 1
        return self.link_id

    def add_group(self, title, bounding, color="#3f789e"):
        self.groups.append({
            "title": title,
            "bounding": bounding,
            "color": color
        })

    def build(self):
        return {
            "last_node_id": self.node_id - 1,
            "last_link_id": self.link_id - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "version": 0.4
        }

def build_base_graph(b: WorkflowBuilder, is_ostris=False):
    unet = b.add_node("UNETLoader", [0, 0], [300, 100], widgets_values={"unet_name": "krea2_model.safetensors", "weight_dtype": "default"})
    clip = b.add_node("CLIPLoader", [0, 150], [300, 100], widgets_values={"clip_name": "qwen3_vl.safetensors", "type": "krea2", "device": "default"})
    vae = b.add_node("VAELoader", [0, 300], [300, 100], widgets_values={"vae_name": "ae.safetensors"})

    lora_name = "krea2_ostris_edit_lora.safetensors" if is_ostris else "krea2_edit_lora.safetensors"

    lora = b.add_node("CcCKrea2LoRAStack", [350, 0], [300, 250], inputs={"model": None}, widgets_values={
        "enabled": True,
        "global_strength": 1.0,
        "lora_1_enabled": True,
        "lora_1_name": lora_name,
        "lora_1_strength": 1.0
    })

    b.link(unet, "MODEL", lora, "model")

    sampler = b.add_node("KSampler", [1500, 0], [300, 200], inputs={"model": None, "positive": None, "negative": None, "latent_image": None}, widgets_values={
        "seed": 0,
        "control_after_generate": "randomize",
        "steps": 20,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0
    })
    decode = b.add_node("VAEDecode", [1850, 0], [300, 100], inputs={"samples": None, "vae": None})
    save = b.add_node("SaveImage", [2200, 0], [300, 200], inputs={"images": None}, widgets_values={"filename_prefix": "CcCKrea2"})

    b.link(sampler, "LATENT", decode, "samples")
    b.link(vae, "VAE", decode, "vae")
    b.link(decode, "IMAGE", save, "images")

    return unet, clip, vae, lora, sampler

def build_easy_workflow(filename, preset, outfit_source, style_source, is_ostris=False, has_subj=True, has_scene=False, has_outfit=False, has_style=False):
    b = WorkflowBuilder()
    b.add_group("Loaders", [0, -50, 700, 500])
    b.add_group("Easy Edit", [750, -50, 700, 500])
    b.add_group("Sampling", [1480, -50, 1100, 500])

    unet, clip, vae, lora, sampler = build_base_graph(b, is_ostris)

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
        "preset": preset,
        "outfit_source": outfit_source,
        "style_source": style_source,
        "negative_prompt": "bad quality"
    }

    if is_ostris:
        easy_widgets["apply_ostris_edit_patch"] = True
        easy_widgets["ostris_kv_cache"] = False
    else:
        easy_widgets["apply_krea2_edit_patch"] = True

    easy = b.add_node(easy_type, [800, 0], [400, 400], inputs=easy_inputs, widgets_values=easy_widgets)

    b.link(lora, "model", easy, "model")
    b.link(clip, "CLIP", easy, "clip")
    b.link(vae, "VAE", easy, "vae")

    y = 500
    if has_subj:
        sub = b.add_node("LoadImage", [0, y], [300, 200], widgets_values={"image": "subject.jpg"})
        b.link(sub, "IMAGE", easy, "subject")
        y += 250
    if has_scene:
        scn = b.add_node("LoadImage", [0, y], [300, 200], widgets_values={"image": "scene.jpg"})
        b.link(scn, "IMAGE", easy, "scene")
        y += 250
    if has_outfit:
        outf = b.add_node("LoadImage", [0, y], [300, 200], widgets_values={"image": "outfit.jpg"})
        b.link(outf, "IMAGE", easy, "outfit")
        y += 250
    if has_style:
        sty = b.add_node("LoadImage", [0, y], [300, 200], widgets_values={"image": "style.jpg"})
        b.link(sty, "IMAGE", easy, "style")

    b.link(easy, "patched_model", sampler, "model")
    b.link(easy, "positive", sampler, "positive")
    b.link(easy, "negative", sampler, "negative")
    b.link(easy, "latent", sampler, "latent_image")

    save_json(b.build(), f"workflows/{filename}")

def build_advanced_workflow(filename, is_ostris=False, is_native=False):
    b = WorkflowBuilder()
    b.add_group("Loaders", [0, -50, 700, 500])
    b.add_group("References", [0, 500, 1400, 600])
    b.add_group("Edit Orchestrator", [1450, -50, 500, 700])
    b.add_group("Sampling", [2000, -50, 1100, 500])

    unet, clip, vae, lora, sampler = build_base_graph(b, is_ostris)

    # Subject ref
    sub_img = b.add_node("LoadImage", [0, 550], [300, 200], widgets_values={"image": "subject.jpg"})
    sub_prep = b.add_node("CcCKrea2QwenVisionImagePrep", [350, 550], [300, 200], inputs={"clip": None, "image": None}, widgets_values={
        "mode": "native",
        "min_mp": 0.0,
        "max_mp": 1.0,
        "fixed_mp": 1.0,
        "downscale_method": "auto",
        "upscale_method": "auto"
    })
    sub_ref = b.add_node("CcCKrea2ReferenceImage", [700, 550], [300, 200], inputs={"prepared_image": None}, widgets_values={
        "reference_path": "edit",
        "vision_slot": "auto",
        "alias": "subject",
        "vision_instruction": "Use this reference for subject identity, facial features, hair, anatomy, body shape and body proportions.",
        "attention_boost": 1.0,
        "masked_attention_boost": 1.0,
        "visual_reference_fit": "auto",
        "style_fidelity": 0.5,
        "style_processing": "2x2",
        "indirect_style_transfer": True
    })

    b.link(clip, "CLIP", sub_prep, "clip")
    b.link(sub_img, "IMAGE", sub_prep, "image")
    b.link(sub_prep, "prepared_image", sub_ref, "prepared_image")

    # Outfit ref
    outf_img = b.add_node("LoadImage", [0, 800], [300, 200], widgets_values={"image": "outfit.jpg"})
    outf_prep = b.add_node("CcCKrea2QwenVisionImagePrep", [350, 800], [300, 200], inputs={"clip": None, "image": None}, widgets_values={
        "mode": "native",
        "min_mp": 0.0,
        "max_mp": 1.0,
        "fixed_mp": 1.0,
        "downscale_method": "auto",
        "upscale_method": "auto"
    })
    outf_ref = b.add_node("CcCKrea2ReferenceImage", [700, 800], [300, 200], inputs={"prepared_image": None, "previous_references": None}, widgets_values={
        "reference_path": "edit",
        "vision_slot": "auto",
        "alias": "outfit",
        "vision_instruction": "Use this reference for clothing, garments and accessories.\nDo not use the wearer's identity as the subject identity.",
        "attention_boost": 1.0,
        "masked_attention_boost": 1.0,
        "visual_reference_fit": "auto",
        "style_fidelity": 0.5,
        "style_processing": "2x2",
        "indirect_style_transfer": True
    })

    b.link(clip, "CLIP", outf_prep, "clip")
    b.link(outf_img, "IMAGE", outf_prep, "image")
    b.link(outf_prep, "prepared_image", outf_ref, "prepared_image")
    b.link(sub_ref, "reference_chain", outf_ref, "previous_references")

    # Target Latent
    t_latent = b.add_node("CcCKrea2TargetLatent", [1050, 550], [300, 250], inputs={"vae": None, "geometry_image": None}, widgets_values={
        "target_content": "empty",
        "geometry_mode": "favor_image",
        "target_megapixels": 2.0,
        "fixed_megapixels": 2.0,
        "aspect_ratio": "1:1",
        "batch_size": 1,
        "include_in_vision": "auto",
        "target_vision_slot": "auto",
        "target_alias": "",
        "target_vision_instruction": ""
    })
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
        "ostris_kv_cache": False
    }
    edit = b.add_node("CcCKrea2Edit", [1500, 0], [400, 400], inputs=edit_inputs, widgets_values=edit_widgets)

    b.link(lora, "model", edit, "model")
    b.link(clip, "CLIP", edit, "clip")
    b.link(vae, "VAE", edit, "vae")
    b.link(outf_ref, "reference_chain", edit, "references")
    b.link(t_latent, "target_latent", edit, "target_latent")

    b.link(edit, "patched_model", sampler, "model")
    b.link(edit, "positive", sampler, "positive")
    b.link(edit, "negative", sampler, "negative")
    b.link(edit, "latent", sampler, "latent_image")

    save_json(b.build(), f"workflows/{filename}")

def main():
    os.makedirs("workflows", exist_ok=True)

    # 01
    build_easy_workflow("01_easy_subject.json", "balanced", "outfit image", "style image", has_subj=True)
    # 02
    build_easy_workflow("02_easy_subject_scene.json", "balanced", "outfit image", "style image", has_subj=True, has_scene=True)
    # 03
    build_easy_workflow("03_easy_subject_outfit.json", "balanced", "outfit image", "style image", has_subj=True, has_outfit=True)
    # 04
    build_easy_workflow("04_easy_subject_scene_outfit.json", "balanced", "outfit image", "style image", has_subj=True, has_scene=True, has_outfit=True)
    # 05
    build_easy_workflow("05_easy_outfit_from_scene.json", "outfit_transfer", "scene image", "style image", has_subj=True, has_scene=True)
    # 06
    build_easy_workflow("06_easy_style_transfer.json", "style_transfer", "outfit image", "style image", has_subj=True, has_style=True)
    # 07
    build_easy_workflow("07_easy_ostris.json", "balanced", "outfit image", "style image", is_ostris=True, has_subj=True, has_scene=True)
    # 08
    build_advanced_workflow("08_advanced_krea2_edit.json", is_ostris=False)
    # 09
    build_advanced_workflow("09_advanced_native.json", is_native=True)
    # 10
    build_advanced_workflow("10_advanced_ostris.json", is_ostris=True)

if __name__ == "__main__":
    main()
