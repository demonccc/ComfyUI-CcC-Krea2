import json
import os
import sys

# Ensure ccc_krea2 is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS

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
        widgets_values = widgets_values or []

        # Schema awareness
        if node_type in NODE_CLASS_MAPPINGS:
            cls = NODE_CLASS_MAPPINGS[node_type]
            in_types = cls.INPUT_TYPES()
            req = in_types.get("required", {})
            opt = in_types.get("optional", {})
            all_in = {**req, **opt}

            # Format inputs
            formatted_inputs = []
            for name in inputs:
                if name not in all_in:
                    raise ValueError(f"Invalid input '{name}' for node '{node_type}'. Valid: {list(all_in.keys())}")
                formatted_inputs.append({"name": name})

            # Format outputs
            formatted_outputs = []
            if hasattr(cls, "RETURN_TYPES"):
                for i, t in enumerate(cls.RETURN_TYPES):
                    name = cls.RETURN_NAMES[i] if hasattr(cls, "RETURN_NAMES") and i < len(cls.RETURN_NAMES) else t
                    formatted_outputs.append({"name": name, "type": t, "links": []})
        else:
            # Fallback for standard ComfyUI nodes (LoadImage, UNETLoader, etc)
            formatted_inputs = [{"name": k} for k in inputs.keys()]
            if node_type == "UNETLoader":
                formatted_outputs = [{"name": "MODEL", "type": "MODEL", "links": []}]
            elif node_type == "CLIPLoader":
                formatted_outputs = [{"name": "CLIP", "type": "CLIP", "links": []}]
            elif node_type == "VAELoader":
                formatted_outputs = [{"name": "VAE", "type": "VAE", "links": []}]
            elif node_type == "LoadImage":
                formatted_outputs = [{"name": "IMAGE", "type": "IMAGE", "links": []}, {"name": "MASK", "type": "MASK", "links": []}]
            elif node_type == "KSampler":
                formatted_outputs = [{"name": "LATENT", "type": "LATENT", "links": []}]
            elif node_type == "VAEDecode":
                formatted_outputs = [{"name": "IMAGE", "type": "IMAGE", "links": []}]
            elif node_type == "SaveImage":
                formatted_outputs = []
            elif node_type == "EmptyLatentImage":
                formatted_outputs = [{"name": "LATENT", "type": "LATENT", "links": []}]
            elif node_type == "CLIPTextEncode":
                formatted_outputs = [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}]
            else:
                formatted_outputs = []

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
            "widgets_values": widgets_values
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

        type_name = from_node["outputs"][from_idx]["type"]

        lnk = [self.link_id, from_node["id"], from_idx, to_node["id"], to_idx, type_name]
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
    unet = b.add_node("UNETLoader", [0, 0], [300, 100], widgets_values=["krea2_model.safetensors", "default"])
    clip = b.add_node("CLIPLoader", [0, 150], [300, 100], widgets_values=["qwen3_vl.safetensors", "krea2"])
    vae = b.add_node("VAELoader", [0, 300], [300, 100], widgets_values=["ae.safetensors"])

    lora_name = "krea2_ostris_edit_lora.safetensors" if is_ostris else "krea2_edit_lora.safetensors"

    # Correct LoRA Stack widget ordering:
    # enabled (bool), global_strength (float), lora_1_enabled (bool), lora_1_name (str), lora_1_strength (float)...
    lora = b.add_node("CcCKrea2LoRAStack", [350, 0], [300, 250], inputs={"model": None}, widgets_values=[
        True, 1.0,
        True, lora_name, 1.0,
        False, "None", 1.0,
        False, "None", 1.0,
        False, "None", 1.0
    ])

    b.link(unet, "MODEL", lora, "model")

    sampler = b.add_node("KSampler", [1500, 0], [300, 200], inputs={"model": None, "positive": None, "negative": None, "latent_image": None}, widgets_values=[20, 1.0, "euler", "normal", 1.0])
    decode = b.add_node("VAEDecode", [1850, 0], [300, 100], inputs={"samples": None, "vae": None})
    save = b.add_node("SaveImage", [2200, 0], [300, 200], inputs={"images": None}, widgets_values=["CcCKrea2"])

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

    easy_widgets = [preset, outfit_source, style_source, True] # apply patch = True
    if is_ostris:
        easy_widgets.append(False) # ostris_kv_cache

    # positive_prompt
    easy_widgets.append("A photo of a person")
    # negative_prompt
    easy_widgets.append("bad quality")

    easy = b.add_node(easy_type, [800, 0], [400, 400], inputs=easy_inputs, widgets_values=easy_widgets)

    b.link(lora, "model", easy, "model")
    b.link(clip, "CLIP", easy, "clip")
    b.link(vae, "VAE", easy, "vae")

    y = 500
    if has_subj:
        sub = b.add_node("LoadImage", [0, y], [300, 200], widgets_values=["subject.jpg"])
        b.link(sub, "IMAGE", easy, "subject")
        y += 250
    if has_scene:
        scn = b.add_node("LoadImage", [0, y], [300, 200], widgets_values=["scene.jpg"])
        b.link(scn, "IMAGE", easy, "scene")
        y += 250
    if has_outfit:
        outf = b.add_node("LoadImage", [0, y], [300, 200], widgets_values=["outfit.jpg"])
        b.link(outf, "IMAGE", easy, "outfit")
        y += 250
    if has_style:
        sty = b.add_node("LoadImage", [0, y], [300, 200], widgets_values=["style.jpg"])
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
    sub_img = b.add_node("LoadImage", [0, 550], [300, 200], widgets_values=["subject.jpg"])
    sub_prep = b.add_node("CcCKrea2QwenVisionImagePrep", [350, 550], [300, 200], inputs={"clip": None, "image": None}, widgets_values=["native", 0.0, 1.0, 1.0, "auto", "auto"])
    sub_ref = b.add_node("CcCKrea2ReferenceImage", [700, 550], [300, 200], inputs={"prepared_image": None}, widgets_values=["edit", 0, "subject", "", 1.0, 1.0, "auto", 0.5, "2x2", True])

    b.link(clip, "CLIP", sub_prep, "clip")
    b.link(sub_img, "IMAGE", sub_prep, "image")
    b.link(sub_prep, "prepared_image", sub_ref, "prepared_image")

    # Outfit ref
    outf_img = b.add_node("LoadImage", [0, 800], [300, 200], widgets_values=["outfit.jpg"])
    outf_prep = b.add_node("CcCKrea2QwenVisionImagePrep", [350, 800], [300, 200], inputs={"clip": None, "image": None}, widgets_values=["native", 0.0, 1.0, 1.0, "auto", "auto"])
    outf_ref = b.add_node("CcCKrea2ReferenceImage", [700, 800], [300, 200], inputs={"prepared_image": None, "previous_references": None}, widgets_values=["edit", 0, "outfit", "", 1.0, 1.0, "auto", 0.5, "2x2", True])

    b.link(clip, "CLIP", outf_prep, "clip")
    b.link(outf_img, "IMAGE", outf_prep, "image")
    b.link(outf_prep, "prepared_image", outf_ref, "prepared_image")
    b.link(sub_ref, "reference_chain", outf_ref, "previous_references")

    # Target Latent
    t_latent = b.add_node("CcCKrea2TargetLatent", [1050, 550], [300, 250], inputs={"vae": None, "geometry_image": None}, widgets_values=["empty", "favor_image", 2.0, 2.0, "1:1", 1, "auto", "auto", "", ""])
    b.link(vae, "VAE", t_latent, "vae")
    b.link(sub_prep, "prepared_image", t_latent, "geometry_image")

    # Edit Orchestrator
    edit_inputs = {"model": None, "clip": None, "vae": None, "references": None, "target_latent": None}

    ref_method = "krea2_edit"
    if is_native:
        ref_method = "native"
    elif is_ostris:
        ref_method = "ostris_edit"

    edit_widgets = [ref_method, True, False, "A photo", "bad quality", ""] # method, apply_patch, ostris_kv, pos, neg, global
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
