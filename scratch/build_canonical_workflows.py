import json
import os
import shutil
from typing import Any, Dict, List

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

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

def move_old_workflows():
    os.makedirs("workflows/additional", exist_ok=True)
    for f in os.listdir("workflows"):
        if f.startswith(tuple(f"{i:02d}_" for i in range(1, 20))) and f.endswith(".json"):
            path = os.path.join("workflows", f)
            if os.path.isfile(path) and f not in CANONICAL_NAMES:
                target = os.path.join("workflows/additional", f)
                if not os.path.exists(target):
                    shutil.move(path, target)
                else:
                    os.remove(path) # avoid duplicate accumulation

class WorkflowBuilder:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.groups = []
        self.node_id = 1
        self.link_id = 1

    def add_node(self, node_type, pos, size, **kwargs):
        node = {
            "id": self.node_id,
            "type": node_type,
            "pos": pos,
            "size": size,
            "flags": {},
            "order": self.node_id,
            "mode": 0,
            "inputs": kwargs.get("inputs", []),
            "outputs": kwargs.get("outputs", []),
            "properties": {"Node name for S&R": node_type},
            "widgets_values": kwargs.get("widgets_values", [])
        }
        self.nodes.append(node)
        self.node_id += 1
        return node

    def link(self, from_node, from_out_idx, to_node, to_in_idx, type_name):
        l = [self.link_id, from_node["id"], from_out_idx, to_node["id"], to_in_idx, type_name]
        self.links.append(l)
        
        while len(to_node["inputs"]) <= to_in_idx:
            to_node["inputs"].append({})
        to_node["inputs"][to_in_idx]["link"] = self.link_id
        
        while len(from_node["outputs"]) <= from_out_idx:
            from_node["outputs"].append({"links": []})
        if "links" not in from_node["outputs"][from_out_idx]:
            from_node["outputs"][from_out_idx]["links"] = []
        from_node["outputs"][from_out_idx]["links"].append(self.link_id)
        
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

def build_model_lora(b: WorkflowBuilder, include_lora=True, lora_name="krea2_edit_lora.safetensors", lora_enabled=True):
    # Models Group
    b.add_group("Models", [50, 50, 400, 200])
    model_node = b.add_node("UNETLoader", [100, 100], [315, 80], outputs=[{"name": "MODEL", "type": "MODEL"}])
    clip_node = b.add_node("CLIPLoader", [100, 200], [315, 80], outputs=[{"name": "CLIP", "type": "CLIP"}], widgets_values=["clip_l.safetensors", "krea2", False])
    
    if include_lora:
        b.add_group("LoRA", [500, 50, 400, 250])
        lora_node = b.add_node("CcCKrea2LoRAStack", [550, 100], [315, 200], 
            inputs=[{"name": "model", "type": "MODEL"}],
            outputs=[{"name": "model", "type": "MODEL"}, {"name": "prompt_augmentation", "type": "CCC_KREA2_PROMPT_AUGMENTATION"}],
            widgets_values=[lora_enabled, lora_name, 1.0, False, "None", 1.0, False, "None", 1.0, False, "None", 1.0, False, "None", 1.0]
        )
        b.link(model_node, 0, lora_node, 0, "MODEL")
        return model_node, clip_node, lora_node
    return model_node, clip_node, None

def build_image(b: WorkflowBuilder, pos, name):
    return b.add_node("LoadImage", pos, [315, 315], outputs=[{"name": "IMAGE", "type": "IMAGE"}, {"name": "MASK", "type": "MASK"}], widgets_values=[name, "image"])

def build_easy_scenario(filename, has_subject=False, has_scene=False, has_outfit=False, has_style=False, preset="balanced", outfit_source="outfit image", style_source="style image", is_ostris=False):
    b = WorkflowBuilder()
    m, c, lora = build_model_lora(b, lora_name="krea2_ostris_edit_lora.safetensors" if is_ostris else "krea2_edit_lora.safetensors")
    
    b.add_group("Images / References", [50, 320, 400, 1500])
    y = 370
    images = {}
    if has_subject:
        images["subject"] = build_image(b, [100, y], "subject.jpg")
        y += 350
    if has_scene:
        images["scene"] = build_image(b, [100, y], "scene.jpg")
        y += 350
    if has_outfit:
        images["outfit"] = build_image(b, [100, y], "outfit.jpg")
        y += 350
    if has_style:
        images["style"] = build_image(b, [100, y], "style.jpg")
        y += 350

    b.add_group("Easy Edit + Sampling", [500, 320, 600, 800])
    
    inputs = [
        {"name": "subject_image", "type": "IMAGE"},
        {"name": "scene_image", "type": "IMAGE"},
        {"name": "outfit_image", "type": "IMAGE"},
        {"name": "style_image", "type": "IMAGE"},
        {"name": "model", "type": "MODEL"},
        {"name": "prompt_augmentation", "type": "CCC_KREA2_PROMPT_AUGMENTATION"},
    ]
    
    node_type = "CcCKrea2EasyEditOstris" if is_ostris else "CcCKrea2EasyEdit"
    
    if is_ostris:
        widgets = [preset, outfit_source, style_source, True, 1.0, False] # apply patch=True, ostris_kv_cache=False
    else:
        widgets = [preset, outfit_source, style_source, True, 1.0] # apply patch=True
        
    edit = b.add_node(node_type, [550, 370], [400, 300], inputs=inputs, outputs=[{"name": "MODEL", "type": "MODEL"}], widgets_values=widgets)
    
    b.link(lora, 0, edit, 4, "MODEL")
    b.link(lora, 1, edit, 5, "CCC_KREA2_PROMPT_AUGMENTATION")
    
    if has_subject: b.link(images["subject"], 0, edit, 0, "IMAGE")
    if has_scene: b.link(images["scene"], 0, edit, 1, "IMAGE")
    if has_outfit: b.link(images["outfit"], 0, edit, 2, "IMAGE")
    if has_style: b.link(images["style"], 0, edit, 3, "IMAGE")
    
    b.add_group("Output", [1150, 320, 400, 300])
    sampler = b.add_node("KSampler", [1200, 370], [315, 200], inputs=[{"name": "model", "type": "MODEL"}])
    b.link(edit, 0, sampler, 0, "MODEL")

    save_json(b.build(), os.path.join("workflows", filename))

def build_advanced_scenario(filename, mode="krea2_edit"):
    b = WorkflowBuilder()
    
    lora_enabled = (mode != "native")
    lora_name = "krea2_ostris_edit_lora.safetensors" if mode == "ostris_edit" else "krea2_edit_lora.safetensors"
    
    m, c, lora = build_model_lora(b, include_lora=True, lora_name=lora_name, lora_enabled=lora_enabled)
    
    if mode == "native":
        b.add_group("Native Reference Runtime Required", [500, 50, 400, 250])
    
    b.add_group("Vision Prep", [50, 320, 400, 800])
    img1 = build_image(b, [100, 370], "ref1.jpg")
    prep1 = b.add_node("CcCKrea2QwenVisionImagePrep", [100, 720], [315, 200], inputs=[{"name": "image", "type": "IMAGE"}], outputs=[{"name": "PREPARED_IMAGE", "type": "CCC_KREA2_PREPARED_IMAGE"}])
    b.link(img1, 0, prep1, 0, "IMAGE")
    
    img2 = build_image(b, [100, 950], "ref2.jpg")
    prep2 = b.add_node("CcCKrea2QwenVisionImagePrep", [100, 1300], [315, 200], inputs=[{"name": "image", "type": "IMAGE"}], outputs=[{"name": "PREPARED_IMAGE", "type": "CCC_KREA2_PREPARED_IMAGE"}])
    b.link(img2, 0, prep2, 0, "IMAGE")

    b.add_group("Reference Chain", [500, 320, 400, 800])
    ref1 = b.add_node("CcCKrea2ReferenceImage", [550, 370], [315, 200], 
        inputs=[{"name": "prepared_image", "type": "CCC_KREA2_PREPARED_IMAGE"}], 
        outputs=[{"name": "REFERENCE_CHAIN", "type": "CCC_KREA2_REFERENCE_CHAIN"}],
        widgets_values=["edit", "subject", "subject identity", 1.0, True, 2] # path, alias, instruction, boost, appearance, style_proc
    )
    b.link(prep1, 0, ref1, 0, "CCC_KREA2_PREPARED_IMAGE")
    
    ref2 = b.add_node("CcCKrea2ReferenceImage", [550, 700], [315, 200], 
        inputs=[{"name": "prepared_image", "type": "CCC_KREA2_PREPARED_IMAGE"}, {"name": "reference_chain", "type": "CCC_KREA2_REFERENCE_CHAIN"}], 
        outputs=[{"name": "REFERENCE_CHAIN", "type": "CCC_KREA2_REFERENCE_CHAIN"}],
        widgets_values=["edit", "outfit", "clothing and accessories", 1.0, True, 2]
    )
    b.link(prep2, 0, ref2, 0, "CCC_KREA2_PREPARED_IMAGE")
    b.link(ref1, 0, ref2, 1, "CCC_KREA2_REFERENCE_CHAIN")

    b.add_group("Target", [950, 320, 400, 300])
    target = b.add_node("CcCKrea2TargetLatent", [1000, 370], [315, 200],
        outputs=[{"name": "TARGET_LATENT", "type": "CCC_KREA2_TARGET_LATENT"}],
        widgets_values=["empty", "fixed", 1.0] # content, geometry, scale
    )

    b.add_group("Advanced Edit + Sampling", [950, 650, 400, 400])
    edit_inputs = [
        {"name": "model", "type": "MODEL"},
        {"name": "prompt_augmentation", "type": "CCC_KREA2_PROMPT_AUGMENTATION"},
        {"name": "target_vision", "type": "CCC_KREA2_TARGET_VISION"},
        {"name": "target_latent", "type": "CCC_KREA2_TARGET_LATENT"},
        {"name": "reference_latents", "type": "LATENT"},
        {"name": "edit_reference_path", "type": "CCC_KREA2_REFERENCE_CHAIN"},
        {"name": "style_reference_path", "type": "CCC_KREA2_REFERENCE_CHAIN"},
    ]
    
    if mode == "ostris_edit":
        widgets = [mode, True, 1.0, False] # apply patch=True, ostris_kv_cache=False
    else:
        widgets = [mode, True, 1.0] # apply patch=True

    edit = b.add_node("CcCKrea2Edit", [1000, 700], [315, 300], inputs=edit_inputs, outputs=[{"name": "MODEL", "type": "MODEL"}], widgets_values=widgets)
    
    b.link(lora, 0, edit, 0, "MODEL")
    b.link(lora, 1, edit, 1, "CCC_KREA2_PROMPT_AUGMENTATION")
    b.link(target, 0, edit, 3, "CCC_KREA2_TARGET_LATENT")
    b.link(ref2, 0, edit, 5, "CCC_KREA2_REFERENCE_CHAIN")

    b.add_group("Output", [1400, 320, 400, 300])
    sampler = b.add_node("KSampler", [1450, 370], [315, 200], inputs=[{"name": "model", "type": "MODEL"}])
    b.link(edit, 0, sampler, 0, "MODEL")

    save_json(b.build(), os.path.join("workflows", filename))

def build_all():
    # 01. Subject
    build_easy_scenario("01_easy_subject.json", has_subject=True)
    # 02. Subject + Scene
    build_easy_scenario("02_easy_subject_scene.json", has_subject=True, has_scene=True)
    # 03. Subject + Outfit
    build_easy_scenario("03_easy_subject_outfit.json", has_subject=True, has_outfit=True)
    # 04. Subject + Scene + Outfit
    build_easy_scenario("04_easy_subject_scene_outfit.json", has_subject=True, has_scene=True, has_outfit=True)
    # 05. Outfit from Scene
    build_easy_scenario("05_easy_outfit_from_scene.json", has_subject=True, has_scene=True, outfit_source="scene image", preset="outfit_transfer")
    # 06. Style Transfer
    build_easy_scenario("06_easy_style_transfer.json", has_subject=True, has_style=True, preset="style_transfer")
    # 07. Easy Ostris
    build_easy_scenario("07_easy_ostris.json", has_subject=True, has_scene=True, is_ostris=True)
    
    # 08, 09, 10
    build_advanced_scenario("08_advanced_krea2_edit.json", mode="krea2_edit")
    build_advanced_scenario("09_advanced_native.json", mode="native")
    build_advanced_scenario("10_advanced_ostris.json", mode="ostris_edit")

if __name__ == "__main__":
    move_old_workflows()
    build_all()
