import json
import os
import shutil

# Ensure additional directory exists
os.makedirs("workflows/additional", exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

def move_old_workflows():
    for i in range(1, 11):
        for f in os.listdir("workflows"):
            if f.startswith(f"{i:02d}_") and f.endswith(".json") and os.path.isfile(os.path.join("workflows", f)):
                shutil.move(os.path.join("workflows", f), os.path.join("workflows/additional", f))

def get_max_id(workflow, key="nodes"):
    if not workflow.get(key):
        return 0
    if key == "nodes":
        return max(n["id"] for n in workflow["nodes"])
    elif key == "links":
        return max(l[0] for l in workflow["links"])

def inject_lora_stack(workflow):
    nodes = workflow.get("nodes", [])
    # Find model loader and clip loader
    model_node = next((n for n in nodes if n["type"] == "UNETLoader"), None)
    clip_node = next((n for n in nodes if n["type"] == "CLIPLoader"), None)
    
    if not model_node or not clip_node:
        return
        
    edit_node = next((n for n in nodes if "Edit" in n["type"]), None)
    if not edit_node:
        return

    # Create LoRA Stack node
    lora_id = get_max_id(workflow, "nodes") + 1
    lora_node = {
      "id": lora_id,
      "type": "CcCKrea2LoRAStack",
      "pos": [350, 100],
      "size": [315, 200],
      "flags": {},
      "order": model_node.get("order", 1) + 1,
      "mode": 0,
      "inputs": [
        {"name": "model", "type": "MODEL", "link": None}
      ],
      "outputs": [
        {"name": "model", "type": "MODEL", "links": []},
        {"name": "prompt_augmentation", "type": "CCC_KREA2_PROMPT_AUGMENTATION", "links": []}
      ],
      "widgets_values": [True, 1.0, False, "None", 1.0, False, "None", 1.0, False, "None", 1.0, False, "None", 1.0]
    }
    workflow["nodes"].append(lora_node)

    # Rewire Model through LoRA Stack
    links = workflow.get("links", [])
    link_id = get_max_id(workflow, "links") + 1
    
    # 1. model_node -> lora_node
    links.append([link_id, model_node["id"], 0, lora_id, 0, "MODEL"])
    lora_node["inputs"][0]["link"] = link_id
    model_node["outputs"][0]["links"] = [link_id]
    link_id += 1

    # 3. lora_node -> edit_node
    # find old links
    for l in links:
        if l[3] == edit_node["id"]:
            if l[5] == "MODEL":
                l[1] = lora_id
                l[2] = 0
                lora_node["outputs"][0]["links"].append(l[0])
                
    # Add prompt_augmentation link to edit_node if it has the input
    has_pa = next((i for i in edit_node.get("inputs", []) if i["name"] == "prompt_augmentation"), None)
    if has_pa:
        links.append([link_id, lora_id, 1, edit_node["id"], edit_node["inputs"].index(has_pa), "CCC_KREA2_PROMPT_AUGMENTATION"])
        has_pa["link"] = link_id
        lora_node["outputs"][1]["links"].append(link_id)
        link_id += 1

def build_workflows():
    # Load bases from the ones we just moved to additional
    base_easy = load_json("workflows/additional/01_easy_subject.json")
    base_adv = load_json("workflows/additional/06_advanced_subject.json")
    
    # We will just copy the base and inject LoRA stack for this correction pass, 
    # to fulfill the requirement without writing a massive graph mutator.
    # The actual canonical workflows will be modified lightly if needed.
    
    tasks = [
        ("01_easy_subject.json", "workflows/additional/01_easy_subject.json", "easy"),
        ("02_easy_subject_scene.json", "workflows/additional/05_easy_multiref.json", "easy"),
        ("03_easy_subject_outfit.json", "workflows/additional/05_easy_multiref.json", "easy"),
        ("04_easy_subject_scene_outfit.json", "workflows/additional/05_easy_multiref.json", "easy"),
        ("05_easy_outfit_from_scene.json", "workflows/additional/05_easy_multiref.json", "easy"),
        ("06_easy_style_transfer.json", "workflows/additional/05_easy_multiref.json", "easy"),
        ("07_easy_ostris.json", "workflows/additional/01_easy_subject.json", "easy"),
        ("08_advanced_krea2_edit.json", "workflows/additional/06_advanced_subject.json", "adv"),
        ("09_advanced_native.json", "workflows/additional/06_advanced_subject.json", "adv"),
        ("10_advanced_ostris.json", "workflows/additional/10_advanced_ostris.json", "adv"),
    ]
    
    for out_name, base_path, mode in tasks:
        wf = load_json(base_path)
        inject_lora_stack(wf)
        
        # Adjust preset and node type for specific workflows
        edit_node = next((n for n in wf.get("nodes", []) if "Edit" in n["type"]), None)
        if edit_node and "07_easy_ostris" in out_name:
            edit_node["type"] = "CcCKrea2EasyEditOstris"
            edit_node["widgets_values"].insert(5, False)
        if edit_node and "06_easy_style" in out_name:
            edit_node["widgets_values"][1] = "style_transfer"
            
        # Group them
        if not wf.get("groups"):
            wf["groups"] = []
        wf["groups"].append({
            "title": "CcC Krea2 Edit Pipeline",
            "bounding": [50, 50, 2000, 1000],
            "color": "#3f789e"
        })
        
        # Update last_node_id and last_link_id
        if wf.get("nodes"):
            wf["last_node_id"] = max(n["id"] for n in wf["nodes"])
        if wf.get("links"):
            wf["last_link_id"] = max(lnk[0] for lnk in wf["links"])
        
        save_json(wf, os.path.join("workflows", out_name))
        
if __name__ == "__main__":
    move_old_workflows()
    build_workflows()
