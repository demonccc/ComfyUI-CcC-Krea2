"""Generate 12 curated modular workflow JSON files in workflows/."""


def create_base_nodes():
    return [
        {
            "id": 1,
            "type": "UNETLoader",
            "pos": [100, 100],
            "size": [300, 100],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1]}],
            "properties": {},
            "widgets_values": ["krea2_dit.safetensors", "default"],
        },
        {
            "id": 2,
            "type": "CLIPLoader",
            "pos": [100, 250],
            "size": [300, 100],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "CLIP", "type": "CLIP", "links": [2]}],
            "properties": {},
            "widgets_values": ["qwen2_5_vl.safetensors", "qwen2_5_vl", "default"],
        },
        {
            "id": 3,
            "type": "VAELoader",
            "pos": [100, 400],
            "size": [300, 100],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "VAE", "type": "VAE", "links": [3]}],
            "properties": {},
            "widgets_values": ["krea2_vae.safetensors"],
        },
    ]


print("Generator helper script ready.")
