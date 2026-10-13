"""Ten-slot model-only LoRA loader for Krea2 and other ComfyUI MODEL pipelines."""

from typing import Any

from .constants import NODE_CATEGORY
from .lora import SlotLoraLoader, get_lora_names


class CcCKrea2LoRALoader:
    """Apply up to ten LoRAs in slot order without modifying CLIP or prompts."""

    CATEGORY = NODE_CATEGORY
    FUNCTION = "apply_loras"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "Load up to 10 model-only LoRAs. Each slot has its own enabled switch "
        "and strength from -10 to 10. Zero strength applies no changes."
    )

    def __init__(self):
        self.loaders = [SlotLoraLoader(i) for i in range(1, 11)]

    @classmethod
    def INPUT_TYPES(cls):
        names = get_lora_names()
        required = {"model": ("MODEL",)}
        for index in range(1, 11):
            required[f"lora_{index}_name"] = (names, {"default": "None"})
            required[f"lora_{index}_enabled"] = ("BOOLEAN", {"default": False})
            required[f"lora_{index}_strength"] = (
                "FLOAT",
                {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05, "display": "slider"},
            )
        return {"required": required}

    def apply_loras(self, model: Any, **kwargs):
        current_model = model
        for index, loader in enumerate(self.loaders, start=1):
            if not kwargs.get(f"lora_{index}_enabled", False):
                continue
            name = kwargs.get(f"lora_{index}_name", "None")
            strength = float(kwargs.get(f"lora_{index}_strength", 0.0))
            if not name or name == "None" or strength == 0.0:
                continue
            current_model = loader.load(current_model, str(name), strength)
        return (current_model,)
