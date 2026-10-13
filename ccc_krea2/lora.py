"""CcC Krea2 - LoRA Stack and CcC Krea2 - LoRA Prompt Settings nodes."""

import logging
from typing import Tuple, Optional, Any, List

from .constants import NODE_CATEGORY
from .prompt_augmentation import (
    CCC_KREA2_LORA_PROMPT_SETTINGS,
    CCC_KREA2_PROMPT_AUGMENTATION,
    LoRAPromptSlot,
    LoRAPromptSettings,
    PromptAugmentation,
    EMPTY_PROMPT_AUGMENTATION,
)

logger = logging.getLogger("CcCKrea2")


def get_lora_names() -> List[str]:
    """Retrieve list of available LoRA filenames from ComfyUI folder_paths, ensuring 'None' is included once."""
    try:
        import folder_paths

        raw_list = folder_paths.get_filename_list("loras")
        names = list(raw_list) if raw_list else []
        if "None" not in names:
            return ["None"] + names
        return names
    except Exception:
        return ["None"]


class SlotLoraLoader:
    """Independent LoRA loader instance maintaining a loaded-LoRA cache per slot."""

    def __init__(self, slot_index: int):
        self.slot_index = slot_index
        self.native_loader = None
        self._cache = {}
        try:
            import nodes

            if hasattr(nodes, "LoraLoaderModelOnly"):
                self.native_loader = nodes.LoraLoaderModelOnly()
        except Exception:
            pass

    def load(self, model: Any, lora_name: str, strength_model: float) -> Any:
        if self.native_loader is not None:
            try:
                res = self.native_loader.load_lora(model, lora_name, strength_model)
                if isinstance(res, tuple):
                    return res[0]
                return res
            except Exception as e:
                logger.debug(
                    f"Native LoraLoaderModelOnly call failed in slot {self.slot_index}, falling back to API: {e}"
                )

        # Fallback adapter using ComfyUI APIs
        import folder_paths
        import comfy.sd
        import comfy.utils

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if lora_path in self._cache:
            lora = self._cache[lora_path]
        else:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self._cache[lora_path] = lora

        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        return model_lora


class CcCKrea2LoRAPromptSettings:
    """Configures prompt augmentation settings for up to 4 LoRA slots."""

    RETURN_TYPES = (CCC_KREA2_LORA_PROMPT_SETTINGS,)
    RETURN_NAMES = ("lora_prompt_settings",)
    FUNCTION = "build_settings"
    CATEGORY = NODE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "enabled": ("BOOLEAN", {"default": True}),
        }
        for i in range(1, 5):
            req[f"lora_{i}_prompt_enabled"] = ("BOOLEAN", {"default": False})
            req[f"lora_{i}_prompt_position"] = (["append", "prepend"], {"default": "append"})
            req[f"lora_{i}_positive_prompt"] = ("STRING", {"multiline": True, "default": ""})
            req[f"lora_{i}_negative_prompt"] = ("STRING", {"multiline": True, "default": ""})
        return {"required": req}

    def build_settings(self, enabled: bool = True, **kwargs) -> Tuple[LoRAPromptSettings]:
        slots = []
        for i in range(1, 5):
            slot = LoRAPromptSlot(
                enabled=bool(kwargs.get(f"lora_{i}_prompt_enabled", False)),
                position=str(kwargs.get(f"lora_{i}_prompt_position", "append")),
                positive_prompt=str(kwargs.get(f"lora_{i}_positive_prompt", "")),
                negative_prompt=str(kwargs.get(f"lora_{i}_negative_prompt", "")),
            )
            slots.append(slot)
        return (LoRAPromptSettings(enabled=bool(enabled), slots=tuple(slots)),)


class CcCKrea2LoRAStack:
    """Apply up to ten model-only LoRAs, in order, without prompt processing."""

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_loras"
    CATEGORY = NODE_CATEGORY

    def __init__(self):
        self.loaders = [SlotLoraLoader(i) for i in range(1, 11)]

    @classmethod
    def INPUT_TYPES(cls):
        names = get_lora_names()
        required = {"model": ("MODEL",)}
        for i in range(1, 11):
            required[f"lora_{i}_enabled"] = ("BOOLEAN", {"default": False})
            required[f"lora_{i}_name"] = (names, {"default": "None"})
            required[f"lora_{i}_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}
            )
        return {"required": required}

    def apply_loras(self, model: Any, **kwargs):
        current_model = model
        for i, loader in enumerate(self.loaders, 1):
            if not kwargs.get(f"lora_{i}_enabled", False):
                continue
            name = kwargs.get(f"lora_{i}_name", "None")
            strength = float(kwargs.get(f"lora_{i}_strength", 1.0))
            if name and name != "None" and strength != 0:
                current_model = loader.load(current_model, str(name), strength)
        return (current_model,)
