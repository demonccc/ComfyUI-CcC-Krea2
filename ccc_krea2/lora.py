"""CcC Krea2 - LoRA Stack and CcC Krea2 - LoRA Prompt Settings nodes."""

import logging
from typing import Any, List

from .constants import NODE_CATEGORY
from .prompt_augmentation import CCC_KREA2_PROMPT_AUGMENTATION, PromptAugmentation

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

class CcCKrea2LoRAStackWithPrompt:
    """Five model-only LoRAs with strictly positive prompt prepend/append triggers."""

    CATEGORY = NODE_CATEGORY
    FUNCTION = "apply_loras"
    RETURN_TYPES = ("MODEL", CCC_KREA2_PROMPT_AUGMENTATION)
    RETURN_NAMES = ("model", "prompt_augmentation")

    def __init__(self):
        self.loaders = [SlotLoraLoader(i) for i in range(1, 6)]

    @classmethod
    def INPUT_TYPES(cls):
        names = get_lora_names()
        required = {"model": ("MODEL",)}
        for i in range(1, 6):
            required[f"lora_{i}_enabled"] = ("BOOLEAN", {"default": False})
            required[f"lora_{i}_name"] = (names, {"default": "None"})
            required[f"lora_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05})
            required[f"lora_{i}_prompt_position"] = (["append", "prepend"], {"default": "append"})
            required[f"lora_{i}_prompt"] = ("STRING", {"multiline": True, "default": "", "rows": 2})
        return {"required": required, "optional": {
            "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
        }}

    def apply_loras(self, model, prompt_augmentation=None, **kwargs):
        current_model = model
        incoming = prompt_augmentation if prompt_augmentation is not None else PromptAugmentation()
        prepend = list(incoming.positive_prepend)
        append = list(incoming.positive_append)
        for i, loader in enumerate(self.loaders, 1):
            if not kwargs.get(f"lora_{i}_enabled", False):
                continue
            name = kwargs.get(f"lora_{i}_name", "None")
            strength = float(kwargs.get(f"lora_{i}_strength", 1.0))
            if not name or name == "None" or strength == 0:
                continue
            current_model = loader.load(current_model, str(name), strength)
            prompt = str(kwargs.get(f"lora_{i}_prompt", "")).strip()
            if prompt:
                position = kwargs.get(f"lora_{i}_prompt_position", "append")
                (prepend if position == "prepend" else append).append(prompt)
        return current_model, PromptAugmentation(
            positive_prepend=tuple(prepend), positive_append=tuple(append),
            negative_prepend=incoming.negative_prepend,
            negative_append=incoming.negative_append,
        )
