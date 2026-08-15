"""CcC Krea2 LoRA Stack and LoRA Prompt Settings custom nodes."""

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
    """Loads up to four model-only LoRAs and accumulates prompt augmentations."""

    RETURN_TYPES = ("MODEL", CCC_KREA2_PROMPT_AUGMENTATION)
    RETURN_NAMES = ("model", "prompt_augmentation")
    FUNCTION = "apply_loras"
    CATEGORY = NODE_CATEGORY

    def __init__(self):
        self.loaders = [SlotLoraLoader(i) for i in range(1, 5)]

    @classmethod
    def INPUT_TYPES(cls):
        lora_choices = get_lora_names()
        req = {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "global_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
        }
        for i in range(1, 5):
            req[f"lora_{i}_enabled"] = ("BOOLEAN", {"default": False})
            req[f"lora_{i}_name"] = (lora_choices, {"default": "None"})
            req[f"lora_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05})

        opt = {
            "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
            "lora_prompt_settings": (CCC_KREA2_LORA_PROMPT_SETTINGS,),
        }
        return {"required": req, "optional": opt}

    def apply_loras(
        self,
        model: Any,
        enabled: bool = True,
        global_strength: float = 1.0,
        prompt_augmentation: Optional[PromptAugmentation] = None,
        lora_prompt_settings: Optional[LoRAPromptSettings] = None,
        **kwargs,
    ) -> Tuple[Any, PromptAugmentation]:
        # Handle disabled stack
        if not enabled:
            out_aug = prompt_augmentation if prompt_augmentation is not None else EMPTY_PROMPT_AUGMENTATION
            return (model, out_aug)

        current_model = model
        inc_aug = prompt_augmentation if prompt_augmentation is not None else EMPTY_PROMPT_AUGMENTATION

        new_pos_prepend: List[str] = []
        new_pos_append: List[str] = []
        new_neg_prepend: List[str] = []
        new_neg_append: List[str] = []

        settings_enabled = lora_prompt_settings is not None and lora_prompt_settings.enabled

        for i in range(1, 5):
            slot_idx = i - 1
            l_enabled = bool(kwargs.get(f"lora_{i}_enabled", False))
            l_name = kwargs.get(f"lora_{i}_name", None)
            l_strength = float(kwargs.get(f"lora_{i}_strength", 1.0))

            effective_strength = l_strength * global_strength

            # Active LoRA condition
            is_active = l_enabled and l_name is not None and str(l_name) != "None" and effective_strength != 0.0

            if is_active:
                current_model = self.loaders[slot_idx].load(current_model, str(l_name), effective_strength)

                # Active LoRA prompt filtering
                if settings_enabled and lora_prompt_settings is not None and slot_idx < len(lora_prompt_settings.slots):
                    pslot = lora_prompt_settings.slots[slot_idx]
                    if pslot.enabled:
                        pos_text = pslot.positive_prompt
                        neg_text = pslot.negative_prompt

                        if pslot.position == "prepend":
                            if pos_text and pos_text.strip():
                                new_pos_prepend.append(pos_text.strip())
                            if neg_text and neg_text.strip():
                                new_neg_prepend.append(neg_text.strip())
                        else:  # append
                            if pos_text and pos_text.strip():
                                new_pos_append.append(pos_text.strip())
                            if neg_text and neg_text.strip():
                                new_neg_append.append(neg_text.strip())

        # Construct chained immutable PromptAugmentation
        pos_pre = inc_aug.positive_prepend + tuple(new_pos_prepend)
        pos_app = inc_aug.positive_append + tuple(new_pos_append)
        neg_pre = inc_aug.negative_prepend + tuple(new_neg_prepend)
        neg_app = inc_aug.negative_append + tuple(new_neg_append)

        merged_aug = PromptAugmentation(
            positive_prepend=pos_pre,
            positive_append=pos_app,
            negative_prepend=neg_pre,
            negative_append=neg_app,
        )

        return (current_model, merged_aug)
