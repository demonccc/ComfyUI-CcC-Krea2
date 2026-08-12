"""CcC Krea2 - Edit node."""

from ..edit_engine import run_krea2_edit_orchestrator
from ..prompt_augmentation import CCC_KREA2_PROMPT_AUGMENTATION


class CcCKrea2Edit:
    """Core 5-layer modular orchestrator node executing Krea 2 Edit pipeline."""

    CATEGORY = "CcC/Krea2"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Input Diffusion MODEL to edit."}),
                "clip": ("CLIP", {"tooltip": "Krea2 Qwen CLIP text/vision encoder."}),
                "vae": ("VAE", {"tooltip": "VAE encoder/decoder."}),
                "references": ("REFERENCE_CHAIN", {"tooltip": "Linked list of ReferenceSpecs defining appearance, semantic, and style conditioning."}),
                "target_latent": ("LATENT", {"tooltip": "Target dimensions and noise initialization."}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True, "tooltip": "User prompt describing the desired edit."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True, "tooltip": "Negative prompt."}),
            },
            "optional": {
                "global_vision_directive": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional global instruction pre-pended to all reference prompts."}),
                "reference_method": (["native", "krea2_edit", "ostris_edit"], {"default": "krea2_edit", "tooltip": "Backend engine used to transport reference information."}),
                "ostris_kv_cache": ("BOOLEAN", {"default": False, "tooltip": "Unsupported. Reserved for future ai-toolkit kv-cache models."}),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION, {"tooltip": "Slot-based LoRA prompt fragment injection."}),
            }
        }


    def edit(
        self,
        model,
        clip,
        vae,
        references,
        target_latent,
        positive_prompt="",
        negative_prompt="",
        reference_method="krea2_edit",
        ostris_kv_cache=False,
        prompt_augmentation=None,
        global_vision_directive="",
        **kwargs
    ):
        patched_model, pos_cond, neg_cond, out_latent, edit_info = run_krea2_edit_orchestrator(
            model=model,
            clip=clip,
            vae=vae,
            references=references,
            target_latent=target_latent,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            reference_method=reference_method,
            ostris_kv_cache=ostris_kv_cache,
            prompt_augmentation=prompt_augmentation,
            global_vision_directive=global_vision_directive,
            **kwargs
        )
        return (patched_model, pos_cond, neg_cond, out_latent, edit_info)

    process = edit
