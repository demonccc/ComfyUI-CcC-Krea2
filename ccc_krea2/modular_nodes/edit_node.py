"""CcC Krea2 - Edit orchestrator node."""

from ccc_krea2.edit_engine import run_krea2_edit_orchestrator
from ccc_krea2.reference_specs import ReferenceChain


class CcCKrea2Edit:
    """Final Krea 2 Edit orchestrator node performing visual reference encoding, conditioning, and patching."""

    CATEGORY = "CcC/Krea2"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "references": ("CCC_KREA2_REFERENCE_CHAIN",),
                "target_latent": ("LATENT",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "prompt_augmentation": ("CCC_KREA2_PROMPT_AUGMENTATION",),
                "global_vision_directive": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def process(
        self,
        model,
        clip,
        vae,
        references,
        target_latent,
        positive_prompt,
        negative_prompt,
        prompt_augmentation=None,
        global_vision_directive=""
    ):
        ref_chain = references if isinstance(references, ReferenceChain) else ReferenceChain()

        patched_model, positive, negative, latent_out, edit_info = run_krea2_edit_orchestrator(
            model=model,
            clip=clip,
            vae=vae,
            references=ref_chain,
            target_latent=target_latent,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            prompt_augmentation=prompt_augmentation,
            global_vision_directive=global_vision_directive
        )

        return (patched_model, positive, negative, latent_out, edit_info)
