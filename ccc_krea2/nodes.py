"""ComfyUI Custom Node classes for CcC Krea2 suite."""

from .constants import (
    NODE_CATEGORY,
    ReferenceRole,
    ROLE_ORDER_SUBJECT,
    ROLE_ORDER_SUBJECT_OUTFIT,
    ROLE_ORDER_SUBJECT_SCENE,
    ROLE_ORDER_SUBJECT_SCENE_OUTFIT,
    ROLE_ORDER_INPAINT,
    ROLE_ORDER_INPAINT_SUBJECT_OUTFIT,
    ROLE_ORDER_INPAINT_SUBJECT_SCENE,
    GROUNDING_RESIZE_MODES,
    SAMPLING_RESIZE_MODES,
    LEGACY_REFERENCE_FIT_MODES,
    ATTENTION_MASK_MODES,
)
from .settings import (
    CCC_KREA2_IMAGE_ADVANCED_SETTINGS,
    CCC_KREA2_EDIT_ADVANCED_SETTINGS,
    PRESET_CHOICES,
    RESIZE_METHODS,
    ROLE_CHOICES,
    ImageRoleSettings,
    ImageAdvancedSettingsBundle,
    EditAdvancedSettings,
)
from .prompt_augmentation import CCC_KREA2_PROMPT_AUGMENTATION
from .lora import CcCKrea2LoRAPromptSettings, CcCKrea2LoRAStack
from .engine import Krea2EditEngine, NodeExecutionRequest
from .t2i import CcCKrea2TextToImage
from .modular_nodes.vision_prep_node import CcCKrea2QwenVisionImagePrep
from .modular_nodes.target_latent_node import CcCKrea2TargetLatent
from .modular_nodes.reference_node import CcCKrea2ReferenceImage
from .modular_nodes.easy_edit_node import CcCKrea2EasyEdit, CcCKrea2EasyEditOstris
from .modular_nodes.subject_node import CcCKrea2SubjectImage
from .modular_nodes.scene_node import CcCKrea2SceneImage
from .modular_nodes.outfit_node import CcCKrea2OutfitImage
from .modular_nodes.style_node import CcCKrea2StyleImage
from .modular_nodes.edit_node import CcCKrea2Edit


class BaseKrea2Node:
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent")
    FUNCTION = "process"
    CATEGORY = NODE_CATEGORY


class CcCKrea2ImageAdvancedSettings:
    """Image Advanced Settings node for per-role attention, grounding, and geometry overrides."""

    RETURN_TYPES = (CCC_KREA2_IMAGE_ADVANCED_SETTINGS,)
    RETURN_NAMES = ("image_advanced_settings",)
    FUNCTION = "process"
    CATEGORY = NODE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": (ROLE_CHOICES, {"default": "subject"}),
                "boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "mask_invert": ("BOOLEAN", {"default": False}),
                "grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "grounding_px": ("INT", {"default": 768, "min": 128, "max": 4096, "step": 16}),
                "grounding_min_px": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 16}),
                "grounding_max_px": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "grounding_resize_method": (RESIZE_METHODS, {"default": "auto"}),
                "reference_fit_mode": (LEGACY_REFERENCE_FIT_MODES, {"default": "fit"}),
                "reference_resize_method": (RESIZE_METHODS, {"default": "auto"}),
            },
            "optional": {
                "image_advanced_settings": (CCC_KREA2_IMAGE_ADVANCED_SETTINGS,),
            },
        }

    def process(
        self,
        role: str,
        boost: float,
        mask_invert: bool,
        grounding_resize_mode: str,
        grounding_px: int,
        grounding_min_px: int,
        grounding_max_px: int,
        grounding_resize_method: str,
        reference_fit_mode: str,
        reference_resize_method: str,
        image_advanced_settings=None,
    ):
        role_set = ImageRoleSettings(
            boost=boost,
            mask_invert=mask_invert,
            grounding_resize_mode=grounding_resize_mode,
            grounding_px=grounding_px,
            grounding_min_px=grounding_min_px,
            grounding_max_px=grounding_max_px,
            grounding_resize_method=grounding_resize_method,
            reference_fit_mode=reference_fit_mode,
            reference_resize_method=reference_resize_method,
        )
        bundle = image_advanced_settings if image_advanced_settings is not None else ImageAdvancedSettingsBundle()
        return (bundle.with_role(role, role_set),)


class CcCKrea2EditAdvancedSettings:
    """Edit Advanced Settings node for sampling, mask modes, and aspect ratio overrides."""

    RETURN_TYPES = (CCC_KREA2_EDIT_ADVANCED_SETTINGS,)
    RETURN_NAMES = ("edit_advanced_settings",)
    FUNCTION = "process"
    CATEGORY = NODE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampling_resize_mode": (SAMPLING_RESIZE_MODES, {"default": "fit"}),
                "sampling_resize_method": (RESIZE_METHODS, {"default": "auto"}),
                "attention_mask_mode": (ATTENTION_MASK_MODES, {"default": "hard"}),
                "role_resolution_limit_mode": (["off", "max_megapixels"], {"default": "max_megapixels"}),
                "role_resolution_max_megapixels": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 12.0, "step": 0.25}),
                "custom_aspect_source": (["auto", "subject", "scene", "source"], {"default": "auto"}),
                "prompt_instructions_mode": (["automatic", "append"], {"default": "automatic"}),
                "prompt_instructions": ("STRING", {"multiline": True, "default": ""}),
                "inpaint_mask_invert": ("BOOLEAN", {"default": False}),
                "inpaint_mask_grow": ("INT", {"default": 0, "min": 0, "max": 256}),
                "inpaint_mask_blur": ("INT", {"default": 0, "min": 0, "max": 256}),
            }
        }

    def process(
        self,
        batch_size: int,
        sampling_resize_mode: str,
        sampling_resize_method: str,
        attention_mask_mode: str,
        role_resolution_limit_mode: str,
        role_resolution_max_megapixels: float,
        custom_aspect_source: str,
        prompt_instructions_mode: str,
        prompt_instructions: str,
        inpaint_mask_invert: bool,
        inpaint_mask_grow: int,
        inpaint_mask_blur: int,
    ):
        return (
            EditAdvancedSettings(
                batch_size=batch_size,
                sampling_resize_mode=sampling_resize_mode,
                sampling_resize_method=sampling_resize_method,
                attention_mask_mode=attention_mask_mode,
                role_resolution_limit_mode=role_resolution_limit_mode,
                role_resolution_max_megapixels=role_resolution_max_megapixels,
                custom_aspect_source=custom_aspect_source,
                prompt_instructions_mode=prompt_instructions_mode,
                prompt_instructions=prompt_instructions,
                inpaint_mask_invert=inpaint_mask_invert,
                inpaint_mask_grow=inpaint_mask_grow,
                inpaint_mask_blur=inpaint_mask_blur,
            ),
        )


class CcCKrea2Subject(BaseKrea2Node):
    """CcC Krea2 - Subject node (Single-Reference Workflow)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "Required VAE for reference latent encoding and target latent creation."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "subject_image": ("IMAGE", {"tooltip": "Primary subject reference image."}),
                "preset": (PRESET_CHOICES, {"default": "balanced"}),
                "output_resolution": (["subject", "custom"], {"default": "subject"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_attention_mask": ("MASK",),
                "latent_source": (["empty", "subject"], {"default": "empty"}),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
                "image_advanced_settings": (CCC_KREA2_IMAGE_ADVANCED_SETTINGS,),
                "edit_advanced_settings": (CCC_KREA2_EDIT_ADVANCED_SETTINGS,),
            },
        }

    def process(self, model, clip, vae, prompt, subject_image, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Subject",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            preset=kwargs.get("preset", "balanced"),
            output_resolution=kwargs.get("output_resolution", "subject"),
            megapixels=kwargs.get("megapixels", 1.0),
            prompt_augmentation=kwargs.get("prompt_augmentation", None),
            image_advanced_settings=kwargs.get("image_advanced_settings", None),
            edit_advanced_settings=kwargs.get("edit_advanced_settings", None),
            subject_image=subject_image,
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            latent_source=kwargs.get("latent_source", "empty"),
            role_order=ROLE_ORDER_SUBJECT,
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2SubjectOutfit(BaseKrea2Node):
    """CcC Krea2 - Subject + Outfit node (Dual-Reference Workflow)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "subject_image": ("IMAGE",),
                "outfit_image": ("IMAGE",),
                "preset": (PRESET_CHOICES, {"default": "balanced"}),
                "output_resolution": (["subject", "custom"], {"default": "subject"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_attention_mask": ("MASK",),
                "outfit_attention_mask": ("MASK",),
                "latent_source": (["empty", "subject"], {"default": "empty"}),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
                "image_advanced_settings": (CCC_KREA2_IMAGE_ADVANCED_SETTINGS,),
                "edit_advanced_settings": (CCC_KREA2_EDIT_ADVANCED_SETTINGS,),
            },
        }

    def process(self, model, clip, vae, prompt, subject_image, outfit_image, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Subject + Outfit",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            preset=kwargs.get("preset", "balanced"),
            output_resolution=kwargs.get("output_resolution", "subject"),
            megapixels=kwargs.get("megapixels", 1.0),
            prompt_augmentation=kwargs.get("prompt_augmentation", None),
            image_advanced_settings=kwargs.get("image_advanced_settings", None),
            edit_advanced_settings=kwargs.get("edit_advanced_settings", None),
            subject_image=subject_image,
            outfit_image=outfit_image,
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            outfit_attention_mask=kwargs.get("outfit_attention_mask", None),
            latent_source=kwargs.get("latent_source", "empty"),
            role_order=ROLE_ORDER_SUBJECT_OUTFIT,
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2SubjectScene(BaseKrea2Node):
    """CcC Krea2 - Subject + Scene node (Dual-Reference Workflow)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "subject_image": ("IMAGE",),
                "scene_image": ("IMAGE",),
                "preset": (PRESET_CHOICES, {"default": "balanced"}),
                "output_resolution": (["scene", "custom"], {"default": "scene"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_attention_mask": ("MASK",),
                "scene_attention_mask": ("MASK",),
                "latent_source": (["empty", "subject", "scene"], {"default": "empty"}),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
                "image_advanced_settings": (CCC_KREA2_IMAGE_ADVANCED_SETTINGS,),
                "edit_advanced_settings": (CCC_KREA2_EDIT_ADVANCED_SETTINGS,),
            },
        }

    def process(self, model, clip, vae, prompt, subject_image, scene_image, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Subject + Scene",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            preset=kwargs.get("preset", "balanced"),
            output_resolution=kwargs.get("output_resolution", "scene"),
            megapixels=kwargs.get("megapixels", 1.0),
            prompt_augmentation=kwargs.get("prompt_augmentation", None),
            image_advanced_settings=kwargs.get("image_advanced_settings", None),
            edit_advanced_settings=kwargs.get("edit_advanced_settings", None),
            subject_image=subject_image,
            scene_image=scene_image,
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            scene_attention_mask=kwargs.get("scene_attention_mask", None),
            latent_source=kwargs.get("latent_source", "empty"),
            role_order=ROLE_ORDER_SUBJECT_SCENE,
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2SubjectSceneOutfit(BaseKrea2Node):
    """CcC Krea2 - Subject + Scene + Outfit node (Triple-Reference Workflow)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "subject_image": ("IMAGE",),
                "scene_image": ("IMAGE",),
                "outfit_image": ("IMAGE",),
                "preset": (PRESET_CHOICES, {"default": "balanced"}),
                "output_resolution": (["scene", "custom"], {"default": "scene"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_attention_mask": ("MASK",),
                "scene_attention_mask": ("MASK",),
                "outfit_attention_mask": ("MASK",),
                "latent_source": (["empty", "subject", "scene"], {"default": "empty"}),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
                "image_advanced_settings": (CCC_KREA2_IMAGE_ADVANCED_SETTINGS,),
                "edit_advanced_settings": (CCC_KREA2_EDIT_ADVANCED_SETTINGS,),
            },
        }

    def process(self, model, clip, vae, prompt, subject_image, scene_image, outfit_image, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Subject + Scene + Outfit",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            preset=kwargs.get("preset", "balanced"),
            output_resolution=kwargs.get("output_resolution", "scene"),
            megapixels=kwargs.get("megapixels", 1.0),
            prompt_augmentation=kwargs.get("prompt_augmentation", None),
            image_advanced_settings=kwargs.get("image_advanced_settings", None),
            edit_advanced_settings=kwargs.get("edit_advanced_settings", None),
            subject_image=subject_image,
            scene_image=scene_image,
            outfit_image=outfit_image,
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            scene_attention_mask=kwargs.get("scene_attention_mask", None),
            outfit_attention_mask=kwargs.get("outfit_attention_mask", None),
            latent_source=kwargs.get("latent_source", "empty"),
            role_order=ROLE_ORDER_SUBJECT_SCENE_OUTFIT,
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2Inpaint(BaseKrea2Node):
    """CcC Krea2 - Inpaint node (Single-Reference Inpainting Workflow)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "source_image": ("IMAGE",),
                "inpaint_mask": ("MASK",),
                "preset": (PRESET_CHOICES, {"default": "balanced"}),
                "output_resolution": (["source", "custom"], {"default": "source"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "source_attention_mask": ("MASK",),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
                "image_advanced_settings": (CCC_KREA2_IMAGE_ADVANCED_SETTINGS,),
                "edit_advanced_settings": (CCC_KREA2_EDIT_ADVANCED_SETTINGS,),
            },
        }

    def process(self, model, clip, vae, prompt, source_image, inpaint_mask, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Inpaint",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            preset=kwargs.get("preset", "balanced"),
            output_resolution=kwargs.get("output_resolution", "source"),
            megapixels=kwargs.get("megapixels", 1.0),
            prompt_augmentation=kwargs.get("prompt_augmentation", None),
            image_advanced_settings=kwargs.get("image_advanced_settings", None),
            edit_advanced_settings=kwargs.get("edit_advanced_settings", None),
            source_image=source_image,
            inpaint_mask=inpaint_mask,
            source_attention_mask=kwargs.get("source_attention_mask", None),
            latent_source="source",
            inpaint_base_role=ReferenceRole.SOURCE,
            role_order=ROLE_ORDER_INPAINT,
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2InpaintSubjectOutfit(BaseKrea2Node):
    """CcC Krea2 - Inpaint Subject + Outfit node (Subject & Outfit Inpainting)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "subject_image": ("IMAGE",),
                "outfit_image": ("IMAGE",),
                "preset": (PRESET_CHOICES, {"default": "balanced"}),
                "output_resolution": (["subject", "custom"], {"default": "subject"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "inpaint_mask": ("MASK",),
                "subject_attention_mask": ("MASK",),
                "outfit_attention_mask": ("MASK",),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
                "image_advanced_settings": (CCC_KREA2_IMAGE_ADVANCED_SETTINGS,),
                "edit_advanced_settings": (CCC_KREA2_EDIT_ADVANCED_SETTINGS,),
            },
        }

    def process(self, model, clip, vae, prompt, subject_image, outfit_image, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Inpaint Subject + Outfit",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            preset=kwargs.get("preset", "balanced"),
            output_resolution=kwargs.get("output_resolution", "subject"),
            megapixels=kwargs.get("megapixels", 1.0),
            prompt_augmentation=kwargs.get("prompt_augmentation", None),
            image_advanced_settings=kwargs.get("image_advanced_settings", None),
            edit_advanced_settings=kwargs.get("edit_advanced_settings", None),
            subject_image=subject_image,
            outfit_image=outfit_image,
            inpaint_mask=kwargs.get("inpaint_mask", None),
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            outfit_attention_mask=kwargs.get("outfit_attention_mask", None),
            latent_source="subject",
            inpaint_base_role=ReferenceRole.SUBJECT,
            role_order=ROLE_ORDER_INPAINT_SUBJECT_OUTFIT,
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2InpaintSubjectScene(BaseKrea2Node):
    """CcC Krea2 - Inpaint Subject + Scene node (Subject & Scene Inpainting)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "subject_image": ("IMAGE",),
                "scene_image": ("IMAGE",),
                "preset": (PRESET_CHOICES, {"default": "balanced"}),
                "output_resolution": (["scene", "custom"], {"default": "scene"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "inpaint_mask": ("MASK",),
                "subject_attention_mask": ("MASK",),
                "scene_attention_mask": ("MASK",),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
                "image_advanced_settings": (CCC_KREA2_IMAGE_ADVANCED_SETTINGS,),
                "edit_advanced_settings": (CCC_KREA2_EDIT_ADVANCED_SETTINGS,),
            },
        }

    def process(self, model, clip, vae, prompt, subject_image, scene_image, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Inpaint Subject + Scene",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            preset=kwargs.get("preset", "balanced"),
            output_resolution=kwargs.get("output_resolution", "scene"),
            megapixels=kwargs.get("megapixels", 1.0),
            prompt_augmentation=kwargs.get("prompt_augmentation", None),
            image_advanced_settings=kwargs.get("image_advanced_settings", None),
            edit_advanced_settings=kwargs.get("edit_advanced_settings", None),
            subject_image=subject_image,
            scene_image=scene_image,
            inpaint_mask=kwargs.get("inpaint_mask", None),
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            scene_attention_mask=kwargs.get("scene_attention_mask", None),
            latent_source="scene",
            inpaint_base_role=ReferenceRole.SCENE,
            role_order=ROLE_ORDER_INPAINT_SUBJECT_SCENE,
        )
        return Krea2EditEngine.execute(request)


NODE_CLASS_MAPPINGS = {
    # Modular Easy Nodes
    "CcCKrea2EasyEdit": CcCKrea2EasyEdit,
    "CcCKrea2EasyEditOstris": CcCKrea2EasyEditOstris,
    # Modular Reference Pipeline Nodes
    "CcCKrea2QwenVisionImagePrep": CcCKrea2QwenVisionImagePrep,
    "CcCKrea2TargetLatent": CcCKrea2TargetLatent,
    "CcCKrea2ReferenceImage": CcCKrea2ReferenceImage,
    "CcCKrea2Edit": CcCKrea2Edit,
    # Compatibility Nodes
    "CcCKrea2SubjectImage": CcCKrea2SubjectImage,
    "CcCKrea2SceneImage": CcCKrea2SceneImage,
    "CcCKrea2OutfitImage": CcCKrea2OutfitImage,
    "CcCKrea2StyleImage": CcCKrea2StyleImage,
    # Existing & Utility Nodes
    "CcCKrea2LoRAPromptSettings": CcCKrea2LoRAPromptSettings,
    "CcCKrea2LoRAStack": CcCKrea2LoRAStack,
    "CcCKrea2TextToImage": CcCKrea2TextToImage,
    # Legacy Combined Nodes
    "CcCKrea2Subject": CcCKrea2Subject,
    "CcCKrea2SubjectOutfit": CcCKrea2SubjectOutfit,
    "CcCKrea2SubjectScene": CcCKrea2SubjectScene,
    "CcCKrea2SubjectSceneOutfit": CcCKrea2SubjectSceneOutfit,
    "CcCKrea2Inpaint": CcCKrea2Inpaint,
    "CcCKrea2InpaintSubjectOutfit": CcCKrea2InpaintSubjectOutfit,
    "CcCKrea2InpaintSubjectScene": CcCKrea2InpaintSubjectScene,
    "CcCKrea2ImageAdvancedSettings": CcCKrea2ImageAdvancedSettings,
    "CcCKrea2EditAdvancedSettings": CcCKrea2EditAdvancedSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Modular Easy Nodes
    "CcCKrea2EasyEdit": "CcC Krea2 - Easy Edit",
    "CcCKrea2EasyEditOstris": "CcC Krea2 - Easy Edit Ostris",
    # Modular Reference Pipeline Nodes
    "CcCKrea2QwenVisionImagePrep": "CcC Krea2 - Qwen Vision Image Prep",
    "CcCKrea2TargetLatent": "CcC Krea2 - Target Latent",
    "CcCKrea2ReferenceImage": "CcC Krea2 - Reference Image",
    "CcCKrea2Edit": "CcC Krea2 - Edit",
    # Compatibility Nodes
    "CcCKrea2SubjectImage": "CcC Krea2 - Subject Image (Legacy)",
    "CcCKrea2SceneImage": "CcC Krea2 - Scene Image (Legacy)",
    "CcCKrea2OutfitImage": "CcC Krea2 - Outfit Image (Legacy)",
    "CcCKrea2StyleImage": "CcC Krea2 - Style Image (Legacy)",
    # Existing & Utility Nodes
    "CcCKrea2LoRAPromptSettings": "CcC Krea2 - LoRA Prompt Settings",
    "CcCKrea2LoRAStack": "CcC Krea2 - LoRA Stack",
    "CcCKrea2TextToImage": "CcC Krea2 - Text to Image",
    # Legacy Combined Nodes
    "CcCKrea2Subject": "CcC Krea2 - Subject (Legacy)",
    "CcCKrea2SubjectOutfit": "CcC Krea2 - Subject + Outfit (Legacy)",
    "CcCKrea2SubjectScene": "CcC Krea2 - Subject + Scene (Legacy)",
    "CcCKrea2SubjectSceneOutfit": "CcC Krea2 - Subject + Scene + Outfit (Legacy)",
    "CcCKrea2Inpaint": "CcC Krea2 - Inpaint (Legacy)",
    "CcCKrea2InpaintSubjectOutfit": "CcC Krea2 - Inpaint Subject + Outfit (Legacy)",
    "CcCKrea2InpaintSubjectScene": "CcC Krea2 - Inpaint Subject + Scene (Legacy)",
    "CcCKrea2ImageAdvancedSettings": "CcC Krea2 - Image Advanced Settings (Legacy)",
    "CcCKrea2EditAdvancedSettings": "CcC Krea2 - Edit Advanced Settings (Legacy)",
}
