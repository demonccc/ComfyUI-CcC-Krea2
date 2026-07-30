"""ComfyUI Custom Node classes for CcC Krea2 suite."""

from typing import Tuple, Dict, Any, List
import torch

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
    GROUNDING_PRESETS,
    SAMPLING_RESIZE_MODES,
    REFERENCE_FIT_MODES,
    ATTENTION_MASK_MODES,
    DEFAULT_GROUNDING_PX_SUBJECT,
    DEFAULT_GROUNDING_PX_SCENE,
    DEFAULT_GROUNDING_PX_OUTFIT,
    DEFAULT_GROUNDING_PX_SOURCE,
    DEFAULT_GROUNDING_MIN_PX,
    DEFAULT_GROUNDING_MAX_PX,
    DEFAULT_BOOST_SUBJECT,
    DEFAULT_BOOST_SCENE,
    DEFAULT_BOOST_OUTFIT,
    DEFAULT_BOOST_SOURCE,
)
from .engine import Krea2EditEngine, NodeExecutionRequest


class BaseKrea2Node:
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent")
    FUNCTION = "process"
    CATEGORY = NODE_CATEGORY


class CcCKrea2Subject(BaseKrea2Node):
    """CcC Krea2 - Subject node (Single-Reference Workflow)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "Required VAE for reference latent encoding and target latent creation."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Instruction for identity editing."}),
                "subject_image": ("IMAGE", {"tooltip": "Primary subject reference image."}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_attention_mask": ("MASK",),
                "subject_boost": ("FLOAT", {"default": DEFAULT_BOOST_SUBJECT, "min": 0.0, "max": 8.0, "step": 0.05}),
                "subject_mask_invert": ("BOOLEAN", {"default": False}),
                "attention_mask_mode": (ATTENTION_MASK_MODES, {"default": "hard"}),
                "subject_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "subject_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "subject_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_SUBJECT, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampling_resize_mode": (SAMPLING_RESIZE_MODES, {"default": "fit"}),
                "reference_fit_mode": (REFERENCE_FIT_MODES, {"default": "fit"}),
                "latent_source": (["empty", "subject"], {"default": "empty"}),
            }
        }

    def process(self, model, clip, vae, prompt, subject_image, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Subject",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            subject_image=subject_image,
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            subject_boost=kwargs.get("subject_boost", DEFAULT_BOOST_SUBJECT),
            subject_mask_invert=kwargs.get("subject_mask_invert", False),
            attention_mask_mode=kwargs.get("attention_mask_mode", "hard"),
            subject_grounding_preset=kwargs.get("subject_grounding_preset", "balanced"),
            subject_grounding_resize_mode=kwargs.get("subject_grounding_resize_mode", "normalize"),
            subject_grounding_px=kwargs.get("subject_grounding_px", DEFAULT_GROUNDING_PX_SUBJECT),
            subject_grounding_min_px=kwargs.get("subject_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            subject_grounding_max_px=kwargs.get("subject_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            batch_size=kwargs.get("batch_size", 1),
            sampling_resize_mode=kwargs.get("sampling_resize_mode", "fit"),
            reference_fit_mode=kwargs.get("reference_fit_mode", "fit"),
            latent_source=kwargs.get("latent_source", "empty"),
            role_order=ROLE_ORDER_SUBJECT
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2SubjectOutfit(BaseKrea2Node):
    """CcC Krea2 - Subject + Outfit node (Experimental Outfit Workflow)."""

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
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_attention_mask": ("MASK",),
                "subject_boost": ("FLOAT", {"default": DEFAULT_BOOST_SUBJECT, "min": 0.0, "max": 8.0, "step": 0.05}),
                "subject_mask_invert": ("BOOLEAN", {"default": False}),
                "outfit_attention_mask": ("MASK",),
                "outfit_boost": ("FLOAT", {"default": DEFAULT_BOOST_OUTFIT, "min": 0.0, "max": 8.0, "step": 0.05}),
                "outfit_mask_invert": ("BOOLEAN", {"default": False}),
                "attention_mask_mode": (ATTENTION_MASK_MODES, {"default": "hard"}),
                "subject_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "subject_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "subject_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_SUBJECT, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "outfit_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "outfit_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "outfit_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_OUTFIT, "min": 128, "max": 4096, "step": 16}),
                "outfit_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "outfit_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampling_resize_mode": (SAMPLING_RESIZE_MODES, {"default": "fit"}),
                "reference_fit_mode": (REFERENCE_FIT_MODES, {"default": "fit"}),
                "latent_source": (["empty", "subject"], {"default": "empty"}),
            }
        }

    def process(self, model, clip, vae, prompt, subject_image, outfit_image, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Subject + Outfit",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            subject_image=subject_image,
            outfit_image=outfit_image,
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            subject_boost=kwargs.get("subject_boost", DEFAULT_BOOST_SUBJECT),
            subject_mask_invert=kwargs.get("subject_mask_invert", False),
            outfit_attention_mask=kwargs.get("outfit_attention_mask", None),
            outfit_boost=kwargs.get("outfit_boost", DEFAULT_BOOST_OUTFIT),
            outfit_mask_invert=kwargs.get("outfit_mask_invert", False),
            attention_mask_mode=kwargs.get("attention_mask_mode", "hard"),
            subject_grounding_preset=kwargs.get("subject_grounding_preset", "balanced"),
            subject_grounding_resize_mode=kwargs.get("subject_grounding_resize_mode", "normalize"),
            subject_grounding_px=kwargs.get("subject_grounding_px", DEFAULT_GROUNDING_PX_SUBJECT),
            subject_grounding_min_px=kwargs.get("subject_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            subject_grounding_max_px=kwargs.get("subject_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            outfit_grounding_preset=kwargs.get("outfit_grounding_preset", "balanced"),
            outfit_grounding_resize_mode=kwargs.get("outfit_grounding_resize_mode", "normalize"),
            outfit_grounding_px=kwargs.get("outfit_grounding_px", DEFAULT_GROUNDING_PX_OUTFIT),
            outfit_grounding_min_px=kwargs.get("outfit_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            outfit_grounding_max_px=kwargs.get("outfit_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            batch_size=kwargs.get("batch_size", 1),
            sampling_resize_mode=kwargs.get("sampling_resize_mode", "fit"),
            reference_fit_mode=kwargs.get("reference_fit_mode", "fit"),
            latent_source=kwargs.get("latent_source", "empty"),
            role_order=ROLE_ORDER_SUBJECT_OUTFIT
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2SubjectScene(BaseKrea2Node):
    """CcC Krea2 - Subject + Scene node (Dual Reference Workflow)."""

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
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_attention_mask": ("MASK",),
                "subject_boost": ("FLOAT", {"default": DEFAULT_BOOST_SUBJECT, "min": 0.0, "max": 8.0, "step": 0.05}),
                "subject_mask_invert": ("BOOLEAN", {"default": False}),
                "scene_attention_mask": ("MASK",),
                "scene_boost": ("FLOAT", {"default": DEFAULT_BOOST_SCENE, "min": 0.0, "max": 8.0, "step": 0.05}),
                "scene_mask_invert": ("BOOLEAN", {"default": False}),
                "attention_mask_mode": (ATTENTION_MASK_MODES, {"default": "hard"}),
                "subject_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "subject_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "subject_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_SUBJECT, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "scene_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "scene_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "scene_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_SCENE, "min": 128, "max": 4096, "step": 16}),
                "scene_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "scene_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampling_resize_mode": (SAMPLING_RESIZE_MODES, {"default": "fit"}),
                "reference_fit_mode": (REFERENCE_FIT_MODES, {"default": "fit"}),
                "latent_source": (["empty", "subject", "scene"], {"default": "empty"}),
            }
        }

    def process(self, model, clip, vae, prompt, subject_image, scene_image, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Subject + Scene",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            subject_image=subject_image,
            scene_image=scene_image,
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            subject_boost=kwargs.get("subject_boost", DEFAULT_BOOST_SUBJECT),
            subject_mask_invert=kwargs.get("subject_mask_invert", False),
            scene_attention_mask=kwargs.get("scene_attention_mask", None),
            scene_boost=kwargs.get("scene_boost", DEFAULT_BOOST_SCENE),
            scene_mask_invert=kwargs.get("scene_mask_invert", False),
            attention_mask_mode=kwargs.get("attention_mask_mode", "hard"),
            subject_grounding_preset=kwargs.get("subject_grounding_preset", "balanced"),
            subject_grounding_resize_mode=kwargs.get("subject_grounding_resize_mode", "normalize"),
            subject_grounding_px=kwargs.get("subject_grounding_px", DEFAULT_GROUNDING_PX_SUBJECT),
            subject_grounding_min_px=kwargs.get("subject_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            subject_grounding_max_px=kwargs.get("subject_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            scene_grounding_preset=kwargs.get("scene_grounding_preset", "balanced"),
            scene_grounding_resize_mode=kwargs.get("scene_grounding_resize_mode", "normalize"),
            scene_grounding_px=kwargs.get("scene_grounding_px", DEFAULT_GROUNDING_PX_SCENE),
            scene_grounding_min_px=kwargs.get("scene_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            scene_grounding_max_px=kwargs.get("scene_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            batch_size=kwargs.get("batch_size", 1),
            sampling_resize_mode=kwargs.get("sampling_resize_mode", "fit"),
            reference_fit_mode=kwargs.get("reference_fit_mode", "fit"),
            latent_source=kwargs.get("latent_source", "empty"),
            role_order=ROLE_ORDER_SUBJECT_SCENE
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2SubjectSceneOutfit(BaseKrea2Node):
    """CcC Krea2 - Subject + Scene + Outfit node (Experimental 3-Ref Workflow)."""

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
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_attention_mask": ("MASK",),
                "subject_boost": ("FLOAT", {"default": DEFAULT_BOOST_SUBJECT, "min": 0.0, "max": 8.0, "step": 0.05}),
                "subject_mask_invert": ("BOOLEAN", {"default": False}),
                "scene_attention_mask": ("MASK",),
                "scene_boost": ("FLOAT", {"default": DEFAULT_BOOST_SCENE, "min": 0.0, "max": 8.0, "step": 0.05}),
                "scene_mask_invert": ("BOOLEAN", {"default": False}),
                "outfit_attention_mask": ("MASK",),
                "outfit_boost": ("FLOAT", {"default": DEFAULT_BOOST_OUTFIT, "min": 0.0, "max": 8.0, "step": 0.05}),
                "outfit_mask_invert": ("BOOLEAN", {"default": False}),
                "attention_mask_mode": (ATTENTION_MASK_MODES, {"default": "hard"}),
                "subject_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "subject_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "subject_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_SUBJECT, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "scene_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "scene_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "scene_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_SCENE, "min": 128, "max": 4096, "step": 16}),
                "scene_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "scene_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "outfit_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "outfit_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "outfit_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_OUTFIT, "min": 128, "max": 4096, "step": 16}),
                "outfit_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "outfit_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampling_resize_mode": (SAMPLING_RESIZE_MODES, {"default": "fit"}),
                "reference_fit_mode": (REFERENCE_FIT_MODES, {"default": "fit"}),
                "latent_source": (["empty", "subject", "scene"], {"default": "empty"}),
            }
        }

    def process(self, model, clip, vae, prompt, subject_image, scene_image, outfit_image, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Subject + Scene + Outfit",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            subject_image=subject_image,
            scene_image=scene_image,
            outfit_image=outfit_image,
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            subject_boost=kwargs.get("subject_boost", DEFAULT_BOOST_SUBJECT),
            subject_mask_invert=kwargs.get("subject_mask_invert", False),
            scene_attention_mask=kwargs.get("scene_attention_mask", None),
            scene_boost=kwargs.get("scene_boost", DEFAULT_BOOST_SCENE),
            scene_mask_invert=kwargs.get("scene_mask_invert", False),
            outfit_attention_mask=kwargs.get("outfit_attention_mask", None),
            outfit_boost=kwargs.get("outfit_boost", DEFAULT_BOOST_OUTFIT),
            outfit_mask_invert=kwargs.get("outfit_mask_invert", False),
            attention_mask_mode=kwargs.get("attention_mask_mode", "hard"),
            subject_grounding_preset=kwargs.get("subject_grounding_preset", "balanced"),
            subject_grounding_resize_mode=kwargs.get("subject_grounding_resize_mode", "normalize"),
            subject_grounding_px=kwargs.get("subject_grounding_px", DEFAULT_GROUNDING_PX_SUBJECT),
            subject_grounding_min_px=kwargs.get("subject_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            subject_grounding_max_px=kwargs.get("subject_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            scene_grounding_preset=kwargs.get("scene_grounding_preset", "balanced"),
            scene_grounding_resize_mode=kwargs.get("scene_grounding_resize_mode", "normalize"),
            scene_grounding_px=kwargs.get("scene_grounding_px", DEFAULT_GROUNDING_PX_SCENE),
            scene_grounding_min_px=kwargs.get("scene_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            scene_grounding_max_px=kwargs.get("scene_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            outfit_grounding_preset=kwargs.get("outfit_grounding_preset", "balanced"),
            outfit_grounding_resize_mode=kwargs.get("outfit_grounding_resize_mode", "normalize"),
            outfit_grounding_px=kwargs.get("outfit_grounding_px", DEFAULT_GROUNDING_PX_OUTFIT),
            outfit_grounding_min_px=kwargs.get("outfit_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            outfit_grounding_max_px=kwargs.get("outfit_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            batch_size=kwargs.get("batch_size", 1),
            sampling_resize_mode=kwargs.get("sampling_resize_mode", "fit"),
            reference_fit_mode=kwargs.get("reference_fit_mode", "fit"),
            latent_source=kwargs.get("latent_source", "empty"),
            role_order=ROLE_ORDER_SUBJECT_SCENE_OUTFIT
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2Inpaint(BaseKrea2Node):
    """CcC Krea2 - Inpaint node for single image inpainting."""

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
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "source_attention_mask": ("MASK",),
                "source_boost": ("FLOAT", {"default": DEFAULT_BOOST_SOURCE, "min": 0.0, "max": 8.0, "step": 0.05}),
                "source_mask_invert": ("BOOLEAN", {"default": False}),
                "attention_mask_mode": (ATTENTION_MASK_MODES, {"default": "hard"}),
                "source_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "source_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "source_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_SOURCE, "min": 128, "max": 4096, "step": 16}),
                "source_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "source_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "inpaint_mask_invert": ("BOOLEAN", {"default": False}),
                "inpaint_mask_grow": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "inpaint_mask_blur": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampling_resize_mode": (SAMPLING_RESIZE_MODES, {"default": "fit"}),
                "reference_fit_mode": (REFERENCE_FIT_MODES, {"default": "fit"}),
            }
        }

    def process(self, model, clip, vae, prompt, source_image, inpaint_mask, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Inpaint",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            source_image=source_image,
            inpaint_mask=inpaint_mask,
            source_attention_mask=kwargs.get("source_attention_mask", None),
            source_boost=kwargs.get("source_boost", DEFAULT_BOOST_SOURCE),
            source_mask_invert=kwargs.get("source_mask_invert", False),
            attention_mask_mode=kwargs.get("attention_mask_mode", "hard"),
            source_grounding_preset=kwargs.get("source_grounding_preset", "balanced"),
            source_grounding_resize_mode=kwargs.get("source_grounding_resize_mode", "normalize"),
            source_grounding_px=kwargs.get("source_grounding_px", DEFAULT_GROUNDING_PX_SOURCE),
            source_grounding_min_px=kwargs.get("source_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            source_grounding_max_px=kwargs.get("source_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            inpaint_mask_invert=kwargs.get("inpaint_mask_invert", False),
            inpaint_mask_grow=kwargs.get("inpaint_mask_grow", 0),
            inpaint_mask_blur=kwargs.get("inpaint_mask_blur", 0),
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            batch_size=kwargs.get("batch_size", 1),
            sampling_resize_mode=kwargs.get("sampling_resize_mode", "fit"),
            reference_fit_mode=kwargs.get("reference_fit_mode", "fit"),
            role_order=ROLE_ORDER_INPAINT
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2InpaintSubjectOutfit(BaseKrea2Node):
    """CcC Krea2 - Inpaint Subject + Outfit node (Experimental Outfit Inpainting)."""

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
                "inpaint_mask": ("MASK",),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject_attention_mask": ("MASK",),
                "subject_boost": ("FLOAT", {"default": DEFAULT_BOOST_SUBJECT, "min": 0.0, "max": 8.0, "step": 0.05}),
                "subject_mask_invert": ("BOOLEAN", {"default": False}),
                "outfit_attention_mask": ("MASK",),
                "outfit_boost": ("FLOAT", {"default": DEFAULT_BOOST_OUTFIT, "min": 0.0, "max": 8.0, "step": 0.05}),
                "outfit_mask_invert": ("BOOLEAN", {"default": False}),
                "attention_mask_mode": (ATTENTION_MASK_MODES, {"default": "hard"}),
                "subject_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "subject_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "subject_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_SUBJECT, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "outfit_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "outfit_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "outfit_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_OUTFIT, "min": 128, "max": 4096, "step": 16}),
                "outfit_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "outfit_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "inpaint_mask_invert": ("BOOLEAN", {"default": False}),
                "inpaint_mask_grow": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "inpaint_mask_blur": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampling_resize_mode": (SAMPLING_RESIZE_MODES, {"default": "fit"}),
                "reference_fit_mode": (REFERENCE_FIT_MODES, {"default": "fit"}),
            }
        }

    def process(self, model, clip, vae, prompt, subject_image, outfit_image, inpaint_mask, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Inpaint Subject + Outfit",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            subject_image=subject_image,
            outfit_image=outfit_image,
            inpaint_mask=inpaint_mask,
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            subject_boost=kwargs.get("subject_boost", DEFAULT_BOOST_SUBJECT),
            subject_mask_invert=kwargs.get("subject_mask_invert", False),
            outfit_attention_mask=kwargs.get("outfit_attention_mask", None),
            outfit_boost=kwargs.get("outfit_boost", DEFAULT_BOOST_OUTFIT),
            outfit_mask_invert=kwargs.get("outfit_mask_invert", False),
            attention_mask_mode=kwargs.get("attention_mask_mode", "hard"),
            subject_grounding_preset=kwargs.get("subject_grounding_preset", "balanced"),
            subject_grounding_resize_mode=kwargs.get("subject_grounding_resize_mode", "normalize"),
            subject_grounding_px=kwargs.get("subject_grounding_px", DEFAULT_GROUNDING_PX_SUBJECT),
            subject_grounding_min_px=kwargs.get("subject_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            subject_grounding_max_px=kwargs.get("subject_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            outfit_grounding_preset=kwargs.get("outfit_grounding_preset", "balanced"),
            outfit_grounding_resize_mode=kwargs.get("outfit_grounding_resize_mode", "normalize"),
            outfit_grounding_px=kwargs.get("outfit_grounding_px", DEFAULT_GROUNDING_PX_OUTFIT),
            outfit_grounding_min_px=kwargs.get("outfit_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            outfit_grounding_max_px=kwargs.get("outfit_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            inpaint_mask_invert=kwargs.get("inpaint_mask_invert", False),
            inpaint_mask_grow=kwargs.get("inpaint_mask_grow", 0),
            inpaint_mask_blur=kwargs.get("inpaint_mask_blur", 0),
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            batch_size=kwargs.get("batch_size", 1),
            sampling_resize_mode=kwargs.get("sampling_resize_mode", "fit"),
            reference_fit_mode=kwargs.get("reference_fit_mode", "fit"),
            role_order=ROLE_ORDER_INPAINT_SUBJECT_OUTFIT
        )
        return Krea2EditEngine.execute(request)


class CcCKrea2InpaintSubjectScene(BaseKrea2Node):
    """CcC Krea2 - Inpaint Subject + Scene node (Dual-Reference Inpainting)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "scene_image": ("IMAGE",),
                "subject_image": ("IMAGE",),
                "inpaint_mask": ("MASK",),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "scene_attention_mask": ("MASK",),
                "scene_boost": ("FLOAT", {"default": DEFAULT_BOOST_SCENE, "min": 0.0, "max": 8.0, "step": 0.05}),
                "scene_mask_invert": ("BOOLEAN", {"default": False}),
                "subject_attention_mask": ("MASK",),
                "subject_boost": ("FLOAT", {"default": DEFAULT_BOOST_SUBJECT, "min": 0.0, "max": 8.0, "step": 0.05}),
                "subject_mask_invert": ("BOOLEAN", {"default": False}),
                "attention_mask_mode": (ATTENTION_MASK_MODES, {"default": "hard"}),
                "scene_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "scene_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "scene_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_SCENE, "min": 128, "max": 4096, "step": 16}),
                "scene_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "scene_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_preset": (GROUNDING_PRESETS, {"default": "balanced"}),
                "subject_grounding_resize_mode": (GROUNDING_RESIZE_MODES, {"default": "normalize"}),
                "subject_grounding_px": ("INT", {"default": DEFAULT_GROUNDING_PX_SUBJECT, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_min_px": ("INT", {"default": DEFAULT_GROUNDING_MIN_PX, "min": 128, "max": 4096, "step": 16}),
                "subject_grounding_max_px": ("INT", {"default": DEFAULT_GROUNDING_MAX_PX, "min": 128, "max": 4096, "step": 16}),
                "inpaint_mask_invert": ("BOOLEAN", {"default": False}),
                "inpaint_mask_grow": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "inpaint_mask_blur": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sampling_resize_mode": (SAMPLING_RESIZE_MODES, {"default": "fit"}),
                "reference_fit_mode": (REFERENCE_FIT_MODES, {"default": "fit"}),
            }
        }

    def process(self, model, clip, vae, prompt, scene_image, subject_image, inpaint_mask, **kwargs):
        request = NodeExecutionRequest(
            node_name="CcC Krea2 - Inpaint Subject + Scene",
            model=model,
            clip=clip,
            vae=vae,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            scene_image=scene_image,
            subject_image=subject_image,
            inpaint_mask=inpaint_mask,
            scene_attention_mask=kwargs.get("scene_attention_mask", None),
            scene_boost=kwargs.get("scene_boost", DEFAULT_BOOST_SCENE),
            scene_mask_invert=kwargs.get("scene_mask_invert", False),
            subject_attention_mask=kwargs.get("subject_attention_mask", None),
            subject_boost=kwargs.get("subject_boost", DEFAULT_BOOST_SUBJECT),
            subject_mask_invert=kwargs.get("subject_mask_invert", False),
            attention_mask_mode=kwargs.get("attention_mask_mode", "hard"),
            scene_grounding_preset=kwargs.get("scene_grounding_preset", "balanced"),
            scene_grounding_resize_mode=kwargs.get("scene_grounding_resize_mode", "normalize"),
            scene_grounding_px=kwargs.get("scene_grounding_px", DEFAULT_GROUNDING_PX_SCENE),
            scene_grounding_min_px=kwargs.get("scene_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            scene_grounding_max_px=kwargs.get("scene_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            subject_grounding_preset=kwargs.get("subject_grounding_preset", "balanced"),
            subject_grounding_resize_mode=kwargs.get("subject_grounding_resize_mode", "normalize"),
            subject_grounding_px=kwargs.get("subject_grounding_px", DEFAULT_GROUNDING_PX_SUBJECT),
            subject_grounding_min_px=kwargs.get("subject_grounding_min_px", DEFAULT_GROUNDING_MIN_PX),
            subject_grounding_max_px=kwargs.get("subject_grounding_max_px", DEFAULT_GROUNDING_MAX_PX),
            inpaint_mask_invert=kwargs.get("inpaint_mask_invert", False),
            inpaint_mask_grow=kwargs.get("inpaint_mask_grow", 0),
            inpaint_mask_blur=kwargs.get("inpaint_mask_blur", 0),
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            batch_size=kwargs.get("batch_size", 1),
            sampling_resize_mode=kwargs.get("sampling_resize_mode", "fit"),
            reference_fit_mode=kwargs.get("reference_fit_mode", "fit"),
            role_order=ROLE_ORDER_INPAINT_SUBJECT_SCENE
        )
        return Krea2EditEngine.execute(request)


# Mappings for ComfyUI custom node discovery
NODE_CLASS_MAPPINGS = {
    "CcCKrea2Subject": CcCKrea2Subject,
    "CcCKrea2SubjectOutfit": CcCKrea2SubjectOutfit,
    "CcCKrea2SubjectScene": CcCKrea2SubjectScene,
    "CcCKrea2SubjectSceneOutfit": CcCKrea2SubjectSceneOutfit,
    "CcCKrea2Inpaint": CcCKrea2Inpaint,
    "CcCKrea2InpaintSubjectOutfit": CcCKrea2InpaintSubjectOutfit,
    "CcCKrea2InpaintSubjectScene": CcCKrea2InpaintSubjectScene,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CcCKrea2Subject": "CcC Krea2 - Subject",
    "CcCKrea2SubjectOutfit": "CcC Krea2 - Subject + Outfit",
    "CcCKrea2SubjectScene": "CcC Krea2 - Subject + Scene",
    "CcCKrea2SubjectSceneOutfit": "CcC Krea2 - Subject + Scene + Outfit",
    "CcCKrea2Inpaint": "CcC Krea2 - Inpaint",
    "CcCKrea2InpaintSubjectOutfit": "CcC Krea2 - Inpaint Subject + Outfit",
    "CcCKrea2InpaintSubjectScene": "CcC Krea2 - Inpaint Subject + Scene",
}
