"""CcC Krea2 - Target Latent node."""

from ..target_latent import build_target_latent


class CcCKrea2TargetLatent:
    """Prepare target sampling LATENT decoupling Content decisions from Geometry decisions."""

    CATEGORY = "CcC/Krea2"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("target_latent", "latent_info")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_content": (["empty", "image"], {"default": "empty", "tooltip": "Source content for initial latent: empty (noise) or image (encoding target_image)."}),
                "geometry_mode": (["fixed", "favor_image"], {"default": "fixed", "tooltip": "Geometry mode: fixed (use aspect_ratio/fixed_megapixels) or favor_image (match geometry_image or target_image resolution)."}),

                "target_megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 12.0, "step": 0.01, "tooltip": "Megapixels limit when favor_image is active."}),
                "fixed_megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 12.0, "step": 0.01, "tooltip": "Target megapixels when fixed geometry mode is selected."}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"], {"default": "1:1", "tooltip": "Aspect ratio when fixed geometry mode is selected."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Latent batch size."}),
            },
            "optional": {
                "include_in_vision": (["auto", "yes", "no"], {"default": "auto", "tooltip": "Adds target image to Qwen semantic context without making it an appearance reference."}),
                "target_vision_slot": (["auto", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], {"default": "auto", "tooltip": "Controls Qwen semantic image ordering slot for target vision context."}),
                "target_alias": ("STRING", {"default": "", "tooltip": "Alias name for target vision reference."}),
                "target_vision_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional explicit instruction describing what Qwen should use from target vision image."}),
                "vae": ("VAE", {"tooltip": "VAE model for encoding target image."}),
                "target_image": ("PREPARED_VISION_IMAGE", {"tooltip": "Prepared vision image to encode as target content."}),
                "geometry_image": ("PREPARED_VISION_IMAGE", {"tooltip": "Prepared vision image to use for resolution math in favor_image mode."}),
                # Deprecated compatibility inputs
                "subject_image": ("PREPARED_VISION_IMAGE", {"tooltip": "(Legacy/Deprecated) Legacy subject image input socket."}),
                "scene_image": ("PREPARED_VISION_IMAGE", {"tooltip": "(Legacy/Deprecated) Legacy scene image input socket."}),
            }
        }

    def process(
        self,
        vae=None,
        target_content="empty",
        geometry_mode="fixed",
        target_megapixels=2.0,
        fixed_megapixels=2.0,
        aspect_ratio="1:1",
        batch_size=1,
        include_in_vision="auto",
        target_vision_slot="auto",
        target_alias="",
        target_vision_instruction="",
        target_image=None,
        geometry_image=None,
        subject_image=None,
        scene_image=None,
        **kwargs
    ):
        if "target_latent_content" in kwargs:
            target_content = kwargs["target_latent_content"]
        if "target_geometry" in kwargs:
            geometry_mode = kwargs["target_geometry"]
        if "maximum_mp" in kwargs:
            target_megapixels = kwargs["maximum_mp"]

        # Compatibility handling for legacy geometry_mode choices
        if geometry_mode in ("favor_subject", "favor_scene"):
            geometry_mode = "favor_image"

        latent_dict, info_str = build_target_latent(
            vae=vae,
            target_content=target_content,
            geometry_mode=geometry_mode,
            target_megapixels=target_megapixels,
            fixed_megapixels=fixed_megapixels,
            aspect_ratio=aspect_ratio,
            batch_size=batch_size,
            include_in_vision=include_in_vision,
            target_vision_slot=target_vision_slot,
            target_alias=target_alias,
            target_vision_instruction=target_vision_instruction,
            target_image=target_image,
            geometry_image=geometry_image,
            subject_image=subject_image,
            scene_image=scene_image,
        )
        return (latent_dict, info_str)
