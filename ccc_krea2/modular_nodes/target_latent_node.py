"""CcC Krea2 - Target Latent node."""

from ccc_krea2.target_latent import build_target_latent


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
                "target_content": (["empty", "subject", "scene"], {"default": "empty"}),
                "geometry_mode": (["favor_subject", "favor_scene", "fixed"], {"default": "fixed"}),
                "target_megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 12.0, "step": 0.01}),
                "fixed_megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 12.0, "step": 0.01}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"], {"default": "1:1"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "vae": ("VAE",),
                "subject_image": ("PREPARED_VISION_IMAGE",),
                "scene_image": ("PREPARED_VISION_IMAGE",),
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

        latent_dict, info_str = build_target_latent(
            vae=vae,
            target_content=target_content,
            geometry_mode=geometry_mode,
            target_megapixels=target_megapixels,
            fixed_megapixels=fixed_megapixels,
            aspect_ratio=aspect_ratio,
            batch_size=batch_size,
            subject_image=subject_image,
            scene_image=scene_image
        )
        return (latent_dict, info_str)
