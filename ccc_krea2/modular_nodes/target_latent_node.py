"""CcC Krea2 - Target Latent node."""

from ccc_krea2.target_latent import create_target_latent


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
                "target_latent_content": (["empty", "subject", "scene"], {"default": "empty"}),
                "target_geometry": (["favor_subject", "favor_scene", "fixed"], {"default": "fixed"}),
                "maximum_mp": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.05}),
                "fixed_mp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05}),
                "fixed_aspect_ratio": (["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "custom"], {"default": "1:1"}),
                "custom_aspect_width": ("INT", {"default": 1, "min": 1, "max": 100}),
                "custom_aspect_height": ("INT", {"default": 1, "min": 1, "max": 100}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "vae": ("VAE",),
                "subject_image": ("CCC_KREA2_PREPARED_IMAGE",),
                "scene_image": ("CCC_KREA2_PREPARED_IMAGE",),
            }
        }

    def process(
        self,
        target_latent_content="empty",
        target_geometry="fixed",
        maximum_mp=2.0,
        fixed_mp=1.0,
        fixed_aspect_ratio="1:1",
        custom_aspect_width=1,
        custom_aspect_height=1,
        batch_size=1,
        vae=None,
        subject_image=None,
        scene_image=None
    ):
        latent_dict, info_str = create_target_latent(
            vae=vae,
            target_latent_content=target_latent_content,
            target_geometry=target_geometry,
            subject_image=subject_image,
            scene_image=scene_image,
            maximum_mp=maximum_mp,
            fixed_mp=fixed_mp,
            fixed_aspect_ratio=fixed_aspect_ratio,
            custom_aspect_width=custom_aspect_width,
            custom_aspect_height=custom_aspect_height,
            batch_size=batch_size
        )
        return (latent_dict, info_str)
