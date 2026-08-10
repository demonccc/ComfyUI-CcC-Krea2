"""CcC Krea2 - Qwen Vision Image Prep node."""

from ..vision_prep import prepare_vision_image, format_vision_info


class CcCKrea2QwenVisionImagePrep:
    """Prepare a derivative image optimized for Qwen Vision while keeping the original image intact."""

    CATEGORY = "CcC/Krea2"
    RETURN_TYPES = ("PREPARED_VISION_IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("prepared_image", "vision_image", "vision_info")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "mode": (["native", "adaptive", "fixed"], {"default": "native"}),
                "min_mp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 12.0, "step": 0.01}),
                "max_mp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 12.0, "step": 0.01}),
                "fixed_mp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 12.0, "step": 0.01}),
                "downscale_method": (["auto", "area", "bicubic", "bilinear", "lanczos", "nearest-exact"], {"default": "auto"}),
                "upscale_method": (["auto", "bicubic", "bilinear", "lanczos", "nearest-exact"], {"default": "auto"}),
            }
        }

    def process(
        self,
        image,
        mode="native",
        min_mp=0.0,
        max_mp=1.0,
        fixed_mp=1.0,
        downscale_method="auto",
        upscale_method="auto",
        clip=None,
        **kwargs
    ):
        if "vision_preparation_mode" in kwargs:
            mode = kwargs["vision_preparation_mode"]
        if "vision_minimum_mp" in kwargs:
            min_mp = kwargs["vision_minimum_mp"]
        if "vision_maximum_mp" in kwargs:
            max_mp = kwargs["vision_maximum_mp"]
        if "vision_fixed_mp" in kwargs:
            fixed_mp = kwargs["vision_fixed_mp"]
        if "vision_downscale_method" in kwargs:
            downscale_method = kwargs["vision_downscale_method"]
        if "vision_upscale_method" in kwargs:
            upscale_method = kwargs["vision_upscale_method"]

        prep_img = prepare_vision_image(
            image=image,
            clip=clip,
            mode=mode,
            min_mp=min_mp,
            max_mp=max_mp,
            fixed_mp=fixed_mp,
            downscale_method=downscale_method,
            upscale_method=upscale_method
        )
        info_str = format_vision_info(prep_img)
        return (prep_img, prep_img.vision_image, info_str)
