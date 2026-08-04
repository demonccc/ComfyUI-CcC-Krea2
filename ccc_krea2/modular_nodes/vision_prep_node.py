"""CcC Krea2 - Qwen Vision Image Prep node."""

from ccc_krea2.vision_prep import prepare_vision_image, format_vision_info


class CcCKrea2QwenVisionImagePrep:
    """Prepare a derivative image optimized for Qwen Vision while keeping the original image intact."""

    CATEGORY = "CcC/Krea2"
    RETURN_TYPES = ("CCC_KREA2_PREPARED_IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("prepared_image", "vision_image", "vision_info")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "vision_preparation_mode": (["native", "adaptive", "fixed"], {"default": "native"}),
                "vision_minimum_mp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "vision_maximum_mp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05}),
                "vision_fixed_mp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05}),
                "vision_downscale_method": (["auto", "area", "bicubic", "bilinear", "lanczos", "nearest-exact"], {"default": "auto"}),
                "vision_upscale_method": (["auto", "bicubic", "bilinear", "lanczos", "nearest-exact"], {"default": "auto"}),
            }
        }

    def process(
        self,
        clip,
        image,
        vision_preparation_mode="native",
        vision_minimum_mp=0.0,
        vision_maximum_mp=1.0,
        vision_fixed_mp=1.0,
        vision_downscale_method="auto",
        vision_upscale_method="auto"
    ):
        prep_img = prepare_vision_image(
            image=image,
            clip=clip,
            mode=vision_preparation_mode,
            min_mp=vision_minimum_mp,
            max_mp=vision_maximum_mp,
            fixed_mp=vision_fixed_mp,
            downscale_method=vision_downscale_method,
            upscale_method=vision_upscale_method
        )
        info_str = format_vision_info(prep_img)
        return (prep_img, prep_img.vision_image, info_str)
