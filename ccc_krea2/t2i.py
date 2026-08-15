"""Native Text-to-Image helper node for CcC Krea2 suite."""

import math
from typing import Tuple, Dict, Any, Optional
import torch

from .constants import NODE_CATEGORY, LOGGER_PREFIX
from .prompt_augmentation import (
    CCC_KREA2_PROMPT_AUGMENTATION,
    apply_prompt_augmentation,
)

ASPECT_RATIO_MAP: Dict[str, Tuple[float, float]] = {
    "1:1": (1.0, 1.0),
    "4:3": (4.0, 3.0),
    "3:4": (3.0, 4.0),
    "16:9": (16.0, 9.0),
    "9:16": (9.0, 16.0),
    "3:2": (3.0, 2.0),
    "2:3": (2.0, 3.0),
}


def calculate_t2i_resolution(
    aspect_ratio: str,
    megapixels: float,
    custom_aspect_width: int = 1,
    custom_aspect_height: int = 1,
) -> Tuple[int, int]:
    """Calculate 16-pixel aligned dimensions for T2I based on aspect ratio and megapixels.

    Guarantees output width and height are at least 128 and multiples of 16.
    """
    if aspect_ratio in ASPECT_RATIO_MAP:
        rw, rh = ASPECT_RATIO_MAP[aspect_ratio]
    else:
        rw = max(1.0, float(custom_aspect_width))
        rh = max(1.0, float(custom_aspect_height))

    target_pixels = megapixels * 1_000_000.0
    ratio = rw / rh
    w_raw = math.sqrt(target_pixels * ratio)
    h_raw = math.sqrt(target_pixels / ratio)

    out_w = max(128, int(round(w_raw / 16.0) * 16))
    out_h = max(128, int(round(h_raw / 16.0) * 16))

    return out_w, out_h


def create_empty_sd3_latent(width: int, height: int, batch_size: int = 1) -> Dict[str, Any]:
    """Generate empty SD3-compatible LATENT dictionary matching EmptySD3LatentImage contract."""
    try:
        import nodes

        if hasattr(nodes, "EmptySD3LatentImage"):
            res = nodes.EmptySD3LatentImage().generate(width=width, height=height, batch_size=batch_size)
            if isinstance(res, tuple) and len(res) > 0:
                return res[0]
            elif isinstance(res, dict):
                return res
    except Exception:
        pass

    device = "cpu"
    try:
        import comfy.model_management

        device = comfy.model_management.intermediate_device()
    except Exception:
        pass

    samples = torch.zeros((batch_size, 16, height // 8, width // 8), dtype=torch.float32, device=device)
    return {"samples": samples}


class CcCKrea2TextToImage:
    """CcC Krea2 - Text to Image native helper node."""

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "latent")
    FUNCTION = "process"
    CATEGORY = NODE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "aspect_ratio": (
                    ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "custom"],
                    {"default": "1:1"},
                ),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "custom_aspect_width": ("INT", {"default": 1, "min": 1, "max": 32}),
                "custom_aspect_height": ("INT", {"default": 1, "min": 1, "max": 32}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
            },
        }

    def process(
        self,
        model: Any,
        clip: Any,
        prompt: str,
        aspect_ratio: str = "1:1",
        megapixels: float = 1.0,
        batch_size: int = 1,
        custom_aspect_width: int = 1,
        custom_aspect_height: int = 1,
        negative_prompt: str = "",
        prompt_augmentation: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        if clip is None:
            raise ValueError(f"{LOGGER_PREFIX} CLIP text encoder input cannot be None.")

        # Apply immutable prompt augmentation
        eff_pos, eff_neg = apply_prompt_augmentation(
            positive_prompt=prompt,
            negative_prompt=negative_prompt,
            augmentation=prompt_augmentation,
        )

        # Native CLIP text encoding (independent positive and negative)
        pos_tokens = clip.tokenize(eff_pos)
        positive = clip.encode_from_tokens_scheduled(pos_tokens)

        neg_tokens = clip.tokenize(eff_neg)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        # T2I resolution calculation
        width, height = calculate_t2i_resolution(
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            custom_aspect_width=custom_aspect_width,
            custom_aspect_height=custom_aspect_height,
        )

        # Native empty SD3 latent creation
        latent = create_empty_sd3_latent(width=width, height=height, batch_size=batch_size)

        # Return unchanged input MODEL along with positive, negative, and latent
        return (model, positive, negative, latent)
