"""Krea2 CcC Size Resolver node."""

from typing import Tuple

import torch

from ..constants import NODE_CATEGORY
from ..target_latent import get_image_dims


def _resolve_size(long_edge_image: torch.Tensor, aspect_ratio_image: torch.Tensor) -> Tuple[int, int]:
    size_h, size_w = get_image_dims(long_edge_image)
    aspect_h, aspect_w = get_image_dims(aspect_ratio_image)

    long_edge = max(size_w, size_h)
    ratio = aspect_w / float(aspect_h)
    if ratio <= 0:
        raise ValueError("[CcC Krea2] Aspect ratio reference must have valid dimensions.")

    if ratio >= 1.0:
        width = long_edge
        height = max(1, int(round(long_edge / ratio)))
    else:
        height = long_edge
        width = max(1, int(round(long_edge * ratio)))

    return width, height


class CcCKrea2SizeResolver:
    """Resolve width and height from independent long-edge and aspect-ratio images."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "process"
    DESCRIPTION = (
        "Resolves final width and height from two images: Long Edge Image supplies only the longest edge; "
        "Aspect Ratio Image supplies only width-to-height proportions. Alignment to /16 is handled by Latent."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "long_edge_image": ("IMAGE",),
                "aspect_ratio_image": ("IMAGE",),
            }
        }

    def process(self, long_edge_image: torch.Tensor, aspect_ratio_image: torch.Tensor):
        return _resolve_size(long_edge_image, aspect_ratio_image)
