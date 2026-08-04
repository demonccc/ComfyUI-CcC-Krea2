"""Moodboard style reference processing: crops/tiles slicing and preparation recipe application."""

import torch
from typing import List, Tuple, Any
from ccc_krea2.reference_specs import StyleReferenceSpec, PreparedVisionImage
from ccc_krea2.vision_prep import prepare_vision_image


def slice_style_image(
    image: torch.Tensor,
    mode: str = "2x2"
) -> List[torch.Tensor]:
    """Slice an input image tensor into full image, 2x2 crops, or 4x4 tiles."""
    if image.ndim == 3:
        image = image.unsqueeze(0)

    bs, ih, iw, c = image.shape

    if mode == "full":
        return [image]

    crops = []
    grid_size = 2 if mode == "2x2" else 4
    step_h = ih // grid_size
    step_w = iw // grid_size

    for row in range(grid_size):
        for col in range(grid_size):
            y0 = row * step_h
            y1 = (row + 1) * step_h if row < grid_size - 1 else ih
            x0 = col * step_w
            x1 = (col + 1) * step_w if col < grid_size - 1 else iw
            crop = image[:, y0:y1, x0:x1, :]
            crops.append(crop)

    return crops


def expand_style_reference_spans(
    spec: StyleReferenceSpec,
    start_slot: int,
    clip: Any
) -> Tuple[List[PreparedVisionImage], int, int]:
    """Expand a StyleReferenceSpec into contiguous physical vision prep image spans.

    Returns:
        (prepared_crop_images, start_slot, end_slot)
    """
    orig_img = spec.prepared_image.original_image
    slices = slice_style_image(orig_img, mode=spec.style_processing)

    prepared_crops: List[PreparedVisionImage] = []
    prep_spec = spec.prepared_image.prep_spec

    for crop in slices:
        prep_crop = prepare_vision_image(
            image=crop,
            clip=clip,
            mode=prep_spec.mode,
            min_mp=prep_spec.semantic_min_mp,
            max_mp=prep_spec.semantic_max_mp,
            fixed_mp=prep_spec.semantic_fixed_mp,
            downscale_method=prep_spec.downscale_method_requested,
            upscale_method=prep_spec.upscale_method_requested
        )
        prepared_crops.append(prep_crop)

    end_slot = start_slot + len(prepared_crops) - 1
    return prepared_crops, start_slot, end_slot
