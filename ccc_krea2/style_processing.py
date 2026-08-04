"""Moodboard style reference processing: crops/tiles slicing and preparation recipe application."""

import torch
from typing import List, Tuple, Any
from ccc_krea2.reference_specs import StyleReferenceSpec, PreparedVisionImage
from ccc_krea2.vision_prep import prepare_vision_image


SHUFFLE_2X2 = (2, 0, 3, 1)
SHUFFLE_4X4 = (10, 3, 12, 5, 0, 15, 6, 9, 2, 13, 4, 11, 8, 1, 14, 7)


def slice_style_image(
    image: torch.Tensor,
    mode: str = "2x2"
) -> List[torch.Tensor]:
    """Slice an input image tensor into full image, 2x2 crops, or 4x4 tiles using upstream Moodboard shuffled orders."""
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

    # Reorder according to upstream Moodboard shuffle arrays
    if mode == "2x2":
        shuffled = [crops[i] for i in SHUFFLE_2X2]
    elif mode == "4x4":
        shuffled = [crops[i] for i in SHUFFLE_4X4]
    else:
        shuffled = crops

    return shuffled


def apply_statistical_style_fidelity(
    conditioning_tensor: torch.Tensor,
    style_fidelity: float
) -> torch.Tensor:
    """Apply Krea2 Moodboard statistical style fidelity transform to conditioning tensor.

    Transform: output = fidelity * orig + (1 - fidelity) * statistical_target
    where statistical_target repeats (mean, mean + std, mean - std) across feature channels.
    """
    if style_fidelity >= 1.0:
        return conditioning_tensor

    orig = conditioning_tensor.clone()
    mean = torch.mean(orig, dim=-1, keepdim=True)
    std = torch.std(orig, dim=-1, keepdim=True)

    # Construct repeating target stats
    target_stats = torch.cat([mean, mean + std, mean - std], dim=-1)
    if target_stats.shape[-1] < orig.shape[-1]:
        repeats = (orig.shape[-1] // target_stats.shape[-1]) + 1
        target_stats = target_stats.repeat(1, 1, repeats)[:, :, : orig.shape[-1]]
    else:
        target_stats = target_stats[:, :, : orig.shape[-1]]

    blended = style_fidelity * orig + (1.0 - style_fidelity) * target_stats
    return blended


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
