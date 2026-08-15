import torch
from dataclasses import dataclass
from typing import List, Tuple, Any, Union
from .reference_specs import StyleReferenceSpec, PreparedVisionImage
from .vision_prep import prepare_vision_image


SHUFFLE_2X2 = (2, 0, 3, 1)
SHUFFLE_4X4 = (10, 3, 12, 5, 0, 15, 6, 9, 2, 13, 4, 11, 8, 1, 14, 7)


@dataclass(frozen=True)
class StyleSpanOperation:
    logical_reference_id: str
    logical_vision_slot: int
    physical_qwen_index: int
    row_start: int
    row_end: int
    style_fidelity: float
    indirect_style_transfer: bool


VALID_STYLE_PROCESSING_MODES = ("full", "2x2", "4x4")


def get_style_processing_image_count(mode: str) -> int:
    """Return physical vision image count for a given style_processing mode."""
    if mode == "full":
        return 1
    elif mode == "2x2":
        return 4
    elif mode == "4x4":
        return 16
    raise ValueError(f"Invalid style_processing mode '{mode}'. Supported modes: 'full', '2x2', '4x4'.")


def slice_style_image(image: torch.Tensor, mode: str = "2x2") -> List[torch.Tensor]:
    """Slice an input image tensor into full image, 2x2 crops, or 4x4 tiles using upstream Moodboard shuffled orders."""
    if mode not in VALID_STYLE_PROCESSING_MODES:
        raise ValueError(f"Invalid style_processing mode '{mode}'. Supported modes: 'full', '2x2', '4x4'.")

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
        raise ValueError(f"Invalid style_processing mode '{mode}'. Supported modes: 'full', '2x2', '4x4'.")

    return shuffled


def apply_statistical_style_fidelity(
    cond_tensor: torch.Tensor, spans_info: List[Union[StyleSpanOperation, Tuple[Tuple[int, int], float, bool]]]
) -> Tuple[torch.Tensor, bool, List[int]]:
    """Apply Krea2 Moodboard statistical style fidelity and multi-span indirect row removal.

    Fidelity Transform: output = fidelity * orig + (1.0 - fidelity) * target
    Target: per-reference stats (mean, mean + std, mean - std) cycled across span visual rows.
    Indirect: removes all designated indirect style vision rows in ONE operation using a single keep mask.

    Returns:
        (transformed_tensor, indirect_applied, removed_indices)
    """
    b, seq, fused = cond_tensor.shape
    if fused % 12 != 0:
        raise ValueError(f"Conditioning fused dimension {fused} is not divisible by 12.")

    z = cond_tensor.reshape(b, seq, 12, fused // 12).clone()
    keep = torch.ones(seq, dtype=torch.bool, device=cond_tensor.device)
    indirect_applied = False

    # Normalize operations into StyleSpanOperation objects
    ops: List[StyleSpanOperation] = []
    for item in spans_info:
        if isinstance(item, StyleSpanOperation):
            ops.append(item)
        elif isinstance(item, tuple) and len(item) == 3:
            (start, end), fidelity, indirect = item
            ops.append(
                StyleSpanOperation(
                    logical_reference_id="style",
                    logical_vision_slot=0,
                    physical_qwen_index=0,
                    row_start=start,
                    row_end=end,
                    style_fidelity=fidelity,
                    indirect_style_transfer=indirect,
                )
            )

    # Step 1: Apply Style Fidelity to all style spans using original row coordinates
    for op in ops:
        start, end = op.row_start, op.row_end
        if start < 0 or end <= start or end > seq:
            raise ValueError(
                f"Style span validation failed: span range ({start}, {end}) is invalid "
                f"or exceeds sequence length ({seq})."
            )

        if op.style_fidelity < 1.0:
            span = z[:, start:end]  # (B, rows, 12, fused//12)
            mu = span.mean(dim=1, keepdim=True)
            sigma = span.std(dim=1, keepdim=True) + 1e-6
            stats = torch.cat([mu, mu + sigma, mu - sigma], dim=1)  # (B, 3, 12, fused//12)
            idx = torch.arange(end - start, device=span.device) % 3
            target = stats[:, idx]
            z[:, start:end] = op.style_fidelity * span + (1.0 - op.style_fidelity) * target

        if op.indirect_style_transfer:
            keep[start:end] = False
            indirect_applied = True

    # Step 2: Remove all indirect style rows in ONE operation using single keep-mask
    z = z.reshape(b, seq, fused)
    removed_indices = []
    if indirect_applied:
        removed_indices = torch.where(~keep)[0].tolist()
        z = z[:, keep]

    return z, indirect_applied, removed_indices


def expand_style_reference_spans(
    spec: StyleReferenceSpec, start_slot: int, clip: Any
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
            upscale_method=prep_spec.upscale_method_requested,
        )
        prepared_crops.append(prep_crop)

    end_slot = start_slot + len(prepared_crops) - 1
    return prepared_crops, start_slot, end_slot
