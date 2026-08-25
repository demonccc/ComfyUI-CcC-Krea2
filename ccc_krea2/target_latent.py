"""Target latent creation, target vision context, and geometry resolution for Krea 2 modular pipeline."""

import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from .reference_specs import PreparedVisionImage, ReferenceChain
from .geometry import resize_tensor


EASY_GEOMETRY_HARD_CAP_MEGAPIXELS = 2.5
EASY_ASPECT_RATIOS = ("auto", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16")


@dataclass(frozen=True)
class TargetVisionContext:
    """Target Vision Context metadata for contributing vision tokens from target source."""

    include_in_vision: str = "auto"  # "auto", "yes", "no"
    target_vision_slot: Optional[int] = None  # None ("auto") or int 1..10
    target_alias: str = ""
    target_vision_instruction: str = ""
    target_image: Optional[PreparedVisionImage] = None


@dataclass(frozen=True)
class SubjectAwareSceneGeometry:
    """Scene-led target geometry that preserves Subject pixels until the hard cap requires fitting."""

    target_height: int
    target_width: int
    scene_height: int
    scene_width: int
    subject_height: int
    subject_width: int
    aligned_subject_height: int
    aligned_subject_width: int
    max_megapixels: float
    scene_was_downscaled: bool
    latent_was_expanded: bool
    latent_was_capped: bool
    subject_requires_downscale: bool


@dataclass(frozen=True)
class CanvasAspectGeometry:
    """Smallest selected-aspect canvas that contains its anchor until the hard cap is reached."""

    aspect_ratio: str
    target_height: int
    target_width: int
    anchor_height: int
    anchor_width: int
    aligned_anchor_height: int
    aligned_anchor_width: int
    max_megapixels: float
    latent_was_capped: bool
    anchor_requires_downscale: bool


def get_image_dims(image_tensor: torch.Tensor) -> Tuple[int, int]:
    """Helper to extract (height, width) safely from 3D or 4D image tensor."""
    if image_tensor.ndim == 4:
        return int(image_tensor.shape[1]), int(image_tensor.shape[2])
    elif image_tensor.ndim == 3:
        return int(image_tensor.shape[0]), int(image_tensor.shape[1])
    else:
        raise ValueError(f"Unsupported image tensor shape: {image_tensor.shape}")


def _scene_dimensions_at_pixel_budget(
    scene_width: int,
    scene_height: int,
    max_pixels: int,
    allow_upscale: bool = False,
) -> Tuple[int, int]:
    """Preserve Scene aspect ratio while fitting inside a pixel budget and /16 geometry."""
    scene_pixels = scene_width * scene_height
    budget_scale = math.sqrt(max_pixels / float(scene_pixels))
    scale = budget_scale if allow_upscale else min(1.0, budget_scale)
    raw_width = scene_width * scale
    raw_height = scene_height * scale

    if scale < 1.0:
        width = max(128, int(math.floor(raw_width / 16.0)) * 16)
        height = max(128, int(math.floor(raw_height / 16.0)) * 16)
    else:
        width = max(128, int(round(raw_width / 16.0)) * 16)
        height = max(128, int(round(raw_height / 16.0)) * 16)
        if width * height > max_pixels:
            width = max(128, int(math.floor(raw_width / 16.0)) * 16)
            height = max(128, int(math.floor(raw_height / 16.0)) * 16)

    return width, height


def calculate_subject_aware_scene_geometry(
    scene_image: torch.Tensor,
    subject_image: torch.Tensor,
    max_megapixels: float = EASY_GEOMETRY_HARD_CAP_MEGAPIXELS,
) -> SubjectAwareSceneGeometry:
    """Calculate Scene geometry while avoiding Subject downscale unless the hard cap makes it unavoidable."""
    scene_height, scene_width = get_image_dims(scene_image)
    subject_height, subject_width = get_image_dims(subject_image)
    aligned_subject_width = max(16, int(math.ceil(subject_width / 16.0)) * 16)
    aligned_subject_height = max(16, int(math.ceil(subject_height / 16.0)) * 16)
    max_pixels = max(1, int(max_megapixels * 1_000_000))

    base_width, base_height = _scene_dimensions_at_pixel_budget(
        scene_width,
        scene_height,
        max_pixels,
    )
    scene_was_downscaled = base_width < scene_width or base_height < scene_height

    subject_fits_base = aligned_subject_width <= base_width and aligned_subject_height <= base_height
    latent_was_expanded = False
    latent_was_capped = scene_was_downscaled
    target_width, target_height = base_width, base_height

    if not subject_fits_base:
        # Scale the original Scene dimensions so its aspect ratio remains the geometry source.
        expansion = max(
            base_width / float(scene_width),
            base_height / float(scene_height),
            aligned_subject_width / float(scene_width),
            aligned_subject_height / float(scene_height),
        )
        expanded_width = max(128, int(math.ceil((scene_width * expansion) / 16.0)) * 16)
        expanded_height = max(128, int(math.ceil((scene_height * expansion) / 16.0)) * 16)
        if expanded_width * expanded_height <= max_pixels:
            target_width, target_height = expanded_width, expanded_height
        else:
            target_width, target_height = _scene_dimensions_at_pixel_budget(
                scene_width,
                scene_height,
                max_pixels,
                allow_upscale=True,
            )
            latent_was_capped = True

        latent_was_expanded = target_width > base_width or target_height > base_height

    subject_requires_downscale = aligned_subject_width > target_width or aligned_subject_height > target_height

    return SubjectAwareSceneGeometry(
        target_height=target_height,
        target_width=target_width,
        scene_height=scene_height,
        scene_width=scene_width,
        subject_height=subject_height,
        subject_width=subject_width,
        aligned_subject_height=aligned_subject_height,
        aligned_subject_width=aligned_subject_width,
        max_megapixels=max_megapixels,
        scene_was_downscaled=scene_was_downscaled,
        latent_was_expanded=latent_was_expanded,
        latent_was_capped=latent_was_capped,
        subject_requires_downscale=subject_requires_downscale,
    )


def calculate_canvas_aspect_geometry(
    anchor_image: torch.Tensor,
    aspect_ratio: str,
    max_megapixels: float = EASY_GEOMETRY_HARD_CAP_MEGAPIXELS,
) -> CanvasAspectGeometry:
    """Build a /16 canvas of the requested aspect that contains the anchor without resize when possible."""
    if aspect_ratio not in EASY_ASPECT_RATIOS:
        raise ValueError(f"Unsupported explicit canvas aspect ratio: {aspect_ratio!r}")

    anchor_height, anchor_width = get_image_dims(anchor_image)
    aligned_anchor_width = max(16, int(math.ceil(anchor_width / 16.0)) * 16)
    aligned_anchor_height = max(16, int(math.ceil(anchor_height / 16.0)) * 16)

    if aspect_ratio == "auto":
        ratio_width, ratio_height = aligned_anchor_width, aligned_anchor_height
        target_width, target_height = aligned_anchor_width, aligned_anchor_height
    else:
        ratio_width, ratio_height = (int(value) for value in aspect_ratio.split(":"))
        ratio = ratio_width / float(ratio_height)

        # The anchor, especially Subject, governs the minimum canvas size. The selected
        # aspect expands the missing axis instead of resizing or cropping the anchor.
        raw_height = max(aligned_anchor_height, aligned_anchor_width / ratio)
        raw_width = raw_height * ratio
        target_width = max(128, int(math.ceil(raw_width / 16.0)) * 16)
        target_height = max(128, int(math.ceil(raw_height / 16.0)) * 16)

    max_pixels = max(1, int(max_megapixels * 1_000_000))
    latent_was_capped = target_width * target_height > max_pixels
    if latent_was_capped:
        target_width, target_height = _scene_dimensions_at_pixel_budget(
            ratio_width,
            ratio_height,
            max_pixels,
            allow_upscale=True,
        )

    anchor_requires_downscale = (
        aligned_anchor_width > target_width or aligned_anchor_height > target_height
    )
    return CanvasAspectGeometry(
        aspect_ratio=aspect_ratio,
        target_height=target_height,
        target_width=target_width,
        anchor_height=anchor_height,
        anchor_width=anchor_width,
        aligned_anchor_height=aligned_anchor_height,
        aligned_anchor_width=aligned_anchor_width,
        max_megapixels=max_megapixels,
        latent_was_capped=latent_was_capped,
        anchor_requires_downscale=anchor_requires_downscale,
    )


def calculate_target_latent_resolution(
    geometry_mode: str = "fixed",
    target_megapixels: float = 1.0,
    fixed_megapixels: float = 2.0,
    aspect_ratio: str = "1:1",
    target_image: Optional[PreparedVisionImage] = None,
    geometry_image: Optional[PreparedVisionImage] = None,
    subject_image: Optional[PreparedVisionImage] = None,
    scene_image: Optional[PreparedVisionImage] = None,
    force_target_megapixels: bool = False,
    **kwargs: Any,
) -> Tuple[int, int, str, float, Optional[Tuple[int, int]], list]:
    """Calculate target latent pixel dimensions [H, W] and metadata."""
    if "target_geometry" in kwargs:
        geometry_mode = kwargs["target_geometry"]
    if "maximum_mp" in kwargs:
        target_megapixels = kwargs["maximum_mp"]
    if "fixed_mp" in kwargs:
        fixed_megapixels = kwargs["fixed_mp"]
    if "fixed_aspect_ratio" in kwargs:
        aspect_ratio = kwargs["fixed_aspect_ratio"]
    force_target_mp = force_target_megapixels or kwargs.get("force_target_megapixels", False)

    warnings = []
    source_dims = None
    geometry_source = "custom_aspect_ratio"

    if geometry_mode == "favor_subject":
        if geometry_image is None:
            geometry_image = subject_image
        geometry_mode = "favor_image"
    elif geometry_mode == "favor_scene":
        if geometry_image is None:
            geometry_image = scene_image
        geometry_mode = "favor_image"

    ref_img = geometry_image or target_image or subject_image or scene_image

    if geometry_mode in ("favor_image", "favor_subject", "favor_scene") and ref_img is not None:
        ih, iw = get_image_dims(ref_img.original_image)
        source_dims = (ih, iw)
        src_ar = iw / float(ih)
        geometry_source = "geometry_original_image" if geometry_image else "target_original_image"
        src_mp = (ih * iw) / 1_000_000.0
        active_mp = target_megapixels if force_target_mp else min(src_mp, target_megapixels)
    elif geometry_mode in ("fixed", "crop_subject"):
        geometry_source = "fixed_megapixels"
        active_mp = fixed_megapixels
        if fixed_megapixels > 2.0:
            warnings.append(
                f"Warning: Fixed MP is set to {fixed_megapixels:.2f} MP, exceeding recommended 2.0 MP limit."
            )

        if ":" in aspect_ratio:
            parts = aspect_ratio.split(":")
            try:
                src_ar = float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                src_ar = 1.0
        else:
            src_ar = 1.0
    else:
        if ref_img is not None:
            ih, iw = get_image_dims(ref_img.original_image)
            source_dims = (ih, iw)
            src_ar = iw / float(ih)
            geometry_source = "geometry_original_image" if geometry_image else "target_original_image"
            active_mp = min((ih * iw) / 1_000_000.0, target_megapixels)
        else:
            geometry_source = "fixed_megapixels"
            active_mp = fixed_megapixels
            src_ar = 1.0

    target_pixels = int(active_mp * 1_000_000)
    raw_h = math.sqrt(target_pixels / src_ar)
    raw_w = raw_h * src_ar

    # Align to 16-pixel multiples
    target_h = max(128, int(round(raw_h / 16.0)) * 16)
    target_w = max(128, int(round(raw_w / 16.0)) * 16)

    if (
        geometry_mode in ("favor_image", "favor_subject", "favor_scene")
        and (target_h * target_w) / 1_000_000.0 > target_megapixels + 0.05
    ):
        alt_h = max(128, int(math.floor(raw_h / 16.0)) * 16)
        alt_w = max(128, int(math.floor(raw_w / 16.0)) * 16)
        if alt_h >= 128 and alt_w >= 128:
            target_h, target_w = alt_h, alt_w

    return target_h, target_w, geometry_source, active_mp, source_dims, warnings


@dataclass(frozen=True)
class TargetContentTransform:
    source_size: Tuple[int, int]  # (W, H)
    crop_rectangle: Tuple[int, int, int, int]  # (left, top, crop_w, crop_h)
    target_size: Tuple[int, int]  # (target_w, target_h)
    interpolation: str  # "bicubic"
    interpolation_applied: bool


def adapt_target_content_image(
    image: torch.Tensor, target_w: int, target_h: int
) -> Tuple[torch.Tensor, TargetContentTransform]:
    """Deterministically adapt a source pixel image to fill target geometry for target latent initialization."""
    if image.ndim == 3:
        image = image.unsqueeze(0)

    bs, src_h, src_w, c = image.shape
    tgt_ar = target_w / float(target_h)
    src_ar = src_w / float(src_h)

    if src_ar > tgt_ar:
        crop_h = src_h
        crop_w = int(round(src_h * tgt_ar))
    else:
        crop_w = src_w
        crop_h = int(round(src_w / tgt_ar))

    left = (src_w - crop_w) // 2
    top = (src_h - crop_h) // 2

    cropped = image[:, top : top + crop_h, left : left + crop_w, :]

    interp_applied = (crop_w, crop_h) != (target_w, target_h)
    if interp_applied:
        adapted = resize_tensor(cropped, target_h=target_h, target_w=target_w, method="bicubic")
    else:
        adapted = cropped

    adapted = torch.clamp(adapted, 0.0, 1.0)

    transform = TargetContentTransform(
        source_size=(src_w, src_h),
        crop_rectangle=(left, top, crop_w, crop_h),
        target_size=(target_w, target_h),
        interpolation="bicubic",
        interpolation_applied=interp_applied,
    )
    return adapted, transform


def normalize_vae_output(encoded: Any, batch_size: int) -> torch.Tensor:
    """Normalize VAE encode output into a 4D or 5D tensor and expand batch dimension if necessary."""
    if isinstance(encoded, torch.Tensor):
        latent = encoded
    elif isinstance(encoded, dict) and "samples" in encoded:
        latent = encoded["samples"]
    elif hasattr(encoded, "samples"):
        latent = getattr(encoded, "samples")
    elif hasattr(encoded, "sample") and callable(getattr(encoded, "sample")):
        latent = encoded.sample()
    else:
        raise ValueError(f"Unsupported VAE return format: {type(encoded)}")

    if not isinstance(latent, torch.Tensor) or latent.ndim not in (4, 5):
        raise ValueError(f"Normalized VAE latent must be a 4D or 5D tensor, got shape {getattr(latent, 'shape', None)}")

    b = latent.shape[0]
    if b == 1 and batch_size > 1:
        repeats = [batch_size] + [1] * (latent.ndim - 1)
        latent = latent.repeat(*repeats)
    elif b == batch_size:
        pass
    else:
        raise ValueError(f"Encoded latent batch size ({b}) does not match requested batch size ({batch_size}).")

    return latent


resolve_target_geometry = calculate_target_latent_resolution


def parse_vision_slot_input(slot_input: Any) -> Optional[int]:
    """Parse vision slot string/int input ('auto' or None -> None, '1'..'10' -> int)."""
    if slot_input is None or str(slot_input).lower() == "auto":
        return None
    try:
        val = int(slot_input)
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None


def should_include_target_in_vision(ctx: TargetVisionContext, existing_chain: Optional[ReferenceChain] = None) -> bool:
    """Evaluate whether target vision context should contribute a Qwen vision block.

    Auto deduplication logic:
    If include_in_vision == "yes", always True.
    If include_in_vision == "no", always False.
    If include_in_vision == "auto":
      True if target_image is provided and NOT already present in existing_chain edit references.
    """
    if ctx.include_in_vision == "no":
        return False
    if ctx.include_in_vision == "yes":
        return ctx.target_image is not None

    # "auto" mode
    if ctx.target_image is None:
        return False

    if existing_chain is None or not existing_chain.references:
        return True

    target_prep = ctx.target_image
    for ref in existing_chain.references:
        ref_path = getattr(ref, "reference_path", ref.role.lower())
        # Exception: Style path references undergo different processing and are NOT deduplicated
        if ref_path == "style":
            continue

        ref_prep = ref.prepared_image
        if ref_prep is target_prep:
            return False
        if ref_prep is not None and target_prep is not None:
            ref_orig = getattr(ref_prep, "original_image", ref_prep)
            target_orig = getattr(target_prep, "original_image", target_prep)
            if ref_orig is target_orig:
                return False

    return True


def create_target_latent(
    target_latent_content: str = "empty",
    target_geometry: str = "fixed",
    target_image: Optional[PreparedVisionImage] = None,
    geometry_image: Optional[PreparedVisionImage] = None,
    subject_image: Optional[PreparedVisionImage] = None,
    scene_image: Optional[PreparedVisionImage] = None,
    maximum_mp: float = 2.0,
    fixed_mp: float = 2.0,
    fixed_aspect_ratio: str = "1:1",
    custom_aspect_width: int = 1,
    custom_aspect_height: int = 1,
    batch_size: int = 1,
    vae: Any = None,
    content_fit: str = "crop",
    include_in_vision: str = "auto",
    target_vision_slot: Any = "auto",
    target_alias: str = "",
    target_vision_instruction: str = "",
    **kwargs: Any,
) -> Tuple[Dict[str, Any], str]:
    """Backward compatibility alias for build_target_latent."""
    aspect = f"{custom_aspect_width}:{custom_aspect_height}" if fixed_aspect_ratio == "custom" else fixed_aspect_ratio
    return build_target_latent(
        vae=vae,
        target_content=target_latent_content,
        geometry_mode=target_geometry,
        target_megapixels=maximum_mp,
        fixed_megapixels=fixed_mp,
        aspect_ratio=aspect,
        batch_size=batch_size,
        target_image=target_image,
        geometry_image=geometry_image,
        subject_image=subject_image,
        scene_image=scene_image,
        content_fit=content_fit,
        include_in_vision=include_in_vision,
        target_vision_slot=target_vision_slot,
        target_alias=target_alias,
        target_vision_instruction=target_vision_instruction,
        **kwargs,
    )


def build_target_latent(
    vae: Any = None,
    target_content: str = "empty",
    geometry_mode: str = "fixed",
    target_megapixels: float = 2.0,
    fixed_megapixels: float = 2.0,
    aspect_ratio: str = "1:1",
    batch_size: int = 1,
    target_image: Optional[PreparedVisionImage] = None,
    geometry_image: Optional[PreparedVisionImage] = None,
    subject_image: Optional[PreparedVisionImage] = None,
    scene_image: Optional[PreparedVisionImage] = None,
    content_fit: str = "crop",
    include_in_vision: str = "auto",
    target_vision_slot: Any = "auto",
    target_alias: str = "",
    target_vision_instruction: str = "",
    force_target_megapixels: bool = False,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], str]:
    """Build formatted target LATENT dict and latent_info string."""
    if "target_latent_content" in kwargs:
        target_content = kwargs["target_latent_content"]
    if "target_geometry" in kwargs:
        geometry_mode = kwargs["target_geometry"]
    if "maximum_mp" in kwargs:
        target_megapixels = kwargs["maximum_mp"]
    if "fixed_mp" in kwargs:
        fixed_megapixels = kwargs["fixed_mp"]
    content_fit = kwargs.get("content_fit", kwargs.get("target_content_fit", content_fit))
    force_target_mp = force_target_megapixels or kwargs.get("force_target_megapixels", False)
    orig_target_content = target_content
    orig_geometry_mode = geometry_mode

    if target_content == "subject":
        if target_image is None:
            target_image = subject_image
        target_content = "image"
    elif target_content == "scene":
        if target_image is None:
            target_image = scene_image
        target_content = "image"

    if geometry_mode == "favor_subject":
        if geometry_image is None:
            geometry_image = subject_image
        geometry_mode = "favor_image"
    elif geometry_mode == "favor_scene":
        if geometry_image is None:
            geometry_image = scene_image
        geometry_mode = "favor_image"

    if orig_target_content == "subject":
        active_target_image = target_image or subject_image
    elif orig_target_content == "scene":
        active_target_image = target_image or scene_image
    else:
        active_target_image = target_image

    target_h, target_w, geom_src, active_mp, src_dims, warnings = calculate_target_latent_resolution(
        geometry_mode=geometry_mode,
        target_megapixels=target_megapixels,
        fixed_megapixels=fixed_megapixels,
        aspect_ratio=aspect_ratio,
        target_image=active_target_image,
        geometry_image=geometry_image,
        subject_image=subject_image,
        scene_image=scene_image,
        force_target_megapixels=force_target_mp,
    )

    if target_width is not None and target_height is not None:
        target_w = max(128, int(target_width) // 16 * 16)
        target_h = max(128, int(target_height) // 16 * 16)
        active_mp = (target_w * target_h) / 1_000_000.0
        geom_src = f"{geom_src}_explicit_target"

    latent_h = target_h // 8
    latent_w = target_w // 8
    vae_applied = False
    transform_info: Optional[TargetContentTransform] = None
    contained_w: Optional[int] = None
    contained_h: Optional[int] = None
    offset_x: int = 0
    offset_y: int = 0
    content_src_name = "N/A"

    if target_content == "empty":
        samples = torch.zeros((batch_size, 16, latent_h, latent_w), dtype=torch.float32)
    elif target_content in ("image", "subject", "scene"):
        if active_target_image is None:
            if orig_target_content == "subject":
                raise ValueError("Subject image is required when target_content is 'subject'.")
            elif orig_target_content == "scene":
                raise ValueError("Scene image is required when target_content is 'scene'.")
            else:
                raise ValueError("Target image is required when target_content is 'image'.")

        if vae is None:
            raise ValueError(f"VAE is required when target_content is '{target_content}'.")

        orig_img = active_target_image.original_image
        src_h, src_w = get_image_dims(orig_img)

        if content_fit == "contain_no_upscale":
            if src_w <= target_w and src_h <= target_h:
                scale = 1.0
            else:
                scale = min(target_w / float(src_w), target_h / float(src_h))

            raw_contained_w = src_w * scale
            raw_contained_h = src_h * scale

            if scale < 1.0:
                contained_w = max(16, int(round(raw_contained_w / 16.0)) * 16)
                contained_h = max(16, int(round(raw_contained_h / 16.0)) * 16)
                if contained_w > target_w:
                    contained_w = int(math.floor(raw_contained_w / 16.0)) * 16
                if contained_h > target_h:
                    contained_h = int(math.floor(raw_contained_h / 16.0)) * 16
            else:
                contained_w = min(target_w, int(math.ceil(src_w / 16.0)) * 16)
                contained_h = min(target_h, int(math.ceil(src_h / 16.0)) * 16)
                contained_w = max(16, contained_w)
                contained_h = max(16, contained_h)

            if orig_img.ndim == 3:
                img_4d = orig_img.unsqueeze(0)
            else:
                img_4d = orig_img

            if scale < 1.0:
                adapted_img = resize_tensor(img_4d, target_h=contained_h, target_w=contained_w, method="bicubic")
            elif (src_w, src_h) != (contained_w, contained_h):
                pad_h = contained_h - src_h
                pad_w = contained_w - src_w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                adapted_img = F.pad(
                    img_4d.permute(0, 3, 1, 2),
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                ).permute(0, 2, 3, 1)
            else:
                adapted_img = img_4d
            adapted_img = torch.clamp(adapted_img, 0.0, 1.0)

            raw_encoded = vae.encode(adapted_img)
            contained_latent = normalize_vae_output(raw_encoded, batch_size=batch_size)

            if contained_latent.ndim == 5:
                t_dim = contained_latent.shape[2]
                contained_lh = contained_latent.shape[3]
                contained_lw = contained_latent.shape[4]
                samples = torch.zeros(
                    (batch_size, 16, t_dim, latent_h, latent_w),
                    dtype=contained_latent.dtype,
                    device=contained_latent.device,
                )
                offset_y = (latent_h - contained_lh) // 2
                offset_x = (latent_w - contained_lw) // 2
                samples[:, :, :, offset_y : offset_y + contained_lh, offset_x : offset_x + contained_lw] = (
                    contained_latent
                )
            else:
                contained_lh = contained_latent.shape[2]
                contained_lw = contained_latent.shape[3]
                samples = torch.zeros(
                    (batch_size, 16, latent_h, latent_w),
                    dtype=contained_latent.dtype,
                    device=contained_latent.device,
                )
                offset_y = (latent_h - contained_lh) // 2
                offset_x = (latent_w - contained_lw) // 2
                samples[:, :, offset_y : offset_y + contained_lh, offset_x : offset_x + contained_lw] = contained_latent

            vae_applied = True
            content_src_name = f"Target original_image ({target_alias})" if target_alias else "Target original_image"
        else:
            adapted_img, transform_info = adapt_target_content_image(orig_img, target_w=target_w, target_h=target_h)
            raw_encoded = vae.encode(adapted_img)
            samples = normalize_vae_output(raw_encoded, batch_size=batch_size)
            vae_applied = True
            content_src_name = f"Target original_image ({target_alias})" if target_alias else "Target original_image"
    else:
        raise ValueError(f"Unknown target_content mode: '{target_content}'. Expected 'empty' or 'image'.")

    slot_val = parse_vision_slot_input(target_vision_slot)

    vision_ctx = TargetVisionContext(
        include_in_vision=include_in_vision,
        target_vision_slot=slot_val,
        target_alias=target_alias,
        target_vision_instruction=target_vision_instruction,
        target_image=active_target_image,
    )

    latent_dict = {
        "samples": samples,
        "batch_index": list(range(batch_size)),
        "target_vision_context": vision_ctx,
    }

    # Format latent_info string
    src_size_str = f"{src_dims[1]} x {src_dims[0]}" if src_dims else "N/A"
    actual_mp = (target_h * target_w) / 1_000_000.0

    if content_fit == "contain_no_upscale" and target_content in ("image", "subject", "scene"):
        orig_img = active_target_image.original_image
        sh, sw = get_image_dims(orig_img)
        lines = [
            f"Latent Content: {orig_target_content}",
            f"Geometry Strategy: {orig_geometry_mode}",
            f"Geometry Source: {geom_src}",
            f"Content Source: {content_src_name}",
            "Content Fit: contain_no_upscale",
            f"Content Source Size: {sw} x {sh}",
            f"Content Resolved Size: {contained_w} x {contained_h}",
            "Content Crop Rectangle: none",
            "Content Latent Placement: centered",
            f"Content Latent Offset: X={offset_x}, Y={offset_y}",
            f"Content Target Size: {target_w} x {target_h}",
            f"VAE Encode Applied: {'yes' if vae_applied else 'no'}",
            f"Include Target in Vision: {include_in_vision}",
            f"Target Vision Slot: {'auto' if slot_val is None else slot_val}",
        ]
    else:
        lines = [
            f"Latent Content: {orig_target_content}",
            f"Geometry Strategy: {orig_geometry_mode}",
            f"Geometry Source: {geom_src}",
            f"Content Source: {content_src_name}",
            f"Content Source Size: {f'{transform_info.source_size[0]} x {transform_info.source_size[1]}' if transform_info else src_size_str}",
            f"Content Crop Rectangle: {transform_info.crop_rectangle if transform_info else 'N/A'}",
            f"Content Target Size: {f'{transform_info.target_size[0]} x {transform_info.target_size[1]}' if transform_info else f'{target_w} x {target_h}'}",
            f"Content Interpolation: {transform_info.interpolation if transform_info else 'none'}",
            f"VAE Encode Applied: {'yes' if vae_applied else 'no'}",
            f"Include Target in Vision: {include_in_vision}",
            f"Target Vision Slot: {'auto' if slot_val is None else slot_val}",
        ]

    if target_alias:
        lines.append(f"Target Alias: {target_alias}")

    if geometry_mode in ("favor_image", "favor_subject", "favor_scene"):
        lines.append(f"Maximum MP: {target_megapixels:.2f} MP")
    elif geometry_mode == "fixed":
        lines.append(f"Fixed MP: {fixed_megapixels:.2f} MP")
        lines.append(f"Fixed Aspect Ratio: {aspect_ratio}")

    lines.extend(
        [
            f"Target Pixel Size: {target_w} x {target_h}",
            f"Target MP: {actual_mp:.3f} MP",
            f"Target Latent Size: {latent_w} x {latent_h}",
            f"Batch Size: {batch_size}",
            f"Warnings: {'; '.join(warnings) if warnings else 'none'}",
        ]
    )

    latent_info = "\n".join(lines)
    return latent_dict, latent_info
