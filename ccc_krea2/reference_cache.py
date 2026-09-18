"""Portable reference caches for CcC Krea2.

A cache stores prompt-independent work only:
- the raw VAE appearance latent for one target geometry;
- Qwen3-VL visual features (merged tokens, grid and DeepStack tensors).

Prompt-dependent Qwen language conditioning is always recomputed at runtime.
"""

from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .grounding import resize_grounding_image
from .krea2edit_geometry import (
    ResolvedGeometry,
    process_image_and_mask_geometry,
    resolve_krea2edit_geometry,
)


CACHE_FORMAT = "ccc_krea2_reference_cache"
CACHE_VERSION = "1.0"
CACHE_FOLDER = "krea2_ccc_cache"
_QWEN_CACHE_LOCK = RLock()


@dataclass(frozen=True)
class QwenVisualCache:
    """Prompt-independent Qwen3-VL visual features."""

    merged: torch.Tensor
    grid: torch.Tensor
    deepstack: Tuple[torch.Tensor, ...]
    model_type: str
    input_size: Tuple[int, int]
    grounding_px: int

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.merged) or self.merged.ndim != 2:
            raise ValueError("[CcC Krea2] Qwen cache merged tensor must be 2D.")
        if not torch.is_tensor(self.grid) or self.grid.ndim != 2 or self.grid.shape[-1] != 3:
            raise ValueError("[CcC Krea2] Qwen cache grid must have shape N x 3.")
        if not self.deepstack or not all(torch.is_tensor(t) and t.ndim == 2 for t in self.deepstack):
            raise ValueError("[CcC Krea2] Qwen cache requires DeepStack tensors.")
        object.__setattr__(self, "merged", self.merged.detach().cpu().contiguous().clone())
        object.__setattr__(self, "grid", self.grid.detach().cpu().contiguous().clone())
        object.__setattr__(
            self,
            "deepstack",
            tuple(t.detach().cpu().contiguous().clone() for t in self.deepstack),
        )


@dataclass(frozen=True)
class CcCKrea2ReferenceCache:
    """Portable cached representation of one CcC visual reference."""

    appearance_latent: torch.Tensor
    geometry: ResolvedGeometry
    qwen_visual: QwenVisualCache
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.appearance_latent) or self.appearance_latent.ndim not in (4, 5):
            raise ValueError("[CcC Krea2] Cached appearance latent must be a 4D or 5D tensor.")
        object.__setattr__(
            self,
            "appearance_latent",
            self.appearance_latent.detach().cpu().contiguous().clone(),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def target_size(self) -> Tuple[int, int]:
        return int(self.metadata["target_width"]), int(self.metadata["target_height"])


@dataclass(frozen=True)
class CachedQwenImage:
    """Tokenization placeholder carrying a prompt-independent Qwen visual cache."""

    cache: QwenVisualCache


def _qwen_transformer(clip: Any) -> Any:
    stage = getattr(clip, "cond_stage_model", None)
    encoder = getattr(stage, "qwen3vl_4b", None)
    transformer = getattr(encoder, "transformer", None)
    if transformer is None or not callable(getattr(transformer, "preprocess_embed", None)):
        raise ValueError(
            "[CcC Krea2] Reference cache requires the Krea2 Qwen3-VL 4B encoder "
            "with preprocess_embed support."
        )
    return transformer


def _prepare_qwen_image(
    image: torch.Tensor,
    semantic_resize: bool,
    grounding_px: int,
    resize_method: str,
) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    prepared = image
    if semantic_resize:
        prepared = resize_grounding_image(
            image=image,
            resize_mode="downscale_only",
            grounding_px=int(grounding_px),
            grounding_preset="custom",
            resize_method=resize_method,
        )
    return prepared[..., :3].clamp(0.0, 1.0)


def _clip_device_context(clip: Any):
    load_model = getattr(clip, "load_model", None)
    if callable(load_model):
        load_model()
    patcher = getattr(clip, "patcher", None)
    device = getattr(patcher, "load_device", torch.device("cpu"))
    try:
        from comfy import model_management

        return device, model_management.cuda_device_context(device)
    except (ImportError, AttributeError):
        return device, nullcontext()


@torch.no_grad()
def build_qwen_visual_cache(
    clip: Any,
    image: torch.Tensor,
    semantic_resize: bool = True,
    grounding_px: int = 768,
    resize_method: str = "lanczos",
) -> QwenVisualCache:
    """Run only the prompt-independent Qwen vision path once."""

    transformer = _qwen_transformer(clip)
    prepared = _prepare_qwen_image(image, semantic_resize, grounding_px, resize_method)
    device, device_context = _clip_device_context(clip)

    with _QWEN_CACHE_LOCK, device_context:
        merged, extra = transformer.preprocess_embed(
            {"type": "image", "data": prepared},
            device=device,
        )

    if not isinstance(extra, dict) or "grid" not in extra or "deepstack" not in extra:
        raise ValueError("[CcC Krea2] Qwen vision preprocessing did not return grid and DeepStack.")
    deepstack = tuple(extra["deepstack"])
    return QwenVisualCache(
        merged=merged,
        grid=extra["grid"],
        deepstack=deepstack,
        model_type=str(getattr(transformer, "model_type", "qwen3vl_4b")),
        input_size=(int(prepared.shape[2]), int(prepared.shape[1])),
        grounding_px=int(grounding_px) if semantic_resize else 0,
    )


@contextmanager
def use_cached_qwen_images(clip: Any, physical_images):
    """Serve cached visual features only for cached image placeholders in one encode call."""

    cached = [item for item in physical_images if isinstance(item, CachedQwenImage)]
    if not cached:
        yield
        return

    transformer = _qwen_transformer(clip)
    original = transformer.preprocess_embed
    had_instance_attribute = "preprocess_embed" in vars(transformer)

    def preprocess_embed(embed, device):
        payload = embed.get("data") if isinstance(embed, dict) else None
        if isinstance(payload, CachedQwenImage):
            cache = payload.cache
            current_model_type = str(getattr(transformer, "model_type", "qwen3vl_4b"))
            if cache.model_type != current_model_type:
                raise ValueError(
                    "[CcC Krea2] Cached Qwen model type does not match the loaded encoder: "
                    f"{cache.model_type} != {current_model_type}."
                )
            return cache.merged.to(device=device, copy=True), {
                "grid": cache.grid.clone(),
                "deepstack": [tensor.to(device=device, copy=True) for tensor in cache.deepstack],
            }
        return original(embed, device=device)

    with _QWEN_CACHE_LOCK:
        try:
            transformer.preprocess_embed = preprocess_embed
            yield
        finally:
            if had_instance_attribute:
                transformer.preprocess_embed = original
            else:
                delattr(transformer, "preprocess_embed")


@torch.no_grad()
def create_reference_cache(
    image: torch.Tensor,
    clip: Any,
    vae: Any,
    target_latent: Dict[str, torch.Tensor],
    reference_fit: str = "native",
    grid_horizontal_position: str = "center",
    grid_vertical_position: str = "center",
    resize_method: str = "lanczos",
    semantic_resize: bool = True,
    semantic_grounding_px: int = 768,
    semantic_resize_method: str = "lanczos",
) -> CcCKrea2ReferenceCache:
    """Create appearance and Qwen visual caches for one reference."""

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("[CcC Krea2] Reference cache creation requires exactly one IMAGE.")
    samples = target_latent.get("samples") if isinstance(target_latent, dict) else None
    if not torch.is_tensor(samples) or samples.ndim not in (4, 5):
        raise ValueError("[CcC Krea2] Reference cache requires a valid target LATENT.")
    if vae is None or not callable(getattr(vae, "encode", None)):
        raise ValueError("[CcC Krea2] Reference cache requires a Krea2-compatible VAE.")

    target_h = int(samples.shape[-2]) * 8
    target_w = int(samples.shape[-1]) * 8
    source_h, source_w = int(image.shape[1]), int(image.shape[2])

    geometry = resolve_krea2edit_geometry(
        src_h=source_h,
        src_w=source_w,
        tgt_h=target_h,
        tgt_w=target_w,
        fit_mode=reference_fit,
        grid_horizontal_position=grid_horizontal_position,
        grid_vertical_position=grid_vertical_position,
        resize_method=resize_method,
    )
    fit_image, _ = process_image_and_mask_geometry(image=image, mask=None, geom=geometry)
    encoded = vae.encode(fit_image)
    if isinstance(encoded, dict):
        appearance = encoded["samples"]
    elif hasattr(encoded, "sample"):
        appearance = encoded.sample()
    else:
        appearance = encoded
    if not torch.is_tensor(appearance):
        raise ValueError("[CcC Krea2] VAE did not return a tensor reference latent.")

    qwen = build_qwen_visual_cache(
        clip=clip,
        image=image,
        semantic_resize=semantic_resize,
        grounding_px=semantic_grounding_px,
        resize_method=semantic_resize_method,
    )

    metadata = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "source_width": str(source_w),
        "source_height": str(source_h),
        "target_width": str(target_w),
        "target_height": str(target_h),
        "reference_fit": reference_fit,
        "resize_method": resize_method,
        "semantic_resize": "true" if semantic_resize else "false",
        "semantic_grounding_px": str(semantic_grounding_px),
        "semantic_resize_method": semantic_resize_method,
        "qwen_model_type": qwen.model_type,
        "qwen_input_width": str(qwen.input_size[0]),
        "qwen_input_height": str(qwen.input_size[1]),
    }
    return CcCKrea2ReferenceCache(
        appearance_latent=appearance,
        geometry=geometry,
        qwen_visual=qwen,
        metadata=metadata,
    )


def validate_cache_target(cache: CcCKrea2ReferenceCache, target_w: int, target_h: int) -> None:
    """Appearance caches are target-specific; Qwen visual caches are not."""

    if cache.target_size != (int(target_w), int(target_h)):
        raise ValueError(
            "[CcC Krea2] Cached appearance geometry does not match the current target. "
            f"Cache: {cache.target_size[0]}x{cache.target_size[1]}, "
            f"target: {target_w}x{target_h}. Create another cache for this target geometry."
        )


def _geometry_metadata(geometry: ResolvedGeometry) -> Dict[str, Any]:
    return {
        "mode_requested": geometry.mode_requested,
        "mode_resolved": geometry.mode_resolved,
        "source_size": list(geometry.source_size),
        "crop_rectangle": list(geometry.crop_rectangle),
        "vae_input_pixel_size": list(geometry.vae_input_pixel_size),
        "vae_latent_grid_size": list(geometry.vae_latent_grid_size),
        "target_grid_size": list(geometry.target_grid_size),
        "centered_fractional_offset": list(geometry.centered_fractional_offset),
        "interpolation_method": geometry.interpolation_method,
        "whether_interpolation_occurred": geometry.whether_interpolation_occurred,
    }


def _geometry_from_metadata(data: Mapping[str, Any]) -> ResolvedGeometry:
    return ResolvedGeometry(
        mode_requested=str(data["mode_requested"]),
        mode_resolved=str(data["mode_resolved"]),
        source_size=tuple(int(v) for v in data["source_size"]),
        crop_rectangle=tuple(int(v) for v in data["crop_rectangle"]),
        vae_input_pixel_size=tuple(int(v) for v in data["vae_input_pixel_size"]),
        vae_latent_grid_size=tuple(int(v) for v in data["vae_latent_grid_size"]),
        target_grid_size=tuple(int(v) for v in data["target_grid_size"]),
        centered_fractional_offset=tuple(float(v) for v in data["centered_fractional_offset"]),
        interpolation_method=str(data["interpolation_method"]),
        whether_interpolation_occurred=bool(data["whether_interpolation_occurred"]),
    )


def save_reference_cache(cache: CcCKrea2ReferenceCache, path: Path) -> None:
    """Serialize a cache losslessly as safetensors."""

    from safetensors.torch import save_file

    tensors = {
        "appearance_latent": cache.appearance_latent.detach().cpu().contiguous(),
        "qwen_merged": cache.qwen_visual.merged,
        "qwen_grid": cache.qwen_visual.grid,
    }
    for index, tensor in enumerate(cache.qwen_visual.deepstack):
        tensors[f"qwen_deepstack_{index}"] = tensor

    payload = {
        "cache": dict(cache.metadata),
        "geometry": _geometry_metadata(cache.geometry),
        "qwen": {
            "model_type": cache.qwen_visual.model_type,
            "input_size": list(cache.qwen_visual.input_size),
            "grounding_px": cache.qwen_visual.grounding_px,
            "deepstack_count": len(cache.qwen_visual.deepstack),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"ccc_krea2_cache": json.dumps(payload, sort_keys=True)})


def load_reference_cache(path: Path) -> CcCKrea2ReferenceCache:
    """Load and validate a portable CcC reference cache."""

    from safetensors import safe_open
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    raw = metadata.get("ccc_krea2_cache")
    if not raw:
        raise ValueError("[CcC Krea2] File is not a CcC reference cache.")
    payload = json.loads(raw)
    cache_meta = payload.get("cache", {})
    if cache_meta.get("format") != CACHE_FORMAT:
        raise ValueError("[CcC Krea2] Unsupported reference cache format.")

    qwen_meta = payload["qwen"]
    count = int(qwen_meta["deepstack_count"])
    deepstack = tuple(tensors[f"qwen_deepstack_{index}"] for index in range(count))
    qwen = QwenVisualCache(
        merged=tensors["qwen_merged"],
        grid=tensors["qwen_grid"],
        deepstack=deepstack,
        model_type=str(qwen_meta["model_type"]),
        input_size=tuple(int(v) for v in qwen_meta["input_size"]),
        grounding_px=int(qwen_meta["grounding_px"]),
    )
    return CcCKrea2ReferenceCache(
        appearance_latent=tensors["appearance_latent"],
        geometry=_geometry_from_metadata(payload["geometry"]),
        qwen_visual=qwen,
        metadata=cache_meta,
    )


def default_cache_directory() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / CACHE_FOLDER
    except (ImportError, AttributeError):
        return Path(CACHE_FOLDER)


def list_reference_cache_files() -> Tuple[str, ...]:
    root = default_cache_directory()
    if not root.exists():
        return ("",)
    files = sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*.safetensors"))
    return tuple(files) if files else ("",)


def resolve_cache_path(filename: str) -> Path:
    root = default_cache_directory().resolve()
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("[CcC Krea2] Cache filename must be a safe relative path.")
    path = (root / relative).resolve()
    if root not in path.parents and path != root:
        raise ValueError("[CcC Krea2] Cache path escapes the cache directory.")
    if path.suffix.lower() != ".safetensors":
        path = path.with_suffix(".safetensors")
    return path


def format_reference_cache_info(cache: CcCKrea2ReferenceCache) -> str:
    geometry = cache.geometry
    return "\n".join(
        [
            "CcC Krea2 Reference Cache",
            f"Source: {cache.metadata['source_width']} x {cache.metadata['source_height']}",
            f"Target: {cache.metadata['target_width']} x {cache.metadata['target_height']}",
            f"Appearance latent: {tuple(cache.appearance_latent.shape)} {cache.appearance_latent.dtype}",
            f"Reference fit: {cache.metadata['reference_fit']}",
            f"VAE input: {geometry.vae_input_pixel_size[0]} x {geometry.vae_input_pixel_size[1]}",
            f"Qwen input: {cache.qwen_visual.input_size[0]} x {cache.qwen_visual.input_size[1]}",
            f"Qwen visual tokens: {cache.qwen_visual.merged.shape[0]}",
            f"Qwen DeepStack: {len(cache.qwen_visual.deepstack)} tensors",
            f"Qwen model: {cache.qwen_visual.model_type}",
        ]
    )
