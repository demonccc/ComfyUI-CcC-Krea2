"""Public Krea2 CcC Edit reference-cache nodes."""

from __future__ import annotations

from typing import Optional, Tuple

from ..attention_regions import REFERENCE_ATTENTION_SCOPES
from ..constants import NODE_CATEGORY
from ..reference_cache import (
    CcCKrea2ReferenceCache,
    create_reference_cache,
    format_reference_cache_info,
    list_reference_cache_files,
    load_reference_cache,
    resolve_cache_path,
    save_reference_cache,
)
from .edit_reference_types import VisualReferenceChain, VisualReferenceEntry


REFERENCE_FIT = ("crop", "resize", "contain", "native")
PLACEMENT_GRIDS = ("inside", "outside")
GRID_HORIZONTAL = ("center", "left", "right")
GRID_VERTICAL = ("center", "up", "down")
RESIZE_METHODS = ("lanczos", "bicubic", "bilinear", "area")


class CcCKrea2ReferenceCacheCreate:
    """Precompute one visual reference for repeated Krea2 edits."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_REFERENCE_CACHE", "STRING")
    RETURN_NAMES = ("cache", "cache_info")
    FUNCTION = "create"
    DESCRIPTION = (
        "Caches prompt-independent work for one visual reference: the raw VAE appearance latent "
        "and Qwen3-VL visual features. The appearance part is target-geometry-specific; the Qwen "
        "visual part remains prompt-independent."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "target_latent": ("LATENT",),
                "reference_fit": (REFERENCE_FIT, {"default": "native"}),
                "grid_horizontal_position": (GRID_HORIZONTAL, {"default": "center"}),
                "grid_vertical_position": (GRID_VERTICAL, {"default": "center"}),
                "resize_method": (RESIZE_METHODS, {"default": "lanczos"}),
                "semantic_resize": ("BOOLEAN", {"default": True}),
                "semantic_grounding_px": (
                    "INT",
                    {"default": 768, "min": 32, "max": 4096, "step": 32},
                ),
                "semantic_resize_method": (RESIZE_METHODS, {"default": "lanczos"}),
            }
        }

    def create(
        self,
        image,
        clip,
        vae,
        target_latent,
        reference_fit="native",
        grid_horizontal_position="center",
        grid_vertical_position="center",
        resize_method="lanczos",
        semantic_resize=True,
        semantic_grounding_px=768,
        semantic_resize_method="lanczos",
    ):
        cache = create_reference_cache(
            image=image,
            clip=clip,
            vae=vae,
            target_latent=target_latent,
            reference_fit=reference_fit,
            grid_horizontal_position=grid_horizontal_position,
            grid_vertical_position=grid_vertical_position,
            resize_method=resize_method,
            semantic_resize=semantic_resize,
            semantic_grounding_px=semantic_grounding_px,
            semantic_resize_method=semantic_resize_method,
        )
        return cache, format_reference_cache_info(cache)


class CcCKrea2ReferenceCacheSave:
    """Save one portable cache under ComfyUI/models/krea2_ccc_cache."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves a Krea2 CcC Edit reference cache as a portable safetensors file."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache": ("KREA2_REFERENCE_CACHE",),
                "filename": (
                    "STRING",
                    {
                        "default": "reference_cache",
                        "tooltip": "Relative path under models/krea2_ccc_cache.",
                    },
                ),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {"default": False}),
            },
        }

    def save(
        self,
        cache: CcCKrea2ReferenceCache,
        filename: str,
        overwrite: bool = False,
    ) -> Tuple[str]:
        path = resolve_cache_path(filename.strip())
        if path.exists() and not overwrite:
            base = path.with_suffix("")
            suffix = path.suffix
            index = 1
            while True:
                candidate = base.parent / f"{base.name}_{index:03d}{suffix}"
                if not candidate.exists():
                    path = candidate
                    break
                index += 1
        save_reference_cache(cache, path)

        from ..reference_cache import default_cache_directory

        relative = str(path.relative_to(default_cache_directory().resolve())).replace("\\", "/")
        return (relative,)


class CcCKrea2ReferenceCacheLoad:
    """Load a portable Krea2 CcC Edit reference cache."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_REFERENCE_CACHE", "STRING")
    RETURN_NAMES = ("cache", "cache_info")
    FUNCTION = "load"
    DESCRIPTION = "Loads a Krea2 CcC Edit reference cache from models/krea2_ccc_cache."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_file": (list_reference_cache_files(),),
            }
        }

    @classmethod
    def IS_CHANGED(cls, cache_file: str):
        path = resolve_cache_path(cache_file)
        if not path.exists():
            return cache_file
        stat = path.stat()
        return cache_file, stat.st_mtime_ns, stat.st_size

    def load(self, cache_file: str):
        cache = load_reference_cache(resolve_cache_path(cache_file))
        return cache, format_reference_cache_info(cache)


class CcCKrea2CachedVisualReference:
    """Add a cached visual reference to the normal ordered reference chain."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_VISUAL_REFERENCE_CHAIN",)
    RETURN_NAMES = ("visual_references",)
    FUNCTION = "process"
    DESCRIPTION = (
        "Adds a precomputed visual reference. Boost, RoPE placement, semantic enablement and prompt "
        "annotation remain runtime controls; VAE and Qwen Vision preprocessing are reused from cache."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache": ("KREA2_REFERENCE_CACHE",),
                "boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "placement_grid": (PLACEMENT_GRIDS, {"default": "inside"}),
                "grid_horizontal_position": (GRID_HORIZONTAL, {"default": "center"}),
                "grid_vertical_position": (GRID_VERTICAL, {"default": "center"}),
                "semantic": ("BOOLEAN", {"default": True}),
                "prompt_annotation": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional text appended as Image N annotation.",
                    },
                ),
                "attention_scope": (REFERENCE_ATTENTION_SCOPES, {"default": "global"}),
                "region_tag": ("STRING", {"default": ""}),
            },
            "optional": {
                "previous_references": ("KREA2_VISUAL_REFERENCE_CHAIN",),
            },
        }

    def process(
        self,
        cache: CcCKrea2ReferenceCache,
        boost: float = 1.0,
        placement_grid: str = "inside",
        grid_horizontal_position: str = "center",
        grid_vertical_position: str = "center",
        semantic: bool = True,
        prompt_annotation: str = "",
        attention_scope: str = "global",
        region_tag: str = "",
        previous_references: Optional[VisualReferenceChain] = None,
    ) -> Tuple[VisualReferenceChain]:
        chain = previous_references if previous_references is not None else VisualReferenceChain()
        attention_scope = str(attention_scope)
        if attention_scope not in REFERENCE_ATTENTION_SCOPES:
            raise ValueError(f"[Krea2 CcC Cached Visual Reference] Invalid attention_scope '{attention_scope}'.")
        resolved_region_tag = str(region_tag or "").strip()
        if attention_scope != "global" and not resolved_region_tag:
            raise ValueError(
                "[Krea2 CcC Cached Visual Reference] regional attention requires a non-empty region_tag."
            )
        reference_fit = cache.metadata.get("reference_fit", "native")
        entry = VisualReferenceEntry(
            image=None,
            cache=cache,
            boost=float(boost),
            reference_fit=reference_fit,
            placement_grid="inside" if reference_fit == "crop" else placement_grid,
            grid_horizontal_position=grid_horizontal_position,
            grid_vertical_position=grid_vertical_position,
            resize_method=cache.metadata.get("resize_method", "lanczos"),
            semantic=bool(semantic),
            semantic_resize=False,
            semantic_grounding_px=int(cache.metadata.get("semantic_grounding_px", "768")),
            semantic_resize_method=cache.metadata.get("semantic_resize_method", "lanczos"),
            prompt_annotation=prompt_annotation.strip() if semantic else "",
            attention_scope=attention_scope,
            region_tag=resolved_region_tag if attention_scope != "global" else "",
        )
        return (chain.append(entry),)
