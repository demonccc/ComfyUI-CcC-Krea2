"""Reusable chained masks for Paint Prepare and regional Reference attention.

SAM 3 is optional: external masks remain usable without its dependencies.
"""
from dataclasses import dataclass
from typing import Tuple
import torch
from ..constants import NODE_CATEGORY


@dataclass(frozen=True)
class AttentionRegion:
    mask: torch.Tensor
    boost: float


@dataclass(frozen=True)
class MaskChain:
    geometry: Tuple[int, int, int]  # batch, height, width
    combined: torch.Tensor
    regions: Tuple[AttentionRegion, ...]


def _sam31_mask(image, prompt, confidence, model, clip):
    """Use the built-in ComfyUI SAM3 detector and its CLIP text conditioning."""
    if model is None or clip is None:
        raise ValueError("SAM 3.1 requires MODEL and CLIP from CheckpointLoaderSimple.")
    try:
        import nodes
        from comfy_extras.nodes_sam3 import SAM3_Detect
    except ImportError as exc:
        raise RuntimeError("This ComfyUI version does not include native SAM3_Detect.") from exc
    conditioning = nodes.CLIPTextEncode().encode(clip, prompt)[0]
    result = SAM3_Detect.execute(
        model=model, image=image, conditioning=conditioning,
        threshold=float(confidence), refine_iterations=2, individual_masks=False,
    )
    # ComfyUI NodeOutput wraps the returned native socket values.
    if hasattr(result, "result"):
        return result.result[0]
    if isinstance(result, (tuple, list)):
        return result[0]
    raise TypeError("Unexpected SAM3_Detect output shape; update ComfyUI core.")


def _offload_sam31(model):
    """Request ComfyUI's model manager to unload only SAM3; never unload Krea2."""
    import comfy.model_management as mm
    # ModelPatcher.offload_model() is not portable across ComfyUI versions.
    # Its manager removes only the requested model from the GPU-resident list.
    loaded = getattr(mm, "current_loaded_models", None)
    if loaded is None:
        raise RuntimeError("ComfyUI model manager does not expose targeted unload.")
    for wrapper in list(loaded):
        if wrapper.model is model:
            if wrapper.model_unload():
                loaded.remove(wrapper)
            mm.soft_empty_cache()
            return


def _normalize_mask(mask, batch, height, width, device):
    if not torch.is_tensor(mask):
        raise ValueError("Mask must be a ComfyUI MASK tensor.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or tuple(mask.shape[-2:]) != (height, width):
        raise ValueError("Mask and original image must have identical pixel dimensions.")
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1)
    elif mask.shape[0] != batch:
        raise ValueError("Mask batch size does not match image.")
    return mask.to(device=device, dtype=torch.float32).clamp(0, 1)


class CcCKrea2Mask:
    CATEGORY = NODE_CATEGORY
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "MASK", "KREA2_MASK_CHAIN", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "mask", "attention", "preview", "info")
    DESCRIPTION = "Chain union masks for Paint Prepare and region-specific reference boosts for Edit."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source": (["external_mask", "sam3"], {"default": "external_mask"}),
                "prompt": ("STRING", {"default": "face"}),
                "boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "unload_sam_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "external_mask": ("MASK",),
                "sam_model": ("MODEL",),
                "sam_clip": ("CLIP",),
                "previous_masks": ("KREA2_MASK_CHAIN",),
            },
        }

    def process(self, image, source="external_mask", prompt="face", boost=1.0,
                confidence=0.5, external_mask=None, previous_masks=None,
                sam_model=None, sam_clip=None, unload_sam_model=False):
        if image.ndim != 4:
            raise ValueError("Image must have ComfyUI BHWC dimensions.")
        batch, height, width = image.shape[:3]
        geometry = (batch, height, width)
        if source == "external_mask":
            if external_mask is None:
                raise ValueError("External mask mode requires external_mask.")
            mask = external_mask
        elif source == "sam3":
            if not str(prompt).strip():
                raise ValueError("SAM 3 requires a text prompt.")
            mask = _sam31_mask(image, prompt, confidence, sam_model, sam_clip)
            if unload_sam_model:
                _offload_sam31(sam_model)
        else:
            raise ValueError(f"Unknown segmentation source: {source}")
        mask = _normalize_mask(mask, batch, height, width, image.device)
        if previous_masks is not None:
            if not isinstance(previous_masks, MaskChain) or previous_masks.geometry != geometry:
                raise ValueError("Chained masks must belong to the same image geometry.")
            combined = torch.maximum(previous_masks.combined.to(mask.device), mask)
            regions = previous_masks.regions
        else:
            combined = mask
            regions = ()
        chain = MaskChain(geometry, combined, regions + (AttentionRegion(mask, float(boost)),))
        overlay = image[..., :3].clone()
        overlay = overlay * (1 - combined.unsqueeze(-1) * 0.4)
        overlay[..., 0] += combined * 0.4
        overlay = overlay.clamp(0, 1)
        info = f"Mask source: {source}; regions: {len(chain.regions)}; last boost: {boost:.2f}"
        return image, combined, chain, overlay, info
