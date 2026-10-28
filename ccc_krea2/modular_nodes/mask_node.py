"""Reusable chained masks for Paint Prepare and regional Reference attention.

SAM 3 is optional: external masks remain usable without its dependencies.
"""
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F
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


_SAM3_PROCESSOR = None


def _get_sam3():
    global _SAM3_PROCESSOR
    if _SAM3_PROCESSOR is None:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 3 is not installed. Install the official facebookresearch/sam3 package "
                "and obtain checkpoint access, or select external_mask."
            ) from exc
        model = build_sam3_image_model()
        _SAM3_PROCESSOR = Sam3Processor(model)
    return _SAM3_PROCESSOR


def _sam3_mask(image, prompt, confidence):
    import numpy as np
    from PIL import Image
    processor = _get_sam3()
    processor.confidence_threshold = float(confidence)
    output_masks = []
    for frame in image:
        pil = Image.fromarray((frame[..., :3].detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8))
        with torch.inference_mode():
            result = processor.set_text_prompt(
                state=processor.set_image(pil), prompt=str(prompt).strip()
            )
        candidates = result["masks"]
        if not torch.is_tensor(candidates):
            candidates = torch.as_tensor(candidates)
        if candidates.numel() == 0:
            selected = torch.zeros(pil.height, pil.width)
        else:
            candidates = candidates.float()
            while candidates.ndim > 3 and candidates.shape[1] == 1:
                candidates = candidates[:, 0]
            if candidates.ndim == 2:
                candidates = candidates.unsqueeze(0)
            selected = candidates.amax(dim=0)
            if selected.shape != (pil.height, pil.width):
                selected = F.interpolate(
                    selected[None, None], size=(pil.height, pil.width), mode="bilinear", align_corners=False
                )[0, 0]
        output_masks.append(selected.cpu().clamp(0, 1))
    return torch.stack(output_masks, dim=0).to(image.device)


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
            },
            "optional": {
                "external_mask": ("MASK",),
                "previous_masks": ("KREA2_MASK_CHAIN",),
            },
        }

    def process(self, image, source="external_mask", prompt="face", boost=1.0,
                confidence=0.5, external_mask=None, previous_masks=None):
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
            mask = _sam3_mask(image, prompt, confidence)
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
