"""Preview accumulated region weights on top of the source image."""
import torch
from ..constants import NODE_CATEGORY
from .mask_node import MaskChain


class CcCKrea2AttentionPreview:
    CATEGORY = NODE_CATEGORY
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "attention_info")
    DESCRIPTION = "Visualize cumulative mask regions and absolute reference boosts."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "attention": ("KREA2_MASK_CHAIN",),
            "reference_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            "opacity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
            "palette": (["regions", "boost_heatmap"], {"default": "boost_heatmap"}),
        }}

    def render(self, image, attention, reference_boost=1.0, opacity=0.55, palette="boost_heatmap"):
        if not isinstance(attention, MaskChain):
            raise TypeError("Attention input requires a Krea2 CcC Mask chain.")
        if tuple(image.shape[:3]) != attention.geometry:
            raise ValueError("Attention preview image geometry must match source masks.")
        weights = torch.full(attention.geometry, float(reference_boost), device=image.device)
        # Unmasked pixels retain the reference boost. Later regions win.
        region_ids = torch.full(attention.geometry, -1, device=image.device, dtype=torch.long)
        for index, region in enumerate(attention.regions):
            selected = region.mask.to(device=image.device) > 0.5
            weights = torch.where(selected, torch.full_like(weights, region.boost), weights)
            region_ids = torch.where(selected, torch.full_like(region_ids, index), region_ids)
        active = region_ids >= 0
        if palette == "regions":
            colors = torch.tensor([
                [0.95, 0.30, 0.25], [0.20, 0.70, 0.95], [0.35, 0.90, 0.40],
                [0.90, 0.70, 0.18], [0.75, 0.35, 0.85], [0.95, 0.50, 0.70]
            ], device=image.device, dtype=image.dtype)
            tint = colors[region_ids.clamp_min(0) % len(colors)]
        else:
            # Cold=attenuated/blocked, warm=boosted. Neutral green.
            denom = max(1., float(reference_boost), float(weights.max()))
            intensity = (weights / denom).clamp(0, 1).to(image.dtype)
            tint = torch.stack((intensity, 1 - (intensity - 0.5).abs() * 1.5, 1 - intensity), dim=-1).clamp(0, 1)
        blend = (active.to(image.dtype) * float(opacity)).unsqueeze(-1)
        preview = (image[..., :3] * (1 - blend) + tint * blend).clamp(0, 1)
        info = ["=== Krea2 CcC Attention Preview ===", f"Global boost: {reference_boost:.2f}",
                f"Regions: {len(attention.regions)}", "Last overlapping region wins."]
        for index, region in enumerate(attention.regions):
            count = int((region_ids == index).sum().item())
            info.append(f"Region {index+1}: boost={region.boost:.2f}; visible selected pixels={count}")
        return preview, "\n".join(info)
