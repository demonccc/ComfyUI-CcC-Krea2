"""RoPE geometry views including regional attention weights attached to references."""

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..krea2edit_geometry import resolve_krea2edit_geometry, process_image_and_mask_geometry
from .rope_position import plan_rope_positions


PALETTE = ((36, 209, 188), (250, 169, 53), (147, 133, 255),
           (236, 104, 149), (86, 177, 241))


def _rgb(image):
    if image is None:
        return None
    if image.ndim == 4:
        image = image[0]
    arr = image.detach().to("cpu").float().clamp(0, 1).numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray((arr[:, :, :3] * 255).round().astype(np.uint8), "RGB")



def _boost_visibility_preview(image, weights):
    """Darken low-boost pixels: 0 is 90% black, 10 retains 95% source.

    The weights are the final, already-composed per-pixel regional boost map.
    Mask order must be resolved upstream before rendering this image.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    values = np.asarray(weights, dtype=np.float32)
    if values.shape != rgb.shape[:2]:
        raise ValueError("Effective boost map must match the prepared reference geometry.")
    alpha = 0.90 - 0.85 * (np.clip(values, 0.0, 10.0) / 10.0)
    visible = np.clip(rgb * (1.0 - alpha[..., None]), 0.0, 255.0)
    return Image.fromarray(np.rint(visible).astype(np.uint8), "RGB")


def render_rope_preview(visual_entries, target_latent, max_side=1024):
    """Return spatial views [layout, boxes, visuals, boosts, per-ref boosts...] as BHWC floats.

    The visual references are pixel-processed with the production geometry
    implementation. Layout is calculated with the same planner as runtime RoPE.
    Collapsed references are always displayed as a point. Per-reference
    boost views retain the shared RoPE coordinates while avoiding occlusion.
    """
    samples = target_latent["samples"]
    target_h, target_w = int(samples.shape[-2]) // 2, int(samples.shape[-1]) // 2
    target_image = _rgb(target_latent.get("ccc_krea2_target_preview_image"))
    ref_grids, ref_images, ref_boost_images = [], [], []
    for entry in visual_entries:
        if entry.image is None:
            raise ValueError("RoPE preview requires a Visual Reference image.")
        geom = resolve_krea2edit_geometry(
            src_h=int(entry.image.shape[1]), src_w=int(entry.image.shape[2]),
            tgt_h=int(samples.shape[-2]) * 8, tgt_w=int(samples.shape[-1]) * 8,
            fit_mode=entry.reference_fit,
            grid_horizontal_position=entry.grid_horizontal_position,
            grid_vertical_position=entry.grid_vertical_position,
        )
        fitted, fitted_weights = process_image_and_mask_geometry(
            entry.image, entry.attention_weight_map, geom
        )
        ref_images.append(_rgb(fitted))
        # Use the same reference geometry as VAE preparation, not a second
        # transformation derived from the target's displayed position.
        global_boost = float(entry.boost)
        weights = fitted_weights[0].detach().float().cpu().numpy() if fitted_weights is not None else None
        base = _rgb(fitted).convert("RGBA")
        if weights is None:
            weights = np.full((base.height, base.width), global_boost, dtype=np.float32)
        # Use the effective weight map composed with Last Mask Wins upstream.
        boost_image = _boost_visibility_preview(base.convert("RGB"), weights)
        ref_boost_images.append(boost_image)
        ref_grids.append((geom.vae_latent_grid_size[1] // 2,
                          geom.vae_latent_grid_size[0] // 2))
    layout = plan_rope_positions(ref_grids, (target_h, target_w),
                                 [entry.rope_position for entry in visual_entries])

    bounds = [(0., 0., float(target_w), float(target_h))]
    bounds.extend((ref.x, ref.y, ref.x + ref.width, ref.y + ref.height) for ref in layout)
    min_x = min(b[0] for b in bounds)
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds)
    max_y = max(b[3] for b in bounds)
    padding = 48
    scale = min((max_side - 2 * padding) / max(1, max_x - min_x),
                (max_side - 2 * padding) / max(1, max_y - min_y), 8.0)
    width = max(240, int((max_x - min_x) * scale + 2 * padding))
    height = max(160, int((max_y - min_y) * scale + 2 * padding))
    size = (width, height)

    def point(x, y):
        return round(padding + (x - min_x) * scale), round(padding + (y - min_y) * scale)

    target_box = (*point(0, 0), *point(target_w, target_h))
    views = []
    # Four canonical views followed by one spatial boost view per reference.
    # Independent spatial frames prevent overlapping references from hiding
    # each other, without losing the RoPE position on the shared canvas.
    for view_index in range(4 + len(ref_images)):
        canvas = Image.new("RGB", size, (30, 34, 43))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(target_box, fill=(58, 66, 78), outline=(240, 240, 240), width=3)
        if view_index != 0 and target_image is not None:
            tw, th = max(1, target_box[2] - target_box[0]), max(1, target_box[3] - target_box[1])
            canvas.paste(target_image.resize((tw, th), Image.Resampling.BICUBIC),
                         (target_box[0], target_box[1]))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(target_box, outline=(240, 240, 240), width=3)
        draw.text((target_box[0] + 5, target_box[1] + 5),
                  f"TARGET {target_w}x{target_h} tokens", fill="white")

        isolated_reference = view_index - 4 if view_index >= 4 else None
        for index, (entry, ref, image) in enumerate(zip(visual_entries, layout, ref_images)):
            if isolated_reference is not None and index != isolated_reference:
                continue
            color = PALETTE[index % len(PALETTE)]
            label = f"REF {index + 1} / {entry.reference_fit} / {entry.rope_mode} / {entry.reference_stack}"
            if view_index >= 3:
                label += f" / BOOST {entry.boost:g}"
                if entry.attention_weight_map is not None:
                    label += " / REGIONAL"
            if ref.collapsed:
                cx, cy = point(ref.x + ref.width / 2, ref.y + ref.height / 2)
                draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=color)
                draw.line((cx - 14, cy, cx + 14, cy), fill=color, width=2)
                draw.line((cx, cy - 14, cx, cy + 14), fill=color, width=2)
                draw.text((cx + 12, cy - 16), label, fill=color)
                continue
            a, b = point(ref.x, ref.y)
            c, d = point(ref.x + ref.width, ref.y + ref.height)
            if view_index == 1:
                overlay = Image.new("RGBA", size, (0, 0, 0, 0))
                ImageDraw.Draw(overlay).rectangle((a, b, c, d), fill=(*color, 65))
                canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
            elif view_index >= 2 and image is not None:
                w, h = max(1, c - a), max(1, d - b)
                selected = ref_boost_images[index] if view_index >= 3 else image
                # Don't alpha-blend the selected image into the canvas: doing
                # so applies a second attenuation, destroying fine detail.
                canvas.paste(selected.resize((w, h), Image.Resampling.LANCZOS), (a, b))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((a, b, c, d), outline=color, width=3)
            draw.text((a + 5, b + 18), label, fill=color)
        views.append(torch.from_numpy(np.asarray(canvas).copy()).float() / 255.)
    return torch.stack(views, dim=0)
