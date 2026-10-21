"""Annotated RoPE placement image using the runtime's coordinate planner."""

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..krea2edit_geometry import resolve_krea2edit_geometry
from .rope_position import plan_rope_positions


def render_rope_preview(visual_entries, target_latent, max_side=1024):
    samples = target_latent["samples"]
    target_h, target_w = int(samples.shape[-2]) // 2, int(samples.shape[-1]) // 2
    ref_grids = []
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
        ref_grids.append((geom.vae_latent_grid_size[1] // 2, geom.vae_latent_grid_size[0] // 2))
    layout = plan_rope_positions(ref_grids, (target_h, target_w),
                                 [entry.rope_position for entry in visual_entries])

    extents = [(0., 0., float(target_w), float(target_h))]
    extents.extend((r.x, r.y, r.x+r.width, r.y+r.height) for r in layout)
    min_x, min_y = min(v[0] for v in extents), min(v[1] for v in extents)
    max_x, max_y = max(v[2] for v in extents), max(v[3] for v in extents)
    pad = 48
    scale = min((max_side - 2*pad) / max(1, max_x-min_x),
                (max_side - 2*pad) / max(1, max_y-min_y), 8.0)
    width = max(240, int((max_x-min_x)*scale+2*pad))
    height = max(160, int((max_y-min_y)*scale+2*pad))
    canvas = Image.new("RGB", (width, height), (30, 34, 43))
    draw = ImageDraw.Draw(canvas)
    def xy(x, y):
        return (round(pad + (x-min_x)*scale), round(pad + (y-min_y)*scale))

    x0,y0=xy(0,0)
    x1,y1=xy(target_w,target_h)
    draw.rectangle([x0,y0,x1,y1],outline=(240,240,240),width=3,fill=(58,66,78))
    draw.text((x0+6,y0+6),f"TARGET {target_w}x{target_h} tokens",fill="white")
    palette=((36,209,188),(250,169,53),(147,133,255),(236,104,149),(86,177,241))
    for i,(entry,ref) in enumerate(zip(visual_entries,layout)):
        color=palette[i%len(palette)]
        label=f"REF {i+1} / {entry.rope_mode} / {entry.reference_stack}"
        if ref.collapsed:
            cx,cy=xy(ref.x+ref.width/2,ref.y+ref.height/2)
            draw.ellipse((cx-7,cy-7,cx+7,cy+7),fill=color)
            draw.line((cx-14,cy,cx+14,cy),fill=color,width=2)
            draw.line((cx,cy-14,cx,cy+14),fill=color,width=2)
            draw.text((cx+12,cy-16),label,fill=color)
        else:
            a,b=xy(ref.x,ref.y)
            c,d=xy(ref.x+ref.width,ref.y+ref.height)
            draw.rectangle((a,b,c,d),outline=color,width=3)
            draw.text((a+5,b+18),label,fill=color)
    return torch.from_numpy(np.asarray(canvas).copy()).float().unsqueeze(0)/255.0
