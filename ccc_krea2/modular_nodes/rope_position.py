"""Reference RoPE layout planner and runtime coordinate builder."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass(frozen=True)
class RopePlacement:
    frame: int
    x: float
    y: float
    width: float
    height: float
    collapsed: bool = False


def parse_rope_position(position: str):
    aliases = {
        "none": ("inside", "center", "center", "overlay", "spatial", "latent"),
        "up": ("outside", "center", "up", "overlay", "spatial", "latent"),
        "down": ("outside", "center", "down", "overlay", "spatial", "latent"),
        "left": ("outside", "left", "center", "overlay", "spatial", "latent"),
        "right": ("outside", "right", "center", "overlay", "spatial", "latent"),
    }
    if position in aliases:
        return aliases[position]
    parts = position.split(":")
    if len(parts) == 3:
        parts += ["overlay", "spatial", "latent"]
    if len(parts) == 8:
        parts = parts[:6]
    if len(parts) != 6:
        raise ValueError("RoPE position must be grid:horizontal:vertical:stack:mode:geometry")
    grid, horizontal, vertical, stack, mode, geometry = parts
    allowed = (
        ("inside", "outside"),
        ("center", "left", "right"),
        ("center", "up", "down"),
        ("overlay", "stacked"),
        ("spatial", "collapsed"),
        ("latent", "source"),
    )
    for value, values in zip(parts, allowed):
        if value not in values:
            raise ValueError(f"Invalid RoPE value {value!r}; expected {values}")
    if grid == "outside" and horizontal == "center" and vertical == "center":
        raise ValueError("outside + center + center is not a valid RoPE placement")
    return grid, horizontal, vertical, stack, mode, geometry


def resolve_rope_axes(position: str) -> Tuple[str, str, str]:
    return parse_rope_position(position)[:3]


def plan_rope_positions(
    ref_token_grids: List[Tuple[int, int]],
    target_grid: Tuple[int, int],
    ref_rope_positions: Optional[List[str]] = None,
    source_token_grids: Optional[List[Tuple[int, int]]] = None,
) -> List[RopePlacement]:
    """Shared source of truth for runtime RoPE offsets and annotated preview.

    Units are model RoPE tokens, not RGB pixels. Native references use their
    own source dimensions; geometry mode chooses layout dimensions for others.
    """
    th, tw = target_grid
    cursors = {"left": 0.0, "right": float(tw), "up": 0.0, "down": float(th)}
    result = []
    for i, (rh, rw) in enumerate(ref_token_grids):
        value = ref_rope_positions[i] if ref_rope_positions and i < len(ref_rope_positions) else "none"
        grid, horizontal, vertical, stack, mode, geometry = parse_rope_position(value)
        encoded = value.split(":")
        source_dims = (float(encoded[6]), float(encoded[7])) if len(encoded) == 8 else None
        layout_h, layout_w = (source_token_grids[i] if geometry == "source" and source_token_grids else source_dims if geometry == "source" and source_dims else (rh, rw))
        layout_h, layout_w = float(layout_h), float(layout_w)

        if grid == "inside":
            x = {"left": 0.0, "center": (tw - layout_w) / 2.0, "right": tw - layout_w}[horizontal]
            y = {"up": 0.0, "center": (th - layout_h) / 2.0, "down": th - layout_h}[vertical]
        else:
            x = {"left": -layout_w, "center": (tw - layout_w) / 2.0, "right": float(tw)}[horizontal]
            y = {"up": -layout_h, "center": (th - layout_h) / 2.0, "down": float(th)}[vertical]

        # Stacking advances each selected directional axis, in reference order.
        # Inside stacking is allowed to overflow beyond the target.
        if stack == "stacked":
            for direction, active, size in (
                ("left", horizontal == "left", layout_w),
                ("right", horizontal == "right", layout_w),
                ("up", vertical == "up", layout_h),
                ("down", vertical == "down", layout_h),
            ):
                if not active:
                    continue
                if direction == "left":
                    x = cursors[direction] - size if grid == "outside" else cursors[direction]
                    cursors[direction] = x if grid == "outside" else x + size
                elif direction == "right":
                    x = cursors[direction] if grid == "outside" else cursors[direction] - size
                    cursors[direction] = x + size if grid == "outside" else x
                elif direction == "up":
                    y = cursors[direction] - size if grid == "outside" else cursors[direction]
                    cursors[direction] = y if grid == "outside" else y + size
                else:
                    y = cursors[direction] if grid == "outside" else cursors[direction] - size
                    cursors[direction] = y + size if grid == "outside" else y
        result.append(RopePlacement(i + 1, x, y, float(rw), float(rh), mode == "collapsed"))
    return result


def build_incontext_3d_rope_pos_ids(
    batch_size: int,
    txt_len: int,
    ref_token_grids: List[Tuple[int, int]],
    target_grid: Tuple[int, int],
    device: torch.device,
    ref_rope_positions: Optional[List[str]] = None,
    source_token_grids: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
    layout = plan_rope_positions(ref_token_grids, target_grid, ref_rope_positions, source_token_grids)
    parts = []
    if txt_len:
        parts.append(torch.zeros((txt_len, 3), device=device, dtype=torch.float32))
    for (rh, rw), ref in zip(ref_token_grids, layout):
        if ref.collapsed:
            ys = torch.full((rh,), ref.y + ref.height / 2, device=device)
            xs = torch.full((rw,), ref.x + ref.width / 2, device=device)
        else:
            ys = torch.arange(rh, device=device, dtype=torch.float32) + ref.y
            xs = torch.arange(rw, device=device, dtype=torch.float32) + ref.x
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        tt = torch.full_like(yy, float(ref.frame))
        parts.append(torch.stack((tt.flatten(), yy.flatten(), xx.flatten()), dim=-1))
    th, tw = target_grid
    yy, xx = torch.meshgrid(torch.arange(th, device=device, dtype=torch.float32),
                            torch.arange(tw, device=device, dtype=torch.float32), indexing="ij")
    parts.append(torch.stack((torch.zeros_like(yy).flatten(), yy.flatten(), xx.flatten()), dim=-1))
    return torch.cat(parts, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)


def install_krea2_rope_positioning() -> None:
    from .. import patch as patch_module
    patch_module._build_incontext_3d_rope_pos_ids = build_incontext_3d_rope_pos_ids
