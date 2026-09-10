"""Three-axis RoPE positioning for Krea2 visual references."""

from typing import List, Optional, Tuple

import torch


def resolve_rope_axes(position: str) -> Tuple[str, str, str]:
    """Resolve three-axis RoPE controls while preserving legacy single-position values."""
    legacy = {
        "none": ("inside", "center", "center"),
        "up": ("outside", "center", "up"),
        "down": ("outside", "center", "down"),
        "left": ("outside", "left", "center"),
        "right": ("outside", "right", "center"),
    }
    if position in legacy:
        return legacy[position]

    parts = position.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid reference RoPE position '{position}'. Expected grid:horizontal:vertical."
        )
    grid, horizontal, vertical = parts
    if grid not in ("inside", "outside"):
        raise ValueError(f"Invalid RoPE grid '{grid}'. Expected inside or outside.")
    if horizontal not in ("center", "left", "right"):
        raise ValueError(f"Invalid RoPE horizontal '{horizontal}'. Expected center, left, or right.")
    if vertical not in ("center", "up", "down"):
        raise ValueError(f"Invalid RoPE vertical '{vertical}'. Expected center, up, or down.")
    return grid, horizontal, vertical


def build_incontext_3d_rope_pos_ids(
    batch_size: int,
    txt_len: int,
    ref_token_grids: List[Tuple[int, int]],
    target_grid: Tuple[int, int],
    device: torch.device,
    ref_rope_positions: Optional[List[str]] = None,
) -> torch.Tensor:
    """Build Krea2 3D RoPE IDs from v1.2-fitted refs plus optional coordinate displacement.

    Reference sizing is not controlled here. Every visual reference is fitted to the target
    latent first by the canonical Krea2 Edit v1.2 pixel-space geometry. This function only
    changes the coordinate placement used by RoPE.

    `inside:center:center` intentionally matches the proven RedNode/upstream behavior:
    stride-1 reference coordinates with integer centered offsets.
    """
    tgt_gh, tgt_gw = target_grid
    list_pos = []

    if txt_len > 0:
        list_pos.append(torch.zeros((txt_len, 3), device=device, dtype=torch.float32))

    for i, (r_gh, r_gw) in enumerate(ref_token_grids):
        frame_idx = i + 1
        position = (
            ref_rope_positions[i]
            if ref_rope_positions is not None and i < len(ref_rope_positions)
            else "none"
        )
        grid, horizontal, vertical = resolve_rope_axes(position)

        centered_x = float(max(0, (tgt_gw - r_gw) // 2))
        centered_y = float(max(0, (tgt_gh - r_gh) // 2))

        if horizontal == "left":
            x_off = 0.0 if grid == "inside" else -float(r_gw)
        elif horizontal == "right":
            x_off = float(max(0, tgt_gw - r_gw)) if grid == "inside" else float(tgt_gw)
        else:
            x_off = centered_x

        if vertical == "up":
            y_off = 0.0 if grid == "inside" else -float(r_gh)
        elif vertical == "down":
            y_off = float(max(0, tgt_gh - r_gh)) if grid == "inside" else float(tgt_gh)
        else:
            y_off = centered_y

        grid_y = torch.arange(r_gh, device=device, dtype=torch.float32) + y_off
        grid_x = torch.arange(r_gw, device=device, dtype=torch.float32) + x_off
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        mesh_t = torch.full_like(mesh_y, fill_value=float(frame_idx))
        list_pos.append(torch.stack([mesh_t.flatten(), mesh_y.flatten(), mesh_x.flatten()], dim=-1))

    tgt_y = torch.arange(tgt_gh, device=device, dtype=torch.float32)
    tgt_x = torch.arange(tgt_gw, device=device, dtype=torch.float32)
    mesh_ty, mesh_tx = torch.meshgrid(tgt_y, tgt_x, indexing="ij")
    mesh_tt = torch.zeros_like(mesh_ty)
    list_pos.append(torch.stack([mesh_tt.flatten(), mesh_ty.flatten(), mesh_tx.flatten()], dim=-1))

    return torch.cat(list_pos, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)


def install_krea2_rope_positioning() -> None:
    """Install only the optional RoPE placement extension.

    Krea2 Edit reference sizing remains owned by the canonical v1.2 fit geometry.
    """
    from .. import patch as patch_module

    patch_module._build_incontext_3d_rope_pos_ids = build_incontext_3d_rope_pos_ids
