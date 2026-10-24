"""Composable grid layouts for Krea2-sized image sheets."""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from ..constants import NODE_CATEGORY
from ..geometry import resize_tensor
from ..krea_geometry import DEFAULT_KREA_PRESET_SIZE, KREA_PRESET_SIZES, preset_geometry

PLACEMENTS = ("cell", "full_row", "full_column")
FITS = ("contain", "stretch", "crop", "fit")
HORIZONTAL = ("left", "center", "right")
VERTICAL = ("up", "center", "down")


@dataclass(frozen=True)
class LayoutEntry:
    image: torch.Tensor
    placement: str
    row: int
    column: int
    fit: str
    horizontal_position: str
    vertical_position: str
    offset_x: int
    offset_y: int


@dataclass(frozen=True)
class LayoutChain:
    entries: Tuple[LayoutEntry, ...] = ()

    def append(self, entry: LayoutEntry) -> "LayoutChain":
        return LayoutChain(self.entries + (entry,))


def _image(image):
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
        raise ValueError("Layout Cell requires one IMAGE in BHWC format.")
    return image[..., :3].float().clamp(0, 1)


class CcCKrea2LayoutCell:
    CATEGORY = NODE_CATEGORY
    FUNCTION = "add"
    RETURN_TYPES = ("KREA2_LAYOUT_CHAIN",)
    RETURN_NAMES = ("layout",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "placement": (PLACEMENTS, {"default": "cell"}),
            "row": ("INT", {"default": 1, "min": 1, "max": 100}),
            "column": ("INT", {"default": 1, "min": 1, "max": 100}),
            "fit": (FITS, {"default": "contain"}),
            "horizontal_position": (HORIZONTAL, {"default": "center"}),
            "vertical_position": (VERTICAL, {"default": "center"}),
            "offset_x": ("INT", {"default": 0, "min": -16384, "max": 16384}),
            "offset_y": ("INT", {"default": 0, "min": -16384, "max": 16384}),
        }, "optional": {"previous_layout": ("KREA2_LAYOUT_CHAIN",)}}

    def add(self, image, placement="cell", row=1, column=1, fit="contain",
            horizontal_position="center", vertical_position="center",
            offset_x=0, offset_y=0, previous_layout=None):
        if placement not in PLACEMENTS or fit not in FITS:
            raise ValueError("Invalid layout placement or fit.")
        chain = previous_layout if previous_layout is not None else LayoutChain()
        if not isinstance(chain, LayoutChain):
            raise TypeError("previous_layout must be KREA2_LAYOUT_CHAIN.")
        return (chain.append(LayoutEntry(
            _image(image), placement, int(row), int(column), fit,
            horizontal_position, vertical_position, int(offset_x), int(offset_y)
        )),)


def _split(total, count, gap):
    available = total - gap * (count - 1)
    if available < count:
        raise ValueError("Layout Composer: padding/gap leaves insufficient room for cells.")
    base, remainder = divmod(available, count)
    return [base + (i < remainder) for i in range(count)]


def _place(canvas, image, rect, entry):
    x, y, w, h = rect
    ih, iw = image.shape[1:3]
    if entry.fit == "stretch":
        scaled = resize_tensor(image, h, w, "bicubic")
    elif entry.fit == "crop":
        scaled = image
    else:
        ratio = min(w / iw, h / ih) if entry.fit == "contain" else max(w / iw, h / ih)
        sw, sh = max(1, round(iw * ratio)), max(1, round(ih * ratio))
        scaled = resize_tensor(image, sh, sw, "bicubic")
    sh, sw = scaled.shape[1:3]
    if entry.fit in ("contain", "stretch"):
        px, py = (w - sw) // 2, (h - sh) // 2
    else:
        px = {"left": 0, "center": (w - sw) // 2, "right": w - sw}[entry.horizontal_position] + entry.offset_x
        py = {"up": 0, "center": (h - sh) // 2, "down": h - sh}[entry.vertical_position] + entry.offset_y
    dx0, dy0 = max(0, px), max(0, py)
    dx1, dy1 = min(w, px + sw), min(h, py + sh)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx0, sy0 = dx0 - px, dy0 - py
    canvas[:, y + dy0:y + dy1, x + dx0:x + dx1, :] = scaled[:, sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0, :].to(canvas.device)


class CcCKrea2LayoutComposer:
    CATEGORY = NODE_CATEGORY
    FUNCTION = "compose"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "layout_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "layout": ("KREA2_LAYOUT_CHAIN",),
            "preset_size": (KREA_PRESET_SIZES, {"default": DEFAULT_KREA_PRESET_SIZE}),
            "padding": ("INT", {"default": 16, "min": 0, "max": 1024}),
            "gap": ("INT", {"default": 8, "min": 0, "max": 1024}),
            "background_color": (("white", "gray", "black"), {"default": "white"}),
        }}

    def compose(self, layout, preset_size=DEFAULT_KREA_PRESET_SIZE, padding=16, gap=8, background_color="white"):
        if not isinstance(layout, LayoutChain) or not layout.entries:
            raise ValueError("Layout Composer requires a nonempty layout chain.")
        entries = layout.entries
        rows = max(e.row for e in entries if e.placement != "full_column") if any(e.placement != "full_column" for e in entries) else 1
        cols = max(e.column for e in entries if e.placement != "full_row") if any(e.placement != "full_row" for e in entries) else 1
        occupied = {}
        for index, e in enumerate(entries, 1):
            cells = ([(e.row, c) for c in range(1, cols + 1)] if e.placement == "full_row"
                     else [(r, e.column) for r in range(1, rows + 1)] if e.placement == "full_column"
                     else [(e.row, e.column)])
            for cell in cells:
                if cell in occupied:
                    raise ValueError(f"Layout Composer: Image {index} conflicts with Image {occupied[cell]} at row {cell[0]}, column {cell[1]}.")
                occupied[cell] = index
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                if (row, col) not in occupied:
                    raise ValueError(f"Layout Composer: empty cell at row {row}, column {col}.")
        width, height = preset_geometry(preset_size)
        usable_w, usable_h = width - 2 * padding, height - 2 * padding
        if usable_w <= 0 or usable_h <= 0:
            raise ValueError("Layout Composer: padding exceeds canvas.")
        widths, heights = _split(usable_w, cols, gap), _split(usable_h, rows, gap)
        xs, ys = [padding], [padding]
        for w in widths[:-1]:
            xs.append(xs[-1] + w + gap)
        for h in heights[:-1]:
            ys.append(ys[-1] + h + gap)
        value = {"white": 1., "gray": 0.5, "black": 0.}[background_color]
        canvas = torch.full((1, height, width, 3), value, dtype=torch.float32)
        for e in entries:
            start_col = 1 if e.placement == "full_row" else e.column
            end_col = cols if e.placement == "full_row" else e.column
            start_row = 1 if e.placement == "full_column" else e.row
            end_row = rows if e.placement == "full_column" else e.row
            x, y = xs[start_col - 1], ys[start_row - 1]
            w = xs[end_col - 1] + widths[end_col - 1] - x
            h = ys[end_row - 1] + heights[end_row - 1] - y
            _place(canvas, e.image, (x, y, w, h), e)
        return canvas, f"Canvas: {width} x {height}; Grid: {rows} rows x {cols} columns; Images: {len(entries)}"
