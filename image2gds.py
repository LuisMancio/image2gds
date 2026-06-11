# image2gds.py
"""
Created on Jun 08 13:49 2026

@author: muehlenstaedt, mancio, claude 
"""

# -*- coding: utf-8 -*-
"""Convert an image file to a GDS file.
Pixels are merged into large rectangles via a scanline + vertical RLE algorithm.
Much faster than shapely.unary_union — no union calls needed.
"""

# ── Dependencies ──────────────────────────────────────────────────────────────
from gdshelpers.geometry.chip import Cell
from shapely.geometry import box
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Dithering algorithms ──────────────────────────────────────────────────────

def floyd_steinberg(data: np.ndarray) -> np.ndarray:
    """
    Floyd-Steinberg dithering.
    Diffuses quantization error to 4 neighbours.
    Input : uint8 array in [0, 255]
    Output: uint8 binary array (0 or 1)
    """
    img = data.astype(np.float32).copy()
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            old = img[i, j]
            new = 255.0 if old >= 128.0 else 0.0
            img[i, j] = new
            err = old - new

            if j + 1 < cols:
                img[i,     j + 1] += err * 7 / 16
            if i + 1 < rows:
                if j - 1 >= 0:
                    img[i + 1, j - 1] += err * 3 / 16
                img[i + 1, j    ]     += err * 5 / 16
                if j + 1 < cols:
                    img[i + 1, j + 1] += err * 1 / 16

    return (img >= 128).astype(np.uint8)


def ordered_dither_4x4(data: np.ndarray) -> np.ndarray:
    """
    Ordered (Bayer) 4x4 dithering.
    Good for a regular, patterned halftone look.
    """
    bayer = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5],
    ], dtype=np.float32) / 16.0 * 255.0

    rows, cols = data.shape
    tiled = np.tile(bayer, (rows // 4 + 1, cols // 4 + 1))[:rows, :cols]
    return (data.astype(np.float32) > tiled).astype(np.uint8)


def simple_threshold(data: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Binary threshold (no dithering)."""
    return (data > threshold).astype(np.uint8)


# ── Scanline rectangle builder ────────────────────────────────────────────────

def build_rectangles_scanline(data: np.ndarray, pixel_size: float, border: float) -> list:
    """
    Convert binary image to non-overlapping rectangles without any union calls.

    Per row: np.diff finds horizontal runs in O(cols) vectorized.
    Runs are extended vertically as long as the same (col_start, col_end) span
    continues in the next row; when a run breaks, one box is emitted.

    Result: far fewer Shapely objects than one-box-per-pixel, and zero unary_union.
    """
    rows, _ = data.shape
    boxes  = []
    active = {}  # (col_start, col_end) -> row_start_index

    for i in range(rows):
        padded = np.concatenate(([0], data[i], [0]))
        diffs  = np.diff(padded.astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0]
        current = set(zip(starts.tolist(), ends.tolist()))

        # Close runs that ended in this row
        for run in list(active):
            if run not in current:
                rs = active.pop(run)
                s, e = run
                boxes.append(box(
                    border + s * pixel_size, border + rs * pixel_size,
                    border + e * pixel_size, border + i  * pixel_size,
                ))

        # Open runs that are new in this row
        for run in current:
            if run not in active:
                active[run] = i

    # Flush runs still open at the last row
    for (s, e), rs in active.items():
        boxes.append(box(
            border + s * pixel_size, border + rs   * pixel_size,
            border + e * pixel_size, border + rows * pixel_size,
        ))

    return boxes


# ── Configuration ─────────────────────────────────────────────────────────────

IMAGE_FILE   = "IAP_logo.png"
OUTPUT_FILE  = "IAP_logo.gds"
PIXEL_SIZE   = 2*13      # micrometers per pixel
BORDER       = 5      # border offset in micrometers
GDS_LAYER    = 8     # layer for image pixels
BORDER_LAYER = 2      # layer for chip border

# Choose dithering method: 'floyd_steinberg' | 'ordered' | 'threshold'
DITHER_MODE  = "ordered"

# ── Load & process image ──────────────────────────────────────────────────────

img      = Image.open(IMAGE_FILE)
img_gray = img.convert('L')
data_raw = np.array(img_gray)         # uint8, shape (rows, cols)

if DITHER_MODE == "floyd_steinberg":
    data = floyd_steinberg(data_raw)
elif DITHER_MODE == "ordered":
    data = ordered_dither_4x4(data_raw)
else:
    data = simple_threshold(data_raw)

data = np.flipud(data)                # flip so origin is bottom-left
rows, cols = data.shape

print(f"Image        : {rows} x {cols} = {rows*cols} total pixels")

# ── Build rectangles via scanline ─────────────────────────────────────────────

print("Building rectangles (scanline)...")
rectangles = build_rectangles_scanline(data, PIXEL_SIZE, BORDER)
print(f"Generated {len(rectangles)} rectangles  ({rows}×{cols} px, mode={DITHER_MODE})")

# ── Write GDS ─────────────────────────────────────────────────────────────────

cell = Cell('Chip_1')

# Border rectangle
"""
w = cols * PIXEL_SIZE + 2 * BORDER
h = rows * PIXEL_SIZE + 2 * BORDER
border_rect = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
cell.add_to_layer(BORDER_LAYER, border_rect)
"""

print("Writing GDS...")
for rect in tqdm(rectangles, desc="Writing polygons"):
    cell.add_to_layer(GDS_LAYER, rect)

cell.save(OUTPUT_FILE)
print(f"Saved → {OUTPUT_FILE}")