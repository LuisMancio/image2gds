# image2gds.py
"""
Created on Jun 08 13:49 2026

@author: muehlenstaedt, mancio, claude 
"""

# -*- coding: utf-8 -*-
"""Convert an image file to a GDS file.
Pixels are merged into large polygons via shapely.unary_union
for compact file size — no gdspy required.
"""

# ── Dependencies ──────────────────────────────────────────────────────────────
from gdshelpers.geometry.chip import Cell
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
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


# ── Configuration ─────────────────────────────────────────────────────────────

IMAGE_FILE   = "CGH_IAP.png"
OUTPUT_FILE  = "CGH_IAP.gds"
PIXEL_SIZE   = 2      # micrometers per pixel
BORDER       = 5      # border offset in micrometers
GDS_LAYER    = 2      # layer for image pixels
BORDER_LAYER = 2      # layer for chip border

# Choose dithering method: 'floyd_steinberg' | 'ordered' | 'threshold'
DITHER_MODE  = "threshold"

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

# ── Build pixel polygons row by row and merge ─────────────────────────────────
# Merging is done row by row to keep memory usage under control.
# Each row produces one union, then all row-unions are merged at the end.

print("Merging polygons...")
row_unions = []

with tqdm(total=rows, desc=f"Merging rows ({DITHER_MODE})") as pbar:
    for i in range(rows):
        cols_on = np.where(data[i] > 0)[0]
        if len(cols_on) == 0:
            pbar.update(1)
            continue

        # Build all pixel boxes for this row at once
        row_boxes = [
            box(
                BORDER + j * PIXEL_SIZE,
                BORDER + i * PIXEL_SIZE,
                BORDER + j * PIXEL_SIZE + PIXEL_SIZE,
                BORDER + i * PIXEL_SIZE + PIXEL_SIZE,
            )
            for j in cols_on
        ]
        row_unions.append(unary_union(row_boxes))
        pbar.update(1)

print("Final union across rows...")
merged = unary_union(row_unions)

# ── Write GDS ─────────────────────────────────────────────────────────────────

cell = Cell('Chip_1')

# Border rectangle
"""
w = cols * PIXEL_SIZE + 2 * BORDER
h = rows * PIXEL_SIZE + 2 * BORDER
border_rect = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
cell.add_to_layer(BORDER_LAYER, border_rect)
"""

# Add merged geometry — can be a Polygon or MultiPolygon
print("Writing GDS...")
if merged.geom_type == 'Polygon':
    cell.add_to_layer(GDS_LAYER, merged)
else:
    # MultiPolygon: add each part individually
    for geom in tqdm(merged.geoms, desc="Writing polygons"):
        cell.add_to_layer(GDS_LAYER, geom)

cell.save(OUTPUT_FILE)
print(f"Saved → {OUTPUT_FILE}  ({rows}×{cols} px, mode={DITHER_MODE})")