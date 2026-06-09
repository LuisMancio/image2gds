# image2gds.py
"""
Created on Jun 08 13:49 2026

@author: muehlenstaedt, mancio
"""

#Import dependencies
from gdshelpers.geometry.chip import Cell
from shapely.geometry import Polygon
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Dithering algorithms ────────────────────────────────────────────────────

def floyd_steinberg(data: np.ndarray) -> np.ndarray:
    """
    Floyd-Steinberg dithering.
    Diffuses quantization error to 4 neighbours:
        * →  7/16
        ↙ ↓  ↘
        3/16  5/16  1/16
    Input : float32 array in [0, 255]
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
                img[i + 1, j    ] += err * 5 / 16
                if j + 1 < cols:
                    img[i + 1, j + 1] += err * 1 / 16

    return (img >= 128).astype(np.uint8)


def ordered_dither_4x4(data: np.ndarray) -> np.ndarray:
    """
    Ordered (Bayer) 4×4 dithering.
    Good for a regular, patterned halftone look.
    """
    bayer = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5],
    ], dtype=np.float32) / 16.0 * 255.0   # scale to [0, 255]

    rows, cols = data.shape
    tiled = np.tile(bayer, (rows // 4 + 1, cols // 4 + 1))[:rows, :cols]
    return (data.astype(np.float32) > tiled).astype(np.uint8)


def simple_threshold(data: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Original binary threshold (no dithering)."""
    return (data > threshold).astype(np.uint8)


# ── Configuration ───────────────────────────────────────────────────────────

IMAGE_FILE  = "IAP_logo.png"
OUTPUT_FILE = "IAP_logo_threshold_mode.gds"
PIXEL_SIZE  = 2        # micrometers per pixel
BORDER      = 5        # border offset in micrometers
GDS_LAYER   = 1        # layer for image pixels
BORDER_LAYER= 4        # layer for the chip border

# Choose dithering method: 'floyd_steinberg' | 'ordered' | 'threshold'
DITHER_MODE = "floyd_steinberg"

# ── Load & process image ────────────────────────────────────────────────────

img      = Image.open(IMAGE_FILE)
img_gray = img.convert('L')
data_raw = np.array(img_gray)          # uint8, shape (rows, cols)

if DITHER_MODE == "floyd_steinberg":
    data = floyd_steinberg(data_raw)
elif DITHER_MODE == "ordered":
    data = ordered_dither_4x4(data_raw)
else:
    data = simple_threshold(data_raw)

data = np.flipud(data)                 # flip so origin is bottom-left
rows, cols = data.shape

# ── Build GDS ───────────────────────────────────────────────────────────────

cell = Cell('Chip_1')

# Border rectangle
w = cols * PIXEL_SIZE + 2 * BORDER
h = rows * PIXEL_SIZE + 2 * BORDER
border_rect = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
cell.add_to_layer(BORDER_LAYER, border_rect)

# Draw pixels
total_pixels = rows * cols
with tqdm(total=total_pixels, desc=f"Rendering ({DITHER_MODE})") as pbar:
    for i in range(rows):
        for j in range(cols):
            if data[i, j] > 0:
                x0 = BORDER + j * PIXEL_SIZE
                y0 = BORDER + i * PIXEL_SIZE
                rect = Polygon([
                    (x0,              y0),
                    (x0 + PIXEL_SIZE, y0),
                    (x0 + PIXEL_SIZE, y0 + PIXEL_SIZE),
                    (x0,              y0 + PIXEL_SIZE),
                ])
                cell.add_to_layer(GDS_LAYER, rect)
            pbar.update(1)

cell.save(OUTPUT_FILE)
print(f"Saved → {OUTPUT_FILE}  ({rows}×{cols} px, mode={DITHER_MODE})")