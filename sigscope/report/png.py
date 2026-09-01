"""Minimal PNG encoder and the viridis colormap (CLAUDE.md §3 Stage 6, §7).

The HTML report embeds its spectrogram as a base64 PNG, so something has to encode one.
Matplotlib is a **dev-only** dependency (``pyproject.toml``: "scripts/plot_classes.py only
-- not used by the package"), and §2's offline rule rules out fetching an image library at
demo time, so this writes PNG bytes directly with ``zlib`` and ``struct`` from the standard
library. It is about sixty lines and has no dependencies beyond numpy.

The colormap is viridis, as §7 requires: perceptually uniform, standard in the field, and
colourblind-safe. §7 is equally clear about what not to use -- "Not jet -- jet invents
features that are not there and any RF engineer will notice". The 32 anchor points below
are sampled from matplotlib's viridis and interpolated to 256 levels at import, which
avoids vendoring a 256-row table while staying visually indistinguishable from the real
thing.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

__all__ = ["VIRIDIS", "colormap", "encode_png", "draw_rect", "draw_hline", "draw_vline"]

# viridis sampled at 32 points (matplotlib), interpolated to 256 below
_VIRIDIS_ANCHORS: tuple[tuple[int, int, int], ...] = (
    (68, 1, 84), (71, 13, 96), (72, 24, 106), (72, 35, 116),
    (71, 46, 124), (69, 56, 130), (66, 65, 134), (62, 74, 137),
    (58, 84, 140), (54, 93, 141), (50, 101, 142), (46, 109, 142),
    (43, 117, 142), (40, 125, 142), (37, 132, 142), (34, 140, 141),
    (31, 148, 140), (30, 156, 137), (32, 163, 134), (37, 171, 130),
    (46, 179, 124), (58, 186, 118), (72, 193, 110), (88, 199, 101),
    (108, 205, 90), (127, 211, 78), (147, 215, 65), (168, 219, 52),
    (192, 223, 37), (213, 226, 26), (234, 229, 26), (253, 231, 37),
)


def _build_viridis() -> np.ndarray:
    """Interpolate the anchors to a 256x3 uint8 lookup table."""
    anchors = np.asarray(_VIRIDIS_ANCHORS, dtype=np.float64)
    src = np.linspace(0.0, 1.0, anchors.shape[0])
    dst = np.linspace(0.0, 1.0, 256)
    table = np.stack([np.interp(dst, src, anchors[:, c]) for c in range(3)], axis=1)
    return np.clip(table.round(), 0, 255).astype(np.uint8)


VIRIDIS: np.ndarray = _build_viridis()


def colormap(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    table: np.ndarray | None = None,
) -> np.ndarray:
    """Map a 2-D float array to an ``(H, W, 3)`` uint8 RGB image.

    ``vmin`` / ``vmax`` default to the 5th and 99.5th percentiles, which keeps a single
    strong carrier from flattening the whole spectrogram into the bottom of the colormap.
    """
    table = VIRIDIS if table is None else table
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    if vmin is None:
        vmin = float(np.percentile(finite, 5.0))
    if vmax is None:
        vmax = float(np.percentile(finite, 99.5))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max()) or (float(finite.min()) + 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0

    scaled = (values - vmin) / (vmax - vmin)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    idx = np.clip((scaled * 255.0).round(), 0, 255).astype(np.uint8)
    return table[idx]


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 array as PNG bytes (filter type 0, zlib deflate)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("encode_png: expected an (H, W, 3) uint8 array")
    height, width = rgb.shape[:2]

    # each scanline is prefixed with its filter byte (0 = None)
    raw = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), rgb.reshape(height, width * 3)], axis=1
    ).tobytes()

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit truecolour
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def draw_hline(rgb: np.ndarray, y: int, x0: int, x1: int, colour: tuple[int, int, int]) -> None:
    """Draw a 1-pixel horizontal line, clipped to the image."""
    height, width = rgb.shape[:2]
    if not 0 <= y < height:
        return
    x0, x1 = max(0, min(x0, x1)), min(width - 1, max(x0, x1))
    if x1 >= x0:
        rgb[y, x0 : x1 + 1] = colour


def draw_vline(rgb: np.ndarray, x: int, y0: int, y1: int, colour: tuple[int, int, int]) -> None:
    """Draw a 1-pixel vertical line, clipped to the image."""
    height, width = rgb.shape[:2]
    if not 0 <= x < width:
        return
    y0, y1 = max(0, min(y0, y1)), min(height - 1, max(y0, y1))
    if y1 >= y0:
        rgb[y0 : y1 + 1, x] = colour


def draw_rect(
    rgb: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    colour: tuple[int, int, int],
) -> None:
    """Draw a 1-pixel rectangle outline -- the §7 hairline detection box."""
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    draw_hline(rgb, y0, x0, x1, colour)
    draw_hline(rgb, y1, x0, x1, colour)
    draw_vline(rgb, x0, y0, y1, colour)
    draw_vline(rgb, x1, y0, y1, colour)
