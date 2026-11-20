"""RAW decoding and moon detection.

Everything here runs inside worker processes, so this module must stay light
(no numba import) and return only small, picklable results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

RAW_EXTS = {".arw", ".dng", ".nef", ".cr2", ".cr3", ".rw2", ".orf", ".pef", ".srw", ".raw"}
SYNTH_EXT = ".npz"  # synthetic frames written by moonstack.synth

# Moon detection works on a 1/8-scale brightness image.
_DETECT_SCALE = 8


@dataclass
class RawFrame:
    """Undemosaiced sensor data plus what is needed to interpret it."""

    bayer: np.ndarray  # uint16 (H, W), H and W even
    cfa: np.ndarray  # (2, 2) int64: 0=R, 1=G, 2=B at each position of the 2x2 tile
    black: np.ndarray  # (2, 2) float32 black level at each position of the tile
    white: float  # saturation level, same scale as bayer
    wb_as_shot: np.ndarray | None  # (3,) multipliers, G = 1
    wb_daylight: np.ndarray | None  # (3,) multipliers, G = 1
    cam_xyz: np.ndarray | None  # (3, 3) XYZ(D65) -> camera RGB
    flip: int = 0  # LibRaw orientation code (0, 3, 5, 6)


class UnsupportedRaw(RuntimeError):
    pass


def _norm_wb(wb) -> np.ndarray | None:
    if wb is None:
        return None
    wb = np.asarray(wb, dtype=np.float64)[:3]
    if not np.all(np.isfinite(wb)) or np.any(wb <= 0):
        return None
    return (wb / wb[1]).astype(np.float32)


def load_raw(path: Path) -> RawFrame:
    path = Path(path)
    if path.suffix.lower() == SYNTH_EXT:
        z = np.load(path)
        return RawFrame(
            bayer=z["bayer"],
            cfa=z["cfa"].astype(np.int64),
            black=z["black"].astype(np.float32),
            white=float(z["white"]),
            wb_as_shot=_norm_wb(z["wb"]),
            wb_daylight=_norm_wb(z["wb"]),
            cam_xyz=None,
        )

    import rawpy

    with rawpy.imread(str(path)) as raw:
        if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
            raise UnsupportedRaw(f"{path.name}: not a 2x2 Bayer sensor")
        img = raw.raw_image_visible
        if img.ndim != 2:
            raise UnsupportedRaw(f"{path.name}: raw data is not a single mosaic plane")
        h, w = img.shape
        img = np.ascontiguousarray(img[: h - h % 2, : w - w % 2])
        colors = np.asarray(raw.raw_colors_visible[:2, :2]).astype(np.int64)
        desc = raw.color_desc.decode()
        cfa = np.array([["RGB".index(desc[i]) for i in row] for row in colors], dtype=np.int64)
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)[colors]
        cam_xyz = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3]
        if not np.any(cam_xyz):
            cam_xyz = None
        try:
            daylight = _norm_wb(raw.daylight_whitebalance)
        except Exception:
            daylight = None
        return RawFrame(
            bayer=img,
            cfa=cfa,
            black=black,
            white=float(raw.white_level),
            wb_as_shot=_norm_wb(raw.camera_whitebalance),
            wb_daylight=daylight,
            cam_xyz=cam_xyz,
            flip=int(raw.sizes.flip),
        )


def half_luma(bayer: np.ndarray, black: np.ndarray) -> np.ndarray:
    """Half-resolution brightness: mean of each black-subtracted 2x2 Bayer tile."""
    c = bayer.astype(np.float32)
    out = c[0::2, 0::2] - black[0, 0]
    out += c[0::2, 1::2] - black[0, 1]
    out += c[1::2, 0::2] - black[1, 0]
    out += c[1::2, 1::2] - black[1, 1]
    out *= 0.25
    return out


def find_moon(bayer: np.ndarray, black: np.ndarray) -> dict | None:
    """Locate the lit part of the moon. Coordinates are in sensor pixels."""
    luma = half_luma(bayer, black)
    f = 2.0 / _DETECT_SCALE
    small = cv2.resize(luma, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    bg = float(np.median(small))
    hi = float(np.percentile(small, 99.99))
    noise = 1.4826 * float(np.median(np.abs(small - bg))) + 1e-6
    if hi - bg < max(20.0 * noise, 8.0):
        return None
    mask = (small > bg + 0.04 * (hi - bg)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in stats[k])
    if area < 30:
        return None
    s = _DETECT_SCALE
    sh, sw = small.shape
    return {
        "bbox": (x * s, y * s, w * s, h * s),
        "centroid": (float(cents[k][0] + 0.5) * s, float(cents[k][1] + 0.5) * s),
        "touches_border": x <= 0 or y <= 0 or x + w >= sw or y + h >= sh,
        "peak": hi,
    }


def ingest_one(path_str: str) -> dict:
    """Decode one file and return a generous crop around the moon.

    Runs in a worker process. The crop origin is always even so the Bayer
    pattern of the crop matches the pattern of the full frame.
    """
    path = Path(path_str)
    try:
        frame = load_raw(path)
    except Exception as e:  # report and let the caller decide
        return {"path": path_str, "ok": False, "error": f"{type(e).__name__}: {e}"}

    det = find_moon(frame.bayer, frame.black)
    if det is None:
        return {"path": path_str, "ok": False, "error": "could not find the moon in this frame"}

    H, W = frame.bayer.shape
    bx, by, bw, bh = det["bbox"]
    m = int(max(96, 0.2 * max(bw, bh)))
    x0 = max(0, (bx - m) // 2 * 2)
    y0 = max(0, (by - m) // 2 * 2)
    x1 = min(W, (bx + bw + m + 1) // 2 * 2)
    y1 = min(H, (by + bh + m + 1) // 2 * 2)
    crop = np.ascontiguousarray(frame.bayer[y0:y1, x0:x1])
    clipped = float(np.mean(crop >= 0.985 * frame.white))

    frame.bayer = None  # do not ship the full frame back
    return {
        "path": path_str,
        "ok": True,
        "crop": crop,
        "origin": (x0, y0),
        "sensor_shape": (H, W),
        "clipped_frac": clipped,
        "meta": frame,
        **det,
    }
