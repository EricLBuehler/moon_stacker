"""RAW decoding and moon detection.

Everything here runs inside worker processes, so this module must stay light
(no numba import) and return only small, picklable results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import sonytiles

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
    cam_srgb: np.ndarray | None = None  # (3, 3) white-balanced camera RGB -> linear sRGB, from the camera itself
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


def _sky_stats(small: np.ndarray) -> tuple[float, float]:
    """Background level and noise of a 1/8-scale brightness image."""
    bg = float(np.median(small))
    return bg, 1.4826 * float(np.median(np.abs(small - bg))) + 1e-6


def _small_luma(bayer: np.ndarray, black: np.ndarray) -> np.ndarray:
    f = 2.0 / _DETECT_SCALE
    return cv2.resize(half_luma(bayer, black), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def find_moon(bayer: np.ndarray, black: np.ndarray, frame_pixels: int | None = None, sky=None) -> dict | None:
    """Locate the lit part of the moon. Coordinates are in pixels of `bayer`.

    A full frame is mostly sky, so its median is the background. When `bayer`
    is only the part of the sensor around the moon that no longer holds, so
    pass `sky` (level, noise) measured elsewhere, and the full frame's pixel
    count as `frame_pixels` so "the brightest 0.01% of the frame" keeps its
    meaning.
    """
    small = _small_luma(bayer, black)
    bg, noise = sky if sky is not None else _sky_stats(small)
    if frame_pixels is None:
        hi = float(np.percentile(small, 99.99))
    else:
        k = min(small.size, max(1, int(round(1e-4 * frame_pixels / _DETECT_SCALE**2))))
        hi = float(np.partition(small.ravel(), small.size - k)[small.size - k])
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


def _crop_bounds(bbox, H: int, W: int) -> tuple[int, int, int, int]:
    """Generous, even-aligned crop rectangle (x0, y0, x1, y1) around a moon bbox."""
    bx, by, bw, bh = bbox
    m = int(max(96, 0.2 * max(bw, bh)))
    x0 = max(0, (bx - m) // 2 * 2)
    y0 = max(0, (by - m) // 2 * 2)
    x1 = min(W, (bx + bw + m + 1) // 2 * 2)
    y1 = min(H, (by + bh + m + 1) // 2 * 2)
    return x0, y0, x1, y1


def _result(path_str, crop, origin, shape, meta, det, decoder) -> dict:
    return {
        "path": path_str,
        "ok": True,
        "crop": crop,
        "origin": origin,
        "sensor_shape": shape,
        "clipped_frac": float(np.mean(crop >= 0.985 * meta.white)),
        "meta": meta,
        "decoder": decoder,
        **det,
    }


def tiles_reference(path: Path) -> tuple[tuple, RawFrame] | None:
    """Decide whether the fast tile decoder may be used for this run.

    Decodes the first file both ways and requires the tile decoder to
    reproduce LibRaw's pixels exactly. Returns what workers need (the layout
    signature other files must share, and the metadata LibRaw reported), or
    None to stay on LibRaw.
    """
    path = Path(path)
    if path.suffix.lower() != ".arw":
        return None
    try:
        import imagecodecs  # noqa: F401

        with open(path, "rb") as f:
            lay = sonytiles.read_layout(f)
            if lay is None:
                return None
            frame = load_raw(path)
            if frame.bayer.shape != (lay.height, lay.width):
                return None
            ny, nx = lay.grid
            tx0, ty0, tx1, ty1 = lay.bright_tiles() or (nx // 2, ny // 2, nx // 2 + 1, ny // 2 + 1)
            # A few tiles are enough: under the moon if we can tell, plus a corner.
            for rect in ((tx0, ty0, min(tx0 + 2, tx1), min(ty0 + 2, ty1)), (nx - 1, ny - 1, nx, ny)):
                x0, y0 = rect[0] * lay.tile_w, rect[1] * lay.tile_h
                got = lay.decode(f, rect)
                if not np.array_equal(got, frame.bayer[y0 : y0 + got.shape[0], x0 : x0 + got.shape[1]]):
                    return None
        frame.bayer = None
        return lay.signature, frame
    except Exception:
        return None


def _ingest_tiles(path_str: str, signature: tuple, meta: RawFrame) -> dict | None:
    """Fast path: decode only the tiles around the moon. None means "use LibRaw"."""
    with open(path_str, "rb") as f:
        lay = sonytiles.read_layout(f)
        if lay is None or lay.signature != signature:
            return None
        rect = lay.bright_tiles()
        if rect is None:
            return None
        H, W, tw, th = lay.height, lay.width, lay.tile_w, lay.tile_h
        # The decoded area is mostly moon, so take the sky level from a tile of plain sky.
        sx, sy = lay.sky_tile(rect)
        sky = _sky_stats(_small_luma(lay.decode(f, (sx, sy, sx + 1, sy + 1)), meta.black))
        for _ in range(6):
            sub = lay.decode(f, rect)
            ox, oy = rect[0] * tw, rect[1] * th
            det = find_moon(sub, meta.black, frame_pixels=H * W, sky=sky)
            if det is None:
                return None
            bx, by, bw, bh = det["bbox"]
            bbox = (bx + ox, by + oy, bw, bh)
            x0, y0, x1, y1 = _crop_bounds(bbox, H, W)
            need = (x0 // tw, y0 // th, -(-x1 // tw), -(-y1 // th))
            if need[0] >= rect[0] and need[1] >= rect[1] and need[2] <= rect[2] and need[3] <= rect[3]:
                break
            # The moon runs past what we decoded: widen and look again.
            rect = (min(rect[0], need[0]), min(rect[1], need[1]), max(rect[2], need[2]), max(rect[3], need[3]))
        else:
            return None
    s = _DETECT_SCALE
    det["bbox"] = bbox
    det["centroid"] = (det["centroid"][0] + ox, det["centroid"][1] + oy)
    # "Cut off" must mean the edge of the sensor, not the edge of what we decoded.
    det["touches_border"] = (
        bbox[0] <= 0 or bbox[1] <= 0 or bbox[0] + bw >= (W // s) * s or bbox[1] + bh >= (H // s) * s
    )
    crop = np.ascontiguousarray(sub[y0 - oy : y1 - oy, x0 - ox : x1 - ox])
    return _result(path_str, crop, (x0, y0), (H, W), meta, det, "tiles")


def ingest_one(path_str: str, tiles: tuple | None = None) -> dict:
    """Decode one file and return a generous crop around the moon.

    Runs in a worker process. The crop origin is always even so the Bayer
    pattern of the crop matches the pattern of the full frame. `tiles` is the
    result of tiles_reference(); when given, the fast tile decoder is tried
    first and LibRaw is the fallback.
    """
    if tiles is not None:
        try:
            fast = _ingest_tiles(path_str, *tiles)
        except Exception:
            fast = None
        if fast is not None:
            return fast

    path = Path(path_str)
    try:
        frame = load_raw(path)
    except Exception as e:  # report and let the caller decide
        return {"path": path_str, "ok": False, "error": f"{type(e).__name__}: {e}"}

    # Same definition of the brightness peak as the tile path, so both give the same numbers.
    det = find_moon(frame.bayer, frame.black, frame_pixels=frame.bayer.size)
    if det is None:
        return {"path": path_str, "ok": False, "error": "could not find the moon in this frame"}

    H, W = frame.bayer.shape
    x0, y0, x1, y1 = _crop_bounds(det["bbox"], H, W)
    crop = np.ascontiguousarray(frame.bayer[y0:y1, x0:x1])
    frame.bayer = None  # do not ship the full frame back
    return _result(path_str, crop, (x0, y0), (H, W), frame, det, "libraw")


def sony_colour(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """The camera's own daylight white balance and colour matrix from a Sony ARW.

    Sony stores both in an encrypted metadata block that LibRaw does not expose,
    so this asks exiftool (if installed). For bodies newer than LibRaw's colour
    tables this is the only trustworthy source. Returns (wb, matrix) or None.
    """
    import json
    import shutil
    import subprocess

    exe = shutil.which("exiftool")
    if exe is None or Path(path).suffix.lower() != ".arw":
        return None
    try:
        out = subprocess.run(
            [exe, "-j", "-n", "-SR2SubIFD:WB_RGBLevelsDaylight", "-SR2SubIFD:ColorMatrix", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout  # fmt: skip
        tags = json.loads(out)[0]
        wb = np.array([float(v) for v in str(tags["WB_RGBLevelsDaylight"]).split()])
        m = np.array([float(v) for v in str(tags["ColorMatrix"]).split()]).reshape(3, 3)
    except Exception:
        return None
    if wb.shape != (3,) or np.any(wb <= 0) or not np.allclose(m.sum(axis=1), m.sum(axis=1)[0], rtol=0.02):
        return None
    return (wb / wb[1]).astype(np.float32), (m / m.sum(axis=1, keepdims=True)).astype(np.float32)
