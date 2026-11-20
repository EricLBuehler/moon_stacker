"""Partial decoding of Sony tiled lossless ARW files.

Recent Sony bodies store "Lossless Compressed RAW" as a grid of independent
512x512 tiles, each a lossless-JPEG stream with its own entry in an offset
table. LibRaw always decodes the whole frame; for a moon that covers a few
percent of it, decoding just the tiles under the moon is an order of magnitude
less work.

This module only understands that one layout. `read_layout` returns None for
anything else, and callers fall back to LibRaw.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

# TIFF tag ids
_MAKE, _SUBIFD = 0x010F, 0x014A
_WIDTH, _HEIGHT, _BITS, _COMPRESSION, _SPP = 0x0100, 0x0101, 0x0102, 0x0103, 0x0115
_TILE_W, _TILE_H, _TILE_OFFSETS, _TILE_COUNTS = 0x0142, 0x0143, 0x0144, 0x0145
_CFA_DIM, _CFA_PATTERN = 0x828D, 0x828E
_SONY_BLACK, _WHITE = 0x7310, 0xC61D
_JPEG = 7

_TYPE_FMT = {1: "B", 2: "c", 3: "H", 4: "I", 7: "B", 13: "I"}
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 13: 4}


def _read_ifd(f, bo: str, offset: int, wanted: set[int]) -> dict[int, tuple]:
    """Values of the wanted tags in the IFD at `offset`."""
    f.seek(offset)
    (n,) = struct.unpack(bo + "H", f.read(2))
    if n > 4096:
        raise ValueError("implausible IFD")
    raw = f.read(12 * n)
    tags = {}
    for i in range(n):
        tag, typ, count = struct.unpack(bo + "HHI", raw[12 * i : 12 * i + 8])
        if tag not in wanted or typ not in _TYPE_FMT:
            continue
        size = _TYPE_SIZE[typ] * count
        if size > 4:
            (where,) = struct.unpack(bo + "I", raw[12 * i + 8 : 12 * i + 12])
            f.seek(where)
            data = f.read(size)
        else:
            data = raw[12 * i + 8 : 12 * i + 8 + size]
        if len(data) != size:
            raise ValueError("truncated tag")
        tags[tag] = struct.unpack(bo + _TYPE_FMT[typ] * count, data)
    return tags


@dataclass
class TileLayout:
    width: int
    height: int
    tile_w: int
    tile_h: int
    offsets: np.ndarray  # (ny, nx) byte offset of each tile
    counts: np.ndarray  # (ny, nx) compressed size of each tile
    # Everything that must match between two files for them to share metadata.
    signature: tuple = ()
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def grid(self) -> tuple[int, int]:
        return self.offsets.shape  # (ny, nx)

    def bright_tiles(self) -> tuple[int, int, int, int] | None:
        """Tile rectangle (tx0, ty0, tx1, ty1) that holds the bright subject.

        Needs no decoding: lossless compression cannot squeeze noise, and a
        sunlit surface has far more photon noise than black sky, so the tiles
        under the moon are measurably larger in the file.
        """
        sizes = self.counts.astype(np.float64)
        # Tiles on the last row/column can be partly masked and smaller; the
        # median is still the size of an ordinary sky tile.
        ys, xs = np.nonzero(sizes > 1.02 * np.median(sizes))
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def sky_tile(self, rect: tuple[int, int, int, int]) -> tuple[int, int]:
        """A tile of ordinary sky: typical size, as far from `rect` as possible."""
        ny, nx = self.grid
        ys, xs = np.mgrid[:ny, :nx]
        cx, cy = (rect[0] + rect[2] - 1) / 2, (rect[1] + rect[3] - 1) / 2
        typical = np.abs(self.counts / np.median(self.counts) - 1) < 0.01
        dist = np.where(typical, np.hypot(xs - cx, ys - cy), -1.0)
        ty, tx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        return int(tx), int(ty)

    def decode(self, f, rect: tuple[int, int, int, int]) -> np.ndarray:
        """Bayer mosaic of a rectangle of tiles, uint16."""
        import imagecodecs

        tx0, ty0, tx1, ty1 = rect
        tw, th = self.tile_w, self.tile_h
        out = np.empty(((ty1 - ty0) * th, (tx1 - tx0) * tw), np.uint16)
        for ty in range(ty0, ty1):
            for tx in range(tx0, tx1):
                tile = self._cache.get((tx, ty))
                if tile is None:
                    f.seek(int(self.offsets[ty, tx]))
                    quads = imagecodecs.ljpeg_decode(f.read(int(self.counts[ty, tx])))
                    if quads.shape != (th // 2, tw // 2, 4) or quads.dtype != np.uint16:
                        raise ValueError(f"unexpected tile layout {quads.shape} {quads.dtype}")
                    # Each JPEG "pixel" carries one 2x2 Bayer cell as four components.
                    tile = np.empty((th, tw), np.uint16)
                    tile[0::2, 0::2] = quads[..., 0]
                    tile[0::2, 1::2] = quads[..., 1]
                    tile[1::2, 0::2] = quads[..., 2]
                    tile[1::2, 1::2] = quads[..., 3]
                    self._cache[(tx, ty)] = tile
                out[(ty - ty0) * th : (ty - ty0 + 1) * th, (tx - tx0) * tw : (tx - tx0 + 1) * tw] = tile
        return out


def read_layout(f) -> TileLayout | None:
    """Parse an open file. None unless it is a Sony tiled lossless Bayer RAW."""
    try:
        f.seek(0)
        head = f.read(8)
        bo = {b"II": "<", b"MM": ">"}.get(head[:2])
        if bo is None or struct.unpack(bo + "H", head[2:4])[0] != 42:
            return None
        ifd0 = _read_ifd(f, bo, struct.unpack(bo + "I", head[4:8])[0], {_MAKE, _SUBIFD})
        make = b"".join(ifd0.get(_MAKE, ())).rstrip(b"\0 ").upper()
        if not make.startswith(b"SONY"):
            return None
        wanted = {_WIDTH, _HEIGHT, _BITS, _COMPRESSION, _SPP, _TILE_W, _TILE_H, _TILE_OFFSETS, _TILE_COUNTS,
                  _CFA_DIM, _CFA_PATTERN, _SONY_BLACK, _WHITE}  # fmt: skip
        for sub_offset in ifd0.get(_SUBIFD, ()):
            t = _read_ifd(f, bo, sub_offset, wanted)
            if not wanted - {_SONY_BLACK, _WHITE} <= t.keys():
                continue
            w, h, tw, th = t[_WIDTH][0], t[_HEIGHT][0], t[_TILE_W][0], t[_TILE_H][0]
            ok = (
                t[_COMPRESSION][0] == _JPEG
                and t[_SPP][0] == 1
                and t[_BITS][0] <= 16
                and tuple(t[_CFA_DIM]) == (2, 2)
                and len(t[_CFA_PATTERN]) == 4
                and tw % 2 == 0 and th % 2 == 0 and tw > 0 and th > 0
                and w % tw == 0 and h % th == 0
                and len(t[_TILE_OFFSETS]) == len(t[_TILE_COUNTS]) == (w // tw) * (h // th)
            )  # fmt: skip
            if not ok:
                continue
            shape = (h // th, w // tw)
            sig = (w, h, tw, th, t[_BITS][0], tuple(t[_CFA_PATTERN]), tuple(t.get(_SONY_BLACK, ())), tuple(t.get(_WHITE, ())))
            return TileLayout(
                w, h, tw, th,
                np.array(t[_TILE_OFFSETS], np.int64).reshape(shape),
                np.array(t[_TILE_COUNTS], np.int64).reshape(shape),
                sig,
            )  # fmt: skip
    except (OSError, ValueError, struct.error):
        return None
    return None
