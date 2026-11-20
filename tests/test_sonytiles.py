"""The fast Sony tile decoder must refuse anything it does not fully understand.

The bit-exact comparison against LibRaw needs a real camera file. Point
MOONSTACK_SONY_SAMPLE at a Sony "Lossless Compressed" ARW to run it.
"""
import io
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from moonstack import ingest, sonytiles


def _tiff(**kw) -> io.BytesIO:
    buf = io.BytesIO()
    tifffile.imwrite(buf, np.zeros((1024, 1024), np.uint16), tile=(512, 512), **kw)
    return buf


def test_rejects_files_that_are_not_sony_tiled_raws(tmp_path):
    assert sonytiles.read_layout(io.BytesIO(b"")) is None
    assert sonytiles.read_layout(io.BytesIO(b"not a tiff at all, just some bytes")) is None
    assert sonytiles.read_layout(io.BytesIO(b"II*\0" + b"\xff" * 64)) is None  # points nowhere
    assert sonytiles.read_layout(_tiff()) is None  # no camera maker
    assert sonytiles.read_layout(_tiff(extratags=[(0x010F, "s", 0, "Canon", True)])) is None
    # Sony by name, but an ordinary uncompressed TIFF with no raw sub-image.
    assert sonytiles.read_layout(_tiff(extratags=[(0x010F, "s", 0, "SONY", True)])) is None

    # Only .arw files are even considered for the fast path.
    fake = tmp_path / "frame.dng"
    fake.write_bytes(_tiff().getvalue())
    assert ingest.tiles_reference(fake) is None
    bad = tmp_path / "frame.arw"
    bad.write_bytes(b"II*\0garbage")
    assert ingest.tiles_reference(bad) is None


@pytest.mark.skipif("MOONSTACK_SONY_SAMPLE" not in os.environ, reason="needs a real Sony ARW")
def test_tiles_match_libraw_exactly():
    path = Path(os.environ["MOONSTACK_SONY_SAMPLE"])
    ref = ingest.tiles_reference(path)
    assert ref is not None, "file was not accepted by the tile decoder"
    fast = ingest.ingest_one(str(path), tiles=ref)
    slow = ingest.ingest_one(str(path))
    assert fast["decoder"] == "tiles" and slow["decoder"] == "libraw"
    assert fast["origin"] == slow["origin"] and fast["bbox"] == slow["bbox"]
    assert fast["touches_border"] == slow["touches_border"]
    assert np.array_equal(fast["crop"], slow["crop"])

    # And the whole frame, tile by tile.
    full = ingest.load_raw(path).bayer
    with open(path, "rb") as f:
        lay = sonytiles.read_layout(f)
        ny, nx = lay.grid
        assert np.array_equal(lay.decode(f, (0, 0, nx, ny)), full)
