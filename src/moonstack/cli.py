"""Command line: moonstack FOLDER [options]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="moonstack",
        description="Align and stack a burst of RAW moon photos into one clean image.",
    )
    ap.add_argument("folder", type=Path, help="folder containing the RAW files")
    ap.add_argument("-o", "--out", type=Path, help="output folder (default: FOLDER/moonstack_out)")
    ap.add_argument("--keep", type=float, default=0.8,
                    help="fraction of frames to stack, sharpest first (default 0.8)")  # fmt: skip
    ap.add_argument("--scale", type=float, default=1.0,
                    help="output pixels per sensor pixel; 1.5 or 2 can recover extra detail (default 1)")  # fmt: skip
    ap.add_argument("--radius", type=float, default=0.75,
                    help="stacking kernel half-width in sensor pixels, 0.3-1; smaller is sharper but noisier (default 0.75)")  # fmt: skip
    ap.add_argument("--kappa", type=float, default=3.0,
                    help="outlier rejection threshold in sigmas; 0 disables (default 3)")  # fmt: skip
    ap.add_argument("--reference", help="file name of the frame to align to (default: the sharpest)")
    ap.add_argument("--no-local", action="store_true", help="skip the local turbulence correction")
    ap.add_argument("--wb", choices=["daylight", "as-shot"], default="daylight",
                    help="white balance (default daylight: the moon is sunlit rock)")  # fmt: skip
    ap.add_argument("--psf-sigma", type=float, help="blur to remove, in output pixels (default: measured on the limb)")
    ap.add_argument("--sharpen-iters", type=int, default=25, help="deconvolution iterations; 0 disables (default 25)")
    ap.add_argument("--saturation", type=float, default=6.0, help="colour boost of the mineral image (default 6)")
    ap.add_argument("-j", "--jobs", type=int, default=0, help="worker count (default: all cores)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    if not a.folder.is_dir():
        ap.error(f"{a.folder} is not a folder")
    if not 0 < a.keep <= 1:
        ap.error("--keep must be between 0 and 1")

    from .pipeline import Config, run, write_outputs

    cfg = Config(
        keep=a.keep, scale=a.scale, radius=a.radius, kappa=a.kappa, local=not a.no_local, wb=a.wb,
        psf_sigma=a.psf_sigma, reference=a.reference, sharpen_iters=a.sharpen_iters, saturation=a.saturation, jobs=a.jobs, quiet=a.quiet,
    )  # fmt: skip
    try:
        res = run(a.folder, cfg)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = a.out or a.folder / "moonstack_out"
    files = write_outputs(res, out, cfg, flip=res.info.get("flip", 0))
    if not a.quiet:
        print(f"wrote {len(files)} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
