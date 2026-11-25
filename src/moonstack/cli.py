"""Command line: moonstack FOLDER [options]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _scale(text: str):
    if text.lower() == "auto":
        return "auto"
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number or 'auto'") from None
    if not 0.5 <= v <= 3:
        raise argparse.ArgumentTypeError("must be between 0.5 and 3")
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="moonstack",
        description="Align and stack a burst of RAW moon photos into one clean image.",
    )
    ap.add_argument("folder", type=Path, help="folder containing the RAW files")
    ap.add_argument("-o", "--out", type=Path, help="output folder (default: FOLDER/moonstack_out)")
    ap.add_argument("--rerender", action="store_true",
                    help="reuse the stack saved by a previous run and only redo colour, sharpening and "
                         "rendering (--wb, --psf-sigma, --sharpen-iters, --saturation, --no-colour-flatten, --pair)")  # fmt: skip
    ap.add_argument("--keep", type=float, default=0.5,
                    help="fraction of frames to stack, sharpest first (default 0.5). Lower is sharper, higher is less grainy")  # fmt: skip
    ap.add_argument("--local-keep", type=float, default=0.5,
                    help="regional frame selection: in each small region of the moon, stack only the fraction of "
                         "the kept frames that were sharpest there (default 0.5; 1 turns it off)")  # fmt: skip
    ap.add_argument("--scale", type=_scale, default="auto",
                    help="output pixels per sensor pixel. The default, 'auto', measures the blur and picks the grid "
                         "on which the sharpened image renders smoothly; 1, 1.5 or 2 fixes it")  # fmt: skip
    ap.add_argument("--radius", type=float, default=0.75,
                    help="stacking kernel half-width in sensor pixels, 0.3-1; smaller is sharper but noisier (default 0.75)")  # fmt: skip
    ap.add_argument("--kappa", type=float, default=3.0,
                    help="outlier rejection threshold in sigmas; 0 disables (default 3)")  # fmt: skip
    ap.add_argument("--reference", help="file name of the frame to align to (default: the sharpest)")
    ap.add_argument("--no-local", action="store_true", help="skip the local turbulence correction")
    ap.add_argument("--no-chroma-align", action="store_true", help="do not register red and blue to green")
    ap.add_argument("--wb", choices=["daylight", "as-shot"], default="daylight",
                    help="white balance (default daylight: the moon is sunlit rock)")  # fmt: skip
    ap.add_argument("--psf-sigma", type=float, help="blur to remove, in output pixels (default: measured on the limb)")
    ap.add_argument("--sharpen-iters", type=int, default=25, help="deconvolution iterations at 1x scale; 0 disables (default 25)")
    ap.add_argument("--saturation", type=float, default=12.0, help="colour boost of the mineral image (default 12)")
    ap.add_argument("--no-colour-flatten", action="store_true",
                    help="keep the smooth colour slope across the disc in the mineral image")  # fmt: skip
    ap.add_argument("--pair", action="store_true",
                    help="also write moon_pair.jpg: the natural and mineral images side by side, cropped to the moon")  # fmt: skip
    ap.add_argument("--no-partial-decode", action="store_true",
                    help="always decode whole frames with LibRaw (slower; for troubleshooting)")  # fmt: skip
    ap.add_argument("-j", "--jobs", type=int, default=0, help="worker count (default: all cores)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    if not a.folder.is_dir():
        ap.error(f"{a.folder} is not a folder")
    if not 0 < a.keep <= 1:
        ap.error("--keep must be between 0 and 1")
    if a.local_keep is not None and not 0.05 <= a.local_keep <= 1:
        ap.error("--local-keep must be between 0.05 and 1")

    from .pipeline import Config, rerender, run, write_outputs

    cfg = Config(
        keep=a.keep, local_keep=a.local_keep, scale=a.scale, radius=a.radius, kappa=a.kappa, local=not a.no_local, chroma_align=not a.no_chroma_align, wb=a.wb,
        psf_sigma=a.psf_sigma, reference=a.reference, sharpen_iters=a.sharpen_iters, saturation=a.saturation, flatten_colour=not a.no_colour_flatten, pair=a.pair, partial_decode=not a.no_partial_decode, jobs=a.jobs, quiet=a.quiet,
    )  # fmt: skip
    out = a.out or a.folder / "moonstack_out"
    try:
        res = rerender(out, cfg) if a.rerender else run(a.folder, cfg)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    files = write_outputs(res, out, cfg, save_stack=not a.rerender)
    if not a.quiet:
        print(f"wrote {len(files)} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
