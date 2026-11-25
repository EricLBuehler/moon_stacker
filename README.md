# moonstack

Aligns and stacks a burst of RAW moon photos into one clean, sharp image.

```
uv run moonstack /path/to/folder/of/raws
```

Results go to `moonstack_out/` inside that folder (or `-o DIR`):

| File | What it is |
| --- | --- |
| `moon_linear.tif` | The stack itself: 32-bit float, linear, white-balanced, unsharpened. Edit this one. |
| `moon_natural.tif` / `.jpg` | Deconvolved and tone-mapped, 16-bit sRGB. |
| `moon_mineral.tif` / `.jpg` | Same, with the moon's faint colour differences exaggerated. |
| `report.json` | Per-frame sharpness, position, rotation, and whether it was stacked. |
| `moon_stack.npz` | The saved stack, so `--rerender` can redo the finishing without re-stacking. |

## Tuning the look quickly

A full run saves the stack. After that, `--rerender` redoes only colour, sharpening and rendering,
in about a second:

```
uv run moonstack /path/to/folder/of/raws --rerender --saturation 30
```

It honours `--saturation`, `--sharpen-iters`, `--psf-sigma`, `--wb` and `--no-colour-flatten`.
Stacking options (`--keep`, `--scale`, `--radius`, ...) need a full run.

## How it works

1. **Decode** each RAW to its undemosaiced Bayer mosaic (LibRaw via `rawpy`), find the moon, crop around it.
   Sony tiled lossless ARW files are decoded only around the moon (about ten times faster); this is
   verified bit-for-bit against LibRaw on the first file of every run, and anything else falls back to
   LibRaw. `--no-partial-decode` forces the slow path.
2. **Score** every frame for sharpness; the sharpest `--keep` fraction is stacked.
3. **Align** to the sharpest frame: shift and rotation to sub-pixel accuracy, then a local
   warp field measured on a grid of patches to undo atmospheric distortion.
4. **Stack** by Bayer drizzle: each raw red, green and blue sample is placed at its true aligned
   position, so no colour value is ever interpolated. Outliers are rejected against a noise model
   fitted to the data.
5. **Register colours**: red and blue are shifted against green by the lens (by an amount that depends on
   where the moon sits on the sensor) and by the atmosphere. Both are measured from the stack and corrected
   per frame while stacking.
6. **Finish**: measure the blur on the lunar limb, deconvolve brightness (Richardson-Lucy), and take colour
   from channels whose blurs have been matched on the limb. The mineral image also removes the smooth
   colour slope across the disc and holds the boost back at hard edges, where colour is unreliable.

## Colour

For Sony ARW files the daylight white balance and colour matrix are read from the camera's own metadata
via `exiftool` (`brew install exiftool`). Without it, colour falls back to LibRaw's tables, which were
badly off for the a7R VI.

## Options worth knowing

- `--pair` also writes `moon_pair.jpg`: natural and mineral side by side, cropped to the moon. Works with `--rerender`.
- `--keep 0.5` fraction of frames stacked, sharpest first. Lower is sharper, higher is less grainy.
  With a short burst (under a hundred frames or so) try `--keep 0.8`.
- `--local-keep 0.5` regional frame selection (`--local-keep 1` turns it off). Turbulence blurs different parts of the moon in
  different frames, so each small region is stacked from the frames that were sharpest *there*, blended
  smoothly. It only acts where frames measurably differ in sharpness (by more than a few percent), and
  uses every frame elsewhere. Sharper, a little noisier.
- `--scale auto` (the default) measures the blur and picks the output grid on which the sharpened image
  renders smoothly, often 1.5x the sensor grid. `--scale 1` or `--scale 1.5` fixes it.
- `--radius 0.75` stacking kernel width. Smaller is sharper but noisier.
- `--sharpen-iters 25`, `--psf-sigma` deconvolution strength and blur override.
- `--saturation 12` colour boost of the mineral image.
- `--wb daylight|as-shot`, `--no-local`, `--no-chroma-align`, `--kappa`, `--reference FILE`. See `--help`.

## Testing

```
uv run pytest                           # end-to-end on a synthetic burst
MOONSTACK_SONY_SAMPLE=/path/to/file.ARW uv run pytest   # also check the Sony tile decoder against LibRaw
uv run python tests/bench_synth.py      # accuracy against ground truth vs a conventional stack
```

`moonstack-synth DIR` writes a synthetic burst (procedural moon with simulated drift, shake,
turbulence and sensor noise) that `moonstack` can process like real files.
