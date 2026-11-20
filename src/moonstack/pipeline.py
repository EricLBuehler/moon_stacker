"""End-to-end pipeline: folder of RAW files -> stacked images."""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import align, finish
from .ingest import RAW_EXTS, SYNTH_EXT, half_luma, ingest_one, sony_colour


@dataclass
class Config:
    keep: float = 0.8  # fraction of frames to stack, sharpest first
    scale: float | str = 1.0  # output pixels per sensor pixel, or "auto" to choose from the measured blur
    radius: float = 0.75  # drizzle kernel half-width in sensor pixels
    kappa: float = 3.0  # outlier rejection threshold in sigmas (0 = off)
    local: bool = True  # correct turbulence warping with a local field
    chroma_align: bool = True  # register red and blue to green (lens colour fringing, atmospheric dispersion)
    wb: str = "daylight"  # daylight | as-shot
    psf_sigma: float | None = None  # override the measured blur
    reference: str | None = None  # file name to align everything to (default: sharpest)
    sharpen_iters: int = 25
    saturation: float = 12.0
    flatten_colour: bool = True  # remove the smooth colour slope across the disc in the mineral image
    jobs: int = 0  # 0 = all cores
    quiet: bool = False


@dataclass
class Result:
    stack_cam: np.ndarray  # linear camera RGB, data numbers
    linear: np.ndarray  # linear sRGB, white balanced
    sharp: np.ndarray  # linear sRGB after deconvolution
    psf_sigma: float
    frames: list[dict] = field(default_factory=list)  # per-frame report
    info: dict = field(default_factory=dict)
    colour: dict = field(default_factory=dict)  # what develop() needs besides the stack


STACK_FILE = "moon_stack.npz"
# Options that only affect develop(); a re-render may change these.
FINISH_OPTIONS = ("wb", "psf_sigma", "sharpen_iters", "saturation", "flatten_colour")


def develop(stack_cam: np.ndarray, colour: dict, cfg: Config, say=lambda *_a, **_k: None):
    """Stack -> linear image -> sharpened image. Everything after stacking.

    Returns (linear, sharp, info). Cheap enough to repeat with new settings.
    """
    scale = float(colour["scale"])
    daylight, as_shot = colour.get("wb_daylight"), colour.get("wb_as_shot")
    wb, wb_used = (daylight, "daylight") if cfg.wb == "daylight" else (as_shot, "as-shot")
    if wb is None:
        wb, wb_used = (as_shot, "as-shot") if as_shot is not None else ([1.0, 1.0, 1.0], "none")
    linear = finish.to_linear_srgb(stack_cam, colour["white_span"], wb, colour["matrix"])

    measured = finish.estimate_psf_sigma(finish.luminance(linear))
    if cfg.psf_sigma is not None:
        sigma = cfg.psf_sigma
    elif measured is not None:
        sigma = float(np.clip(0.9 * measured, 0.5 * scale, 3.0 * scale))
    else:
        sigma = 1.0 * scale
        say("could not measure blur on the limb; assuming a typical value")
    # Deconvolution converges more slowly when the blur spans more pixels, so
    # the iteration count is given for 1x and grows with the area of a pixel.
    sharp = finish.sharpen(linear, sigma, int(round(cfg.sharpen_iters * scale * scale)))
    info = {
        "psf_sigma_measured": measured,
        "psf_sigma_used": sigma,
        "white_balance": wb_used,
        "white_balance_multipliers": [float(v) for v in wb],
    }
    return linear, sharp, info


def rerender(out: Path, cfg: Config) -> Result:
    """Redo colour, sharpening and rendering from a saved stack (seconds, not a full run)."""
    path = Path(out) / STACK_FILE
    if not path.exists():
        raise RuntimeError(f"no saved stack at {path}; run moonstack on the RAW folder once first")
    z = np.load(path)
    saved = json.loads(str(z["meta"]))
    colour = saved["colour"]
    stack_cam = z["stack_cam"]
    say = (lambda *_a, **_k: None) if cfg.quiet else print
    linear, sharp, dev = develop(stack_cam, colour, cfg, say)
    info = {**saved["summary"], **dev}
    info["config"] = {**info.get("config", {}), **{k: getattr(cfg, k) for k in FINISH_OPTIONS}}
    return Result(stack_cam, linear, sharp, dev["psf_sigma_used"], saved["frames"], info, colour)


# Blur (Gaussian sigma, in output pixels) at which detail is comfortably
# sampled: finer than this and the pixel grid, not the optics, limits detail.
_WELL_SAMPLED_SIGMA = 1.3
# Deconvolution shrinks the blur to roughly this fraction (measured on
# synthetic data at the default strength), so the finished image holds finer
# detail than the stack it was made from and needs a finer grid to show it.
_SHARPENED_FRACTION = 0.6


def choose_scale(sigma: float | None, sharpening: bool = True) -> float:
    """Output scale for --scale auto, from the blur measured at sensor scale.

    Picks the grid on which the *finished* (sharpened) image is well sampled:
    enough pixels across its finest detail that edges render smoothly instead
    of blocky. Steps of 0.25, between 1x and 2x.
    """
    if sigma is None or sigma <= 0:
        return 1.0
    final = sigma * (_SHARPENED_FRACTION if sharpening else 1.0)
    return float(np.clip(np.round(_WELL_SAMPLED_SIGMA / final * 4) / 4, 1.0, 2.0))


def list_inputs(folder: Path) -> list[Path]:
    exts = RAW_EXTS | {SYNTH_EXT}
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith((".", "truth"))
    )  # fmt: skip


def _even(v: float) -> int:
    return int(round(v / 2.0)) * 2


def _ingest(paths, jobs, log):
    """Decode every file and gather moon-centred crops on a common canvas."""
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for r in tqdm(ex.map(ingest_one, [str(p) for p in paths]), total=len(paths),
                      desc="decode", disable=log is None, unit="frame"):  # fmt: skip
            results.append(r)

    good, dropped = [], []
    for r in results:
        if not r["ok"]:
            dropped.append({"file": Path(r["path"]).name, "reason": r["error"]})
        elif r["touches_border"]:
            dropped.append({"file": Path(r["path"]).name, "reason": "moon is cut off by the edge of the frame"})
        else:
            good.append(r)
    if len(good) < 2:
        reasons = "; ".join(sorted({d["reason"] for d in dropped})) or "no files"
        raise RuntimeError(f"need at least 2 usable frames, got {len(good)} ({reasons})")

    meta = good[0]["meta"]
    for r in good:
        m = r["meta"]
        if not (np.array_equal(m.cfa, meta.cfa) and r["sensor_shape"] == good[0]["sensor_shape"]):
            raise RuntimeError(f"{Path(r['path']).name}: sensor layout differs from the first frame")

    bw = float(np.median([r["bbox"][2] for r in good]))
    bh = float(np.median([r["bbox"][3] for r in good]))
    pad = 2 * (0.12 * max(bw, bh) + 48)
    cw = int(np.ceil((bw + pad) / 32.0)) * 32
    ch = int(np.ceil((bh + pad) / 32.0)) * 32

    n = len(good)
    tmp = tempfile.NamedTemporaryFile(prefix="moonstack_", suffix=".u16", delete=False)
    tmp.close()
    frames = np.memmap(tmp.name, dtype=np.uint16, mode="w+", shape=(n, ch, cw))
    black_tile = np.tile(np.round(meta.black).astype(np.uint16), (ch // 2, cw // 2))
    for i, r in enumerate(good):
        frames[i] = black_tile
        bx, by, bwi, bhi = r["bbox"]
        # Centre the canvas on the bounding box (even origin keeps the Bayer phase).
        X0 = _even(bx + bwi / 2.0 - cw / 2.0)
        Y0 = _even(by + bhi / 2.0 - ch / 2.0)
        ox, oy = r["origin"]
        crop = r["crop"]
        sx0, sy0 = max(X0, ox), max(Y0, oy)
        sx1, sy1 = min(X0 + cw, ox + crop.shape[1]), min(Y0 + ch, oy + crop.shape[0])
        frames[i, sy0 - Y0 : sy1 - Y0, sx0 - X0 : sx1 - X0] = crop[sy0 - oy : sy1 - oy, sx0 - ox : sx1 - ox]
        r["canvas_origin"] = (X0, Y0)
        r["crop"] = None
    return frames, tmp.name, good, dropped, meta


def run(folder: Path, cfg: Config) -> Result:
    log = None if cfg.quiet else print
    say = log or (lambda *_a, **_k: None)
    jobs = cfg.jobs or os.cpu_count() or 4
    t_start = time.time()

    paths = list_inputs(folder)
    if not paths:
        raise RuntimeError(f"no RAW files found in {folder}")
    say(f"{len(paths)} files in {folder}")

    frames, tmp_path, recs, dropped, meta = _ingest(paths, min(jobs, 12), log)
    try:
        return _process(frames, recs, dropped, meta, cfg, say, jobs, t_start)
    finally:
        del frames
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _process(frames, recs, dropped, meta, cfg, say, jobs, t_start) -> Result:
    n, ch, cw = frames.shape
    black = meta.black
    say(f"{n} usable frames, working area {cw}x{ch} px"
        + (f", {len(dropped)} dropped" if dropped else ""))  # fmt: skip
    clipped = float(np.mean([r["clipped_frac"] for r in recs]))
    if clipped > 1e-4:
        say(f"warning: {100 * clipped:.2f}% of pixels near the moon are saturated")

    threads = ThreadPoolExecutor(max_workers=jobs)
    progress = lambda it, desc: tqdm(it, total=n, desc=desc, disable=cfg.quiet, unit="frame")  # noqa: E731
    # Brightness images are capped a little above the moon's own peak, so a
    # stray bright streak cannot pull the alignment.
    cap = np.float32(1.25 * np.median([r["peak"] for r in recs]))
    luma = lambda i: np.minimum(half_luma(frames[i], black), cap)  # noqa: E731

    # 1. Sharpness and brightness of every frame.
    def score(i):
        L = luma(i)
        return align.sharpness(L), float(L.sum())

    scored = list(progress(threads.map(score, range(n)), "score"))
    quality = np.array([s[0] for s in scored])
    flux = np.array([s[1] for s in scored])
    order = np.argsort(-quality)
    ref_i = int(order[0])
    if cfg.reference:
        names = [Path(r["path"]).name for r in recs]
        if cfg.reference not in names:
            raise RuntimeError(f"reference frame {cfg.reference} is not among the usable frames")
        ref_i = names.index(cfg.reference)
        order = np.array([ref_i] + [i for i in order if i != ref_i])
    gains = float(np.median(flux)) / flux

    # 2. Global shift + rotation against the sharpest frame.
    ref0 = luma(ref_i) * np.float32(gains[ref_i])
    ref_norm = np.float32(1.0 / (np.percentile(ref0, 99.5) + 1e-6))
    hann = cv2.createHanningWindow((cw // 2, ch // 2), cv2.CV_32F)

    def do_global(i):
        if i == ref_i:
            return np.eye(2, 3, dtype=np.float32), 1.0
        return align.global_align(ref0 * ref_norm, luma(i) * (np.float32(gains[i]) * ref_norm), hann)

    glob = list(progress(threads.map(do_global, range(n)), "align"))
    warps = [g[0] for g in glob]
    cc = np.array([g[1] for g in glob])

    # Frames whose alignment failed outright are unusable.
    usable = cc > 0.5
    n_keep = max(2, int(round(cfg.keep * n)))
    kept = [int(i) for i in order if usable[i]][:n_keep]

    # 3. Local (turbulence) field against the mean of the best frames, which
    #    has the per-frame distortions averaged out.
    fields = [np.zeros((1, 1, 2), np.float32)] * n
    f_off, f_step = 0.0, 1e12
    local_conf = np.full(n, np.nan)
    if cfg.local:
        best = kept[: max(2, len(kept) // 2)]
        acc = np.zeros((ch // 2, cw // 2), np.float64)
        for w in threads.map(lambda i: align.warp_global(luma(i) * np.float32(gains[i]), warps[i]), best):
            acc += w
        ref1 = (acc / len(best)).astype(np.float32)
        disc = finish.disc_mask(ref1, 0.05)
        aligner = align.LocalAligner(ref1, disc)
        if aligner.valid.sum() >= 9:
            f_off, f_step = aligner.full_offset, aligner.full_step

            def do_local(i):
                if i not in kept_set:
                    return None
                return aligner.solve(luma(i) * np.float32(gains[i]), warps[i])

            kept_set = set(kept)
            for i, res in enumerate(progress(threads.map(do_local, range(n)), "local")):
                if res is not None:
                    fields[i] = 2.0 * res[0]  # half-res pixels -> full-res pixels
                    local_conf[i] = res[1]
        else:
            say("moon too small for local alignment; using global alignment only")
    threads.shutdown()

    # 4. Bayer drizzle in three passes: the first gives the mean and scatter that
    #    the following passes use to throw out outliers.
    from .stack import Stacker, measure_channel_shift  # imported late: numba start-up is slow

    full = [align.half_to_full(w) for w in warps]
    scale_note = None
    if cfg.scale == "auto":
        # A quick sensor-resolution stack tells us how blurred the data is.
        probe = Stacker((ch, cw), meta.cfa, black, 1.0, cfg.radius)
        for i in tqdm(kept, desc="probe", disable=cfg.quiet, unit="frame"):
            probe.add(frames[i], gains[i], full[i], fields[i], f_off, f_step)
        probe_sigma = finish.estimate_psf_sigma(probe.result()[..., 1])
        chosen = choose_scale(probe_sigma, cfg.sharpen_iters > 0)
        scale_note = {"requested": "auto", "blur_sigma_at_sensor_scale": probe_sigma}
        if probe_sigma is None:
            say("auto scale: could not measure blur on the limb; staying at 1x")
        elif chosen == 1.0:
            say(f"auto scale: blur is {probe_sigma:.2f} px; the sensor grid shows the sharpened image fully, staying at 1x")
        else:
            say(f"auto scale: blur is {probe_sigma:.2f} px; stacking at {chosen:g}x so the sharpened image has "
                f"enough pixels to render smoothly")  # fmt: skip
        cfg = replace(cfg, scale=chosen)
        del probe
    stacker = Stacker((ch, cw), meta.cfa, black, cfg.scale, cfg.radius)

    # Red and blue land slightly displaced from green: the lens magnifies each
    # colour differently (so the displacement depends on where the moon sits on
    # the sensor, which changes through the burst) and the atmosphere adds a
    # constant offset. Model both per colour, in sensor coordinates, as
    #     position_in_colour = mag * position_in_green + b
    origins = np.array([r["canvas_origin"] for r in recs], dtype=np.float64)
    mag, b = np.ones(3), np.zeros((3, 2))

    def chan_for(i):
        return np.column_stack([mag, (mag - 1)[:, None] * origins[i] + b])

    def one_pass(desc, reject):
        for i in tqdm(kept, desc=desc, disable=cfg.quiet, unit="frame"):
            stacker.add(frames[i], gains[i], full[i], fields[i], f_off, f_step, reject, cfg.kappa, chan_for(i))

    one_pass("stack", False)
    chroma_residual = None
    if cfg.chroma_align:
        # Where the reference area's origin falls on the sensor, averaged over the burst.
        t_bar = np.mean([full[i][:, 2] + origins[i] for i in kept], axis=0)
        for round_ in range(2):
            m = measure_channel_shift(stacker.result(), cfg.scale)
            if m is None:
                if round_ == 0:
                    say("could not measure colour-channel alignment; leaving channels as they are")
                break
            # Largest displacement anywhere in the working area, in pixels.
            chroma_residual = float(np.abs(m[:, 1:]).max() + np.abs(m[:, 0] - 1).max() * max(ch, cw) / 2)
            # Small, noisy stacks can show ~0.1 px of spurious shift; real lens and
            # atmosphere effects worth correcting are larger.
            if round_ == 1 or chroma_residual < 0.2:
                break
            # m describes the uncorrected stack; turn it into the sensor model.
            b = b + m[:, 1:] - (m[:, 0] - 1)[:, None] * t_bar
            mag = mag * m[:, 0]
            say(f"colour channels vs green: red {np.hypot(*m[0, 1:]):.2f} px, blue {np.hypot(*m[2, 1:]):.2f} px; correcting")
            stacker.reset()
            one_pass("restack", False)
    if cfg.kappa > 0:
        stacker.finish_first_pass()
        one_pass("reject", True)
        stacker.refine()
        one_pass("refine", True)
    stack_cam = stacker.result()
    rej_frac = stacker.rejected / max(1.0, 4.0 * len(kept) * stack_cam.shape[0] * stack_cam.shape[1])

    # 5. Colour and sharpening.
    # Prefer the camera's own colour data over LibRaw's tables, which lag new bodies.
    colour_source = "LibRaw"
    own = sony_colour(Path(recs[ref_i]["path"]))
    if own is not None:
        meta.wb_daylight, meta.cam_srgb = own
        colour_source = "camera metadata"
    if meta.cam_srgb is not None:
        matrix = meta.cam_srgb
    else:
        matrix = finish.camera_to_srgb(meta.cam_xyz)
        if meta.cam_xyz is None:
            colour_source = "none"
            say("note: no colour matrix for this camera; colours are white-balanced sensor RGB")
        elif Path(recs[ref_i]["path"]).suffix.lower() == ".arw":
            say("note: exiftool not found, so colour comes from LibRaw's tables, which can be wrong for new cameras")
    as_list = lambda a: None if a is None else [float(v) for v in a]  # noqa: E731
    colour = {
        "wb_daylight": as_list(meta.wb_daylight),
        "wb_as_shot": as_list(meta.wb_as_shot),
        "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
        "white_span": float(meta.white - black.mean()),
        "scale": cfg.scale,
        "flip": meta.flip,
    }
    linear, sharp, dev = develop(stack_cam, colour, cfg, say)
    sigma = dev["psf_sigma_used"]

    kept_set = set(kept)
    report = []
    for i, r in enumerate(recs):
        X0, Y0 = r["canvas_origin"]
        # Working-area coordinates of the reference -> sensor coordinates of this frame.
        M = full[i] + np.array([[0, 0, X0], [0, 0, Y0]])
        mx, my = M @ np.array([(cw - 1) / 2, (ch - 1) / 2, 1.0])
        report.append({
            "file": Path(r["path"]).name,
            "stacked": i in kept_set,
            "reference": i == ref_i,
            "sharpness": float(quality[i] / quality.max()),
            "brightness": float(1.0 / gains[i]),
            "moon_x": float(mx),
            "moon_y": float(my),
            "rotation_deg": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
            "matrix": M.tolist(),
            "align_score": float(cc[i]),
            "local_rms_px": float(np.sqrt(np.mean(fields[i] ** 2))) if i in kept_set else None,
        })  # fmt: skip
    for d in dropped:
        report.append({"file": d["file"], "stacked": False, "dropped": d["reason"]})

    xs = np.array([r["moon_x"] for r in report if "moon_x" in r])
    ys = np.array([r["moon_y"] for r in report if "moon_y" in r])
    info = {
        "frames_found": len(recs) + len(dropped),
        "frames_stacked": len(kept),
        "frames_dropped": len(dropped),
        "reference": Path(recs[ref_i]["path"]).name,
        "moon_drift_px": float(np.hypot(np.ptp(xs), np.ptp(ys))),
        "moon_size_px": [int(np.median([r["bbox"][2] for r in recs])), int(np.median([r["bbox"][3] for r in recs]))],
        "output_size": [int(stack_cam.shape[1]), int(stack_cam.shape[0])],
        "rejected_sample_fraction": float(rej_frac),
        **dev,
        "colour_source": colour_source,
        "channel_magnification": [float(v) for v in mag],
        "channel_offset_px": b.tolist(),
        "channel_residual_px": chroma_residual,
        "saturated_fraction": clipped,
        "flip": meta.flip,
        "seconds": round(time.time() - t_start, 1),
        "scale": cfg.scale,
        "scale_auto": scale_note,
        "config": asdict(cfg),
    }
    say(f"stacked {len(kept)} of {n} frames; moon drifted {info['moon_drift_px']:.0f} px across the burst; "
        f"blur sigma {sigma:.2f} px; {info['seconds']} s")  # fmt: skip
    return Result(stack_cam, linear, sharp, sigma, report, info, colour)


def write_outputs(res: Result, out: Path, cfg: Config, save_stack: bool = True) -> dict[str, Path]:
    import tifffile
    from PIL import Image, ImageCms

    out.mkdir(parents=True, exist_ok=True)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    files: dict[str, Path] = {}

    def tiff(name, arr, **kw):
        p = out / name
        try:
            tifffile.imwrite(p, arr, photometric="rgb", compression="zlib", **kw)
        except Exception:
            tifffile.imwrite(p, arr, photometric="rgb", **kw)
        files[name] = p

    def u16(x):
        return np.round(np.clip(x, 0, 1) * 65535).astype(np.uint16)

    def jpeg(name, x):
        p = out / name
        Image.fromarray(np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)).save(
            p, quality=95, subsampling=0, icc_profile=icc
        )
        files[name] = p

    flip = int(res.colour.get("flip", 0))
    f = lambda a: finish.apply_flip(a, flip)  # noqa: E731
    tiff("moon_linear.tif", f(res.linear).astype(np.float32))
    natural = f(finish.render_natural(res.sharp))
    mineral = f(finish.render_mineral(res.sharp, cfg.saturation, 2.0 * float(res.colour.get("scale", 1.0)), cfg.flatten_colour))
    tiff("moon_natural.tif", u16(natural), iccprofile=icc)
    tiff("moon_mineral.tif", u16(mineral), iccprofile=icc)
    jpeg("moon_natural.jpg", natural)
    jpeg("moon_mineral.jpg", mineral)
    p = out / "report.json"
    p.write_text(json.dumps({"summary": res.info, "frames": res.frames}, indent=2))
    files["report.json"] = p
    if save_stack:  # lets --rerender redo everything above without re-stacking
        p = out / STACK_FILE
        meta = json.dumps({"colour": res.colour, "summary": res.info, "frames": res.frames})
        np.savez(p, stack_cam=res.stack_cam.astype(np.float32), meta=np.array(meta))
        files[STACK_FILE] = p
    return files
