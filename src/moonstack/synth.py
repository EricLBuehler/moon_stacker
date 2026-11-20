"""Synthetic moon bursts with known ground truth, for testing the pipeline.

Renders a procedural cratered, faintly coloured moon, then simulates a burst:
drift, hand shake, slight rotation, atmospheric warping and blur that vary
from frame to frame, Bayer sampling, shot noise and read noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

OVERSAMPLE = 2
CFA = np.array([[0, 1], [1, 2]])  # RGGB
BLACK = 512.0
WHITE = 16383.0
WB = np.array([2.0, 1.0, 1.5])  # camera multipliers that the pipeline must undo


def _fnoise(shape, sigmas, rng, power=0.0):
    out = np.zeros(shape, np.float32)
    for s in sigmas:
        n = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), s)
        out += n / n.std() * s**power
    return out / out.std()


def make_truth(size: int, radius: float, rng) -> np.ndarray:
    """Linear RGB moon, (size, size, 3), true colours, peak about 0.8."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    c = (size - 1) / 2.0
    x, y = (xx - c) / radius, (yy - c) / radius
    r2 = x * x + y * y
    mu = np.sqrt(np.clip(1 - r2, 0, 1))

    height = 2.0 * _fnoise((size, size), [1.5, 3, 6, 12, 24], rng, power=1.0)
    for _ in range(int(900 * (radius / 600) ** 2) + 200):
        R = min(3.0 * rng.uniform(0.02, 1) ** -0.8, 0.12 * radius)
        cx, cy = rng.uniform(c - radius, c + radius, 2)
        m = int(R * 2 + 3)
        x0, x1 = int(max(cx - m, 0)), int(min(cx + m + 1, size))
        y0, y1 = int(max(cy - m, 0)), int(min(cy + m + 1, size))
        if x1 <= x0 or y1 <= y0:
            continue
        d = np.hypot(xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy) / R
        height[y0:y1, x0:x1] += R * (-0.6 * np.clip(1 - d * d, 0, 1) + 0.35 * np.exp(-(((d - 1) / 0.18) ** 2)))
    shade = np.clip(1 + 0.5 * np.gradient(height, axis=1), 0.15, 2.2)

    albedo = 0.6 + 0.2 * np.tanh(1.5 * _fnoise((size, size), [40, 80], rng)) + 0.04 * _fnoise((size, size), [2, 4, 8], rng)
    a = np.radians(35)
    lam = np.clip(x * np.sin(a) + mu * np.cos(a), 0, 1) ** 0.6
    I = albedo * shade * lam * (r2 < 1)

    c1 = _fnoise((size, size), [25, 50], rng)
    c2 = _fnoise((size, size), [15, 40], rng)
    rgb = I[..., None] * np.stack([1.06 * (1 + 0.05 * c1), np.ones_like(c1), 0.90 * (1 - 0.05 * c1 + 0.04 * c2)], -1)
    return (rgb * (0.8 / np.percentile(rgb, 99.9))).astype(np.float32)


def generate(out: Path, frames=40, sensor=(1000, 1400), diameter=600, seed=0, peak_dn=7000.0,
             drift=(3.0, 1.2), jitter=6.0, seeing=0.4, blur=(0.7, 1.1), gain_e_per_dn=0.5, read_dn=3.0):  # fmt: skip
    """Write `frames` synthetic .npz raw files plus truth.npz into `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    H, W = sensor
    S = int(diameter * 1.15) // 2 * 2  # truth size in sensor pixels
    hi = make_truth(S * OVERSAMPLE, diameter / 2 * OVERSAMPLE, rng)
    levels = np.linspace(blur[0], blur[1], 5)
    blurred = []
    for s in levels:
        b = cv2.GaussianBlur(hi, (0, 0), s * OVERSAMPLE)
        blurred.append(cv2.blur(b, (OVERSAMPLE, OVERSAMPLE), anchor=(0, 0)))  # pixel aperture
    truth = cv2.resize(hi, (S, S), interpolation=cv2.INTER_AREA)

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    ct = (S - 1) / 2.0
    start = np.array([W / 2 - drift[0] * frames / 2, H / 2 - drift[1] * frames / 2])
    params = []
    for i in range(frames):
        centre = start + np.array(drift) * i + rng.normal(0, jitter, 2)
        theta = np.radians(0.02 * i + rng.normal(0, 0.02))
        lvl = int(rng.integers(len(levels)))
        g = 1 + rng.normal(0, 0.02)
        lo = rng.standard_normal((H // 32 + 2, W // 32 + 2, 2)).astype(np.float32)
        lo = cv2.GaussianBlur(lo, (0, 0), 1.0)
        D = cv2.resize(lo / lo.std() * seeing, (W, H), interpolation=cv2.INTER_CUBIC)
        co, si = np.cos(theta), np.sin(theta)
        px, py = xx - centre[0], yy - centre[1]
        qx = co * px - si * py + ct + D[..., 0]
        qy = si * px + co * py + ct + D[..., 1]
        # Sensor pixel centre q covers oversampled pixels 2q and 2q+1; the box
        # filter above is anchored at its top-left, so sample at 2q.
        img = cv2.remap(blurred[lvl], (OVERSAMPLE * qx).astype(np.float32), (OVERSAMPLE * qy).astype(np.float32), cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # fmt: skip
        cam = np.clip(img, 0, None) / WB.astype(np.float32) * (peak_dn / 0.8 * g)
        mosaic = np.empty((H, W), np.float32)
        for ty in range(2):
            for tx in range(2):
                mosaic[ty::2, tx::2] = cam[ty::2, tx::2, CFA[ty, tx]]
        e = rng.poisson(mosaic * gain_e_per_dn).astype(np.float32) / gain_e_per_dn
        dn = e + rng.normal(0, read_dn, e.shape).astype(np.float32) + BLACK
        bayer = np.clip(np.round(dn), 0, WHITE).astype(np.uint16)
        np.savez(out / f"synth_{i:04d}.npz", bayer=bayer, cfa=CFA, black=np.full((2, 2), BLACK), white=WHITE, wb=WB)
        params.append([centre[0], centre[1], theta, levels[lvl], g])
    np.savez(out / "truth.npz", truth=truth, params=np.array(params), peak_dn=peak_dn)
    return truth, np.array(params)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="moonstack-synth", description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--diameter", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    generate(a.out, a.frames, (a.height, a.width), a.diameter, a.seed)
    print(f"wrote {a.frames} frames to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
