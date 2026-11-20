"""Bayer drizzle: stack raw colour samples without demosaicing.

For every output pixel we work out where it lands in each frame's raw mosaic
and take the raw samples next to that position. A sample only ever contributes
to its own colour channel, so no colour value is ever interpolated from its
neighbours. Because the frames are offset from each other by random sub-pixel
amounts, every output pixel ends up with real red, green and blue measurements.
"""

from __future__ import annotations

import cv2
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _accumulate(
    raw,  # uint16 (H, W)
    cfa,  # int64 (2, 2)
    black,  # float32 (2, 2)
    gain,  # brightness normalisation for this frame
    M,  # float64 (2, 3) reference -> frame, full-res raw pixel coords
    field,  # float32 (gh, gw, 2) local (dx, dy) in full-res pixels
    f_off,
    f_step,
    scale,  # output pixels per raw pixel
    radius,  # kernel half-width in raw pixels (<= 1)
    num,
    den,
    sq,  # float32 (Ho, Wo, 3) accumulators
    vmax,
    wmax,
    vmin,
    wmin,  # float32 (Ho, Wo, 3): each pixel's most extreme frame and its weight
    use_rej,
    mean,
    sigma,
    kappa,
):
    H, W = raw.shape
    Ho, Wo = num.shape[0], num.shape[1]
    gh, gw = field.shape[0], field.shape[1]
    inv_s = 1.0 / scale
    inv_r = 1.0 / radius
    # A faint full-width kernel guarantees coverage where the sharp kernel
    # happens to find no sample of some colour.
    eps = 1e-3
    rejected = 0
    for v in prange(Ho):
        ry = (v + 0.5) * inv_s - 0.5
        gy = (ry - f_off) / f_step
        if gy < 0.0:
            gy = 0.0
        elif gy > gh - 1:
            gy = gh - 1.0
        iy0 = int(gy)
        iy1 = min(iy0 + 1, gh - 1)
        ty = gy - iy0
        ln = np.zeros(3, np.float32)  # this frame's contribution to one pixel
        ld = np.zeros(3, np.float32)
        for u in range(Wo):
            ln[:] = 0.0
            ld[:] = 0.0
            rx = (u + 0.5) * inv_s - 0.5
            gx = (rx - f_off) / f_step
            if gx < 0.0:
                gx = 0.0
            elif gx > gw - 1:
                gx = gw - 1.0
            ix0 = int(gx)
            ix1 = min(ix0 + 1, gw - 1)
            tx = gx - ix0
            w00 = (1 - tx) * (1 - ty)
            w01 = tx * (1 - ty)
            w10 = (1 - tx) * ty
            w11 = tx * ty
            dx = w00 * field[iy0, ix0, 0] + w01 * field[iy0, ix1, 0] + w10 * field[iy1, ix0, 0] + w11 * field[iy1, ix1, 0]
            dy = w00 * field[iy0, ix0, 1] + w01 * field[iy0, ix1, 1] + w10 * field[iy1, ix0, 1] + w11 * field[iy1, ix1, 1]
            px = rx + dx
            py = ry + dy
            fx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
            fy = M[1, 0] * px + M[1, 1] * py + M[1, 2]
            x0 = int(np.floor(fx))
            y0 = int(np.floor(fy))
            for yy in range(y0, y0 + 2):
                if yy < 0 or yy >= H:
                    continue
                ay = abs(yy - fy)
                sy = 1.0 - ay * inv_r
                if sy < 0.0:
                    sy = 0.0
                for xx in range(x0, x0 + 2):
                    if xx < 0 or xx >= W:
                        continue
                    ax = abs(xx - fx)
                    sx = 1.0 - ax * inv_r
                    if sx < 0.0:
                        sx = 0.0
                    w = sx * sy + eps * (1.0 - ax) * (1.0 - ay)
                    if w <= 0.0:
                        continue
                    c = cfa[yy & 1, xx & 1]
                    val = (raw[yy, xx] - black[yy & 1, xx & 1]) * gain
                    if use_rej and abs(val - mean[v, u, c]) > kappa * sigma[v, u, c]:
                        rejected += 1
                        continue
                    ln[c] += w * val
                    ld[c] += w
                    sq[v, u, c] += w * val * val
            for c in range(3):
                if ld[c] > 0.0:
                    num[v, u, c] += ln[c]
                    den[v, u, c] += ld[c]
                    if not use_rej:
                        vf = ln[c] / ld[c]
                        if vf > vmax[v, u, c]:
                            vmax[v, u, c] = vf
                            wmax[v, u, c] = ld[c]
                        if vf < vmin[v, u, c]:
                            vmin[v, u, c] = vf
                            wmin[v, u, c] = ld[c]
    return rejected


class Stacker:
    """Multi-pass accumulator: plain mean first, then outlier-rejecting passes."""

    def __init__(self, shape, cfa, black, scale=1.0, radius=0.75):
        h, w = shape
        self.out_shape = (int(round(h * scale)), int(round(w * scale)), 3)
        self.cfa = np.ascontiguousarray(cfa, dtype=np.int64)
        self.black = np.ascontiguousarray(black, dtype=np.float32)
        self.scale = float(scale)
        self.radius = float(min(max(radius, 0.3), 1.0))
        self.mean = np.zeros(self.out_shape, np.float32)
        self.sigma = np.ones(self.out_shape, np.float32)
        self.noise_model = []
        self.reset()

    def reset(self):
        z = lambda v=0.0: np.full(self.out_shape, v, np.float32)  # noqa: E731
        self.num, self.den, self.sq = z(), z(), z()
        self.vmax, self.wmax, self.vmin, self.wmin = z(-np.inf), z(), z(np.inf), z()
        self.rejected = 0

    def add(self, raw, gain, M, field, f_off, f_step, reject=False, kappa=3.0):
        self.rejected += _accumulate(
            raw, self.cfa, self.black, np.float32(gain), np.ascontiguousarray(M, dtype=np.float64),
            np.ascontiguousarray(field, dtype=np.float32), float(f_off), float(f_step),
            self.scale, self.radius, self.num, self.den, self.sq,
            self.vmax, self.wmax, self.vmin, self.wmin,
            reject, self.mean, self.sigma, np.float32(kappa),
        )  # fmt: skip

    def result(self) -> np.ndarray:
        """Weighted mean; pixels with no surviving samples keep the previous estimate."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self.num / self.den
        empty = self.den <= 0
        out[empty] = self.mean[empty]
        return out

    def _trimmed(self) -> np.ndarray:
        """Mean without each pixel's brightest and darkest frame.

        One bad frame (aircraft, satellite, cosmic ray) can then not shift the
        value that the rejection pass compares against.
        """
        d = self.den - self.wmax - self.wmin
        hit = self.den > 0  # untouched pixels still hold the +-inf sentinels
        n = self.num - self.wmax * np.where(hit, self.vmax, 0) - self.wmin * np.where(hit, self.vmin, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(d > 0.2 * self.den, n / d, self.num / self.den)
        out[self.den <= 0] = 0
        return out.astype(np.float32)

    def refine(self):
        """After a rejecting pass: use its result as the next comparison image."""
        self.mean = self.result()
        self.reset()

    def finish_first_pass(self):
        """Freeze the mean and expected scatter used to reject outliers.

        The scatter each pixel shows across frames is too noisy to use
        directly (few samples per pixel, and an outlier inflates its own
        threshold). Instead we fit a three-term model to the whole image, per
        channel, and evaluate it everywhere:

            variance = read noise + a * brightness + (c * gradient)^2

        Shot noise grows with brightness; the gradient term covers small
        position errors, which matter only where the image changes quickly
        (the limb, crater rims).
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            plain = self.num / self.den
            var = self.sq / self.den - plain * plain
        self.mean = self._trimmed()
        ok = (self.den > 0) & np.isfinite(var)
        sigma = np.ones(self.out_shape, np.float32)
        self.noise_model = []
        for ch in range(3):
            m, v, good = self.mean[..., ch], var[..., ch], ok[..., ch]
            g2 = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3) ** 2
            g2 /= 64.0  # Sobel sums to 8x the per-pixel slope
            top = float(np.percentile(m[good], 99.5)) if good.any() else 0.0
            sky, lit = good & (m < 0.02 * top), good & (m > 0.1 * top)
            b = _trimmed_mean(v[sky]) if sky.sum() > 100 else 0.0
            a = c2 = 0.0
            if lit.sum() > 100:
                flat = lit & (g2 <= np.percentile(g2[lit], 50))
                a = max(_trimmed_mean((v[flat] - b) / m[flat]), 0.0)
                steep = lit & (g2 >= np.percentile(g2[lit], 90))
                c2 = max(_trimmed_mean((v[steep] - b - a * m[steep]) / g2[steep]), 0.0)
            model = b + a * np.maximum(m, 0) + c2 * g2
            sigma[..., ch] = np.sqrt(np.maximum(model, max(b, 1e-6)))
            self.noise_model.append((float(b), float(a), float(c2)))
        self.sigma = sigma
        self.reset()


def _trimmed_mean(x: np.ndarray) -> float:
    """Mean of the values below the 99th percentile (ignores contaminated pixels)."""
    if x.size == 0:
        return 0.0
    return float(x[x <= np.percentile(x, 99)].mean())
