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
    frames,  # uint16 (N, H, W): every frame's raw mosaic
    idx,  # int64 (K,): which frames to add, in order
    cfa,  # int64 (2, 2)
    black,  # float32 (2, 2)
    gains,  # float32 (N,) brightness normalisation per frame
    Ms,  # float64 (N, 2, 3) reference -> frame, full-res raw pixel coords
    chans,  # float64 (N, 3, 3) per colour: scale, x offset, y offset in frame coords
    fields,  # float32 (N, gh, gw, 2) local (dx, dy) in full-res pixels
    weights,  # float32 (N, gh, gw) how much each frame counts, on the same grid; (N, 1, 1) = not used
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
    H, W = frames.shape[1], frames.shape[2]
    Ho, Wo = num.shape[0], num.shape[1]
    gh, gw = fields.shape[1], fields.shape[2]
    weighted = weights.shape[1] == gh and weights.shape[2] == gw and gh > 1
    inv_s = 1.0 / scale
    inv_r = 1.0 / radius
    # A faint full-width kernel guarantees coverage where the sharp kernel
    # happens to find no sample of some colour.
    eps = 1e-3
    rejected = 0
    # Rows outside, frames inside: one output row's running totals stay in the
    # CPU cache while all the frames are added to it. The other way round,
    # every frame drags the whole set of totals through memory.
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
        ln = np.zeros(3, np.float32)  # one frame's contribution to one pixel
        ld = np.zeros(3, np.float32)
        lq = np.zeros(3, np.float32)
        for k in range(idx.shape[0]):
            f = idx[k]
            raw = frames[f]
            field = fields[f]
            weight = weights[f]
            M = Ms[f]
            chan = chans[f]
            gain = gains[f]
            for u in range(Wo):
                for c in range(3):
                    ln[c] = 0.0
                    ld[c] = 0.0
                    lq[c] = 0.0
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
                fw = 1.0
                if weighted:
                    fw = w00 * weight[iy0, ix0] + w01 * weight[iy0, ix1] + w10 * weight[iy1, ix0] + w11 * weight[iy1, ix1]
                    if fw <= 0.0:
                        continue
                px = rx + dx
                py = ry + dy
                fx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
                fy = M[1, 0] * px + M[1, 1] * py + M[1, 2]
                # The 2x2 block of raw samples around a position holds exactly
                # one sample from each site of the colour pattern, so go
                # straight to it instead of testing all four for every colour.
                for site in range(4):
                    sy_par = site >> 1
                    sx_par = site & 1
                    ch = cfa[sy_par, sx_par]
                    # Each colour is sampled at its own position: the lens and the
                    # atmosphere displace red and blue slightly relative to green.
                    qx = chan[ch, 0] * fx + chan[ch, 1]
                    qy = chan[ch, 0] * fy + chan[ch, 2]
                    x0 = int(np.floor(qx))
                    y0 = int(np.floor(qy))
                    xx = x0 + ((x0 ^ sx_par) & 1)
                    yy = y0 + ((y0 ^ sy_par) & 1)
                    if xx < 0 or xx >= W or yy < 0 or yy >= H:
                        continue
                    ax = abs(xx - qx)
                    ay = abs(yy - qy)
                    sx = max(1.0 - ax * inv_r, 0.0)
                    sy = max(1.0 - ay * inv_r, 0.0)
                    w = (sx * sy + eps * (1.0 - ax) * (1.0 - ay)) * fw
                    if w <= 0.0:
                        continue
                    val = (raw[yy, xx] - black[sy_par, sx_par]) * gain
                    if use_rej and abs(val - mean[v, u, ch]) > kappa * sigma[v, u, ch]:
                        rejected += 1
                        continue
                    ln[ch] += w * val
                    ld[ch] += w
                    lq[ch] += w * val * val
                for c in range(3):
                    if ld[c] > 0.0:
                        num[v, u, c] += ln[c]
                        den[v, u, c] += ld[c]
                        sq[v, u, c] += lq[c]
                        if not use_rej:
                            vf = ln[c] / ld[c]
                            if vf > vmax[v, u, c]:
                                vmax[v, u, c] = vf
                                wmax[v, u, c] = ld[c]
                            if vf < vmin[v, u, c]:
                                vmin[v, u, c] = vf
                                wmin[v, u, c] = ld[c]
    return rejected


# Frames per call of the kernel: enough to keep each row's totals in cache for
# many frames, few enough that a progress bar still moves.
BATCH = 16


_NO_SHIFT = np.array([[1.0, 0.0, 0.0]] * 3)


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

    def add(self, frames, idx, gains, Ms, fields, f_off, f_step, reject=False, kappa=3.0, chans=None, weights=None):
        """Add frames[i] for each i in idx. All per-frame arrays are indexed like `frames`."""
        n = len(frames)
        if chans is None:
            chans = np.broadcast_to(_NO_SHIFT, (n, 3, 3))
        if weights is None:
            weights = np.ones((n, 1, 1), np.float32)
        self.rejected += _accumulate(
            frames, np.asarray(idx, dtype=np.int64), self.cfa, self.black,
            np.ascontiguousarray(gains, dtype=np.float32), np.ascontiguousarray(Ms, dtype=np.float64),
            np.ascontiguousarray(chans, dtype=np.float64),
            np.ascontiguousarray(fields, dtype=np.float32), np.ascontiguousarray(weights, dtype=np.float32),
            float(f_off), float(f_step),
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


def measure_channel_shift(stack: np.ndarray, scale: float) -> np.ndarray | None:
    """How red and blue are displaced relative to green in a stack.

    Returns (3, 3): per colour, magnification and (x, y) offset such that the
    detail green shows at raw position p appears at ``mag * p + offset`` in
    that colour. None if the measurement fails or looks implausible.
    """
    g = stack[..., 1]
    top = float(np.percentile(g, 99.5))
    if top <= 0:
        return None
    a = (g / top).astype(np.float32)
    out = np.array([[1.0, 0.0, 0.0]] * 3)
    # Converges long before 100 iterations; a tighter tolerance only burns time.
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-7)
    for ch in (0, 2):
        b = (stack[..., ch] / (float(np.percentile(stack[..., ch], 99.5)) + 1e-9)).astype(np.float32)
        W = np.eye(2, 3, dtype=np.float32)
        try:
            _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_TRANSLATION, crit, None, 5)
            _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_AFFINE, crit, None, 5)
        except cv2.error:
            return None
        mag = 0.5 * float(W[0, 0] + W[1, 1])
        # Output-pixel coordinates -> raw-pixel coordinates.
        t = (mag * (0.5 * scale - 0.5) + W[:, 2].astype(np.float64) + 0.5) / scale - 0.5
        if abs(mag - 1) > 0.003 or np.abs(t).max() > 0.003 * max(g.shape) / scale + 4:
            return None
        out[ch] = (mag, t[0], t[1])
    return out
