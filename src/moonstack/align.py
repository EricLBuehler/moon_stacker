"""Frame quality scoring and sub-pixel alignment.

All alignment is measured on half-resolution brightness images (one value per
2x2 Bayer tile), which need no demosaicing. Results are converted to
full-resolution raw pixel coordinates for the stacker.

Conventions
-----------
* A global warp ``W`` (2x3) maps reference coordinates to frame coordinates:
  ``frame(W @ [x, y, 1]) ~ ref(x, y)``.
* A local field ``d(x)`` is the residual left after the global warp, measured
  in reference geometry: ``warped(x + d(x)) ~ ref(x)``.
* So the frame position for reference point ``x`` is ``W @ (x + d(x))``.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def sharpness(luma: np.ndarray) -> float:
    """Gradient energy relative to brightness. Higher is sharper."""
    g = cv2.GaussianBlur(luma, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float((gx * gx + gy * gy).sum() / (float((g * g).sum()) + 1e-12))


def global_align(ref: np.ndarray, img: np.ndarray, hann: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigid (shift + rotation) warp from ref to img, on half-res images."""
    (dx, dy), _ = cv2.phaseCorrelate(ref, img, hann)
    W = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 80, 1e-7)
    try:
        cc, W = cv2.findTransformECC(ref, img, W, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    except cv2.error:
        return W, -1.0
    return W, float(cc)


def warp_global(img: np.ndarray, W: np.ndarray) -> np.ndarray:
    h, w = img.shape
    return cv2.warpAffine(
        img, W, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
    )


def half_to_full(W: np.ndarray) -> np.ndarray:
    """Convert a half-res warp to full-res raw pixel coordinates.

    Half-res pixel i sits at full-res coordinate 2*i + 0.5.
    """
    R = W[:, :2].astype(np.float64)
    t = 2.0 * W[:, 2].astype(np.float64) + 0.5 - R @ np.array([0.5, 0.5])
    return np.hstack([R, t[:, None]])


class LocalAligner:
    """Measures the residual warp of a frame on a grid of patches.

    Each patch is matched to the reference by windowed cross-correlation with a
    parabolic sub-pixel peak fit. The raw per-patch shifts are then turned into
    a smooth field by confidence-weighted Gaussian smoothing, so patches with
    little texture (or none: sky) borrow from their neighbours.
    """

    def __init__(
        self,
        ref: np.ndarray,
        disc: np.ndarray,
        patch: int = 32,
        step: int = 16,
        search: int = 4,
        smooth: float = 1.25,
    ):
        self.patch, self.step, self.search, self.smooth = patch, step, search, smooth
        self.shape = ref.shape
        self.norm = float(np.percentile(ref, 99.5)) + 1e-6
        win = np.hanning(patch + 2)[1:-1].astype(np.float32)
        self.win = np.outer(win, win)

        A = self._patches(self._prep(ref))
        self.FA = np.conj(np.fft.rfft2(A))
        self.eA = (A * A).sum(axis=(-1, -2))
        cover = sliding_window_view(disc.astype(np.float32), (patch, patch))[::step, ::step].mean(axis=(-1, -2))
        inside = cover > 0.9
        floor = 0.02 * float(np.median(self.eA[inside])) if inside.any() else np.inf
        self.valid = inside & (self.eA > floor)
        self.grid_shape = self.valid.shape

        # Affine that resamples the node grid onto the half-res pixel grid.
        c0 = (patch - 1) / 2.0
        self._node_to_pix = np.array([[step, 0, c0], [0, step, c0]], dtype=np.float32)
        yy, xx = np.mgrid[: ref.shape[0], : ref.shape[1]].astype(np.float32)
        self._xx, self._yy = xx, yy

    # Field geometry in full-resolution raw pixel coordinates.
    @property
    def full_offset(self) -> float:
        return float(self.patch - 0.5)

    @property
    def full_step(self) -> float:
        return float(2 * self.step)

    def _prep(self, luma: np.ndarray) -> np.ndarray:
        """Band-pass: drop pixel noise and large-scale shading."""
        x = luma / self.norm
        return cv2.GaussianBlur(x, (0, 0), 0.8) - cv2.GaussianBlur(x, (0, 0), 4.0)

    def _patches(self, img: np.ndarray) -> np.ndarray:
        p = sliding_window_view(img, (self.patch, self.patch))[:: self.step, :: self.step]
        p = p - p.mean(axis=(-1, -2), keepdims=True)
        return p * self.win

    def _measure(self, warped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw per-patch shift (dx, dy) and confidence."""
        B = self._patches(self._prep(warped))
        eB = (B * B).sum(axis=(-1, -2))
        c = np.fft.irfft2(self.FA * np.fft.rfft2(B), s=(self.patch, self.patch))
        c = np.fft.fftshift(c, axes=(-2, -1))
        h, s = self.patch // 2, self.search
        c = c[..., h - s : h + s + 1, h - s : h + s + 1]
        n = 2 * s + 1
        gy, gx = c.shape[:2]
        flat = c.reshape(gy, gx, n * n)
        idx = flat.argmax(axis=-1)
        iy, ix = idx // n, idx % n
        peak = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
        ok = (iy > 0) & (iy < n - 1) & (ix > 0) & (ix < n - 1)
        iyc, ixc = np.clip(iy, 1, n - 2), np.clip(ix, 1, n - 2)
        G, Hh = np.mgrid[:gy, :gx]

        def sub(m1, p1):
            den = m1 - 2 * peak + p1
            with np.errstate(divide="ignore", invalid="ignore"):
                o = 0.5 * (m1 - p1) / den
            return np.where(den < 0, np.clip(o, -0.5, 0.5), 0.0)

        ox = sub(c[G, Hh, iyc, ixc - 1], c[G, Hh, iyc, ixc + 1])
        oy = sub(c[G, Hh, iyc - 1, ixc], c[G, Hh, iyc + 1, ixc])
        d = np.stack([ix - s + ox, iy - s + oy], axis=-1).astype(np.float32)
        conf = peak / (np.sqrt(self.eA * eB) + 1e-12)
        conf = np.where(ok & self.valid, np.clip(conf, 0, 1), 0.0).astype(np.float32)
        return d, conf

    def _smooth(self, d: np.ndarray, conf: np.ndarray) -> np.ndarray:
        w = np.where(conf > 0.3, conf * conf, 0.0).astype(np.float32)
        num = cv2.GaussianBlur(d * w[..., None], (0, 0), self.smooth, borderType=cv2.BORDER_REPLICATE)
        den = cv2.GaussianBlur(w, (0, 0), self.smooth, borderType=cv2.BORDER_REPLICATE)
        return num / (den[..., None] + 0.02)

    def dense(self, field: np.ndarray) -> np.ndarray:
        """Node field -> per-pixel field on the half-res grid."""
        h, w = self.shape
        return cv2.warpAffine(
            field, self._node_to_pix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    def warp(self, img: np.ndarray, W: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Resample a frame into reference geometry using warp + local field."""
        d = self.dense(field)
        px, py = self._xx + d[..., 0], self._yy + d[..., 1]
        mx = W[0, 0] * px + W[0, 1] * py + W[0, 2]
        my = W[1, 0] * px + W[1, 1] * py + W[1, 2]
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def solve(self, img: np.ndarray, W: np.ndarray, iters: int = 2) -> tuple[np.ndarray, float]:
        """Local field for one frame, in half-res pixels on the node grid.

        Returns the field and the mean match confidence over textured patches.
        """
        field = np.zeros((*self.grid_shape, 2), dtype=np.float32)
        conf_mean = 0.0
        for _ in range(iters):
            d, conf = self._measure(self.warp(img, W, field))
            field = field + self._smooth(d, conf)
            if self.valid.any():
                conf_mean = float(conf[self.valid].mean())
        return field, conf_mean
