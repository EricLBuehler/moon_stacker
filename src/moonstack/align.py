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
        # Patches that hold any of the moon, including its edge. The edge is no use
        # for measuring shifts (it only pins one direction) but shows blur well.
        self.lit = cover > 0.05
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


def region_sharpness(plane: np.ndarray, W: np.ndarray, aligner: LocalAligner, detail: float = 0.7) -> np.ndarray:
    """Sharpness of one frame around each node of the aligner's grid.

    Same measure as `sharpness` (edge strength relative to brightness), but
    averaged over a neighbourhood the size of an alignment patch. It is
    computed in the frame's own pixel grid and only the finished, smooth map is
    looked up at the nodes: resampling the image first would blur it by an
    amount that depends on the frame's sub-pixel position.
    """
    g = cv2.GaussianBlur(plane, (0, 0), detail)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    window = aligner.patch / 4.0
    edges = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), window)
    bright = cv2.GaussianBlur(g * g, (0, 0), window)
    gh, gw = aligner.grid_shape
    c0 = (aligner.patch - 1) / 2.0
    ys, xs = np.mgrid[:gh, :gw].astype(np.float32) * aligner.step + c0
    mx = W[0, 0] * xs + W[0, 1] * ys + W[0, 2]
    my = W[1, 0] * xs + W[1, 1] * ys + W[1, 2]
    at = lambda m: cv2.remap(m, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # noqa: E731
    return at(edges) / (at(bright) + (1e-3 * aligner.norm) ** 2)


def selection_weights(qa: np.ndarray, qb: np.ndarray, valid: np.ndarray, keep: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame, per-node stacking weights that favour each region's sharpest frames.

    `qa` and `qb` (frames, gh, gw) are the regional sharpness measured on two
    sets of sensor pixels that share no pixel, so their noise is independent.
    Where the two agree on which frames are sharp, the ranking is real and the
    sharpest `keep` fraction of frames is used. Where they disagree the ranking
    is only noise (smooth ground), and every frame is used. The same goes for
    where frames differ by only a few percent (steady air): too little to be
    worth giving up frames for.

    Returns the weights (frames, gh, gw), the per-node trust in the ranking,
    and the per-node spread of real sharpness between frames (0.1 = 10%).
    """
    n = qa.shape[0]
    a = qa - qa.mean(axis=0)
    b = qb - qb.mean(axis=0)
    shared = (a * b).sum(axis=0)  # what the two measurements agree on
    r = shared / (np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0)) + 1e-20)
    # Agreement between two half-measurements -> reliability of their average.
    trust = np.clip(2 * r / (1 + r), 0, 1)
    spread = np.sqrt(np.maximum(shared, 0) / n) / (0.5 * (qa.mean(axis=0) + qb.mean(axis=0)) + 1e-20)
    # Small differences (below about 2%) come from things other than blur,
    # such as the slight local stretching that turbulence also causes.
    trust = trust * np.clip((spread - 0.02) / 0.03, 0, 1)
    trust = np.where(valid, trust, 0).astype(np.float32)
    trust = cv2.GaussianBlur(trust, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)

    q = qa + qb
    rank = np.argsort(np.argsort(-q, axis=0), axis=0) / max(n - 1, 1)  # 0 = sharpest
    ramp = 0.1  # soft edge of the cut, as a fraction of the frames
    chosen = np.clip((keep + ramp - rank) / (2 * ramp), 0, 1).astype(np.float32)
    weights = np.empty_like(chosen)
    for i in range(n):
        # Blending between neighbouring regions keeps the mosaic seamless.
        smooth = cv2.GaussianBlur(chosen[i], (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
        weights[i] = 1 - trust * (1 - smooth)
    return weights, trust, spread.astype(np.float32)
