"""Turn the linear camera-RGB stack into viewable images.

Colour conversion, PSF estimation from the lunar limb, deconvolution, and the
two renderings (natural and saturated "mineral").
"""

from __future__ import annotations

import cv2
import numpy as np

_XYZ_FROM_SRGB = np.array(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
)
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def camera_to_srgb(cam_xyz: np.ndarray | None) -> np.ndarray:
    """Matrix from white-balanced camera RGB to linear sRGB (dcraw's method)."""
    if cam_xyz is None:
        return np.eye(3, dtype=np.float32)
    cam_rgb = cam_xyz @ _XYZ_FROM_SRGB
    cam_rgb /= cam_rgb.sum(axis=1, keepdims=True)
    return np.linalg.inv(cam_rgb).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ _LUMA


def disc_mask(Y: np.ndarray, frac: float = 0.03) -> np.ndarray:
    return Y > frac * float(np.percentile(Y, 99.5))


def to_linear_srgb(stack, white_minus_black, wb, matrix) -> np.ndarray:
    """Camera-RGB data numbers -> linear sRGB, 1.0 = sensor saturation in green."""
    x = stack / np.float32(white_minus_black)
    Y0 = x[..., 1]
    sky = ~cv2.dilate(disc_mask(Y0).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    if sky.sum() > 1000:  # remove any residual black-level error per channel
        x = x - np.median(x[sky], axis=0).astype(np.float32)
    x = x * np.asarray(wb, dtype=np.float32)
    return x @ np.asarray(matrix, dtype=np.float32).T


def estimate_psf_sigma(Y: np.ndarray) -> float | None:
    """Blur of the stack, measured on the lunar limb.

    The sunlit limb is a natural knife edge against black sky. We cut short
    profiles across it, differentiate them, and read the blur from the width of
    the resulting peak. Returns a Gaussian sigma in pixels, or None.
    """
    top = float(np.percentile(Y, 99.5))
    if top <= 0:
        return None
    Yn = (Y / top).astype(np.float32)
    sm = cv2.GaussianBlur(Yn, (0, 0), 2.0)
    contours, _ = cv2.findContours((sm > 0.08).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    pts = max(contours, key=len)[:, 0, :].astype(np.float32)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
    g = np.stack([gx[yi, xi], gy[yi, xi]], axis=1)
    mag = np.linalg.norm(g, axis=1)
    # Keep only the steep part of the outline: the lit limb, not the terminator.
    keep = mag > 0.6 * np.percentile(mag, 90)
    pts, g, mag = pts[keep], g[keep], mag[keep]
    if len(pts) < 40:
        return None
    sel = np.linspace(0, len(pts) - 1, min(len(pts), 720)).astype(int)
    pts, nrm = pts[sel], g[sel] / mag[sel, None]  # normal points into the disc

    dt = 0.25
    t = np.arange(-12, 12 + dt, dt, dtype=np.float32)
    mx = pts[:, 0:1] + nrm[:, 0:1] * t[None]
    my = pts[:, 1:2] + nrm[:, 1:2] * t[None]
    prof = cv2.remap(Yn, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    lsf = np.gradient(prof, dt, axis=1)

    sigmas = []
    for row in lsf:
        k = int(np.argmax(row))
        pk = row[k]
        if pk <= 0 or k < 8 or k > len(row) - 9:
            continue
        half = 0.5 * pk
        a = k
        while a > 0 and row[a] > half:
            a -= 1
        b = k
        while b < len(row) - 1 and row[b] > half:
            b += 1
        if row[a] > half or row[b] > half:
            continue
        left = a + (half - row[a]) / (row[a + 1] - row[a])
        right = b - (half - row[b]) / (row[b - 1] - row[b])
        sigmas.append((right - left) * dt / 2.3548)
    if len(sigmas) < 20:
        return None
    # Rough terrain on the limb can only widen an edge, so lean to the sharp side.
    return float(np.percentile(sigmas, 35))


def richardson_lucy(Y: np.ndarray, sigma: float, iters: int) -> np.ndarray:
    """Richardson-Lucy deconvolution with a Gaussian PSF."""
    ped = 1e-4 * float(Y.max())
    obs = np.maximum(Y, 0) + ped
    est = obs.copy()

    def blur(a):
        return cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

    for _ in range(iters):
        est *= blur(obs / blur(est))
    return est - ped


def sharpen(rgb: np.ndarray, sigma: float, iters: int = 25) -> np.ndarray:
    """Deconvolve brightness only, so colour noise is not amplified.

    Deconvolution rings where the bright limb meets black sky. Within a few
    sigma of the sky we therefore limit the result to the range of values the
    unsharpened image has nearby.
    """
    Y = luminance(rgb).astype(np.float32)
    Ys = richardson_lucy(Y, sigma, iters)

    k = 2 * int(np.ceil(2.0 * sigma)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clamped = np.clip(Ys, cv2.erode(Y, kernel), cv2.dilate(Y, kernel))
    sky = (~disc_mask(Y, 0.02)).astype(np.uint8)
    dist = cv2.distanceTransform(1 - sky, cv2.DIST_L2, 5)
    w = np.clip(1.5 - dist / (3.0 * sigma + 1.0), 0, 1)
    Ys = w * clamped + (1 - w) * Ys

    ratio = np.where(Y > 1e-5 * float(Y.max()), Ys / np.maximum(Y, 1e-12), 1.0)
    return rgb * np.clip(ratio, 0, 8)[..., None].astype(np.float32)


def srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def render_natural(rgb: np.ndarray) -> np.ndarray:
    """Exposure-normalise and gamma-encode. Returns float in 0..1."""
    Y = luminance(rgb)
    return srgb_encode(rgb * (0.85 / float(np.percentile(Y, 99.9)))).astype(np.float32)


def colour_tilt(rgb: np.ndarray, lit: np.ndarray) -> np.ndarray:
    """Smooth colour slope across the disc, as a per-pixel gain (H, W, 3).

    The atmosphere reddens the lower edge of the moon slightly more than the
    upper edge, leaving a colour ramp across the disc of around a percent. That
    is invisible normally but larger than the moon's real colour differences,
    so a strong saturation boost turns it into "orange on one side, blue on the
    other". We fit a plane to each channel's colour ratio over the bright
    highlands (the most uniformly coloured terrain) and return it.
    """
    Y = luminance(rgb)
    b = cv2.GaussianBlur(rgb, (0, 0), 4.0)
    Yb = np.maximum(cv2.GaussianBlur(Y, (0, 0), 4.0), 1e-12)
    inner = cv2.erode(lit.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    h, w = Y.shape
    gain = np.ones((h, w, 3), np.float32)
    if inner.sum() < 5000:
        return gain
    sel = inner & (Yb > np.median(Yb[inner]))
    ys, xs = np.nonzero(sel)
    ys, xs = ys[::7], xs[::7]
    cx, cy = xs.mean(), ys.mean()
    A = np.stack([xs - cx, ys - cy, np.ones(len(xs))], axis=1).astype(np.float64)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for c in range(3):
        v = np.log(np.maximum(b[ys, xs, c], 1e-12) / Yb[ys, xs]).astype(np.float64)
        wgt = np.ones(len(v))
        for _ in range(8):  # robust fit: craters and rays must not steer the plane
            coef = np.linalg.lstsq(A * wgt[:, None], v * wgt, rcond=None)[0]
            r = v - A @ coef
            scale = 1.4826 * np.median(np.abs(r)) + 1e-12
            wgt = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(r), 1e-12))
        gain[..., c] = np.exp(coef[0] * (xx - cx) + coef[1] * (yy - cy))
    return gain


def render_mineral(rgb: np.ndarray, saturation: float = 12.0, chroma_blur: float = 2.0, flatten: bool = True) -> np.ndarray:
    """Exaggerate the moon's faint colour differences.

    The disc is first balanced to neutral grey on average (and, with `flatten`,
    any smooth colour slope across it is removed), so what remains is how each
    region differs from the mean. That difference is smoothed slightly (the
    colour regions are broad; noise is not) and multiplied up.
    """
    Y = luminance(rgb)
    top = float(np.percentile(Y, 99.9))
    lit = Y > 0.25 * top
    if lit.sum() < 100:
        return render_natural(rgb)
    if flatten:
        rgb = rgb / colour_tilt(rgb, lit)
        Y = luminance(rgb)
    balanced = rgb / (np.median(rgb[lit], axis=0) / np.median(Y[lit])).astype(np.float32)
    Y = luminance(balanced)
    Ysafe = np.maximum(Y, 1e-6 * top)
    # Colour as a ratio to brightness, averaged with brightness weighting.
    Yb = cv2.GaussianBlur(Ysafe, (0, 0), chroma_blur)
    chroma = cv2.GaussianBlur(balanced, (0, 0), chroma_blur) / Yb[..., None] - 1.0
    # Fade the boost out in the dark terminator, where colour is all noise.
    fade = np.clip((Y / top - 0.02) / 0.10, 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    out = Y[..., None] * (1.0 + saturation * fade[..., None] * chroma)
    return srgb_encode(out * (0.85 / top)).astype(np.float32)


def apply_flip(img: np.ndarray, flip: int) -> np.ndarray:
    """Apply the camera orientation tag (LibRaw codes)."""
    if flip == 3:
        return np.ascontiguousarray(img[::-1, ::-1])
    if flip == 5:
        return np.ascontiguousarray(np.rot90(img, 1))
    if flip == 6:
        return np.ascontiguousarray(np.rot90(img, -1))
    return img
