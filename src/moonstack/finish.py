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


# Blur-matching kernel, in sensor pixels: centred Gaussians of these widths,
# plus Gaussians of width w displaced by r in eight directions, as (r, w).
_MATCH_SIGMAS = (0.7, 1.2, 2.0, 3.5)
_MATCH_RINGS = ((2.0, 1.2), (5.0, 2.0), (10.0, 3.5))
_MATCH_TARGET = 1.0
_MATCH_BAND = 16  # half-width of the strip of limb the fit looks at


def limb_band(Y: np.ndarray, half_width: int = 10) -> np.ndarray | None:
    """Mask of a strip straddling the sunlit limb (not the soft terminator)."""
    top = float(np.percentile(Y, 99.5))
    if top <= 0:
        return None
    sm = cv2.GaussianBlur((Y / top).astype(np.float32), (0, 0), 2.0)
    grad = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3))
    contours, _ = cv2.findContours((sm > 0.08).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    pts = max(contours, key=len)[:, 0, :]
    on = grad[pts[:, 1], pts[:, 0]]
    steep = pts[on > 0.6 * np.percentile(on, 90)]
    if len(steep) < 200:
        return None
    mask = np.zeros(Y.shape, np.uint8)
    mask[steep[:, 1], steep[:, 0]] = 1
    k = 2 * half_width + 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


def match_channel_psf(rgb: np.ndarray, scale: float = 1.0) -> np.ndarray | None:
    """Give red, green and blue the same blur, measured on the limb.

    Lens and atmosphere blur each colour a little differently, in shape as
    well as width. The air also spreads each colour into a short spectrum
    along the direction of the horizon, more so for blue, so the blur can be
    lopsided. At any hard edge the channels then rise and fall differently
    and leave a coloured rim, which a saturation boost makes glaring. The
    sunlit limb is a clean edge of constant colour, so there the three
    channels should be exact multiples of one another. For red and blue we
    find the mix of blurred and displaced copies of the channel that best
    reproduces slightly-blurred green across the limb, then apply that mix
    everywhere.

    `scale` is output pixels per sensor pixel. Returns the matched image (a
    little softer than the input; it is only used for colour, never for
    detail), or None if the limb cannot be used.
    """
    band = limb_band(luminance(rgb), int(round(_MATCH_BAND * scale)))
    if band is None or band.sum() < 2000:
        return None
    h, w = rgb.shape[:2]
    blur = lambda a, s: cv2.GaussianBlur(a, (0, 0), s * scale, borderType=cv2.BORDER_REFLECT)  # noqa: E731

    def basis(ch):
        """The candidate copies of one channel, one at a time (they are large)."""
        yield ch
        for s in _MATCH_SIGMAS:
            yield blur(ch, s)
        for r, s in _MATCH_RINGS:
            soft = blur(ch, s)
            for i in range(8):
                dx, dy = r * scale * np.cos(i * np.pi / 4), r * scale * np.sin(i * np.pi / 4)
                move = np.float32([[1, 0, dx], [0, 1, dy]])
                yield cv2.warpAffine(soft, move, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    target = blur(rgb[..., 1], _MATCH_TARGET)
    out = np.empty_like(rgb)
    out[..., 1] = target
    t = target[band].astype(np.float64)
    for c in (0, 2):
        ch = np.ascontiguousarray(rgb[..., c])
        A = np.stack([b[band] for b in basis(ch)], axis=1).astype(np.float64)
        ridge = 1e-4 * np.trace(A.T @ A) / A.shape[1]
        k = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ t)
        if not np.isfinite(k).all() or k.sum() <= 0:
            return None
        k /= k.sum()  # keep the channel's overall level; only its blur changes
        acc = np.zeros((h, w), np.float32)
        for wgt, b in zip(k, basis(ch)):
            acc += np.float32(wgt) * b
        out[..., c] = acc
    return out


def sharpen(rgb: np.ndarray, sigma: float, iters: int = 25, match_psf: bool = True, scale: float = 1.0) -> np.ndarray:
    """Deconvolve brightness only, so colour noise is not amplified.

    Colour is taken from blur-matched channels (see match_channel_psf) and
    brightness from the deconvolved image.

    Deconvolution rings where the bright limb meets black sky. Within a few
    sigma of the sky we therefore limit the result to the range of values the
    unsharpened image has nearby.
    """
    Y = luminance(rgb).astype(np.float32)
    base, Yb = rgb, Y
    matched = match_channel_psf(rgb, scale) if match_psf else None
    if matched is not None:
        base = matched
        Yb = luminance(base).astype(np.float32)
    Ys = Y
    if iters > 0:
        Ys = richardson_lucy(Y, sigma, iters)
        k = 2 * int(np.ceil(2.0 * sigma)) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        clamped = np.clip(Ys, cv2.erode(Y, kernel), cv2.dilate(Y, kernel))
        sky = (~disc_mask(Y, 0.02)).astype(np.uint8)
        dist = cv2.distanceTransform(1 - sky, cv2.DIST_L2, 5)
        w = np.clip(1.5 - dist / (3.0 * sigma + 1.0), 0, 1)
        Ys = w * clamped + (1 - w) * Ys
    if base is rgb and iters <= 0:
        return rgb
    ratio = np.where(Yb > 1e-5 * float(Y.max()), Ys / np.maximum(Yb, 1e-12), 1.0)
    return base * np.clip(ratio, 0, 8)[..., None].astype(np.float32)


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
    out = Y[..., None] * (1.0 + saturation * (fade * colour_trust(Y, top, saturation, chroma_blur))[..., None] * chroma)
    return srgb_encode(out * (0.85 / top)).astype(np.float32)


def colour_trust(Y: np.ndarray, top: float, saturation: float, chroma_blur: float) -> np.ndarray:
    """How much of the saturation boost each pixel can take, 0..1.

    The three colours are registered to a few hundredths of a pixel and their
    blurs matched to about a percent. That is not enough at hard edges once
    colour is multiplied 20-30 times: the leftover shows as bright rims on the
    limb and on crater shadows. Where brightness changes quickly relative to
    its level, we know the colour ratio is unreliable by roughly
    (misregistration x relative slope), and hold the boost back so that the
    amplified error stays small. Next to the sky the boost is faded out
    entirely.
    """
    edge_px = 0.08  # assumed leftover misregistration between colours
    tolerated = 0.06  # amplified colour error we accept
    Yb = np.maximum(cv2.GaussianBlur(Y, (0, 0), chroma_blur), 1e-4 * top)
    slope = np.hypot(cv2.Sobel(Yb, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(Yb, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    err = edge_px * slope / Yb
    trust = 1.0 / np.sqrt(1.0 + (saturation * err / tolerated) ** 2)
    sky = (Y < 0.04 * top).astype(np.uint8)
    dist = cv2.distanceTransform(1 - sky, cv2.DIST_L2, 5)
    near = np.clip((dist - 2.0 * chroma_blur) / (5.0 * chroma_blur), 0, 1)
    trust = trust * near * near * (3 - 2 * near)
    # Smooth so the hold-back itself leaves no visible outline.
    return cv2.GaussianBlur(trust.astype(np.float32), (0, 0), chroma_blur)


def side_by_side(left: np.ndarray, right: np.ndarray, margin: float = 0.07) -> np.ndarray:
    """Two renders of the same stack next to each other on black, cropped to the moon."""
    ys, xs = np.nonzero(disc_mask(luminance(left), 0.05))
    if len(xs) == 0:
        return np.hstack([left, right])
    h, w = left.shape[:2]
    pad = int(margin * max(xs.max() - xs.min(), ys.max() - ys.min()))
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    return np.hstack([left[y0:y1, x0:x1], right[y0:y1, x0:x1]])


def apply_flip(img: np.ndarray, flip: int) -> np.ndarray:
    """Apply the camera orientation tag (LibRaw codes)."""
    if flip == 3:
        return np.ascontiguousarray(img[::-1, ::-1])
    if flip == 5:
        return np.ascontiguousarray(np.rot90(img, 1))
    if flip == 6:
        return np.ascontiguousarray(np.rot90(img, -1))
    return img
