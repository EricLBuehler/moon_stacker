"""Measure the pipeline against synthetic ground truth.

    uv run python tests/bench_synth.py [folder]

Reports alignment error, effective blur and noise of the stack, and compares
with a conventional stack (demosaic each frame, then average) that is given
the *true* frame positions.
"""
import sys, tempfile
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import minimize_scalar
from moonstack import finish, synth
from moonstack.pipeline import Config, run

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def register_truth(truth, img):
    """Warp truth into img geometry (affine ECC on luminance)."""
    a, b = (img @ LUMA).astype(np.float32), (truth @ LUMA).astype(np.float32)
    a, b = a / a.max(), b / b.max()
    sh = np.array([(a.shape[1] - b.shape[1]) / 2, (a.shape[0] - b.shape[0]) / 2])
    W = np.array([[1, 0, -sh[0]], [0, 1, -sh[1]]], np.float32)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-9)
    _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_AFFINE, crit, None, 5)
    return lambda t: cv2.warpAffine(t, W, (a.shape[1], a.shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)


def score(img, truth_w, mask):
    """Best-fit Gaussian blur of truth that explains img, and the leftover error."""
    def err(s):
        t = cv2.GaussianBlur(truth_w, (0, 0), s) if s > 0.05 else truth_w
        e = 0
        for c in range(3):
            g = (img[..., c][mask] * t[..., c][mask]).sum() / (t[..., c][mask] ** 2).sum()
            e += ((img[..., c][mask] - g * t[..., c][mask]) ** 2).mean() / g**2
        return np.sqrt(e / 3)
    r = minimize_scalar(err, bounds=(0.05, 4), method="bounded")
    scale = truth_w[mask].mean()
    return r.x, 100 * r.fun / scale, 100 * err(0.0) / scale


def chroma_err(img, truth_w, mask):
    """RMS error of colour ratios R/G and B/G after a 3 px blur, in percent."""
    out = []
    for im in (img, truth_w):
        b = np.maximum(cv2.GaussianBlur(im, (0, 0), 3), 1e-9)
        out.append(np.stack([b[..., 0] / b[..., 1], b[..., 2] / b[..., 1]], -1)[mask])
    a, t = out
    a = a * (np.median(t, 0) / np.median(a, 0))
    return 100 * float(np.sqrt(((a - t) ** 2).mean()))


def demosaic(bayer, cfa, black):
    x = bayer.astype(np.float32)
    out = np.zeros((*x.shape, 3), np.float32)
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
    for c in range(3):
        m = np.zeros(x.shape, np.float32)
        for ty in range(2):
            for tx in range(2):
                if cfa[ty, tx] == c:
                    m[ty::2, tx::2] = 1
        v = (x - black) * m
        out[..., c] = cv2.filter2D(v, -1, k) / cv2.filter2D(m, -1, k)
    return out


def conventional(folder, truth, params, n):
    """Demosaic-then-average with oracle global alignment, in truth geometry."""
    S = truth.shape[0]; ct = (S - 1) / 2
    acc = np.zeros((S, S, 3), np.float64)
    files = sorted(folder.glob("synth_*.npz"))[:n]
    for f, (cx, cy, th, _, g) in zip(files, params):
        z = np.load(f)
        rgb = demosaic(z["bayer"], z["cfa"], float(z["black"][0, 0])) * synth.WB.astype(np.float32) / g
        co, si = np.cos(th), np.sin(th)
        # truth q = R (p - c) + ct  =>  p = R^T (q - ct) + c
        M = np.array([[co, si, cx - co * ct - si * ct], [-si, co, cy + si * ct - co * ct]], np.float32)
        acc += cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    return (acc / len(files)).astype(np.float32), demosaic_single(files[0], params[0], S)


def demosaic_single(f, p, S):
    z = np.load(f); cx, cy, th, _, g = p; ct = (S - 1) / 2
    rgb = demosaic(z["bayer"], z["cfa"], float(z["black"][0, 0])) * synth.WB.astype(np.float32)
    co, si = np.cos(th), np.sin(th)
    M = np.array([[co, si, cx - co * ct - si * ct], [-si, co, cy + si * ct - co * ct]], np.float32)
    return cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp()) / "synth"
    if not (folder / "truth.npz").exists():
        synth.generate(folder, frames=60)
    z = np.load(folder / "truth.npz"); truth, params = z["truth"], z["params"]
    n = len(params)
    print(f"true per-frame blur sigma: {params[:,3].min():.2f}-{params[:,3].max():.2f} px (+ pixel aperture)")

    conv, single = conventional(folder, truth, params, n)
    mask0 = (truth @ LUMA) > 0.15 * (truth @ LUMA).max()
    mask0 = cv2.erode(mask0.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    rows = [("single frame, demosaiced", single, truth, mask0), ("conventional stack, oracle align, all frames", conv, truth, mask0)]

    variants = {
        "moonstack default": Config(quiet=True),
        "  no local field": Config(quiet=True, local=False),
        "  radius 1.0": Config(quiet=True, radius=1.0),
        "  radius 0.5": Config(quiet=True, radius=0.5),
        "  keep all": Config(quiet=True, keep=1.0),
        "  no rejection": Config(quiet=True, kappa=0),
        "  scale 1.5": Config(quiet=True, scale=1.5),
    }
    sharp_rows = []
    for name, cfg in variants.items():
        res = run(folder, cfg)
        if name == "moonstack default":
            fr = [f for f in res.frames if "moon_x" in f]
            # Map a ring of reference points through each frame's estimated
            # transform and then the true one; perfect alignment lands every
            # frame on the same truth point.
            ct = (truth.shape[0] - 1) / 2
            c0 = np.array(fr[0]["matrix"]) ; cen = np.linalg.solve(c0[:, :2], np.array([params[0,0], params[0,1]]) - c0[:, 2])
            ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            pts = np.concatenate([cen[None], cen[None] + 200 * np.stack([np.cos(ang), np.sin(ang)], 1)])
            T = []
            for f, (cx, cy, th, _, _) in zip(fr, params):
                M = np.array(f["matrix"]); p = pts @ M[:, :2].T + M[:, 2]
                co, si = np.cos(th), np.sin(th)
                d = p - [cx, cy]
                T.append(np.stack([co * d[:, 0] - si * d[:, 1], si * d[:, 0] + co * d[:, 1]], 1) + ct)
            T = np.array(T); dev = T - T.mean(0)
            print(f"global alignment error: centre {np.sqrt((dev[:,0]**2).sum(1).mean()):.3f} px rms, "
                  f"200 px out {np.sqrt((dev[:,1:]**2).sum(2).mean()):.3f} px rms (frames carry 0.4 px rms of turbulence)")
            print(f"limb blur estimate: {res.info['psf_sigma_measured']:.2f} px; time {res.info['seconds']} s")
        lin = res.linear
        if cfg.scale != 1.0:
            lin = cv2.resize(lin, None, fx=1 / cfg.scale, fy=1 / cfg.scale, interpolation=cv2.INTER_AREA)
        warp = register_truth(truth, lin)
        tw = warp(truth); mask = warp(mask0.astype(np.float32)) > 0.99
        rows.append((name, lin, tw, mask))
        if cfg.scale == 1.0:
            sharp_rows.append((name + " + sharpen", res.sharp, tw, mask))
    rows += sharp_rows[:1]
    sg = finish.estimate_psf_sigma(conv @ LUMA)
    rows.append((f"conventional + sharpen (limb sigma {sg:.2f})", finish.sharpen(conv, 0.9 * sg), truth, mask0))

    print(f"\n{'':46s} {'blur':>6s} {'noise%':>7s} {'err vs truth%':>14s} {'colour err%':>12s}")
    for name, img, tw, mask in rows:
        s, resid, raw = score(img, tw, mask)
        print(f"{name:46s} {s:6.2f} {resid:7.2f} {raw:14.2f} {chroma_err(img, tw, mask):12.3f}")


if __name__ == "__main__":
    main()
