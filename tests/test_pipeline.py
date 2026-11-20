"""End-to-end check on a small synthetic burst with known ground truth."""
import cv2
import numpy as np
import pytest

from moonstack import synth
from moonstack.pipeline import Config, run, write_outputs

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


@pytest.fixture(scope="module")
def burst(tmp_path_factory):
    folder = tmp_path_factory.mktemp("burst")
    truth, params = synth.generate(folder, frames=24, sensor=(640, 800), diameter=360, seed=1)
    return folder, truth, params


def _error(img, truth):
    """Percent RMS error against truth after affine registration and gain fit."""
    a, b = img @ LUMA, truth @ LUMA
    a, b = (a / a.max()).astype(np.float32), (b / b.max()).astype(np.float32)
    W = np.array([[1, 0, -(a.shape[1] - b.shape[1]) / 2], [0, 1, -(a.shape[0] - b.shape[0]) / 2]], np.float32)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-9)
    _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_AFFINE, crit, None, 5)
    t = cv2.warpAffine(truth, W, (a.shape[1], a.shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    m = (t @ LUMA) > 0.15 * (t @ LUMA).max()
    g = (img[m] * t[m]).sum() / (t[m] ** 2).sum()
    return 100 * float(np.sqrt(((img[m] / g - t[m]) ** 2).mean()) / t[m].mean())


def test_stack_matches_truth(burst, tmp_path):
    folder, truth, params = burst
    cfg = Config(quiet=True)
    res = run(folder, cfg)
    assert res.info["frames_stacked"] == 19

    # Frame positions agree with the simulated ones to a fraction of a pixel.
    fr = [f for f in res.frames if "moon_x" in f]
    dx = np.array([f["moon_x"] for f in fr]) - params[:, 0]
    dy = np.array([f["moon_y"] for f in fr]) - params[:, 1]
    assert dx.std() < 0.35 and dy.std() < 0.35

    # White balance was undone: the disc has the truth's overall colour.
    lit = (res.linear @ LUMA) > 0.3 * (res.linear @ LUMA).max()
    tl = (truth @ LUMA) > 0.3 * (truth @ LUMA).max()
    got = np.median(res.linear[lit], 0)
    want = np.median(truth[tl], 0)
    assert np.allclose(got / got[1], want / want[1], atol=0.02)

    # The blur measured on the limb is close to what was simulated (0.7-1.1 px).
    assert 0.6 < res.info["psf_sigma_measured"] < 1.4

    # Sharpening moves the result towards the truth, and the result is far
    # better than the stack's own blur would be without it.
    e_lin, e_sharp = _error(res.linear, truth), _error(res.sharp, truth)
    assert e_lin < 12.0
    assert e_sharp < 0.7 * e_lin

    files = write_outputs(res, tmp_path / "out", cfg)
    assert {"moon_linear.tif", "moon_natural.tif", "moon_mineral.tif", "report.json"} <= set(files)
    assert all(p.stat().st_size > 0 for p in files.values())


def test_outlier_is_rejected(burst):
    """A streak painted across one frame must not show in the stack."""
    folder, _, _ = burst
    f = sorted(folder.glob("synth_*.npz"))[5]
    z = dict(np.load(f))
    clean = z["bayer"].copy()
    cfg = dict(quiet=True, keep=1.0, reference="synth_0000.npz")  # pinned: one shared geometry
    try:
        z["bayer"][318:322, 330:470] = 16000  # short streak across the disc
        np.savez(f, **z)
        with_rej = run(folder, Config(**cfg)).stack_cam
        without = run(folder, Config(**cfg, kappa=0)).stack_cam
    finally:
        z["bayer"] = clean
        np.savez(f, **z)
    base = run(folder, Config(**cfg)).stack_cam
    assert np.abs(without - base).max() > 300
    assert np.abs(with_rej - base).max() < 0.1 * np.abs(without - base).max()
