"""End-to-end check on a small synthetic burst with known ground truth."""
import cv2
import numpy as np
import pytest

from moonstack import synth
from moonstack.pipeline import Config, choose_scale, rerender, run, write_outputs

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

    # Re-rendering from the saved stack reproduces the run, and honours new settings.
    again = rerender(tmp_path / "out", cfg)
    assert np.allclose(again.sharp, res.sharp, rtol=1e-4, atol=1e-6)
    assert again.frames == res.frames
    softer = rerender(tmp_path / "out", Config(quiet=True, sharpen_iters=0))
    assert _error(softer.sharp, truth) > e_sharp
    assert softer.info["config"]["sharpen_iters"] == 0


def test_outlier_is_rejected(burst):
    """A streak painted across one frame must not show in the stack."""
    folder, _, _ = burst
    f = sorted(folder.glob("synth_*.npz"))[5]
    z = dict(np.load(f))
    clean = z["bayer"].copy()
    # Pinned reference and no channel re-registration: one shared geometry.
    cfg = dict(quiet=True, keep=1.0, reference="synth_0000.npz", chroma_align=False)
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


def test_colour_fringing_is_corrected(burst, tmp_path):
    """Shift red and blue in every frame; the stack must come out registered."""
    folder, _, _ = burst
    out = tmp_path / "fringed"
    out.mkdir()
    for f in sorted(folder.glob("synth_*.npz")):
        z = dict(np.load(f))
        b = z["bayer"].astype(np.float32)
        h, w = b.shape
        for (ty, tx), (dx, dy) in {(0, 0): (0.6, -0.3), (1, 1): (-0.5, 0.4)}.items():  # R and B sites of RGGB
            plane = b[ty::2, tx::2]
            # A shift of d raw pixels is d/2 pixels of the half-resolution colour plane.
            M = np.float32([[1, 0, dx / 2], [0, 1, dy / 2]])
            b[ty::2, tx::2] = cv2.warpAffine(plane, M, (w // 2, h // 2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        z["bayer"] = np.clip(np.round(b), 0, 16383).astype(np.uint16)
        np.savez(out / f.name, **z)
    res = run(out, Config(quiet=True))
    off = np.array(res.info["channel_offset_px"])
    assert np.allclose(off[0], (0.6, -0.3), atol=0.15) and np.allclose(off[2], (-0.5, 0.4), atol=0.15)
    assert res.info["channel_residual_px"] < 0.15


def test_auto_scale(burst):
    # Blurry data stays at sensor resolution; sharp data is enlarged, within limits.
    assert choose_scale(1.44) == 1.0
    assert choose_scale(0.8) == 1.5
    assert choose_scale(0.3) == 2.0
    assert choose_scale(None) == 1.0
    folder, _, _ = burst
    res = run(folder, Config(quiet=True, scale="auto"))
    want = choose_scale(res.info["scale_auto"]["blur_sigma_at_sensor_scale"])
    assert res.info["scale"] == want
    assert res.stack_cam.shape[0] == round(want * res.info["output_size"][1] / want)
