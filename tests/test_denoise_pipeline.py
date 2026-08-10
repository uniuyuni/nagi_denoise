"""Tests for the Phase 3 production entry point: nagi_denoise.pipeline.denoise.

Run with: python tests/test_denoise_pipeline.py  (invoked by `pixi run test`)

These tests use a tiny randomly-initialized NagiV2 checkpoint (not the real
236MB production weights) so they run fast and don't depend on training
artifacts. They check the public contract of denoise(): shape/dtype in/out,
HDR passthrough, non-finite sanitization, and the seam guarantee (tiled vs.
smaller-tile agreement) on a small synthetic image.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nagi_denoise.models.nagi_v2 import NagiV2
from nagi_denoise.pipeline import denoise as denoise_mod


def _make_tiny_checkpoint() -> Path:
    """Build and save a tiny NagiV2 checkpoint for fast, dependency-free tests."""
    torch.manual_seed(0)
    model = NagiV2(width=8, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1))
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "model": {
                "arch": "nagi_v2",
                "img_channels": 3,
                "width": 8,
                "enc_blk_nums": [1, 1],
                "middle_blk_num": 1,
                "dec_blk_nums": [1, 1],
            }
        },
    }
    tmp_dir = Path(tempfile.mkdtemp(prefix="nagi_denoise_test_"))
    path = tmp_dir / "tiny.pt"
    torch.save(ckpt, path)
    return path


_TINY_CKPT = _make_tiny_checkpoint()


def _clear_cache() -> None:
    denoise_mod._MODEL_CACHE.clear()


def test_denoise_hwc_float32_contract():
    rng = np.random.default_rng(0)
    img = (rng.random((40, 48, 3), dtype=np.float32)) * 0.8
    out = denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=0, chroma_cleanup=False)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_denoise_rejects_bad_shape():
    bad = np.zeros((10, 10), dtype=np.float32)
    try:
        denoise_mod.denoise(bad, weights=_TINY_CKPT, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-(H,W,3) input")

    bad2 = np.zeros((10, 10, 4), dtype=np.float32)
    try:
        denoise_mod.denoise(bad2, weights=_TINY_CKPT, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-3-channel input")


def test_denoise_hdr_values_survive():
    """The identity-initialized tiny model + no chroma cleanup should pass HDR
    values (>1.0) through close to unchanged, and never clip them."""
    img = np.ones((32, 32, 3), dtype=np.float32) * 5.0  # well above 1.0 (HDR)
    out = denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=0, chroma_cleanup=False)
    assert out.max() > 1.0, "HDR values must not be clipped to <=1.0"
    assert np.allclose(out, img, atol=0.5), f"expected near-identity, got max diff {np.abs(out - img).max()}"


def test_denoise_sanitizes_non_finite_input():
    img = np.zeros((24, 24, 3), dtype=np.float32)
    img[0, 0, 0] = np.nan
    img[1, 1, 1] = np.inf
    img[2, 2, 2] = -np.inf
    out = denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=0, chroma_cleanup=False)
    assert np.isfinite(out).all()


def test_denoise_tiled_vs_small_tile_agree_within_tolerance():
    """The seam guarantee as a unit test: denoising the same small synthetic
    image with two different tile grids must agree everywhere, including at
    tile boundary lines, within a tight tolerance."""
    rng = np.random.default_rng(1)
    h, w = 96, 112
    yy, xx = np.meshgrid(np.linspace(0, 6, h), np.linspace(0, 6, w), indexing="ij")
    img = (0.5 + 0.4 * np.sin(yy) * np.cos(xx)).astype(np.float32)
    img = np.stack([img, img * 0.8, img * 1.1], axis=-1)
    img += rng.normal(0, 0.02, size=img.shape).astype(np.float32)
    img = np.clip(img, 0.0, None).astype(np.float32)

    out_a = denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=48, overlap=16, chroma_cleanup=False)
    out_b = denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=32, overlap=16, chroma_cleanup=False)

    diff = np.abs(out_a - out_b)
    assert diff.max() < 1e-3, f"tiled outputs disagree beyond tolerance: max diff {diff.max()}"


def test_denoise_model_cache_reuses_instance():
    _clear_cache()
    img = np.zeros((16, 16, 3), dtype=np.float32)
    denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=0, chroma_cleanup=False)
    assert len(denoise_mod._MODEL_CACHE) == 1
    denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=0, chroma_cleanup=False)
    assert len(denoise_mod._MODEL_CACHE) == 1, "second call with identical (weights, device) must reuse the cached model"


def test_denoise_detail_strength_sets_model_attribute():
    _clear_cache()
    img = np.zeros((16, 16, 3), dtype=np.float32)
    denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=0, chroma_cleanup=False, detail_strength=0.777)
    dn = denoise_mod._get_denoiser(str(_TINY_CKPT), "cpu")
    assert abs(float(dn.model.detail_scale) - 0.777) < 1e-6


def _top1pct_retention(in_img: np.ndarray, out_img: np.ndarray) -> float:
    """Mean output luma / mean input luma over the top 1% brightest input pixels."""
    luma_in = denoise_mod._linear_luma_np(in_img)
    luma_out = denoise_mod._linear_luma_np(out_img)
    thr = np.percentile(luma_in, 99.0)
    mask = luma_in >= thr
    return float(luma_out[mask].mean() / luma_in[mask].mean())


def test_denoise_hdr_highlights_survive_with_conditional_highlight_guard():
    """The Phase 3 HDR-defect regression test: with the conditional highlight
    guard (the default), a synthetic HDR image's brightest 1% of luma must
    survive denoise() at >=0.95 retention, even though the tiny test
    checkpoint has the guard disabled by construction (threshold=0.0,
    transition=0.5, strength=0.0 -- see NagiV2 defaults) until denoise()
    re-arms it."""
    _clear_cache()
    rng = np.random.default_rng(3)
    h, w = 64, 64
    img = (rng.random((h, w, 3), dtype=np.float32)) * 0.3
    # Punch a bright HDR patch (values well above 1.0) into part of the image
    # so there is a clear "top 1%" of luma to track.
    img[4:12, 4:12, :] = 3.0

    out = denoise_mod.denoise(
        img,
        weights=_TINY_CKPT,
        device="cpu",
        tile=0,
        chroma_cleanup=False,
        highlight_guard=True,
    )
    retention = _top1pct_retention(img, out)
    assert retention >= 0.95, f"expected top-1% luma retention >= 0.95 with guard on, got {retention}"
    assert denoise_mod.denoise.last_highlight_guard["mode"] == "conditional-armed"
    assert denoise_mod.denoise.last_highlight_guard["threshold"] > 0.0


def test_denoise_highlight_guard_disarms_on_low_dynamic_range():
    """The guard must stay disarmed when the image has no above-SDR content.
    Arming it there only blends the noisy input back in, which was measured
    as 3.7x more flat-region noise and read visually as uneven denoising."""
    _clear_cache()
    rng = np.random.default_rng(7)
    img = (rng.random((64, 64, 3), dtype=np.float32)) * 0.4  # p99.9 luma well under 1.0
    denoise_mod.denoise(
        img, weights=_TINY_CKPT, device="cpu", tile=0, chroma_cleanup=False, highlight_guard=True
    )
    guard = denoise_mod.denoise.last_highlight_guard
    assert guard["mode"] == "conditional-disarmed", guard
    assert guard["strength"] == 0.0


def test_denoise_input_blend_restores_input_proportionally():
    """input_blend mixes the original input back in. Verify the documented
    default (0.20) and that the mix is exact, so the detail/noise trade-off
    stays predictable."""
    _clear_cache()
    rng = np.random.default_rng(8)
    img = (rng.random((48, 48, 3), dtype=np.float32)) * 0.5
    kw = dict(weights=_TINY_CKPT, device="cpu", tile=0, chroma_cleanup=False, highlight_guard=False)
    pure = denoise_mod.denoise(img, input_blend=0.0, **kw)
    mixed = denoise_mod.denoise(img, input_blend=0.20, **kw)
    expected = pure * 0.80 + img * 0.20
    assert np.allclose(mixed, expected, atol=1e-5), "input_blend must mix exactly"
    # And the default must be 0.20.
    default_mixed = denoise_mod.denoise(img, **kw)
    assert np.allclose(default_mixed, expected, atol=1e-5), "default input_blend must be 0.20"


def test_denoise_conditional_highlight_guard_threshold_is_global_not_per_tile():
    """The guard threshold must be derived once from the whole image, not
    recomputed per tile -- otherwise tiles would disagree on the threshold
    and reintroduce seams. Verify by checking that denoising with two
    different tile grids resolves to the exact same threshold, and that the
    threshold matches a direct whole-image computation."""
    _clear_cache()
    rng = np.random.default_rng(4)
    h, w = 96, 112
    img = (rng.random((h, w, 3), dtype=np.float32)) * 0.3
    img[10:20, 10:20, :] = 2.5  # bright region concentrated in one corner/tile

    expected_luma = denoise_mod._linear_luma_np(np.clip(img, 0.0, None))
    expected_p999 = float(np.percentile(expected_luma, 99.9))
    assert expected_p999 > denoise_mod.HIGHLIGHT_GUARD_MIN_P999, "fixture must arm the guard"
    expected_threshold = denoise_mod.HIGHLIGHT_GUARD_DEFAULT_THRESHOLD

    denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=0, chroma_cleanup=False, highlight_guard=True)
    threshold_full = denoise_mod.denoise.last_highlight_guard["threshold"]

    denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=48, overlap=16, chroma_cleanup=False, highlight_guard=True)
    threshold_tiled = denoise_mod.denoise.last_highlight_guard["threshold"]

    assert abs(threshold_full - expected_threshold) < 1e-6
    assert abs(threshold_tiled - expected_threshold) < 1e-6, (
        "adaptive threshold must be identical regardless of tile grid "
        f"(full={threshold_full}, tiled={threshold_tiled}, expected={expected_threshold})"
    )


def test_denoise_highlight_guard_disabled_matches_legacy_behavior():
    """highlight_guard=False must reproduce the pre-fix (as-shipped) behavior:
    strength pinned to 0.0 so input_highlight_guard is a no-op."""
    _clear_cache()
    img = np.ones((16, 16, 3), dtype=np.float32) * 2.0
    denoise_mod.denoise(img, weights=_TINY_CKPT, device="cpu", tile=0, chroma_cleanup=False, highlight_guard=False)
    info = denoise_mod.denoise.last_highlight_guard
    assert info["mode"] == "disabled"
    assert info["strength"] == 0.0
    dn = denoise_mod._get_denoiser(str(_TINY_CKPT), "cpu")
    assert dn.model.highlight_protect_strength == 0.0


if __name__ == "__main__":
    test_denoise_hwc_float32_contract()
    test_denoise_rejects_bad_shape()
    test_denoise_hdr_values_survive()
    test_denoise_sanitizes_non_finite_input()
    test_denoise_tiled_vs_small_tile_agree_within_tolerance()
    test_denoise_model_cache_reuses_instance()
    test_denoise_detail_strength_sets_model_attribute()
    test_denoise_hdr_highlights_survive_with_conditional_highlight_guard()
    test_denoise_highlight_guard_disarms_on_low_dynamic_range()
    test_denoise_input_blend_restores_input_proportionally()
    test_denoise_conditional_highlight_guard_threshold_is_global_not_per_tile()
    test_denoise_highlight_guard_disabled_matches_legacy_behavior()
    print("all tests passed")
