"""Smoke tests for Nagi Denoise models and transforms.

Run with: python -m pytest tests/  (or just python tests/test_model.py)
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

# Fallback path so `python3 tests/test_model.py` works without an editable install.
# A real `pixi install` / `pip install -e .` is preferred and makes this line a no-op.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nagi_denoise.models.nagi_nr import NagiNR
from nagi_denoise.models.nagi_v2 import NagiV2, build_nagi_v2_preset
from nagi_denoise.transforms import (
    asinh_compress,
    asinh_decompress,
    linear_to_srgb,
    srgb_to_linear,
)
from nagi_denoise.losses import NagiLoss, NagiV2Loss


def test_forward_shape_and_dtype():
    m = NagiNR(base_channels=16, num_blocks=(1, 1, 2, 1, 1))
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float32


def test_size_must_be_multiple_of_8():
    m = NagiNR(base_channels=16, num_blocks=(1, 1, 1, 1, 1))
    bad = torch.randn(1, 3, 30, 30)
    try:
        m(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-multiple-of-8 input")


def test_asinh_reversibility_hdr():
    x = torch.tensor([0.0, 0.001, 0.5, 1.0, 5.0, 25.0, -0.1])
    xc = asinh_compress(x, k=8.0)
    xr = asinh_decompress(xc, k=8.0)
    assert torch.allclose(x, xr, atol=1e-5, rtol=1e-4), (x, xr)


def test_model_compress_matches_transforms():
    m = NagiNR(base_channels=16, num_blocks=(1, 1, 1, 1, 1), asinh_k=8.0)
    x = torch.linspace(0.0, 20.0, 32).view(1, 1, 4, 8).expand(1, 3, 4, 8).contiguous()
    a = m.compress(x)
    b = asinh_compress(x, k=8.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_srgb_linear_roundtrip():
    x = torch.linspace(0.0, 1.0, 257)
    y = linear_to_srgb(srgb_to_linear(x))
    assert torch.allclose(x, y, atol=1e-4)


def test_identity_at_init_is_near_input():
    """Head weights are initialized to 0, so the residual is 0 and the model
    should output (decompress(compress(x))) == x within float precision."""
    torch.manual_seed(0)
    m = NagiNR(base_channels=16, num_blocks=(1, 1, 1, 1, 1)).eval()
    x = torch.rand(1, 3, 64, 64) * 2.0  # include HDR range
    with torch.no_grad():
        y = m(x)
    assert torch.allclose(x, y, atol=1e-4), (x - y).abs().max()


def test_loss_runs_and_is_finite():
    m = NagiNR(base_channels=16, num_blocks=(1, 1, 1, 1, 1))
    crit = NagiLoss(compress_fn=m.compress)
    noisy = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)
    pred = m(noisy)
    out = crit(pred, target)
    assert torch.isfinite(out["total"]).item()
    out["total"].backward()  # ensure gradients flow
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in m.parameters())


def test_nagi_v2_forward_aux_and_identity_init():
    m = NagiV2(width=8, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1)).eval()
    x = torch.rand(1, 3, 35, 41) * 4.0
    with torch.no_grad():
        y = m(x)
        aux = m(x, return_aux=True)
    assert y.shape == x.shape
    assert aux["output"].shape == x.shape
    assert aux["output_pre_chroma"].shape == x.shape
    assert aux["base"].shape == x.shape
    assert aux["detail"].shape == (1, 1, 35, 41)
    assert aux["detail_confidence"].shape == (1, 1, 35, 41)
    assert torch.allclose(x, y, atol=1e-4), (x - y).abs().max()


def test_nagi_v2_loss_runs_and_has_gradients():
    m = NagiV2(width=8, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1))
    crit = NagiV2Loss(compress_fn=m.compress)
    noisy = torch.rand(2, 3, 32, 32) * 3.0
    target = torch.rand(2, 3, 32, 32) * 3.0
    pred = m(noisy, return_aux=True)
    out = crit(pred, target)
    assert torch.isfinite(out["total"]).item()
    out["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in m.parameters())


def test_nagi_v2_preset_builds():
    m = build_nagi_v2_preset("v2-s")
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape


def test_nagi_v2_l_preset_builds_identity_and_shape():
    m = build_nagi_v2_preset("v2-l").eval()
    n_params = m.param_count()
    assert 12.0e6 <= n_params <= 16.0e6, f"v2-l params out of range: {n_params/1e6:.2f}M"
    x = torch.rand(1, 3, 35, 41) * 4.0  # non-multiple size + HDR range
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    # Heads are zero-initialized: the model must start as an identity.
    assert torch.allclose(x, y, atol=1e-4), (x - y).abs().max()


def test_synthetic_poisson_gaussian_linear():
    import numpy as np
    from nagi_denoise.data import apply_poisson_gaussian_linear, sample_poisson_gaussian_params

    rng = np.random.default_rng(0)
    a, b = sample_poisson_gaussian_params(rng, (3.0e-4, 2.0e-2), (1.0e-6, 1.0e-3))
    assert 3.0e-4 <= a <= 2.0e-2 and 1.0e-6 <= b <= 1.0e-3
    clean = torch.rand(3, 64, 64) * 2.0  # includes HDR range
    noisy = apply_poisson_gaussian_linear(rng, clean, a=2.0e-2, b=1.0e-3, chroma_scale=0.35)
    assert noisy.shape == clean.shape
    assert torch.isfinite(noisy).all()
    assert (noisy >= 0.0).all()
    assert not torch.allclose(noisy, clean)  # noise actually applied
    # Heteroscedastic: bright regions must carry more noise than dark ones.
    resid = (noisy - clean).abs()
    bright = resid[clean > 1.0].mean()
    dark = resid[clean < 0.2].mean()
    assert bright > dark


def test_correlated_noise_lag1_autocorrelation():
    """Spatial correlation for synthetic noise (Phase 2B): gaussian_filter at
    a given sigma, renormalized to unit std per channel, should reproduce the
    lag-1 autocorrelation calibrated against real-photo noise. Autocorrelation
    is measured the same way as the real-photo calibration: on the residual
    after a 3x3 median filter (isolates the noise-like high-frequency content
    from any residual low-frequency structure introduced by the blur), lag 1
    along x. See module docstring in nagi_denoise/data.py for the full
    sigma -> lag1 calibration table (0.5 -> ~0, 0.7 -> ~0.22, 0.9 -> ~0.37).
    """
    import numpy as np
    from scipy.ndimage import median_filter
    from nagi_denoise.data import correlated_standard_normal

    def lag1_autocorr(field: np.ndarray) -> float:
        a = field[:, :-1]
        b = field[:, 1:]
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / denom) if denom > 1.0e-8 else 0.0

    rng = np.random.default_rng(0)
    h = w = 256

    # sigma=0 -> plain white noise. On the median residual this is NOT quite
    # 0 (the median filter itself induces a small negative bias on i.i.d.
    # input, ~-0.15 -- this matches the "our synthetic" row (-0.146) in the
    # project's calibration table, i.e. today's uncorrelated generator). The
    # key property here is std preserved and NOT positively correlated.
    white = correlated_standard_normal(rng, (1, h, w), 0.0)[0]
    assert abs(white.std() - 1.0) < 0.05
    white_lag1 = lag1_autocorr(white - median_filter(white, size=3))
    assert -0.25 < white_lag1 < 0.05, white_lag1

    # sigma=0.7 -> calibrated to match Fujifilm X-T5 real-photo luma noise
    # (lag1 +0.235); renormalization must keep std ~= 1 (magnitude preserved).
    corr = correlated_standard_normal(rng, (1, h, w), 0.7)[0]
    assert abs(corr.std() - 1.0) < 0.05
    lag1 = lag1_autocorr(corr - median_filter(corr, size=3))
    assert 0.10 < lag1 < 0.32, lag1

    # Per-channel independence: a multi-channel field is blurred/renormalized
    # per channel, not shared across channels.
    multi = correlated_standard_normal(rng, (3, h, w), 0.7)
    assert multi.shape == (3, h, w)
    for c in range(3):
        assert abs(multi[c].std() - 1.0) < 0.05


def test_nagi_v2_chroma_branch_identity_init():
    m = NagiV2(
        width=8,
        enc_blk_nums=(1, 1),
        middle_blk_num=1,
        dec_blk_nums=(1, 1),
        chroma_branch=True,
        chroma_branch_width=8,
        chroma_branch_blocks=1,
        chroma_branch_use_input=True,
    ).eval()
    x = torch.rand(1, 3, 35, 41) * 2.0
    with torch.no_grad():
        aux = m(x, return_aux=True)
    assert aux["output"].shape == x.shape
    assert aux["output_pre_chroma"].shape == x.shape
    assert aux["chroma_residual"].shape == x.shape
    assert aux["chroma_gate"].shape == (1, 1, 35, 41)
    assert torch.allclose(aux["output"], x, atol=1e-4), (aux["output"] - x).abs().max()


def test_nagi_v2_chroma_smooth_branch_identity_init():
    m = NagiV2(
        width=8,
        enc_blk_nums=(1, 1),
        middle_blk_num=1,
        dec_blk_nums=(1, 1),
        chroma_smooth_branch=True,
        chroma_smooth_strength=0.5,
        chroma_smooth_gate_bias=-12.0,
    ).eval()
    x = torch.rand(1, 3, 35, 41) * 2.0
    with torch.no_grad():
        aux = m(x, return_aux=True)
    assert aux["output"].shape == x.shape
    assert aux["chroma_residual"].shape == x.shape
    assert aux["chroma_smooth_gate"].shape == (1, 1, 35, 41)
    assert torch.allclose(aux["output"], x, atol=1e-4), (aux["output"] - x).abs().max()


def test_nagi_v2_input_highlight_guard_locks_highlights():
    m = NagiV2(
        width=8,
        enc_blk_nums=(1, 1),
        middle_blk_num=1,
        dec_blk_nums=(1, 1),
        highlight_protect_threshold=1.0,
        highlight_protect_transition=0.01,
        highlight_protect_strength=1.0,
    )
    x = torch.rand(1, 3, 32, 32) * 0.25
    x[:, :, 8:16, 8:16] = 4.0
    with torch.no_grad():
        aux = m(x, return_aux=True)
    assert aux["highlight_guard"][:, :, 8:16, 8:16].mean().item() > 0.99
    assert torch.allclose(aux["output"][:, :, 8:16, 8:16], x[:, :, 8:16, 8:16], atol=1e-5)


if __name__ == "__main__":
    test_forward_shape_and_dtype()
    test_size_must_be_multiple_of_8()
    test_asinh_reversibility_hdr()
    test_model_compress_matches_transforms()
    test_srgb_linear_roundtrip()
    test_identity_at_init_is_near_input()
    test_loss_runs_and_is_finite()
    test_nagi_v2_forward_aux_and_identity_init()
    test_nagi_v2_loss_runs_and_has_gradients()
    test_nagi_v2_preset_builds()
    test_nagi_v2_l_preset_builds_identity_and_shape()
    test_synthetic_poisson_gaussian_linear()
    test_correlated_noise_lag1_autocorrelation()
    test_nagi_v2_chroma_branch_identity_init()
    test_nagi_v2_chroma_smooth_branch_identity_init()
    test_nagi_v2_input_highlight_guard_locks_highlights()
    print("all tests passed")
