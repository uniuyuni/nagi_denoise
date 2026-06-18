"""Smoke tests for Nagi NR model and transforms.

Run with: python -m pytest tests/  (or just python tests/test_model.py)
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

# Fallback path so `python3 tests/test_model.py` works without an editable install.
# A real `pip install -e packages/nagi_nr` is preferred and makes this line a no-op.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "nagi_nr" / "src"))

from nagi_nr.model import NagiNR
from nagi_nr.nagiq import NagiQ, build_nagiq_preset
from nagi_nr.nagiperfect import NagiPerfect, build_nagiperfect_preset
from nagi_nr.realfast import NagiRealFast, build_realfast_preset
from nagi_nr.transforms import (
    asinh_compress,
    asinh_decompress,
    linear_to_srgb,
    srgb_to_linear,
)
from nagi_nr.losses import NagiLoss, NagiPerfectLoss


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


def test_nagiq_forward_shape_and_identity_init():
    m = NagiQ(width=8, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1)).eval()
    x = torch.rand(1, 3, 35, 41)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert torch.allclose(x, y, atol=1e-5), (x - y).abs().max()


def test_nagiq_preset_builds():
    m = build_nagiq_preset("q48-fast")
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape


def test_realfast_forward_shape_and_identity_init():
    m = NagiRealFast(width=8, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1)).eval()
    x = torch.rand(1, 3, 35, 41)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert torch.allclose(x, y, atol=1e-5), (x - y).abs().max()


def test_realfast_preset_builds():
    m = build_realfast_preset("realfast-v0-lite")
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape


def test_nagiperfect_forward_aux_and_identity_init():
    m = NagiPerfect(width=8, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1)).eval()
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


def test_nagiperfect_loss_runs_and_has_gradients():
    m = NagiPerfect(width=8, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1))
    crit = NagiPerfectLoss(compress_fn=m.compress)
    noisy = torch.rand(2, 3, 32, 32) * 3.0
    target = torch.rand(2, 3, 32, 32) * 3.0
    pred = m(noisy, return_aux=True)
    out = crit(pred, target)
    assert torch.isfinite(out["total"]).item()
    out["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in m.parameters())


def test_nagiperfect_preset_builds():
    m = build_nagiperfect_preset("perfect-s")
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape


def test_nagiperfect_chroma_branch_identity_init():
    m = NagiPerfect(
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


def test_nagiperfect_input_highlight_guard_locks_highlights():
    m = NagiPerfect(
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
    test_nagiq_forward_shape_and_identity_init()
    test_nagiq_preset_builds()
    test_realfast_forward_shape_and_identity_init()
    test_realfast_preset_builds()
    test_nagiperfect_forward_aux_and_identity_init()
    test_nagiperfect_loss_runs_and_has_gradients()
    test_nagiperfect_preset_builds()
    test_nagiperfect_input_highlight_guard_locks_highlights()
    print("all tests passed")
