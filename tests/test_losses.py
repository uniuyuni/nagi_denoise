"""Unit tests for the texture-statistics loss (Phase 4B).

Run with: python -m pytest tests/test_losses.py  (or python tests/test_losses.py)
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nagi_denoise.losses import NagiV2Loss, texture_stats_loss
from nagi_denoise.transforms import linear_to_srgb, srgb_to_linear


torch.manual_seed(0)


def test_texture_stats_zero_for_identical_inputs():
    x = torch.rand(1, 3, 64, 64) * 0.6 + 0.05
    for asymmetric in (True, False):
        loss = texture_stats_loss(x, x, asymmetric=asymmetric)
        assert loss.item() < 1.0e-6, (asymmetric, loss.item())


def test_texture_stats_positive_when_candidate_blurrier():
    # High-frequency checkerboard-ish target; candidate is a heavily
    # low-pass-filtered version of it (strictly less local HF energy
    # everywhere), which must raise the loss under both formulations.
    target = torch.rand(1, 3, 64, 64) * 0.6 + 0.05
    kernel = torch.ones(3, 1, 7, 7) / 49.0
    padded = torch.nn.functional.pad(target, (3, 3, 3, 3), mode="reflect")
    candidate = torch.nn.functional.conv2d(padded, kernel, groups=3)

    for asymmetric in (True, False):
        loss = texture_stats_loss(candidate, target, asymmetric=asymmetric)
        assert loss.item() > 1.0e-5, (asymmetric, loss.item())


def test_texture_stats_zero_for_pure_chroma_change():
    # Build a target/candidate pair that differ ONLY in sRGB-luma-orthogonal
    # chroma (constructed in sRGB space, where the loss's luma is computed),
    # then round-trip back to linear light as the loss expects.
    target_linear = torch.rand(1, 3, 48, 48) * 0.5 + 0.1
    target_srgb = linear_to_srgb(target_linear)

    weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    raw_delta = torch.randn(1, 3, 48, 48) * 0.05
    luma_component = (raw_delta * weights).sum(dim=1, keepdim=True)
    # Remove the luma-weighted projection so the perturbation is pure chroma.
    delta_chroma = raw_delta - luma_component * weights / weights.pow(2).sum()

    candidate_srgb = (target_srgb + delta_chroma).clamp(0.0, 1.0)
    candidate_linear = srgb_to_linear(candidate_srgb)

    # Sanity check: the perturbation actually changed the image (chroma moved).
    assert not torch.allclose(candidate_linear, target_linear, atol=1e-4)

    loss = texture_stats_loss(candidate_linear, target_linear)
    assert loss.item() < 1.0e-4, loss.item()


def test_texture_stats_asymmetric_ignores_excess_energy():
    # Candidate has MORE local HF energy than target (over-restoration /
    # extra grain). The asymmetric term (deficit-only) must not penalize
    # this; the symmetric term must.
    target = torch.rand(1, 3, 64, 64) * 0.6 + 0.05
    noise = (torch.rand(1, 3, 64, 64) - 0.5) * 0.2
    candidate = (target + noise).clamp(0.0, None)

    loss_asym = texture_stats_loss(candidate, target, asymmetric=True)
    loss_sym = texture_stats_loss(candidate, target, asymmetric=False)
    assert loss_asym.item() < 1.0e-6, loss_asym.item()
    assert loss_sym.item() > 1.0e-5, loss_sym.item()


def test_texture_stats_wired_into_nagi_v2_loss():
    # texture_stats_weight=0 (default) must not add the key or move the loss;
    # texture_stats_weight>0 must add "texture_stats" and backprop cleanly.
    def compress_fn(x):
        return torch.asinh(x * 8.0) / 8.0

    target = torch.rand(2, 3, 32, 32) * 0.5 + 0.05
    pred_output = (target + torch.randn(2, 3, 32, 32) * 0.02).clamp_min(0.0)
    pred_output.requires_grad_(True)
    pred = {"output": pred_output, "base": pred_output}

    loss_off = NagiV2Loss(compress_fn=compress_fn, texture_stats_weight=0.0)
    out_off = loss_off(pred, target)
    assert "texture_stats" not in out_off

    loss_on = NagiV2Loss(compress_fn=compress_fn, texture_stats_weight=0.3)
    out_on = loss_on(pred, target)
    assert "texture_stats" in out_on
    out_on["total"].backward()
    assert pred_output.grad is not None
    assert torch.isfinite(pred_output.grad).all()


if __name__ == "__main__":
    test_texture_stats_zero_for_identical_inputs()
    test_texture_stats_positive_when_candidate_blurrier()
    test_texture_stats_zero_for_pure_chroma_change()
    test_texture_stats_asymmetric_ignores_excess_energy()
    test_texture_stats_wired_into_nagi_v2_loss()
    print("all tests passed")
