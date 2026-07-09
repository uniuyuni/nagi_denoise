"""Choose a coarse SCUNet blend preset before local blending.

Local gates kept confusing "SCUNet restored detail" with "SCUNet restored
noise" on blue-shadow images. This script first computes downsampled scene
statistics, chooses a coarse preset, then runs the corresponding local blend.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_chroma_speckle_filter import PRESETS as CHROMA_SPECKLE_PRESETS
from apply_chroma_speckle_filter import apply_chroma_speckle_filter
from apply_dark_dot_speckle_filter import PRESETS as DARK_DOT_PRESETS
from apply_dark_dot_speckle_filter import apply_dark_dot_speckle_filter
from apply_detail_protected_flat_cleanup import apply_cleanup as apply_detail_protected_flat_cleanup
from apply_luma_hf_shrink_filter import PRESETS as LUMA_HF_PRESETS
from apply_luma_hf_shrink_filter import apply_luma_hf_shrink
from apply_luma_tail_speckle_filter import PRESETS as LUMA_TAIL_PRESETS
from apply_luma_tail_speckle_filter import apply_luma_tail_speckle_filter, sigmoid01
from apply_region_aware_flat_gate import apply_region_aware_gate, build_reopen_map, build_strength_map, _read_gate
from apply_neutral_chroma_dot_filter import PRESETS as NEUTRAL_CHROMA_PRESETS
from apply_neutral_chroma_dot_filter import apply_neutral_chroma_dot_filter
from apply_signed_chroma_outlier_filter import apply_signed_chroma_outlier_filter
from apply_blue_shadow_structure_graft import graft_structure
from apply_blue_structure_protector import blue_structure_mask
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import image_stats, make_preview, read_image
from train_scunet_policy import blend_policy_luma, load_model as load_policy_model, predict_policy_tiled
from train_scunet_selector import (
    LUMA_SRGB,
    RUN_ROOT,
    SCENES,
    blend_luma_with_hdr_restore,
    choose_device,
    display,
    hf_abs,
    load_model as load_selector_model,
    luma,
    make_coherent_structure_mask,
    make_texture_mask,
    predict_gate_tiled,
    signed_blue_magenta_risk,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_SELECTOR = RUN_ROOT / "scunet_selector_pilot_v2/scunet_selector_final.pt"
DEFAULT_V3_SELECTOR = RUN_ROOT / "scunet_selector_pilot_v3_luma/scunet_selector_final.pt"
DEFAULT_POLICY = RUN_ROOT / "scunet_policy_pilot_v3_balanced/scunet_policy_final.pt"
SIGNED_CHROMA_PRESETS = {
    "bm_strong": {
        "strength": 0.72,
        "median_size": 7,
        "low_sigma": 2.4,
        "outlier_threshold": 0.0032,
        "outlier_transition": 0.0022,
        "magenta_weight": 1.0,
        "red_weight": 0.0,
        "blue_weight": 1.05,
    },
}
BLUE_STRUCTURE_PROTECT_PRESETS = {
    "mild": {
        "strength": 0.72,
        "blue_threshold": 0.052,
        "blue_transition": 0.024,
        "chroma_threshold": 0.064,
        "chroma_transition": 0.030,
        "luma_min": 0.045,
        "luma_max": 0.82,
    },
}
POST_DARK_DOT_PRESETS = {
    "sky_tail": {
        "strength": 0.65,
        "median_size": 5,
        "dark_threshold": 0.0026,
        "dark_transition": 0.0018,
        "local_sigma": 2.4,
        "local_gain": 0.08,
        "shadow_boost": 0.55,
        "max_lift": 0.026,
        "chroma_strength": 0.35,
        "chroma_sigma": 1.8,
        "line_preserve_strength": 0.92,
        "blue_structure_inhibit": 0.92,
    },
}
REGION_AWARE_FLAT_GATE_PRESETS = {
    "quality_v3": {
        "base_strength": 0.16,
        "flat_boost": 1.05,
        "skin_strength": 0.08,
        "shadow_flat_boost": 0.0,
        "structure_suppress": 0.98,
        "highlight_suppress": 0.88,
        "flat_threshold": 0.044,
        "flat_transition": 0.018,
        "edge_threshold": 0.034,
        "edge_transition": 0.014,
        "min_strength": 0.02,
        "max_strength": 1.0,
        "blur_sigma": 1.05,
        "smooth_params": {
            "luma_strength": 0.94,
            "chroma_strength": 0.97,
            "luma_sigma": 2.75,
            "chroma_sigma": 3.45,
        },
    },
    "dark_sky": {
        "base_strength": 0.12,
        "flat_boost": 1.02,
        "skin_strength": 0.05,
        "shadow_flat_boost": 0.75,
        "structure_suppress": 0.99,
        "highlight_suppress": 0.90,
        "flat_threshold": 0.044,
        "flat_transition": 0.018,
        "edge_threshold": 0.031,
        "edge_transition": 0.012,
        "min_strength": 0.012,
        "max_strength": 1.0,
        "blur_sigma": 1.05,
        "smooth_params": {
            "luma_strength": 0.94,
            "chroma_strength": 0.97,
            "luma_sigma": 2.75,
            "chroma_sigma": 3.45,
        },
    },
    "dark_sky_wide": {
        "base_strength": 0.12,
        "flat_boost": 1.02,
        "skin_strength": 0.05,
        "shadow_flat_boost": 0.75,
        "structure_suppress": 0.99,
        "highlight_suppress": 0.90,
        "flat_threshold": 0.044,
        "flat_transition": 0.018,
        "edge_threshold": 0.031,
        "edge_transition": 0.012,
        "min_strength": 0.012,
        "max_strength": 1.0,
        "blur_sigma": 1.05,
        "smooth_params": {
            "luma_strength": 0.94,
            "chroma_strength": 0.97,
            "luma_sigma": 4.0,
            "chroma_sigma": 5.0,
        },
    },
    "dark_sky_strict": {
        "base_strength": 0.12,
        "flat_boost": 1.02,
        "skin_strength": 0.05,
        "shadow_flat_boost": 0.75,
        "shadow_luma_threshold": 0.22,
        "shadow_luma_transition": 0.06,
        "structure_suppress": 0.99,
        "highlight_suppress": 0.90,
        "flat_threshold": 0.044,
        "flat_transition": 0.018,
        "edge_threshold": 0.031,
        "edge_transition": 0.012,
        "min_strength": 0.012,
        "max_strength": 1.0,
        "blur_sigma": 1.05,
        "smooth_params": {
            "luma_strength": 0.94,
            "chroma_strength": 0.97,
            "luma_sigma": 2.75,
            "chroma_sigma": 3.45,
        },
    },
    "dark_sky_strict_reopen": {
        "base_strength": 0.12,
        "flat_boost": 1.02,
        "skin_strength": 0.05,
        "shadow_flat_boost": 0.75,
        "shadow_luma_threshold": 0.22,
        "shadow_luma_transition": 0.06,
        "structure_suppress": 0.99,
        "highlight_suppress": 0.90,
        "flat_threshold": 0.044,
        "flat_transition": 0.018,
        "edge_threshold": 0.031,
        "edge_transition": 0.012,
        "min_strength": 0.012,
        "max_strength": 1.0,
        "blur_sigma": 1.05,
        "reopen_strength": 0.85,
        "reopen_shadow_weight": 1.0,
        "reopen_structure_suppress": 1.0,
        "reopen_min": 1.0,
        "reopen_max": 1.45,
        "reopen_shadow_threshold": 0.40,
        "reopen_shadow_transition": 0.12,
        "smooth_params": {
            "luma_strength": 0.94,
            "chroma_strength": 0.97,
            "luma_sigma": 2.75,
            "chroma_sigma": 3.45,
        },
    },
    "dark_sky_strict_reopen_skyonly": {
        "base_strength": 0.04,
        "flat_boost": 0.78,
        "skin_strength": 0.02,
        "shadow_flat_boost": 1.55,
        "shadow_luma_threshold": 0.22,
        "shadow_luma_transition": 0.06,
        "structure_suppress": 0.995,
        "highlight_suppress": 0.92,
        "flat_threshold": 0.044,
        "flat_transition": 0.018,
        "edge_threshold": 0.031,
        "edge_transition": 0.012,
        "min_strength": 0.006,
        "max_strength": 1.0,
        "blur_sigma": 1.05,
        "reopen_strength": 1.05,
        "reopen_shadow_weight": 1.0,
        "reopen_structure_suppress": 1.0,
        "reopen_min": 1.0,
        "reopen_max": 1.65,
        "reopen_shadow_threshold": 0.40,
        "reopen_shadow_transition": 0.12,
        "smooth_params": {
            "luma_strength": 0.94,
            "chroma_strength": 0.97,
            "luma_sigma": 2.75,
            "chroma_sigma": 3.45,
        },
    },
    "dark_sky_strict_reopen_skyonly_soft_limiter": {
        "base_strength": 0.04,
        "flat_boost": 0.78,
        "skin_strength": 0.02,
        "shadow_flat_boost": 1.55,
        "shadow_luma_threshold": 0.22,
        "shadow_luma_transition": 0.06,
        "structure_suppress": 0.995,
        "highlight_suppress": 0.92,
        "flat_threshold": 0.044,
        "flat_transition": 0.018,
        "edge_threshold": 0.031,
        "edge_transition": 0.012,
        "min_strength": 0.006,
        "max_strength": 1.0,
        "blur_sigma": 1.05,
        "reopen_strength": 1.05,
        "reopen_shadow_weight": 1.0,
        "reopen_structure_suppress": 1.0,
        "reopen_min": 1.0,
        "reopen_max": 1.65,
        "reopen_shadow_threshold": 0.40,
        "reopen_shadow_transition": 0.12,
        "limiter_strength": 0.18,
        "limiter_min": 0.82,
        "limiter_flat_threshold": 0.30,
        "limiter_flat_transition": 0.30,
        "limiter_shadow_threshold": 0.18,
        "limiter_shadow_transition": 0.24,
        "limiter_structure_suppress": 0.35,
        "smooth_params": {
            "luma_strength": 0.94,
            "chroma_strength": 0.97,
            "luma_sigma": 2.75,
            "chroma_sigma": 3.45,
        },
    },
}

FLAT_CLEANUP_PRESETS = {
    "strong_v1": {
        "luma_strength": 0.92,
        "chroma_strength": 0.95,
        "luma_sigma": 2.20,
        "chroma_sigma": 3.00,
        "flat_threshold": 0.030,
        "flat_transition": 0.014,
        "edge_threshold": 0.030,
        "edge_transition": 0.014,
        "coherent_protect": 0.95,
        "texture_protect": 0.72,
        "detail_gate_protect": 1.20,
        "skin_protect": 0.40,
        "highlight_threshold": 1.0,
        "highlight_transition": 0.25,
        "hdr_restore_threshold": 0.92,
        "hdr_restore_transition": 0.24,
        "gate_blur": 1.20,
        "auto_detail_protect": 0.0,
        "auto_detail_threshold": 0.018,
        "auto_detail_transition": 0.010,
    },
    "detail_v2": {
        "luma_strength": 0.96,
        "chroma_strength": 0.97,
        "luma_sigma": 2.45,
        "chroma_sigma": 3.25,
        "flat_threshold": 0.033,
        "flat_transition": 0.015,
        "edge_threshold": 0.032,
        "edge_transition": 0.014,
        "coherent_protect": 0.98,
        "texture_protect": 0.76,
        "detail_gate_protect": 1.20,
        "auto_detail_protect": 0.55,
        "auto_detail_threshold": 0.016,
        "auto_detail_transition": 0.010,
        "skin_protect": 0.45,
        "highlight_threshold": 1.0,
        "highlight_transition": 0.25,
        "hdr_restore_threshold": 0.92,
        "hdr_restore_transition": 0.24,
        "gate_blur": 1.25,
    },
}


def downsample(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return image
    return zoom(image, (scale, scale, 1), order=1).astype(np.float32, copy=False)


def downsample_mask(mask: np.ndarray, max_side: int) -> np.ndarray:
    h, w = mask.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return mask
    return zoom(mask, (scale, scale), order=1).astype(np.float32, copy=False)


def scene_metrics(noisy_linear: np.ndarray, current_linear: np.ndarray, scunet_linear: np.ndarray) -> dict[str, float]:
    noisy = display(noisy_linear)
    current = display(current_linear)
    scunet = display(scunet_linear)
    noisy_y = luma(noisy, LUMA_SRGB)
    current_y = luma(current, LUMA_SRGB)
    scunet_y = luma(scunet, LUMA_SRGB)
    current_hf = hf_abs(current_y, 0.8)
    scunet_hf = hf_abs(scunet_y, 0.8)
    signed = signed_blue_magenta_risk(scunet)
    shadow = sigmoid01((0.50 - current_y) / 0.16)
    tail = sigmoid01((scunet_hf - current_hf - 0.0025) / 0.0075)
    coherent = make_coherent_structure_mask(
        noisy,
        coherence_threshold=0.38,
        coherence_transition=0.18,
        energy_threshold=0.0048,
        energy_transition=0.0055,
    )
    current_texture = make_texture_mask(current, texture_threshold=0.014, texture_transition=0.014)
    scunet_texture = make_texture_mask(scunet, texture_threshold=0.014, texture_transition=0.014)
    blue_tail = signed * shadow * tail
    blue_struct = signed * shadow * np.maximum(coherent, scunet_texture)
    luma_benefit = sigmoid01((current_hf - scunet_hf - 0.0015) / 0.006)
    return {
        "signed_mean": float(np.mean(signed)),
        "signed_p90": float(np.quantile(signed, 0.90)),
        "shadow_mean": float(np.mean(shadow)),
        "blue_tail_mean": float(np.mean(blue_tail)),
        "blue_tail_p95": float(np.quantile(blue_tail, 0.95)),
        "blue_struct_mean": float(np.mean(blue_struct)),
        "blue_struct_p95": float(np.quantile(blue_struct, 0.95)),
        "tail_mean": float(np.mean(tail)),
        "tail_p95": float(np.quantile(tail, 0.95)),
        "coherent_mean": float(np.mean(coherent)),
        "current_texture_mean": float(np.mean(current_texture)),
        "scunet_texture_mean": float(np.mean(scunet_texture)),
        "luma_benefit_mean": float(np.mean(luma_benefit)),
        "current_luma_hf_mean": float(np.mean(current_hf)),
        "scunet_luma_hf_mean": float(np.mean(scunet_hf)),
    }


def choose_preset(metrics: dict[str, float]) -> tuple[str, str]:
    if metrics["blue_struct_mean"] >= 0.13 or metrics["blue_tail_p95"] >= 0.32:
        return "blue_shadow_safe", "blue-shadow structure/tail risk is high"
    if metrics["current_texture_mean"] >= 0.50 and metrics["signed_mean"] <= 0.26:
        return "hair_luma", "texture is high and signed chroma risk is low"
    return "sky_luma", "default flat/sky cleanup preset"


def apply_selector_preset(
    checkpoint: Path,
    device: torch.device,
    noisy: np.ndarray,
    current: np.ndarray,
    scunet: np.ndarray,
    *,
    tile: int,
    overlap: int,
    strength: float,
    gate_gamma: float,
    risk_inhibit: float,
    tail_inhibit: float,
    edge_inhibit: float,
) -> tuple[np.ndarray, dict[str, float]]:
    model = load_selector_model(checkpoint, device)
    gate = predict_gate_tiled(model, device, noisy, current, scunet, tile=tile, overlap=overlap)
    out = blend_luma_with_hdr_restore(
        current,
        scunet,
        gate,
        strength=strength,
        gate_gamma=gate_gamma,
        chroma_source_mix=0.0,
        chroma_limit=8.0,
        risk_inhibit=risk_inhibit,
        tail_inhibit=tail_inhibit,
        edge_inhibit=edge_inhibit,
        edge_inhibit_mode="detail_loss",
        edge_threshold=0.026,
        edge_transition=0.014,
        hdr_peak_threshold=0.82,
        hdr_transition=0.25,
    )
    gate_stats = {
        "mean": float(np.mean(gate)),
        "p50": float(np.quantile(gate, 0.50)),
        "p90": float(np.quantile(gate, 0.90)),
        "p99": float(np.quantile(gate, 0.99)),
    }
    return out, gate_stats


def apply_policy_preset(
    checkpoint: Path,
    device: torch.device,
    noisy: np.ndarray,
    current: np.ndarray,
    scunet: np.ndarray,
    *,
    tile: int,
    overlap: int,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    model = load_policy_model(checkpoint, device)
    policy = predict_policy_tiled(model, device, noisy, current, scunet, tile=tile, overlap=overlap)
    out = blend_policy_luma(
        current,
        scunet,
        policy,
        strength=2.4,
        gate_gamma=0.75,
        edge_inhibit=0.80,
        risk_inhibit=0.90,
        hdr_peak_threshold=0.82,
        hdr_transition=0.25,
    )
    return out, {
        label: {
            "mean": float(np.mean(policy[..., idx])),
            "p50": float(np.quantile(policy[..., idx], 0.50)),
            "p90": float(np.quantile(policy[..., idx], 0.90)),
            "p99": float(np.quantile(policy[..., idx], 0.99)),
        }
        for idx, label in enumerate(("luma_gate", "edge_keep", "risk_gate"))
    }


def apply_blue_shadow_graft_preset(
    noisy: np.ndarray,
    current: np.ndarray,
    scunet: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    out, stats, _ = graft_structure(
        noisy,
        current,
        scunet,
        strength=0.80,
        fine_sigma=0.65,
        coarse_sigma=2.20,
        gate_threshold=0.018,
        gate_transition=0.018,
        coherent_weight=0.55,
        texture_weight=0.25,
        correction_limit=0.045,
        tail_suppress=0.65,
        gate_blur=0.60,
    )
    return out, stats


def choose_luma_tail_preset(preset: str, requested: str) -> str | None:
    if requested == "off":
        return None
    if requested != "auto":
        return requested
    if preset == "blue_shadow_safe":
        return "strong"
    return "xstrong"


def apply_luma_tail_postfilter(image: np.ndarray, preset: str) -> tuple[np.ndarray, dict[str, float]]:
    params = dict(LUMA_TAIL_PRESETS[preset])
    out, stats, _ = apply_luma_tail_speckle_filter(
        image,
        image,
        strength=float(params["strength"]),
        median_size=int(params["median_size"]),
        highpass_sigma=0.9,
        local_sigma=3.0,
        local_gain=float(params["local_gain"]),
        tail_threshold=float(params["tail_threshold"]),
        tail_transition=float(params["tail_transition"]),
        structure_sigma=1.2,
        detail_sigma=2.8,
        detail_threshold=0.018,
        detail_transition=0.010,
        edge_sigma=1.0,
        edge_threshold=0.030,
        edge_transition=0.015,
        highlight_threshold=1.0,
        highlight_transition=0.25,
        hdr_restore_peak_threshold=0.95,
        hdr_restore_threshold=0.85,
        hdr_restore_transition=0.25,
        correction_limit=0.035,
    )
    return out, stats


def choose_chroma_speckle_preset(requested: str) -> str | None:
    if requested == "off":
        return None
    if requested == "auto":
        return "axisplus"
    return requested


def apply_chroma_speckle_postfilter(image: np.ndarray, preset: str) -> tuple[np.ndarray, dict[str, float]]:
    params = dict(CHROMA_SPECKLE_PRESETS[preset])
    out, stats, _ = apply_chroma_speckle_filter(
        image,
        image,
        strength=float(params["strength"]),
        chroma_sigma=float(params["chroma_sigma"]),
        median_size=int(params["median_size"]),
        speckle_threshold=float(params["speckle_threshold"]),
        speckle_transition=float(params["speckle_transition"]),
        local_sigma=float(params["local_sigma"]),
        local_gain=float(params["local_gain"]),
        axis_boost=float(params["axis_boost"]),
        axis_threshold=float(params["axis_threshold"]),
        axis_transition=float(params["axis_transition"]),
        magenta_boost=float(params["magenta_boost"]),
        magenta_threshold=float(params["magenta_threshold"]),
        magenta_transition=float(params["magenta_transition"]),
        structure_sigma=1.2,
        detail_sigma=2.8,
        detail_threshold=0.018,
        detail_transition=0.010,
        edge_sigma=1.0,
        edge_threshold=0.030,
        edge_transition=0.015,
        highlight_threshold=float(params["highlight_threshold"]),
        highlight_transition=float(params["highlight_transition"]),
        hdr_restore_peak_threshold=float(params["hdr_restore_peak_threshold"]),
        hdr_restore_threshold=float(params["hdr_restore_threshold"]),
        hdr_restore_transition=float(params["hdr_restore_transition"]),
    )
    return out, stats


def choose_dark_dot_preset(preset: str, requested: str) -> str | None:
    if requested == "off":
        return None
    if requested == "auto":
        if preset == "blue_shadow_safe":
            return "strong"
        return "sky"
    return requested


def apply_dark_dot_postfilter(image: np.ndarray, preset: str) -> tuple[np.ndarray, dict[str, float]]:
    params = dict(DARK_DOT_PRESETS[preset])
    out, stats, _ = apply_dark_dot_speckle_filter(
        image,
        image,
        strength=float(params["strength"]),
        median_size=int(params["median_size"]),
        dark_threshold=float(params["dark_threshold"]),
        dark_transition=float(params["dark_transition"]),
        local_sigma=float(params["local_sigma"]),
        local_gain=float(params["local_gain"]),
        shadow_boost=float(params["shadow_boost"]),
        shadow_threshold=0.24,
        shadow_transition=0.10,
        max_lift=float(params["max_lift"]),
        chroma_strength=float(params["chroma_strength"]),
        chroma_sigma=float(params["chroma_sigma"]),
        structure_sigma=1.2,
        detail_sigma=2.8,
        detail_threshold=0.018,
        detail_transition=0.010,
        edge_sigma=1.0,
        edge_threshold=0.030,
        edge_transition=0.015,
        highlight_threshold=1.0,
        highlight_transition=0.25,
        line_sigma=0.70,
        line_smooth_sigma=1.00,
        line_threshold=0.010,
        line_transition=0.006,
        line_coherence_threshold=0.42,
        line_coherence_transition=0.16,
        line_preserve_strength=float(params["line_preserve_strength"]),
        blue_structure_inhibit=0.0,
        blue_structure_threshold=0.052,
        blue_structure_transition=0.024,
        blue_structure_chroma_threshold=0.064,
        blue_structure_chroma_transition=0.030,
        sky_flat_strength=0.0,
        sky_luma_min=0.025,
        sky_luma_max=0.46,
        sky_luma_transition=0.075,
        sky_neutral_threshold=0.105,
        sky_neutral_transition=0.045,
        sky_blue_abs_threshold=0.045,
        sky_blue_abs_transition=0.024,
        sky_line_max=0.34,
        sky_line_transition=0.16,
        hdr_restore_peak_threshold=0.95,
        hdr_restore_threshold=0.85,
        hdr_restore_transition=0.25,
    )
    return out, stats


def choose_luma_hf_preset(requested: str) -> str | None:
    if requested == "off":
        return None
    if requested == "auto":
        return "grain"
    return requested


def apply_luma_hf_postfilter(image: np.ndarray, preset: str) -> tuple[np.ndarray, dict[str, float]]:
    params = dict(LUMA_HF_PRESETS[preset])
    out, stats, _ = apply_luma_hf_shrink(
        image,
        image,
        strength=float(params["strength"]),
        low_sigma=float(params["low_sigma"]),
        shrink_threshold=float(params["shrink_threshold"]),
        detail_preserve_threshold=float(params["detail_preserve_threshold"]),
        detail_preserve_transition=float(params["detail_preserve_transition"]),
        shadow_boost=float(params["shadow_boost"]),
        line_sigma=float(params["line_sigma"]),
        line_smooth_sigma=float(params["line_smooth_sigma"]),
        line_threshold=float(params["line_threshold"]),
        line_transition=float(params["line_transition"]),
        line_coherence_threshold=float(params["line_coherence_threshold"]),
        line_coherence_transition=float(params["line_coherence_transition"]),
        line_preserve_strength=float(params["line_preserve_strength"]),
        shadow_threshold=0.18,
        shadow_transition=0.08,
        structure_sigma=1.2,
        detail_sigma=2.8,
        detail_threshold=0.018,
        detail_transition=0.010,
        edge_sigma=1.0,
        edge_threshold=0.030,
        edge_transition=0.015,
        highlight_threshold=1.0,
        highlight_transition=0.25,
        hdr_restore_peak_threshold=0.95,
        hdr_restore_threshold=0.85,
        hdr_restore_transition=0.25,
    )
    return out, stats


def choose_signed_chroma_preset(requested: str) -> str | None:
    if requested == "off":
        return None
    if requested == "auto":
        return "bm_strong"
    return requested


def apply_signed_chroma_postfilter(image: np.ndarray, preset: str) -> tuple[np.ndarray, dict[str, float]]:
    params = dict(SIGNED_CHROMA_PRESETS[preset])
    out, stats, _ = apply_signed_chroma_outlier_filter(
        image,
        image,
        strength=float(params["strength"]),
        median_size=int(params["median_size"]),
        low_sigma=float(params["low_sigma"]),
        outlier_threshold=float(params["outlier_threshold"]),
        outlier_transition=float(params["outlier_transition"]),
        magenta_weight=float(params["magenta_weight"]),
        red_weight=float(params["red_weight"]),
        blue_weight=float(params["blue_weight"]),
        structure_sigma=1.2,
        detail_sigma=2.8,
        detail_threshold=0.020,
        detail_transition=0.010,
        edge_sigma=1.0,
        edge_threshold=0.030,
        edge_transition=0.015,
        shadow_threshold=0.58,
        shadow_transition=0.18,
        highlight_threshold=0.95,
        highlight_transition=0.25,
        hdr_restore_peak_threshold=0.95,
        hdr_restore_threshold=0.85,
        hdr_restore_transition=0.25,
    )
    return out, stats


def choose_neutral_chroma_preset(requested: str) -> str | None:
    if requested == "off":
        return None
    if requested == "auto":
        return "neutral_strong"
    return requested


def apply_neutral_chroma_postfilter(image: np.ndarray, preset: str) -> tuple[np.ndarray, dict[str, float]]:
    params = dict(NEUTRAL_CHROMA_PRESETS[preset])
    out, stats, _ = apply_neutral_chroma_dot_filter(
        image,
        image,
        strength=float(params["strength"]),
        median_size=int(params["median_size"]),
        low_sigma=float(params["low_sigma"]),
        outlier_threshold=float(params["outlier_threshold"]),
        outlier_transition=float(params["outlier_transition"]),
        magenta_weight=float(params["magenta_weight"]),
        blue_weight=float(params["blue_weight"]),
        neutral_threshold=float(params["neutral_threshold"]),
        neutral_transition=float(params["neutral_transition"]),
        structure_sigma=1.2,
        detail_sigma=2.8,
        detail_threshold=0.020,
        detail_transition=0.010,
        edge_sigma=1.0,
        edge_threshold=0.030,
        edge_transition=0.015,
        shadow_threshold=float(params["shadow_threshold"]),
        shadow_transition=0.18,
        highlight_threshold=0.95,
        highlight_transition=0.25,
        hdr_restore_peak_threshold=0.95,
        hdr_restore_threshold=0.85,
        hdr_restore_transition=0.25,
    )
    return out, stats


def choose_blue_structure_protect_preset(requested: str) -> str | None:
    if requested == "off":
        return None
    if requested == "auto":
        return "mild"
    return requested


def apply_blue_structure_protect_postfilter(
    base: np.ndarray,
    candidate: np.ndarray,
    preset: str,
) -> tuple[np.ndarray, dict[str, float]]:
    params = dict(BLUE_STRUCTURE_PROTECT_PRESETS[preset])
    mask = blue_structure_mask(
        base,
        blue_threshold=float(params["blue_threshold"]),
        blue_transition=float(params["blue_transition"]),
        chroma_threshold=float(params["chroma_threshold"]),
        chroma_transition=float(params["chroma_transition"]),
        luma_min=float(params["luma_min"]),
        luma_max=float(params["luma_max"]),
    )
    restore = np.clip(mask * float(params["strength"]), 0.0, 1.0).astype(np.float32, copy=False)
    out = candidate * (1.0 - restore[..., None]) + base * restore[..., None]
    stats = {
        "restore_mean": float(np.mean(restore)),
        "restore_p95": float(np.quantile(restore, 0.95)),
        "restore_p99": float(np.quantile(restore, 0.99)),
    }
    return out, stats


def choose_post_dark_dot_preset(preset: str, requested: str) -> str | None:
    if requested == "off":
        return None
    if requested == "auto":
        if preset == "sky_luma":
            return "sky_tail"
        return None
    return requested


def apply_post_dark_dot_filter(image: np.ndarray, preset: str) -> tuple[np.ndarray, dict[str, float]]:
    params = dict(POST_DARK_DOT_PRESETS[preset])
    out, stats, _ = apply_dark_dot_speckle_filter(
        image,
        image,
        strength=float(params["strength"]),
        median_size=int(params["median_size"]),
        dark_threshold=float(params["dark_threshold"]),
        dark_transition=float(params["dark_transition"]),
        local_sigma=float(params["local_sigma"]),
        local_gain=float(params["local_gain"]),
        shadow_boost=float(params["shadow_boost"]),
        shadow_threshold=0.24,
        shadow_transition=0.10,
        max_lift=float(params["max_lift"]),
        chroma_strength=float(params["chroma_strength"]),
        chroma_sigma=float(params["chroma_sigma"]),
        structure_sigma=1.2,
        detail_sigma=2.8,
        detail_threshold=0.018,
        detail_transition=0.010,
        edge_sigma=1.0,
        edge_threshold=0.030,
        edge_transition=0.015,
        highlight_threshold=1.0,
        highlight_transition=0.25,
        line_sigma=0.70,
        line_smooth_sigma=1.00,
        line_threshold=0.010,
        line_transition=0.006,
        line_coherence_threshold=0.42,
        line_coherence_transition=0.16,
        line_preserve_strength=float(params["line_preserve_strength"]),
        blue_structure_inhibit=float(params["blue_structure_inhibit"]),
        blue_structure_threshold=0.052,
        blue_structure_transition=0.024,
        blue_structure_chroma_threshold=0.064,
        blue_structure_chroma_transition=0.030,
        sky_flat_strength=0.0,
        sky_luma_min=0.025,
        sky_luma_max=0.46,
        sky_luma_transition=0.075,
        sky_neutral_threshold=0.105,
        sky_neutral_transition=0.045,
        sky_blue_abs_threshold=0.045,
        sky_blue_abs_transition=0.024,
        sky_line_max=0.34,
        sky_line_transition=0.16,
        hdr_restore_peak_threshold=0.95,
        hdr_restore_threshold=0.85,
        hdr_restore_transition=0.25,
    )
    return out, stats


def summarize_mask(mask: np.ndarray) -> dict[str, float]:
    x = np.asarray(mask, dtype=np.float32)
    return {
        "mean": float(np.mean(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "gt_025": float(np.mean(x > 0.25)),
        "gt_040": float(np.mean(x > 0.40)),
        "gt_055": float(np.mean(x > 0.55)),
    }


def region_aware_reopen_guard(
    reference: np.ndarray,
    image: np.ndarray,
    preset: str,
    *,
    min_candidate_gt040: float,
    max_structure_mean: float,
) -> dict[str, object]:
    params = dict(REGION_AWARE_FLAT_GATE_PRESETS[preset])
    params.pop("smooth_params", None)
    for key in list(params):
        if key.startswith("reopen_") or key.startswith("limiter_"):
            params.pop(key)
    _, _, masks = build_strength_map(reference, image, **params)
    shadow_gate = np.clip((masks["shadow_flat"] - 0.40) / 0.12, 0.0, 1.0)
    safe = np.clip(1.0 - masks["structure_protect"], 0.0, 1.0)
    candidate = np.clip(masks["flat"] * shadow_gate * safe, 0.0, 1.0)
    candidate_stats = summarize_mask(candidate)
    structure_stats = summarize_mask(masks["structure_protect"])
    passed = (
        candidate_stats["gt_040"] >= float(min_candidate_gt040)
        and structure_stats["mean"] <= float(max_structure_mean)
    )
    return {
        "passed": bool(passed),
        "preset": preset,
        "min_candidate_gt040": float(min_candidate_gt040),
        "max_structure_mean": float(max_structure_mean),
        "candidate": candidate_stats,
        "structure_protect": structure_stats,
    }


def region_aware_spill_risk(
    reference: np.ndarray,
    image: np.ndarray,
    gate_path: Path,
    preset: str,
    *,
    max_side: int = 1400,
    soft_ratio_threshold: float = 0.50,
) -> dict[str, object]:
    gate = _read_gate(gate_path, image.shape[:2])
    reference_small = downsample(reference, max_side)
    image_small = downsample(image, max_side)
    gate_small = downsample_mask(gate, max_side)
    params = dict(REGION_AWARE_FLAT_GATE_PRESETS[preset])
    params.pop("smooth_params", None)
    reopen_params = {key: params.pop(key) for key in list(params) if key.startswith("reopen_")}
    for key in list(params):
        if key.startswith("limiter_"):
            params.pop(key)
    strength, build_stats, masks = build_strength_map(reference_small, image_small, **params)
    reopen = build_reopen_map(masks, **reopen_params)
    effective = np.clip(gate_small * strength * reopen, 0.0, 1.0).astype(np.float32, copy=False)
    structure = masks["structure_protect"]
    shadow_flat = masks["shadow_flat"]
    target_weight = np.clip(gate_small * shadow_flat * (1.0 - structure), 0.0, 1.0)
    spill_weight = np.clip(gate_small * structure * (0.35 + 0.65 * shadow_flat), 0.0, 1.0)
    target_mean = float(np.sum(effective * target_weight) / max(float(np.sum(target_weight)), 1.0e-6))
    spill_mean = float(np.sum(effective * spill_weight) / max(float(np.sum(spill_weight)), 1.0e-6))
    spill_ratio = float(spill_mean / max(target_mean, 1.0e-6))
    target_coverage = float(np.mean(target_weight > 0.20))
    spill_coverage = float(np.mean(spill_weight > 0.20))
    use_soft_limiter = spill_ratio >= float(soft_ratio_threshold)
    return {
        "base_preset": preset,
        "chosen_preset": "dark_sky_strict_reopen_skyonly_soft_limiter" if use_soft_limiter else preset,
        "reason": "spill risk is high" if use_soft_limiter else "target cleanup dominates spill risk",
        "soft_ratio_threshold": float(soft_ratio_threshold),
        "target_mean": target_mean,
        "spill_mean": spill_mean,
        "spill_ratio": spill_ratio,
        "target_coverage": target_coverage,
        "spill_coverage": spill_coverage,
        "effective_gate": summarize_mask(effective),
        "target_weight": summarize_mask(target_weight),
        "spill_weight": summarize_mask(spill_weight),
        "structure_protect": summarize_mask(structure),
        "shadow_flat": summarize_mask(shadow_flat),
        "build_stats": build_stats,
    }


def is_reopen_region_aware_preset(preset: str) -> bool:
    params = REGION_AWARE_FLAT_GATE_PRESETS.get(preset)
    return bool(params and float(params.get("reopen_strength", 0.0)) > 0.0)


def choose_region_aware_flat_gate_preset(scene_preset: str, requested: str) -> str | None:
    if requested == "off":
        return None
    if requested == "auto":
        if scene_preset == "sky_luma":
            return "dark_sky_strict"
        return None
    if requested == "auto_reopen":
        if scene_preset == "sky_luma":
            return "dark_sky_strict_reopen"
        return None
    if requested == "auto_reopen_skyonly":
        if scene_preset == "sky_luma":
            return "dark_sky_strict_reopen_skyonly"
        return None
    if requested == "auto_reopen_skyonly_adaptive":
        if scene_preset == "sky_luma":
            return "auto_reopen_skyonly_adaptive"
        return None
    return requested


def flat_gate_path(gate_dir: Path, scene_name: str) -> Path:
    return gate_dir / f"{scene_name}_flat_cleanup_gate_v12_native_pilot_v1_gate.png"


def apply_region_aware_flat_gate_filter(
    reference: np.ndarray,
    image: np.ndarray,
    gate_path: Path,
    preset: str,
) -> tuple[np.ndarray, dict[str, float]]:
    params = dict(REGION_AWARE_FLAT_GATE_PRESETS[preset])
    smooth_params = dict(params.pop("smooth_params"))
    gate = _read_gate(gate_path, image.shape[:2])
    out, stats, _ = apply_region_aware_gate(reference, image, gate, smooth_params=smooth_params, **params)
    stats.update({f"smooth_{name}": float(value) for name, value in smooth_params.items()})
    return out, stats


def choose_flat_cleanup_preset(requested: str) -> str | None:
    if requested == "off":
        return None
    if requested == "auto":
        return "strong_v1"
    return requested


def apply_flat_cleanup_filter(reference: np.ndarray, image: np.ndarray, preset: str) -> tuple[np.ndarray, dict[str, float]]:
    params = dict(FLAT_CLEANUP_PRESETS[preset])
    detail_gate = np.zeros(image.shape[:2], dtype=np.float32)
    out, stats, _ = apply_detail_protected_flat_cleanup(
        reference,
        image,
        detail_gate,
        luma_strength=float(params["luma_strength"]),
        chroma_strength=float(params["chroma_strength"]),
        luma_sigma=float(params["luma_sigma"]),
        chroma_sigma=float(params["chroma_sigma"]),
        flat_threshold=float(params["flat_threshold"]),
        flat_transition=float(params["flat_transition"]),
        edge_threshold=float(params["edge_threshold"]),
        edge_transition=float(params["edge_transition"]),
        coherent_protect=float(params["coherent_protect"]),
        texture_protect=float(params["texture_protect"]),
        detail_gate_protect=float(params["detail_gate_protect"]),
        auto_detail_protect=float(params["auto_detail_protect"]),
        auto_detail_threshold=float(params["auto_detail_threshold"]),
        auto_detail_transition=float(params["auto_detail_transition"]),
        skin_protect=float(params["skin_protect"]),
        highlight_threshold=float(params["highlight_threshold"]),
        highlight_transition=float(params["highlight_transition"]),
        hdr_restore_threshold=float(params["hdr_restore_threshold"]),
        hdr_restore_transition=float(params["hdr_restore_transition"]),
        gate_blur=float(params["gate_blur"]),
    )
    return out, stats


def resolve_inputs(args: argparse.Namespace) -> tuple[str, Path, Path, Path]:
    if args.scene:
        scene = SCENES[args.scene]
        return scene.name, scene.noisy, scene.current, scene.scunet
    if not (args.noisy and args.current and args.scunet):
        raise SystemExit("--scene or all of --noisy/--current/--scunet is required")
    name = args.scene_name or Path(args.noisy).stem
    return name, Path(args.noisy), Path(args.current), Path(args.scunet)


def apply(args: argparse.Namespace) -> None:
    scene_name, noisy_path, current_path, scunet_path = resolve_inputs(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    noisy = read_image(noisy_path)
    current = read_image(current_path)
    scunet = read_image(scunet_path)
    metrics = scene_metrics(
        downsample(noisy, args.metric_max_side),
        downsample(current, args.metric_max_side),
        downsample(scunet, args.metric_max_side),
    )
    preset, reason = choose_preset(metrics)
    if args.force_preset:
        preset = args.force_preset
        reason = "forced by CLI"

    if args.precomputed_base:
        out = read_image(Path(args.precomputed_base))
        if out.shape[:2] != noisy.shape[:2]:
            raise ValueError(f"precomputed base shape mismatch: noisy={noisy.shape} base={out.shape}")
        model_stats = {"precomputed_base": str(Path(args.precomputed_base))}
        model_kind = "precomputed_base"
    elif preset == "hair_luma":
        out, model_stats = apply_selector_preset(
            Path(args.v2_selector),
            device,
            noisy,
            current,
            scunet,
            tile=args.tile,
            overlap=args.overlap,
            strength=2.4,
            gate_gamma=0.75,
            risk_inhibit=0.0,
            tail_inhibit=0.0,
            edge_inhibit=0.0,
        )
        model_kind = "selector_v2_luma"
    elif preset == "sky_luma":
        out, model_stats = apply_selector_preset(
            Path(args.v3_selector),
            device,
            noisy,
            current,
            scunet,
            tile=args.tile,
            overlap=args.overlap,
            strength=2.4,
            gate_gamma=0.75,
            risk_inhibit=0.0,
            tail_inhibit=0.0,
            edge_inhibit=0.80,
        )
        model_kind = "selector_v3_luma"
    elif preset == "blue_shadow_safe":
        out, model_stats = apply_blue_shadow_graft_preset(noisy, current, scunet)
        model_kind = "blue_shadow_structure_graft_v1_mid"
    else:
        raise ValueError(f"unknown preset: {preset!r}")

    tail_preset = choose_luma_tail_preset(preset, args.luma_tail_preset)
    tail_stats = None
    if tail_preset is not None:
        out, tail_stats = apply_luma_tail_postfilter(out, tail_preset)

    chroma_preset = choose_chroma_speckle_preset(args.chroma_speckle_preset)
    chroma_stats = None
    if chroma_preset is not None:
        out, chroma_stats = apply_chroma_speckle_postfilter(out, chroma_preset)

    dark_dot_preset = choose_dark_dot_preset(preset, args.dark_dot_preset)
    dark_dot_stats = None
    if dark_dot_preset is not None:
        out, dark_dot_stats = apply_dark_dot_postfilter(out, dark_dot_preset)

    luma_hf_preset = choose_luma_hf_preset(args.luma_hf_preset)
    luma_hf_stats = None
    if luma_hf_preset is not None:
        out, luma_hf_stats = apply_luma_hf_postfilter(out, luma_hf_preset)

    signed_chroma_preset = choose_signed_chroma_preset(args.signed_chroma_preset)
    signed_chroma_stats = None
    if signed_chroma_preset is not None:
        out, signed_chroma_stats = apply_signed_chroma_postfilter(out, signed_chroma_preset)

    pre_neutral_out = out
    neutral_chroma_preset = choose_neutral_chroma_preset(args.neutral_chroma_preset)
    neutral_chroma_stats = None
    if neutral_chroma_preset is not None:
        out, neutral_chroma_stats = apply_neutral_chroma_postfilter(out, neutral_chroma_preset)

    blue_structure_protect_preset = choose_blue_structure_protect_preset(args.blue_structure_protect_preset)
    blue_structure_protect_stats = None
    if blue_structure_protect_preset is not None and neutral_chroma_preset is not None:
        out, blue_structure_protect_stats = apply_blue_structure_protect_postfilter(
            pre_neutral_out,
            out,
            blue_structure_protect_preset,
        )

    post_dark_dot_preset = choose_post_dark_dot_preset(preset, args.post_dark_dot_preset)
    post_dark_dot_stats = None
    if post_dark_dot_preset is not None:
        out, post_dark_dot_stats = apply_post_dark_dot_filter(out, post_dark_dot_preset)

    flat_cleanup_preset = choose_flat_cleanup_preset(args.flat_cleanup_preset)
    flat_cleanup_stats = None
    if flat_cleanup_preset is not None:
        out, flat_cleanup_stats = apply_flat_cleanup_filter(noisy, out, flat_cleanup_preset)

    region_aware_flat_gate_preset = choose_region_aware_flat_gate_preset(
        preset,
        args.region_aware_flat_gate_preset,
    )
    region_aware_flat_gate_stats = None
    region_aware_flat_gate_guard_stats = None
    region_aware_flat_gate_adaptive_stats = None
    region_aware_flat_gate_path = None
    if region_aware_flat_gate_preset is not None:
        gate_path = flat_gate_path(Path(args.region_aware_flat_gate_dir), scene_name)
        if region_aware_flat_gate_preset == "auto_reopen_skyonly_adaptive" and gate_path.exists():
            region_aware_flat_gate_adaptive_stats = region_aware_spill_risk(
                noisy,
                out,
                gate_path,
                "dark_sky_strict_reopen_skyonly",
                max_side=args.region_aware_flat_gate_adaptive_max_side,
                soft_ratio_threshold=args.region_aware_flat_gate_adaptive_soft_ratio,
            )
            region_aware_flat_gate_preset = str(region_aware_flat_gate_adaptive_stats["chosen_preset"])
        if args.region_aware_flat_gate_guard and is_reopen_region_aware_preset(region_aware_flat_gate_preset):
            region_aware_flat_gate_guard_stats = region_aware_reopen_guard(
                noisy,
                out,
                region_aware_flat_gate_preset,
                min_candidate_gt040=args.region_aware_flat_gate_guard_min_candidate_gt040,
                max_structure_mean=args.region_aware_flat_gate_guard_max_structure_mean,
            )
            if not region_aware_flat_gate_guard_stats["passed"]:
                if args.region_aware_flat_gate_preset == "auto_reopen_skyonly_adaptive":
                    region_aware_flat_gate_preset = None
                else:
                    region_aware_flat_gate_preset = (
                        None
                        if args.region_aware_flat_gate_guard_fallback == "off"
                        else args.region_aware_flat_gate_guard_fallback
                    )
        if region_aware_flat_gate_preset is None:
            pass
        elif gate_path.exists():
            out, region_aware_flat_gate_stats = apply_region_aware_flat_gate_filter(
                noisy,
                out,
                gate_path,
                region_aware_flat_gate_preset,
            )
            region_aware_flat_gate_path = str(gate_path)
        elif args.region_aware_flat_gate_preset in {"auto", "auto_reopen_skyonly_adaptive"}:
            region_aware_flat_gate_preset = None
        else:
            raise FileNotFoundError(f"region-aware flat gate not found: {gate_path}")

    name = args.name or f"{scene_name}_scunet_preset_chooser"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    meta = {
        "scene": scene_name,
        "preset": preset,
        "reason": reason,
        "model_kind": model_kind,
        "luma_tail_preset": tail_preset,
        "chroma_speckle_preset": chroma_preset,
        "dark_dot_preset": dark_dot_preset,
        "luma_hf_preset": luma_hf_preset,
        "signed_chroma_preset": signed_chroma_preset,
        "neutral_chroma_preset": neutral_chroma_preset,
        "blue_structure_protect_preset": blue_structure_protect_preset,
        "post_dark_dot_preset": post_dark_dot_preset,
        "flat_cleanup_preset": flat_cleanup_preset,
        "region_aware_flat_gate_preset": region_aware_flat_gate_preset,
        "region_aware_flat_gate_path": region_aware_flat_gate_path,
        "metrics": metrics,
        "model_stats": model_stats,
        "luma_tail_stats": tail_stats,
        "chroma_speckle_stats": chroma_stats,
        "dark_dot_stats": dark_dot_stats,
        "luma_hf_stats": luma_hf_stats,
        "signed_chroma_stats": signed_chroma_stats,
        "neutral_chroma_stats": neutral_chroma_stats,
        "blue_structure_protect_stats": blue_structure_protect_stats,
        "post_dark_dot_stats": post_dark_dot_stats,
        "flat_cleanup_stats": flat_cleanup_stats,
        "region_aware_flat_gate_stats": region_aware_flat_gate_stats,
        "region_aware_flat_gate_guard_stats": region_aware_flat_gate_guard_stats,
        "region_aware_flat_gate_adaptive_stats": region_aware_flat_gate_adaptive_stats,
        "inputs": {
            "noisy": str(noisy_path),
            "current": str(current_path),
            "scunet": str(scunet_path),
            "precomputed_base": str(Path(args.precomputed_base)) if args.precomputed_base else None,
        },
        "outputs": {"exr": str(exr_path), "preview": str(preview_path)},
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a coarse SCUNet preset chooser.")
    parser.add_argument("--scene", default=None, choices=sorted(SCENES))
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--noisy", default=None)
    parser.add_argument("--current", default=None)
    parser.add_argument("--scunet", default=None)
    parser.add_argument("--precomputed-base", default=None)
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "scunet_preset_chooser_v1_outputs"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--metric-max-side", type=int, default=1400)
    parser.add_argument("--v2-selector", default=str(DEFAULT_V2_SELECTOR))
    parser.add_argument("--v3-selector", default=str(DEFAULT_V3_SELECTOR))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--force-preset", choices=["hair_luma", "sky_luma", "blue_shadow_safe"], default=None)
    parser.add_argument("--luma-tail-preset", choices=["off", "auto", *sorted(LUMA_TAIL_PRESETS)], default="off")
    parser.add_argument(
        "--chroma-speckle-preset",
        choices=["off", "auto", *sorted(CHROMA_SPECKLE_PRESETS)],
        default="off",
    )
    parser.add_argument(
        "--dark-dot-preset",
        choices=["off", "auto", *sorted(DARK_DOT_PRESETS)],
        default="off",
    )
    parser.add_argument(
        "--luma-hf-preset",
        choices=["off", "auto", *sorted(LUMA_HF_PRESETS)],
        default="off",
    )
    parser.add_argument(
        "--signed-chroma-preset",
        choices=["off", "auto", *sorted(SIGNED_CHROMA_PRESETS)],
        default="off",
    )
    parser.add_argument(
        "--neutral-chroma-preset",
        choices=["off", "auto", *sorted(NEUTRAL_CHROMA_PRESETS)],
        default="off",
    )
    parser.add_argument(
        "--blue-structure-protect-preset",
        choices=["off", "auto", *sorted(BLUE_STRUCTURE_PROTECT_PRESETS)],
        default="off",
    )
    parser.add_argument(
        "--post-dark-dot-preset",
        choices=["off", "auto", *sorted(POST_DARK_DOT_PRESETS)],
        default="off",
    )
    parser.add_argument(
        "--flat-cleanup-preset",
        choices=["off", "auto", *sorted(FLAT_CLEANUP_PRESETS)],
        default="off",
    )
    parser.add_argument(
        "--region-aware-flat-gate-preset",
        choices=["off", "auto", "auto_reopen", "auto_reopen_skyonly", "auto_reopen_skyonly_adaptive", *sorted(REGION_AWARE_FLAT_GATE_PRESETS)],
        default="off",
    )
    parser.add_argument("--region-aware-flat-gate-guard", action="store_true")
    parser.add_argument("--region-aware-flat-gate-adaptive-max-side", type=int, default=1400)
    parser.add_argument("--region-aware-flat-gate-adaptive-soft-ratio", type=float, default=0.50)
    parser.add_argument("--region-aware-flat-gate-guard-min-candidate-gt040", type=float, default=0.05)
    parser.add_argument("--region-aware-flat-gate-guard-max-structure-mean", type=float, default=0.62)
    parser.add_argument(
        "--region-aware-flat-gate-guard-fallback",
        choices=["off", "dark_sky_strict"],
        default="dark_sky_strict",
    )
    parser.add_argument(
        "--region-aware-flat-gate-dir",
        default=str(RUN_ROOT / "flat_cleanup_gate_v12_native_pilot_v1_outputs"),
    )
    args = parser.parse_args()
    apply(args)


if __name__ == "__main__":
    main()
