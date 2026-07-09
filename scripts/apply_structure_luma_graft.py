"""Graft denoised reference luma structure back into a denoised base.

This is stronger than high-frequency detail restoration: it borrows low/mid
frequency luma shape from the original noisy reference after edge-aware luma
smoothing, while preserving the denoised base chroma.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, uniform_filter

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_guided_luma_smoother import guided_filter_gray
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import image_stats, make_preview, read_image


PRESETS = {
    "soft": {
        "guide_sigma": 1.2,
        "radius": 3,
        "eps": 0.0015,
        "post_sigma": 0.25,
        "strength": 0.35,
        "edge_threshold": 0.010,
        "edge_transition": 0.012,
        "contrast_threshold": 0.010,
        "contrast_transition": 0.012,
        "contrast_gate_weight": 0.75,
        "correction_limit": 0.070,
        "zero_mean_sigma": 24.0,
        "zero_mean_strength": 0.35,
        "gate_blur_sigma": 0.80,
        "mid_boost": 0.0,
        "mid_sigma_low": 0.7,
        "mid_sigma_high": 3.0,
        "mid_limit": 0.080,
        "fine_boost": 0.0,
        "fine_sigma": 0.7,
        "fine_limit": 0.035,
    },
    "mid": {
        "guide_sigma": 1.4,
        "radius": 4,
        "eps": 0.0025,
        "post_sigma": 0.35,
        "strength": 0.50,
        "edge_threshold": 0.009,
        "edge_transition": 0.011,
        "contrast_threshold": 0.009,
        "contrast_transition": 0.011,
        "contrast_gate_weight": 0.75,
        "correction_limit": 0.090,
        "zero_mean_sigma": 24.0,
        "zero_mean_strength": 0.35,
        "gate_blur_sigma": 0.80,
        "mid_boost": 0.0,
        "mid_sigma_low": 0.7,
        "mid_sigma_high": 3.0,
        "mid_limit": 0.080,
        "fine_boost": 0.0,
        "fine_sigma": 0.7,
        "fine_limit": 0.035,
    },
    "strong": {
        "guide_sigma": 1.6,
        "radius": 5,
        "eps": 0.0035,
        "post_sigma": 0.45,
        "strength": 0.65,
        "edge_threshold": 0.008,
        "edge_transition": 0.010,
        "contrast_threshold": 0.008,
        "contrast_transition": 0.010,
        "contrast_gate_weight": 0.75,
        "correction_limit": 0.110,
        "zero_mean_sigma": 24.0,
        "zero_mean_strength": 0.35,
        "gate_blur_sigma": 0.80,
        "mid_boost": 0.0,
        "mid_sigma_low": 0.7,
        "mid_sigma_high": 3.0,
        "mid_limit": 0.080,
        "fine_boost": 0.0,
        "fine_sigma": 0.7,
        "fine_limit": 0.035,
    },
    "rebuild": {
        "guide_sigma": 0.45,
        "radius": 1,
        "eps": 0.00025,
        "post_sigma": 0.0,
        "strength": 1.00,
        "edge_threshold": 0.0035,
        "edge_transition": 0.009,
        "contrast_threshold": 0.0035,
        "contrast_transition": 0.009,
        "contrast_gate_weight": 0.90,
        "correction_limit": 0.280,
        "zero_mean_sigma": 0.0,
        "zero_mean_strength": 0.0,
        "gate_blur_sigma": 0.45,
        "mid_boost": 0.0,
        "mid_sigma_low": 0.7,
        "mid_sigma_high": 3.0,
        "mid_limit": 0.080,
        "fine_boost": 0.0,
        "fine_sigma": 0.7,
        "fine_limit": 0.035,
    },
    "rebuild_clarity": {
        "guide_sigma": 0.75,
        "radius": 2,
        "eps": 0.00055,
        "post_sigma": 0.0,
        "strength": 0.86,
        "edge_threshold": 0.0035,
        "edge_transition": 0.009,
        "contrast_threshold": 0.0035,
        "contrast_transition": 0.009,
        "contrast_gate_weight": 0.90,
        "correction_limit": 0.280,
        "zero_mean_sigma": 0.0,
        "zero_mean_strength": 0.0,
        "gate_blur_sigma": 0.45,
        "mid_boost": 0.55,
        "mid_sigma_low": 0.7,
        "mid_sigma_high": 3.0,
        "mid_limit": 0.080,
        "fine_boost": 0.22,
        "fine_sigma": 0.7,
        "fine_limit": 0.035,
    },
}


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def apply_structure_luma_graft(
    reference: np.ndarray,
    base: np.ndarray,
    *,
    guide_sigma: float,
    radius: int,
    eps: float,
    post_sigma: float,
    strength: float,
    edge_threshold: float,
    edge_transition: float,
    contrast_threshold: float,
    contrast_transition: float,
    contrast_gate_weight: float,
    correction_limit: float,
    zero_mean_sigma: float,
    zero_mean_strength: float,
    gate_blur_sigma: float,
    mid_boost: float,
    mid_sigma_low: float,
    mid_sigma_high: float,
    mid_limit: float,
    fine_boost: float,
    fine_sigma: float,
    fine_limit: float,
    hdr_restore_threshold: float,
    hdr_restore_transition: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    ref_rgb = np.clip(_safe_rgb(reference), 0.0, None)
    base_rgb = np.clip(_safe_rgb(base), 0.0, None)
    if ref_rgb.shape[:2] != base_rgb.shape[:2]:
        raise ValueError(f"shape mismatch: reference={ref_rgb.shape}, base={base_rgb.shape}")

    ref_display = np.clip(linear_to_srgb_np(ref_rgb), 0.0, 1.0)
    base_display = np.clip(linear_to_srgb_np(base_rgb), 0.0, 1.0)
    ref_y = luma(ref_display, LUMA_SRGB)
    base_y = luma(base_display, LUMA_SRGB)

    guide = gaussian_filter(ref_y, sigma=float(guide_sigma), mode="reflect")
    ref_structure = guided_filter_gray(guide, ref_y, radius=int(radius), eps=float(eps))
    if post_sigma > 0:
        ref_structure = gaussian_filter(ref_structure, sigma=float(post_sigma), mode="reflect")

    edge = gaussian_gradient_magnitude(ref_structure, sigma=0.9, mode="reflect")
    local_mean = uniform_filter(ref_structure, size=13, mode="reflect")
    contrast = np.abs(ref_structure - local_mean)
    edge_gate = _smoothstep01((edge - float(edge_threshold)) / max(float(edge_transition), 1.0e-6))
    contrast_gate = _smoothstep01(
        (contrast - float(contrast_threshold)) / max(float(contrast_transition), 1.0e-6)
    )
    gate = np.clip(np.maximum(edge_gate, float(contrast_gate_weight) * contrast_gate), 0.0, 1.0).astype(
        np.float32, copy=False
    )
    if gate_blur_sigma > 0:
        gate = gaussian_filter(gate, sigma=float(gate_blur_sigma), mode="reflect")

    correction = np.clip(ref_structure - base_y, -float(correction_limit), float(correction_limit))
    correction = correction * gate * float(strength)
    if mid_boost > 0:
        ref_mid = gaussian_filter(ref_structure, sigma=float(mid_sigma_low), mode="reflect") - gaussian_filter(
            ref_structure, sigma=float(mid_sigma_high), mode="reflect"
        )
        base_mid = gaussian_filter(base_y, sigma=float(mid_sigma_low), mode="reflect") - gaussian_filter(
            base_y, sigma=float(mid_sigma_high), mode="reflect"
        )
        correction += np.clip(ref_mid - base_mid, -float(mid_limit), float(mid_limit)) * gate * float(mid_boost)
    if fine_boost > 0:
        ref_fine = ref_structure - gaussian_filter(ref_structure, sigma=float(fine_sigma), mode="reflect")
        base_fine = base_y - gaussian_filter(base_y, sigma=float(fine_sigma), mode="reflect")
        correction += np.clip(ref_fine - base_fine, -float(fine_limit), float(fine_limit)) * gate * float(fine_boost)
    if zero_mean_sigma > 0 and zero_mean_strength > 0:
        correction = correction - gaussian_filter(correction, sigma=float(zero_mean_sigma), mode="reflect") * float(
            zero_mean_strength
        )
    out_y = np.clip(base_y + correction, 0.0, 1.0)

    base_chroma = base_display / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(base_chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    base_peak = np.max(base_rgb, axis=2)
    hdr_restore = _smoothstep01(
        (base_peak - float(hdr_restore_threshold)) / max(float(hdr_restore_transition), 1.0e-6)
    )
    out = out * (1.0 - hdr_restore[..., None]) + base_rgb * hdr_restore[..., None]

    stats = {
        "guide_sigma": float(guide_sigma),
        "radius": int(radius),
        "eps": float(eps),
        "post_sigma": float(post_sigma),
        "strength": float(strength),
        "edge_threshold": float(edge_threshold),
        "contrast_threshold": float(contrast_threshold),
        "contrast_gate_weight": float(contrast_gate_weight),
        "correction_limit": float(correction_limit),
        "gate_blur_sigma": float(gate_blur_sigma),
        "mid_boost": float(mid_boost),
        "fine_boost": float(fine_boost),
        "hdr_restore_threshold": float(hdr_restore_threshold),
        "hdr_restore_transition": float(hdr_restore_transition),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p95": float(np.quantile(hdr_restore, 0.95)),
        "gate_mean": float(np.mean(gate)),
        "gate_p50": float(np.quantile(gate, 0.50)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "correction_abs_mean": float(np.mean(np.abs(correction))),
        "correction_abs_p95": float(np.quantile(np.abs(correction), 0.95)),
        "correction_abs_p99": float(np.quantile(np.abs(correction), 0.99)),
        "correction_abs_max": float(np.max(np.abs(correction))),
    }
    return out, stats, gate.astype(np.float32, copy=False), ref_structure.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Graft reference luma structure into a denoised base.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="mid")
    parser.add_argument("--guide-sigma", type=float, default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--post-sigma", type=float, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--edge-threshold", type=float, default=None)
    parser.add_argument("--edge-transition", type=float, default=None)
    parser.add_argument("--contrast-threshold", type=float, default=None)
    parser.add_argument("--contrast-transition", type=float, default=None)
    parser.add_argument("--contrast-gate-weight", type=float, default=None)
    parser.add_argument("--correction-limit", type=float, default=None)
    parser.add_argument("--zero-mean-sigma", type=float, default=None)
    parser.add_argument("--zero-mean-strength", type=float, default=None)
    parser.add_argument("--gate-blur-sigma", type=float, default=None)
    parser.add_argument("--mid-boost", type=float, default=None)
    parser.add_argument("--mid-sigma-low", type=float, default=None)
    parser.add_argument("--mid-sigma-high", type=float, default=None)
    parser.add_argument("--mid-limit", type=float, default=None)
    parser.add_argument("--fine-boost", type=float, default=None)
    parser.add_argument("--fine-sigma", type=float, default=None)
    parser.add_argument("--fine-limit", type=float, default=None)
    parser.add_argument("--hdr-restore-threshold", type=float, default=0.92)
    parser.add_argument("--hdr-restore-transition", type=float, default=0.24)
    args = parser.parse_args()

    params = dict(PRESETS[args.preset])
    for key in params:
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    params["hdr_restore_threshold"] = args.hdr_restore_threshold
    params["hdr_restore_transition"] = args.hdr_restore_transition

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_structure_luma_graft_{args.preset}"

    reference = read_image(reference_path)
    base = read_image(base_path)
    out, stats, gate, ref_structure = apply_structure_luma_graft(reference, base, **params)

    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    structure_path = out_dir / f"{name}_reference_structure.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    Image.fromarray(np.clip(ref_structure * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(structure_path)
    meta = {
        "reference": str(reference_path),
        "base": str(base_path),
        "outputs": {
            "exr": str(exr_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
            "reference_structure": str(structure_path),
        },
        "params": {"preset": args.preset, **params},
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


if __name__ == "__main__":
    main()
