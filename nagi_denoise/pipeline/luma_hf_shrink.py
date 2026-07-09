"""Texture-aware display-luma high-frequency shrink for residual grain.

The speckle filter only targets isolated outliers. This filter targets the
broader sand-like luma grain that remains after the Guard pipeline by shrinking
small display-luma high-frequency residuals inside flat, non-highlight regions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

from .flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from .detail_guard import write_exr, write_tiff
from .probe import image_stats, make_preview, read_image


PRESETS = {
    "mild": {
        "strength": 0.45,
        "low_sigma": 0.85,
        "shrink_threshold": 0.0045,
        "detail_preserve_threshold": 0.020,
        "detail_preserve_transition": 0.010,
        "shadow_boost": 0.0,
        "line_sigma": 0.70,
        "line_smooth_sigma": 1.00,
        "line_threshold": 0.010,
        "line_transition": 0.006,
        "line_coherence_threshold": 0.42,
        "line_coherence_transition": 0.16,
        "line_preserve_strength": 0.75,
    },
    "strong": {
        "strength": 0.68,
        "low_sigma": 0.95,
        "shrink_threshold": 0.0060,
        "detail_preserve_threshold": 0.024,
        "detail_preserve_transition": 0.011,
        "shadow_boost": 0.0,
        "line_sigma": 0.70,
        "line_smooth_sigma": 1.00,
        "line_threshold": 0.010,
        "line_transition": 0.006,
        "line_coherence_threshold": 0.42,
        "line_coherence_transition": 0.16,
        "line_preserve_strength": 0.82,
    },
    "xstrong": {
        "strength": 0.85,
        "low_sigma": 1.05,
        "shrink_threshold": 0.0075,
        "detail_preserve_threshold": 0.028,
        "detail_preserve_transition": 0.012,
        "shadow_boost": 0.0,
        "line_sigma": 0.70,
        "line_smooth_sigma": 1.00,
        "line_threshold": 0.010,
        "line_transition": 0.006,
        "line_coherence_threshold": 0.42,
        "line_coherence_transition": 0.16,
        "line_preserve_strength": 0.88,
    },
    "ultra": {
        "strength": 1.0,
        "low_sigma": 1.35,
        "shrink_threshold": 0.0120,
        "detail_preserve_threshold": 0.040,
        "detail_preserve_transition": 0.016,
        "shadow_boost": 0.0,
        "line_sigma": 0.70,
        "line_smooth_sigma": 1.00,
        "line_threshold": 0.010,
        "line_transition": 0.006,
        "line_coherence_threshold": 0.42,
        "line_coherence_transition": 0.16,
        "line_preserve_strength": 0.92,
    },
    "shadow": {
        "strength": 1.0,
        "low_sigma": 1.40,
        "shrink_threshold": 0.0128,
        "detail_preserve_threshold": 0.043,
        "detail_preserve_transition": 0.018,
        "shadow_boost": 0.70,
        "line_sigma": 0.70,
        "line_smooth_sigma": 1.00,
        "line_threshold": 0.010,
        "line_transition": 0.006,
        "line_coherence_threshold": 0.42,
        "line_coherence_transition": 0.16,
        "line_preserve_strength": 0.92,
    },
    "grain": {
        "strength": 0.82,
        "low_sigma": 1.00,
        "shrink_threshold": 0.0070,
        "detail_preserve_threshold": 0.027,
        "detail_preserve_transition": 0.012,
        "shadow_boost": 0.0,
        "line_sigma": 0.70,
        "line_smooth_sigma": 1.00,
        "line_threshold": 0.010,
        "line_transition": 0.006,
        "line_coherence_threshold": 0.42,
        "line_coherence_transition": 0.16,
        "line_preserve_strength": 0.88,
    },
}


def sigmoid01(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False)


def make_flat_gate(
    guide_rgb_linear: np.ndarray,
    *,
    structure_sigma: float,
    detail_sigma: float,
    detail_threshold: float,
    detail_transition: float,
    edge_sigma: float,
    edge_threshold: float,
    edge_transition: float,
    highlight_threshold: float,
    highlight_transition: float,
) -> tuple[np.ndarray, dict[str, float]]:
    guide = np.nan_to_num(guide_rgb_linear[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    guide_display = np.clip(linear_to_srgb_np(guide), 0.0, 1.0)
    guide_y = luma(guide_display, LUMA_SRGB)
    guide_y_linear = luma(np.clip(guide, 0.0, None), LUMA_LINEAR)

    structure = gaussian_filter(guide_y, sigma=float(structure_sigma), mode="reflect")
    detail = np.abs(structure - gaussian_filter(structure, sigma=float(detail_sigma), mode="reflect"))
    edge = gaussian_gradient_magnitude(structure, sigma=float(edge_sigma), mode="reflect")
    flat = sigmoid01((float(detail_threshold) - detail) / max(float(detail_transition), 1.0e-6))
    non_edge = sigmoid01((float(edge_threshold) - edge) / max(float(edge_transition), 1.0e-6))
    highlight = smoothstep((guide_y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6))
    gate = (flat * non_edge * (1.0 - highlight)).astype(np.float32, copy=False)
    return gate, {
        "flat_mean": float(np.mean(flat)),
        "non_edge_mean": float(np.mean(non_edge)),
        "highlight_restore_mean": float(np.mean(highlight)),
        "flat_gate_mean": float(np.mean(gate)),
        "flat_gate_p90": float(np.quantile(gate, 0.90)),
        "flat_gate_p99": float(np.quantile(gate, 0.99)),
    }


def make_line_preserve_gate(
    guide_rgb_linear: np.ndarray,
    *,
    line_sigma: float,
    line_smooth_sigma: float,
    line_threshold: float,
    line_transition: float,
    line_coherence_threshold: float,
    line_coherence_transition: float,
) -> tuple[np.ndarray, dict[str, float]]:
    guide = np.nan_to_num(guide_rgb_linear[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    guide_display = np.clip(linear_to_srgb_np(guide), 0.0, 1.0)
    guide_y = luma(guide_display, LUMA_SRGB)
    sigma = float(line_sigma)
    gx = gaussian_filter(guide_y, sigma=sigma, order=(0, 1), mode="reflect")
    gy = gaussian_filter(guide_y, sigma=sigma, order=(1, 0), mode="reflect")
    smooth_sigma = float(line_smooth_sigma)
    jxx = gaussian_filter(gx * gx, sigma=smooth_sigma, mode="reflect")
    jyy = gaussian_filter(gy * gy, sigma=smooth_sigma, mode="reflect")
    jxy = gaussian_filter(gx * gy, sigma=smooth_sigma, mode="reflect")
    energy = np.sqrt(np.maximum(jxx + jyy, 0.0))
    coherence = np.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy) / np.maximum(jxx + jyy, 1.0e-8)
    energy_gate = sigmoid01((energy - float(line_threshold)) / max(float(line_transition), 1.0e-6))
    coherence_gate = sigmoid01(
        (coherence - float(line_coherence_threshold)) / max(float(line_coherence_transition), 1.0e-6)
    )
    line_gate = (energy_gate * coherence_gate).astype(np.float32, copy=False)
    return line_gate, {
        "line_energy_mean": float(np.mean(energy)),
        "line_energy_p95": float(np.quantile(energy, 0.95)),
        "line_energy_p99": float(np.quantile(energy, 0.99)),
        "line_coherence_mean": float(np.mean(coherence)),
        "line_coherence_p95": float(np.quantile(coherence, 0.95)),
        "line_gate_mean": float(np.mean(line_gate)),
        "line_gate_p90": float(np.quantile(line_gate, 0.90)),
        "line_gate_p99": float(np.quantile(line_gate, 0.99)),
    }


def soft_shrink_highpass(
    high: np.ndarray,
    *,
    amount: np.ndarray,
) -> np.ndarray:
    mag = np.abs(high)
    return np.sign(high) * np.maximum(mag - amount, 0.0)


def apply_luma_hf_shrink(
    image: np.ndarray,
    guide_image: np.ndarray,
    *,
    strength: float,
    low_sigma: float,
    shrink_threshold: float,
    detail_preserve_threshold: float,
    detail_preserve_transition: float,
    shadow_boost: float,
    line_sigma: float,
    line_smooth_sigma: float,
    line_threshold: float,
    line_transition: float,
    line_coherence_threshold: float,
    line_coherence_transition: float,
    line_preserve_strength: float,
    shadow_threshold: float,
    shadow_transition: float,
    structure_sigma: float,
    detail_sigma: float,
    detail_threshold: float,
    detail_transition: float,
    edge_sigma: float,
    edge_threshold: float,
    edge_transition: float,
    highlight_threshold: float,
    highlight_transition: float,
    hdr_restore_peak_threshold: float,
    hdr_restore_threshold: float,
    hdr_restore_transition: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    base = np.nan_to_num(image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    display = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)
    y_low = gaussian_filter(y, sigma=float(low_sigma), mode="reflect")
    high = y - y_low
    high_abs = np.abs(high)

    flat_gate, gate_stats = make_flat_gate(
        guide_image,
        structure_sigma=structure_sigma,
        detail_sigma=detail_sigma,
        detail_threshold=detail_threshold,
        detail_transition=detail_transition,
        edge_sigma=edge_sigma,
        edge_threshold=edge_threshold,
        edge_transition=edge_transition,
        highlight_threshold=highlight_threshold,
        highlight_transition=highlight_transition,
    )
    detail_preserve = sigmoid01(
        (high_abs - float(detail_preserve_threshold)) / max(float(detail_preserve_transition), 1.0e-6)
    )
    line_gate, line_stats = make_line_preserve_gate(
        guide_image,
        line_sigma=line_sigma,
        line_smooth_sigma=line_smooth_sigma,
        line_threshold=line_threshold,
        line_transition=line_transition,
        line_coherence_threshold=line_coherence_threshold,
        line_coherence_transition=line_coherence_transition,
    )
    shadow_gate = sigmoid01((float(shadow_threshold) - y_low) / max(float(shadow_transition), 1.0e-6))
    line_keep = 1.0 - np.clip(float(line_preserve_strength) * line_gate, 0.0, 1.0)
    shrink_gate = np.clip(flat_gate * (1.0 - detail_preserve) * line_keep * float(strength), 0.0, 1.0)
    amount = shrink_gate * float(shrink_threshold) * (1.0 + float(shadow_boost) * shadow_gate)
    high_new = soft_shrink_highpass(high, amount=amount)
    out_y = np.clip(y_low + high_new, 0.0, 1.0)

    chroma = display / np.maximum(y[..., None], 1.0e-6)
    out_display = np.clip(chroma * np.maximum(out_y[..., None], 0.0), 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)

    y_linear = luma(np.clip(base, 0.0, None), LUMA_LINEAR)
    peak_linear = np.max(base, axis=2)
    hdr_signal = np.maximum(
        y_linear - float(hdr_restore_threshold),
        peak_linear - float(hdr_restore_peak_threshold),
    )
    hdr_restore = smoothstep(hdr_signal / max(float(hdr_restore_transition), 1.0e-6))
    out = out * (1.0 - hdr_restore[..., None]) + base * hdr_restore[..., None]

    delta = high_new - high
    stats = {
        "strength": float(strength),
        "low_sigma": float(low_sigma),
        "shrink_threshold": float(shrink_threshold),
        "detail_preserve_threshold": float(detail_preserve_threshold),
        "detail_preserve_transition": float(detail_preserve_transition),
        "shadow_boost": float(shadow_boost),
        "line_sigma": float(line_sigma),
        "line_smooth_sigma": float(line_smooth_sigma),
        "line_threshold": float(line_threshold),
        "line_transition": float(line_transition),
        "line_coherence_threshold": float(line_coherence_threshold),
        "line_coherence_transition": float(line_coherence_transition),
        "line_preserve_strength": float(line_preserve_strength),
        "shadow_threshold": float(shadow_threshold),
        "shadow_transition": float(shadow_transition),
        "shadow_gate_mean": float(np.mean(shadow_gate)),
        "shadow_gate_p90": float(np.quantile(shadow_gate, 0.90)),
        "shadow_gate_p99": float(np.quantile(shadow_gate, 0.99)),
        "shrink_gate_mean": float(np.mean(shrink_gate)),
        "shrink_gate_p90": float(np.quantile(shrink_gate, 0.90)),
        "shrink_gate_p99": float(np.quantile(shrink_gate, 0.99)),
        "detail_preserve_mean": float(np.mean(detail_preserve)),
        "delta_abs_mean": float(np.mean(np.abs(delta))),
        "delta_abs_p99": float(np.quantile(np.abs(delta), 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
        **line_stats,
        **gate_stats,
    }
    return out, stats, shrink_gate.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply display-luma high-frequency shrink.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--guide-input", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-tiff", action="store_true")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="strong")
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--low-sigma", type=float, default=None)
    parser.add_argument("--shrink-threshold", type=float, default=None)
    parser.add_argument("--detail-preserve-threshold", type=float, default=None)
    parser.add_argument("--detail-preserve-transition", type=float, default=None)
    parser.add_argument("--shadow-boost", type=float, default=None)
    parser.add_argument("--line-sigma", type=float, default=None)
    parser.add_argument("--line-smooth-sigma", type=float, default=None)
    parser.add_argument("--line-threshold", type=float, default=None)
    parser.add_argument("--line-transition", type=float, default=None)
    parser.add_argument("--line-coherence-threshold", type=float, default=None)
    parser.add_argument("--line-coherence-transition", type=float, default=None)
    parser.add_argument("--line-preserve-strength", type=float, default=None)
    parser.add_argument("--shadow-threshold", type=float, default=0.18)
    parser.add_argument("--shadow-transition", type=float, default=0.08)
    parser.add_argument("--structure-sigma", type=float, default=1.2)
    parser.add_argument("--detail-sigma", type=float, default=2.8)
    parser.add_argument("--detail-threshold", type=float, default=0.018)
    parser.add_argument("--detail-transition", type=float, default=0.010)
    parser.add_argument("--edge-sigma", type=float, default=1.0)
    parser.add_argument("--edge-threshold", type=float, default=0.030)
    parser.add_argument("--edge-transition", type=float, default=0.015)
    parser.add_argument("--highlight-threshold", type=float, default=1.0)
    parser.add_argument("--highlight-transition", type=float, default=0.25)
    parser.add_argument("--hdr-restore-peak-threshold", type=float, default=0.95)
    parser.add_argument("--hdr-restore-threshold", type=float, default=0.85)
    parser.add_argument("--hdr-restore-transition", type=float, default=0.25)
    args = parser.parse_args()

    params = dict(PRESETS[args.preset])
    for attr, key in (
        ("strength", "strength"),
        ("low_sigma", "low_sigma"),
        ("shrink_threshold", "shrink_threshold"),
        ("detail_preserve_threshold", "detail_preserve_threshold"),
        ("detail_preserve_transition", "detail_preserve_transition"),
        ("shadow_boost", "shadow_boost"),
        ("line_sigma", "line_sigma"),
        ("line_smooth_sigma", "line_smooth_sigma"),
        ("line_threshold", "line_threshold"),
        ("line_transition", "line_transition"),
        ("line_coherence_threshold", "line_coherence_threshold"),
        ("line_coherence_transition", "line_coherence_transition"),
        ("line_preserve_strength", "line_preserve_strength"),
    ):
        value = getattr(args, attr)
        if value is not None:
            params[key] = value

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide_input).expanduser() if args.guide_input else input_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_luma_hf_{args.preset}"

    image = read_image(input_path)
    guide = read_image(guide_path)
    if image.shape[:2] != guide.shape[:2]:
        raise ValueError(f"shape mismatch: input={image.shape}, guide={guide.shape}")

    out, stats, shrink_gate = apply_luma_hf_shrink(
        image,
        guide,
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
        shadow_threshold=args.shadow_threshold,
        shadow_transition=args.shadow_transition,
        structure_sigma=args.structure_sigma,
        detail_sigma=args.detail_sigma,
        detail_threshold=args.detail_threshold,
        detail_transition=args.detail_transition,
        edge_sigma=args.edge_sigma,
        edge_threshold=args.edge_threshold,
        edge_transition=args.edge_transition,
        highlight_threshold=args.highlight_threshold,
        highlight_transition=args.highlight_transition,
        hdr_restore_peak_threshold=args.hdr_restore_peak_threshold,
        hdr_restore_threshold=args.hdr_restore_threshold,
        hdr_restore_transition=args.hdr_restore_transition,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}.json"

    write_exr(exr_path, out)
    if not args.no_tiff:
        write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(shrink_gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    meta = {
        "input": str(input_path),
        "guide_input": str(guide_path),
        "preset": args.preset,
        "params": params,
        "outputs": {
            "exr": str(exr_path),
            "tiff": None if args.no_tiff else str(tiff_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
        },
        "filter": stats,
        "input_stats": image_stats(image),
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
