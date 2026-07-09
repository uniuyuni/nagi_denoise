"""Suppress residual display-luma speckles after the Guard NR pipeline.

This is intentionally a post-filter, not a new model. The Guard pipeline already
handles chroma well; the remaining visible defect is a small tail of bright/dark
luma speckles. This filter targets only that tail in flat, non-highlight areas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, median_filter

from .flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from .detail_guard import write_exr, write_tiff
from .probe import image_stats, make_preview, read_image


PRESETS = {
    "mild": {
        "strength": 0.45,
        "tail_threshold": 0.0065,
        "tail_transition": 0.0040,
        "local_gain": 1.40,
        "median_size": 3,
    },
    "balanced": {
        "strength": 0.65,
        "tail_threshold": 0.0055,
        "tail_transition": 0.0035,
        "local_gain": 1.20,
        "median_size": 3,
    },
    "strong": {
        "strength": 0.85,
        "tail_threshold": 0.0045,
        "tail_transition": 0.0030,
        "local_gain": 1.05,
        "median_size": 3,
    },
    "xstrong": {
        "strength": 1.0,
        "tail_threshold": 0.0030,
        "tail_transition": 0.0020,
        "local_gain": 0.80,
        "median_size": 5,
    },
}


def sigmoid01(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False)


def make_structure_gate(
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
    stats = {
        "flat_mean": float(np.mean(flat)),
        "non_edge_mean": float(np.mean(non_edge)),
        "highlight_restore_mean": float(np.mean(highlight)),
        "structure_gate_mean": float(np.mean(gate)),
        "structure_gate_p90": float(np.quantile(gate, 0.90)),
        "structure_gate_p99": float(np.quantile(gate, 0.99)),
    }
    return gate, stats


def apply_luma_tail_speckle_filter(
    image: np.ndarray,
    guide_image: np.ndarray,
    *,
    strength: float,
    median_size: int,
    highpass_sigma: float,
    local_sigma: float,
    local_gain: float,
    tail_threshold: float,
    tail_transition: float,
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
    correction_limit: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    base = np.nan_to_num(image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    display = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)

    structure_gate, structure_stats = make_structure_gate(
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

    y_low = gaussian_filter(y, sigma=float(highpass_sigma), mode="reflect")
    residual = y - y_low
    abs_residual = np.abs(residual)
    local_abs = gaussian_filter(abs_residual, sigma=float(local_sigma), mode="reflect")
    adaptive_threshold = float(tail_threshold) + float(local_gain) * local_abs
    tail_gate = sigmoid01((abs_residual - adaptive_threshold) / max(float(tail_transition), 1.0e-6))

    if int(median_size) <= 1:
        y_target = y_low
    else:
        y_target = median_filter(y, size=int(median_size), mode="reflect")
    correction = np.clip(y_target - y, -float(correction_limit), float(correction_limit))
    blend = np.clip(structure_gate * tail_gate * float(strength), 0.0, 1.0).astype(np.float32, copy=False)
    out_y = np.clip(y + correction * blend, 0.0, 1.0)

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

    stats = {
        "strength": float(strength),
        "median_size": int(median_size),
        "highpass_sigma": float(highpass_sigma),
        "local_sigma": float(local_sigma),
        "local_gain": float(local_gain),
        "tail_threshold": float(tail_threshold),
        "tail_transition": float(tail_transition),
        "tail_gate_mean": float(np.mean(tail_gate)),
        "tail_gate_p90": float(np.quantile(tail_gate, 0.90)),
        "tail_gate_p99": float(np.quantile(tail_gate, 0.99)),
        "blend_mean": float(np.mean(blend)),
        "blend_p90": float(np.quantile(blend, 0.90)),
        "blend_p99": float(np.quantile(blend, 0.99)),
        "correction_abs_mean": float(np.mean(np.abs(correction * blend))),
        "correction_abs_p99": float(np.quantile(np.abs(correction * blend), 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
        **structure_stats,
    }
    return out, stats, blend


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply luma-tail speckle suppression to a denoised image.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--guide-input", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="balanced")
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--median-size", type=int, default=None)
    parser.add_argument("--highpass-sigma", type=float, default=0.9)
    parser.add_argument("--local-sigma", type=float, default=3.0)
    parser.add_argument("--local-gain", type=float, default=None)
    parser.add_argument("--tail-threshold", type=float, default=None)
    parser.add_argument("--tail-transition", type=float, default=None)
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
    parser.add_argument("--correction-limit", type=float, default=0.035)
    args = parser.parse_args()

    params = dict(PRESETS[args.preset])
    for arg_name, key in (
        ("strength", "strength"),
        ("median_size", "median_size"),
        ("local_gain", "local_gain"),
        ("tail_threshold", "tail_threshold"),
        ("tail_transition", "tail_transition"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            params[key] = value

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide_input).expanduser() if args.guide_input else input_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_luma_tail_{args.preset}"

    image = read_image(input_path)
    guide = read_image(guide_path)
    if image.shape[:2] != guide.shape[:2]:
        raise ValueError(f"shape mismatch: input={image.shape}, guide={guide.shape}")

    out, stats, blend = apply_luma_tail_speckle_filter(
        image,
        guide,
        strength=float(params["strength"]),
        median_size=int(params["median_size"]),
        highpass_sigma=args.highpass_sigma,
        local_sigma=args.local_sigma,
        local_gain=float(params["local_gain"]),
        tail_threshold=float(params["tail_threshold"]),
        tail_transition=float(params["tail_transition"]),
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
        correction_limit=args.correction_limit,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    blend_path = out_dir / f"{name}_blend.png"
    meta_path = out_dir / f"{name}.json"

    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(blend * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(blend_path)
    meta = {
        "input": str(input_path),
        "guide_input": str(guide_path),
        "preset": args.preset,
        "params": {**params, "correction_limit": args.correction_limit},
        "outputs": {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path), "blend": str(blend_path)},
        "filter": stats,
        "input_stats": image_stats(image),
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
