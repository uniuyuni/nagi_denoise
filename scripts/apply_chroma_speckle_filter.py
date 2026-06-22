"""Suppress isolated display-chroma speckles after the practical NR pipeline.

The flat chroma smoother removes broad color grain, but real high-ISO photos can
still contain isolated magenta/green pinpoints. This filter keeps display luma
fixed and only replaces chroma outliers in flat, non-highlight regions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from apply_luma_hf_shrink_filter import make_flat_gate, sigmoid01
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


PRESETS = {
    "mild": {
        "strength": 0.40,
        "chroma_sigma": 1.9,
        "median_size": 3,
        "speckle_threshold": 0.018,
        "speckle_transition": 0.009,
        "local_sigma": 1.8,
        "local_gain": 0.45,
        "axis_boost": 0.20,
        "axis_threshold": 0.014,
        "axis_transition": 0.007,
        "magenta_boost": 0.25,
        "magenta_threshold": 0.018,
        "magenta_transition": 0.012,
        "highlight_threshold": 0.50,
        "highlight_transition": 0.12,
        "hdr_restore_peak_threshold": 0.58,
        "hdr_restore_threshold": 0.45,
        "hdr_restore_transition": 0.12,
    },
    "strong": {
        "strength": 0.52,
        "chroma_sigma": 1.9,
        "median_size": 3,
        "speckle_threshold": 0.014,
        "speckle_transition": 0.007,
        "local_sigma": 1.8,
        "local_gain": 0.45,
        "axis_boost": 0.25,
        "axis_threshold": 0.014,
        "axis_transition": 0.007,
        "magenta_boost": 0.30,
        "magenta_threshold": 0.018,
        "magenta_transition": 0.012,
        "highlight_threshold": 0.55,
        "highlight_transition": 0.14,
        "hdr_restore_peak_threshold": 0.62,
        "hdr_restore_threshold": 0.50,
        "hdr_restore_transition": 0.14,
    },
    "xstrong": {
        "strength": 0.70,
        "chroma_sigma": 1.9,
        "median_size": 3,
        "speckle_threshold": 0.014,
        "speckle_transition": 0.007,
        "local_sigma": 1.8,
        "local_gain": 0.45,
        "axis_boost": 0.45,
        "axis_threshold": 0.014,
        "axis_transition": 0.007,
        "magenta_boost": 0.45,
        "magenta_threshold": 0.018,
        "magenta_transition": 0.012,
        "highlight_threshold": 0.70,
        "highlight_transition": 0.18,
        "hdr_restore_peak_threshold": 0.75,
        "hdr_restore_threshold": 0.65,
        "hdr_restore_transition": 0.18,
    },
    "axismax": {
        "strength": 0.72,
        "chroma_sigma": 1.9,
        "median_size": 3,
        "speckle_threshold": 0.012,
        "speckle_transition": 0.006,
        "local_sigma": 1.8,
        "local_gain": 0.38,
        "axis_boost": 0.85,
        "axis_threshold": 0.011,
        "axis_transition": 0.006,
        "magenta_boost": 0.15,
        "magenta_threshold": 0.018,
        "magenta_transition": 0.012,
        "highlight_threshold": 0.70,
        "highlight_transition": 0.18,
        "hdr_restore_peak_threshold": 0.75,
        "hdr_restore_threshold": 0.65,
        "hdr_restore_transition": 0.18,
    },
    "axisplus": {
        "strength": 0.92,
        "chroma_sigma": 2.3,
        "median_size": 3,
        "speckle_threshold": 0.008,
        "speckle_transition": 0.004,
        "local_sigma": 1.8,
        "local_gain": 0.24,
        "axis_boost": 1.45,
        "axis_threshold": 0.0075,
        "axis_transition": 0.004,
        "magenta_boost": 0.25,
        "magenta_threshold": 0.018,
        "magenta_transition": 0.012,
        "highlight_threshold": 0.30,
        "highlight_transition": 0.10,
        "hdr_restore_peak_threshold": 0.42,
        "hdr_restore_threshold": 0.30,
        "hdr_restore_transition": 0.12,
    },
    "quality": {
        "strength": 0.52,
        "chroma_sigma": 1.9,
        "median_size": 3,
        "speckle_threshold": 0.014,
        "speckle_transition": 0.007,
        "local_sigma": 1.8,
        "local_gain": 0.45,
        "axis_boost": 0.25,
        "axis_threshold": 0.014,
        "axis_transition": 0.007,
        "magenta_boost": 0.30,
        "magenta_threshold": 0.018,
        "magenta_transition": 0.012,
        "highlight_threshold": 0.55,
        "highlight_transition": 0.14,
        "hdr_restore_peak_threshold": 0.62,
        "hdr_restore_threshold": 0.50,
        "hdr_restore_transition": 0.14,
    },
}


def apply_chroma_speckle_filter(
    image: np.ndarray,
    guide_image: np.ndarray,
    *,
    strength: float,
    chroma_sigma: float,
    median_size: int,
    speckle_threshold: float,
    speckle_transition: float,
    local_sigma: float,
    local_gain: float,
    axis_boost: float,
    axis_threshold: float,
    axis_transition: float,
    magenta_boost: float,
    magenta_threshold: float,
    magenta_transition: float,
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
    chroma = display - y[..., None]

    flat_gate, flat_stats = make_flat_gate(
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

    low_chroma = gaussian_filter(chroma, sigma=(float(chroma_sigma), float(chroma_sigma), 0.0), mode="reflect")
    if int(median_size) > 1:
        med_chroma = median_filter(chroma, size=(int(median_size), int(median_size), 1), mode="reflect")
        target_chroma = 0.55 * med_chroma + 0.45 * low_chroma
    else:
        target_chroma = low_chroma

    residual = chroma - low_chroma
    residual_mag = np.sqrt(np.sum(residual * residual, axis=2))
    local_mag = gaussian_filter(residual_mag, sigma=float(local_sigma), mode="reflect")
    adaptive_threshold = float(speckle_threshold) + float(local_gain) * local_mag
    speckle_gate = sigmoid01((residual_mag - adaptive_threshold) / max(float(speckle_transition), 1.0e-6))

    residual_r = residual[..., 0]
    residual_g = residual[..., 1]
    residual_b = residual[..., 2]
    magenta_axis = 0.5 * (residual_r + residual_b) - residual_g
    red_axis = residual_r - 0.5 * (residual_g + residual_b)
    blue_axis = residual_b - 0.5 * (residual_r + residual_g)
    axis_signal = np.maximum.reduce(
        (
            np.abs(magenta_axis),
            np.abs(red_axis),
            np.abs(blue_axis),
            np.abs(residual_r - residual_g),
            np.abs(residual_b - 0.5 * (residual_r + residual_g)),
        )
    )
    axis_gate = sigmoid01((axis_signal - float(axis_threshold)) / max(float(axis_transition), 1.0e-6))

    magenta_signal = 0.5 * (display[..., 0] + display[..., 2]) - display[..., 1]
    magenta_gate = sigmoid01((magenta_signal - float(magenta_threshold)) / max(float(magenta_transition), 1.0e-6))
    boost = 1.0 + float(axis_boost) * axis_gate + float(magenta_boost) * magenta_gate
    blend = np.clip(flat_gate * speckle_gate * boost * float(strength), 0.0, 1.0).astype(np.float32, copy=False)

    out_chroma = chroma * (1.0 - blend[..., None]) + target_chroma * blend[..., None]
    out_display = np.clip(y[..., None] + out_chroma, 0.0, 1.0)
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
        "chroma_sigma": float(chroma_sigma),
        "median_size": int(median_size),
        "speckle_threshold": float(speckle_threshold),
        "speckle_transition": float(speckle_transition),
        "local_sigma": float(local_sigma),
        "local_gain": float(local_gain),
        "axis_boost": float(axis_boost),
        "axis_threshold": float(axis_threshold),
        "axis_transition": float(axis_transition),
        "magenta_boost": float(magenta_boost),
        "magenta_threshold": float(magenta_threshold),
        "magenta_transition": float(magenta_transition),
        "residual_mag_mean": float(np.mean(residual_mag)),
        "residual_mag_p95": float(np.quantile(residual_mag, 0.95)),
        "residual_mag_p99": float(np.quantile(residual_mag, 0.99)),
        "axis_signal_p95": float(np.quantile(axis_signal, 0.95)),
        "axis_signal_p99": float(np.quantile(axis_signal, 0.99)),
        "axis_gate_mean": float(np.mean(axis_gate)),
        "axis_gate_p90": float(np.quantile(axis_gate, 0.90)),
        "axis_gate_p99": float(np.quantile(axis_gate, 0.99)),
        "speckle_gate_mean": float(np.mean(speckle_gate)),
        "speckle_gate_p90": float(np.quantile(speckle_gate, 0.90)),
        "speckle_gate_p99": float(np.quantile(speckle_gate, 0.99)),
        "magenta_gate_mean": float(np.mean(magenta_gate)),
        "magenta_gate_p99": float(np.quantile(magenta_gate, 0.99)),
        "blend_mean": float(np.mean(blend)),
        "blend_p90": float(np.quantile(blend, 0.90)),
        "blend_p99": float(np.quantile(blend, 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
        **flat_stats,
    }
    return out, stats, blend


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply isolated display-chroma speckle suppression.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--guide-input", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="strong")
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--chroma-sigma", type=float, default=None)
    parser.add_argument("--median-size", type=int, default=None)
    parser.add_argument("--speckle-threshold", type=float, default=None)
    parser.add_argument("--speckle-transition", type=float, default=None)
    parser.add_argument("--local-sigma", type=float, default=None)
    parser.add_argument("--local-gain", type=float, default=None)
    parser.add_argument("--axis-boost", type=float, default=None)
    parser.add_argument("--axis-threshold", type=float, default=None)
    parser.add_argument("--axis-transition", type=float, default=None)
    parser.add_argument("--magenta-boost", type=float, default=None)
    parser.add_argument("--magenta-threshold", type=float, default=None)
    parser.add_argument("--magenta-transition", type=float, default=None)
    parser.add_argument("--structure-sigma", type=float, default=1.2)
    parser.add_argument("--detail-sigma", type=float, default=2.8)
    parser.add_argument("--detail-threshold", type=float, default=0.018)
    parser.add_argument("--detail-transition", type=float, default=0.010)
    parser.add_argument("--edge-sigma", type=float, default=1.0)
    parser.add_argument("--edge-threshold", type=float, default=0.030)
    parser.add_argument("--edge-transition", type=float, default=0.015)
    parser.add_argument("--highlight-threshold", type=float, default=None)
    parser.add_argument("--highlight-transition", type=float, default=None)
    parser.add_argument("--hdr-restore-peak-threshold", type=float, default=None)
    parser.add_argument("--hdr-restore-threshold", type=float, default=None)
    parser.add_argument("--hdr-restore-transition", type=float, default=None)
    args = parser.parse_args()

    params = dict(PRESETS[args.preset])
    for attr, key in (
        ("strength", "strength"),
        ("chroma_sigma", "chroma_sigma"),
        ("median_size", "median_size"),
        ("speckle_threshold", "speckle_threshold"),
        ("speckle_transition", "speckle_transition"),
        ("local_sigma", "local_sigma"),
        ("local_gain", "local_gain"),
        ("axis_boost", "axis_boost"),
        ("axis_threshold", "axis_threshold"),
        ("axis_transition", "axis_transition"),
        ("magenta_boost", "magenta_boost"),
        ("magenta_threshold", "magenta_threshold"),
        ("magenta_transition", "magenta_transition"),
        ("highlight_threshold", "highlight_threshold"),
        ("highlight_transition", "highlight_transition"),
        ("hdr_restore_peak_threshold", "hdr_restore_peak_threshold"),
        ("hdr_restore_threshold", "hdr_restore_threshold"),
        ("hdr_restore_transition", "hdr_restore_transition"),
    ):
        value = getattr(args, attr)
        if value is not None:
            params[key] = value

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide_input).expanduser() if args.guide_input else input_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_chroma_speckle_{args.preset}"

    image = read_image(input_path)
    guide = read_image(guide_path)
    if image.shape[:2] != guide.shape[:2]:
        raise ValueError(f"shape mismatch: input={image.shape}, guide={guide.shape}")

    out, stats, blend = apply_chroma_speckle_filter(
        image,
        guide,
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
        structure_sigma=args.structure_sigma,
        detail_sigma=args.detail_sigma,
        detail_threshold=args.detail_threshold,
        detail_transition=args.detail_transition,
        edge_sigma=args.edge_sigma,
        edge_threshold=args.edge_threshold,
        edge_transition=args.edge_transition,
        highlight_threshold=float(params["highlight_threshold"]),
        highlight_transition=float(params["highlight_transition"]),
        hdr_restore_peak_threshold=float(params["hdr_restore_peak_threshold"]),
        hdr_restore_threshold=float(params["hdr_restore_threshold"]),
        hdr_restore_transition=float(params["hdr_restore_transition"]),
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
        "params": params,
        "outputs": {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path), "blend": str(blend_path)},
        "filter": stats,
        "input_stats": image_stats(image),
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
