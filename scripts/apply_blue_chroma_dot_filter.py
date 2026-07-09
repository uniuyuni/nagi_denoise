"""Suppress one-sided blue chroma pin dots after the quality NR pipeline.

The signed chroma outlier pass reduces general blue/magenta axis outliers, but
some Ice-like samples still keep positive blue pin dots. This pass is narrower:
it preserves display luma and only pulls positive blue-axis impulses toward a
robust local blue-axis surface in dark, mostly flat regions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, maximum_filter

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from apply_luma_hf_shrink_filter import sigmoid01
from apply_signed_chroma_outlier_filter import AXES, flat_dark_gate, normalize_axis
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


PRESETS = {
    "pin": {
        "strength": 0.68,
        "median_size": 5,
        "low_sigma": 1.6,
        "positive_threshold": 0.0040,
        "positive_transition": 0.0020,
        "local_gain": 0.10,
        "max_correction": 0.030,
        "isolated_weight": 0.35,
        "blue_signal_threshold": 0.010,
        "blue_signal_transition": 0.006,
        "shadow_threshold": 0.62,
    },
    "pin_strong": {
        "strength": 0.86,
        "median_size": 5,
        "low_sigma": 1.8,
        "positive_threshold": 0.0032,
        "positive_transition": 0.0018,
        "local_gain": 0.08,
        "max_correction": 0.040,
        "isolated_weight": 0.45,
        "blue_signal_threshold": 0.008,
        "blue_signal_transition": 0.005,
        "shadow_threshold": 0.66,
    },
}


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def apply_blue_chroma_dot_filter(
    image: np.ndarray,
    guide_image: np.ndarray,
    *,
    strength: float,
    median_size: int,
    low_sigma: float,
    positive_threshold: float,
    positive_transition: float,
    local_gain: float,
    max_correction: float,
    isolated_weight: float,
    blue_signal_threshold: float,
    blue_signal_transition: float,
    structure_sigma: float,
    detail_sigma: float,
    detail_threshold: float,
    detail_transition: float,
    edge_sigma: float,
    edge_threshold: float,
    edge_transition: float,
    shadow_threshold: float,
    shadow_transition: float,
    highlight_threshold: float,
    highlight_transition: float,
    hdr_restore_peak_threshold: float,
    hdr_restore_threshold: float,
    hdr_restore_transition: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    base = _safe_rgb(image)
    display = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)
    chroma = display - y[..., None]

    gate, gate_stats = flat_dark_gate(
        guide_image,
        structure_sigma=structure_sigma,
        detail_sigma=detail_sigma,
        detail_threshold=detail_threshold,
        detail_transition=detail_transition,
        edge_sigma=edge_sigma,
        edge_threshold=edge_threshold,
        edge_transition=edge_transition,
        shadow_threshold=shadow_threshold,
        shadow_transition=shadow_transition,
        highlight_threshold=highlight_threshold,
        highlight_transition=highlight_transition,
    )

    blue_axis = normalize_axis(AXES["blue"])
    axis_value = np.sum(chroma * blue_axis.reshape(1, 1, 3), axis=2)
    med = median_filter(axis_value, size=int(median_size), mode="reflect")
    low = gaussian_filter(axis_value, sigma=float(low_sigma), mode="reflect")
    target = 0.75 * med + 0.25 * low
    positive = np.maximum(axis_value - target, 0.0)

    local = gaussian_filter(positive, sigma=1.4, mode="reflect")
    threshold = float(positive_threshold) + float(local_gain) * local
    impulse_gate = sigmoid01((positive - threshold) / max(float(positive_transition), 1.0e-6))

    local_max = maximum_filter(positive, size=3, mode="reflect")
    isolated = sigmoid01(
        (positive - 0.72 * local_max - 0.0006) / max(float(positive_transition), 1.0e-6)
    )
    blue_signal = display[..., 2] - 0.5 * (display[..., 0] + display[..., 1])
    blue_signal_gate = sigmoid01(
        (blue_signal - float(blue_signal_threshold)) / max(float(blue_signal_transition), 1.0e-6)
    )

    blend = np.clip(
        gate
        * impulse_gate
        * (1.0 + float(isolated_weight) * isolated)
        * blue_signal_gate
        * float(strength),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    correction = np.minimum(positive, float(max_correction)) * blend
    out_chroma = chroma - correction[..., None] * blue_axis.reshape(1, 1, 3)
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
        "median_size": int(median_size),
        "low_sigma": float(low_sigma),
        "positive_threshold": float(positive_threshold),
        "positive_transition": float(positive_transition),
        "local_gain": float(local_gain),
        "max_correction": float(max_correction),
        "isolated_weight": float(isolated_weight),
        "blue_signal_threshold": float(blue_signal_threshold),
        "blue_signal_transition": float(blue_signal_transition),
        "positive_p95": float(np.quantile(positive, 0.95)),
        "positive_p99": float(np.quantile(positive, 0.99)),
        "positive_p999": float(np.quantile(positive, 0.999)),
        "impulse_gate_mean": float(np.mean(impulse_gate)),
        "impulse_gate_p99": float(np.quantile(impulse_gate, 0.99)),
        "blue_signal_gate_mean": float(np.mean(blue_signal_gate)),
        "blend_mean": float(np.mean(blend)),
        "blend_p90": float(np.quantile(blend, 0.90)),
        "blend_p99": float(np.quantile(blend, 0.99)),
        "correction_mean": float(np.mean(correction)),
        "correction_p99": float(np.quantile(correction, 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
        **gate_stats,
    }
    return out.astype(np.float32, copy=False), stats, blend


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply one-sided blue chroma pin-dot suppression.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--guide", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="pin")
    parser.add_argument("--no-tiff", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide).expanduser() if args.guide else input_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_blue_chroma_dot"

    image = read_image(input_path)
    guide = read_image(guide_path)
    params = dict(PRESETS[args.preset])
    out, stats, blend = apply_blue_chroma_dot_filter(
        image,
        guide,
        strength=float(params["strength"]),
        median_size=int(params["median_size"]),
        low_sigma=float(params["low_sigma"]),
        positive_threshold=float(params["positive_threshold"]),
        positive_transition=float(params["positive_transition"]),
        local_gain=float(params["local_gain"]),
        max_correction=float(params["max_correction"]),
        isolated_weight=float(params["isolated_weight"]),
        blue_signal_threshold=float(params["blue_signal_threshold"]),
        blue_signal_transition=float(params["blue_signal_transition"]),
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

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    blend_path = out_dir / f"{name}_blend.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    if not args.no_tiff:
        write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out)).save(preview_path)
    Image.fromarray((np.clip(blend, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(blend_path)
    stats["preset"] = args.preset
    stats["input"] = str(input_path)
    stats["guide"] = str(guide_path)
    stats["output"] = str(exr_path)
    stats["tiff"] = None if args.no_tiff else str(tiff_path)
    stats["preview"] = str(preview_path)
    stats["blend"] = str(blend_path)
    stats["output_stats"] = image_stats(out)
    meta_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")
    if not args.no_tiff:
        print(f"wrote {tiff_path}")
    print(f"wrote {preview_path}")
    print(f"wrote {blend_path}")


if __name__ == "__main__":
    main()
