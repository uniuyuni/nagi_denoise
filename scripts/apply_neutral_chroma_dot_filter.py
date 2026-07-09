"""Suppress chroma pin dots only in neutral flat regions.

This is a follow-up to the broad signed-chroma pass. Ice-like scenes contain
real cyan/blue structures, so global blue p999 metrics mix subject color with
noise. This filter first gates to low-frequency neutral chroma, then applies a
stronger signed blue/magenta outlier correction inside that safer region.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from apply_luma_hf_shrink_filter import sigmoid01
from apply_signed_chroma_outlier_filter import AXES, flat_dark_gate, normalize_axis
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


PRESETS = {
    "neutral": {
        "strength": 0.82,
        "median_size": 7,
        "low_sigma": 2.2,
        "outlier_threshold": 0.0028,
        "outlier_transition": 0.0018,
        "magenta_weight": 1.05,
        "blue_weight": 1.05,
        "neutral_threshold": 0.070,
        "neutral_transition": 0.025,
        "shadow_threshold": 0.72,
    },
    "neutral_strong": {
        "strength": 0.96,
        "median_size": 7,
        "low_sigma": 2.4,
        "outlier_threshold": 0.0024,
        "outlier_transition": 0.0016,
        "magenta_weight": 1.15,
        "blue_weight": 1.20,
        "neutral_threshold": 0.080,
        "neutral_transition": 0.028,
        "shadow_threshold": 0.76,
    },
}


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def apply_neutral_chroma_dot_filter(
    image: np.ndarray,
    guide_image: np.ndarray,
    *,
    strength: float,
    median_size: int,
    low_sigma: float,
    outlier_threshold: float,
    outlier_transition: float,
    magenta_weight: float,
    blue_weight: float,
    neutral_threshold: float,
    neutral_transition: float,
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

    low_chroma = gaussian_filter(chroma, sigma=(2.0, 2.0, 0.0), mode="reflect")
    low_chroma_mag = np.sqrt(np.sum(low_chroma * low_chroma, axis=2))
    neutral_gate = sigmoid01(
        (float(neutral_threshold) - low_chroma_mag) / max(float(neutral_transition), 1.0e-6)
    )
    gate = (gate * neutral_gate).astype(np.float32, copy=False)

    out_chroma = chroma.copy()
    blend_max = np.zeros_like(y, dtype=np.float32)
    axis_stats: dict[str, float] = {}
    for name, raw_axis, weight in (
        ("magenta", AXES["magenta"], magenta_weight),
        ("blue", AXES["blue"], blue_weight),
    ):
        axis = normalize_axis(raw_axis)
        axis_value = np.sum(out_chroma * axis.reshape(1, 1, 3), axis=2)
        med = median_filter(axis_value, size=int(median_size), mode="reflect")
        low = gaussian_filter(axis_value, sigma=float(low_sigma), mode="reflect")
        target = 0.70 * med + 0.30 * low
        outlier = axis_value - target
        outlier_gate = sigmoid01(
            (np.abs(outlier) - float(outlier_threshold)) / max(float(outlier_transition), 1.0e-6)
        )
        blend = np.clip(gate * outlier_gate * float(strength) * float(weight), 0.0, 1.0).astype(
            np.float32, copy=False
        )
        out_chroma -= (outlier * blend)[..., None] * axis.reshape(1, 1, 3)
        blend_max = np.maximum(blend_max, blend)
        axis_stats[f"{name}_outlier_p99"] = float(np.quantile(np.abs(outlier), 0.99))
        axis_stats[f"{name}_blend_mean"] = float(np.mean(blend))
        axis_stats[f"{name}_blend_p99"] = float(np.quantile(blend, 0.99))

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
        "outlier_threshold": float(outlier_threshold),
        "outlier_transition": float(outlier_transition),
        "magenta_weight": float(magenta_weight),
        "blue_weight": float(blue_weight),
        "neutral_threshold": float(neutral_threshold),
        "neutral_transition": float(neutral_transition),
        "neutral_gate_mean": float(np.mean(neutral_gate)),
        "neutral_gate_p90": float(np.quantile(neutral_gate, 0.90)),
        "neutral_gate_p99": float(np.quantile(neutral_gate, 0.99)),
        "low_chroma_mag_p90": float(np.quantile(low_chroma_mag, 0.90)),
        "low_chroma_mag_p99": float(np.quantile(low_chroma_mag, 0.99)),
        "blend_mean": float(np.mean(blend_max)),
        "blend_p90": float(np.quantile(blend_max, 0.90)),
        "blend_p99": float(np.quantile(blend_max, 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
        **gate_stats,
        **axis_stats,
    }
    return out.astype(np.float32, copy=False), stats, blend_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply neutral-region chroma pin-dot suppression.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--guide", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="neutral")
    parser.add_argument("--no-tiff", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide).expanduser() if args.guide else input_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_neutral_chroma_dot"

    image = read_image(input_path)
    guide = read_image(guide_path)
    params = dict(PRESETS[args.preset])
    out, stats, blend = apply_neutral_chroma_dot_filter(
        image,
        guide,
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
