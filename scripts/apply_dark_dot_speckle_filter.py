"""Suppress isolated dark display-luma dots after the current NR candidate.

The luma tail filter reduces symmetric high-frequency speckles. Some real
photos still show dark purple/blue pin dots in otherwise flat regions, especially
sky and blue shadows. This pass only raises pixels that are darker than their
local median, while using the existing flat/line gates to avoid hair, branches,
and other coherent fine structure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from apply_blue_structure_protector import blue_structure_mask
from apply_luma_hf_shrink_filter import make_flat_gate, make_line_preserve_gate, sigmoid01
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


PRESETS = {
    "quality": {
        "strength": 0.62,
        "median_size": 5,
        "dark_threshold": 0.0060,
        "dark_transition": 0.0030,
        "local_sigma": 2.0,
        "local_gain": 0.22,
        "shadow_boost": 0.30,
        "max_lift": 0.030,
        "chroma_strength": 0.36,
        "chroma_sigma": 1.4,
        "line_preserve_strength": 0.90,
    },
    "strong": {
        "strength": 0.78,
        "median_size": 5,
        "dark_threshold": 0.0048,
        "dark_transition": 0.0025,
        "local_sigma": 2.0,
        "local_gain": 0.18,
        "shadow_boost": 0.45,
        "max_lift": 0.038,
        "chroma_strength": 0.48,
        "chroma_sigma": 1.6,
        "line_preserve_strength": 0.86,
    },
    "sky": {
        "strength": 0.92,
        "median_size": 5,
        "dark_threshold": 0.0038,
        "dark_transition": 0.0022,
        "local_sigma": 2.4,
        "local_gain": 0.12,
        "shadow_boost": 0.55,
        "max_lift": 0.045,
        "chroma_strength": 0.56,
        "chroma_sigma": 1.8,
        "line_preserve_strength": 0.80,
    },
}


def apply_dark_dot_speckle_filter(
    image: np.ndarray,
    guide_image: np.ndarray,
    *,
    strength: float,
    median_size: int,
    dark_threshold: float,
    dark_transition: float,
    local_sigma: float,
    local_gain: float,
    shadow_boost: float,
    shadow_threshold: float,
    shadow_transition: float,
    max_lift: float,
    chroma_strength: float,
    chroma_sigma: float,
    structure_sigma: float,
    detail_sigma: float,
    detail_threshold: float,
    detail_transition: float,
    edge_sigma: float,
    edge_threshold: float,
    edge_transition: float,
    highlight_threshold: float,
    highlight_transition: float,
    line_sigma: float,
    line_smooth_sigma: float,
    line_threshold: float,
    line_transition: float,
    line_coherence_threshold: float,
    line_coherence_transition: float,
    line_preserve_strength: float,
    blue_structure_inhibit: float,
    blue_structure_threshold: float,
    blue_structure_transition: float,
    blue_structure_chroma_threshold: float,
    blue_structure_chroma_transition: float,
    sky_flat_strength: float,
    sky_luma_min: float,
    sky_luma_max: float,
    sky_luma_transition: float,
    sky_neutral_threshold: float,
    sky_neutral_transition: float,
    sky_blue_abs_threshold: float,
    sky_blue_abs_transition: float,
    sky_line_max: float,
    sky_line_transition: float,
    hdr_restore_peak_threshold: float,
    hdr_restore_threshold: float,
    hdr_restore_transition: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    base = np.nan_to_num(image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    display = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)

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
    line_gate, line_stats = make_line_preserve_gate(
        guide_image,
        line_sigma=line_sigma,
        line_smooth_sigma=line_smooth_sigma,
        line_threshold=line_threshold,
        line_transition=line_transition,
        line_coherence_threshold=line_coherence_threshold,
        line_coherence_transition=line_coherence_transition,
    )

    med_size = max(3, int(median_size) | 1)
    y_med = median_filter(y, size=med_size, mode="reflect")
    dark_residual = np.maximum(y_med - y, 0.0)
    local_dark = gaussian_filter(dark_residual, sigma=float(local_sigma), mode="reflect")
    adaptive_threshold = float(dark_threshold) + float(local_gain) * local_dark
    dark_gate = sigmoid01((dark_residual - adaptive_threshold) / max(float(dark_transition), 1.0e-6))

    shadow_gate = sigmoid01((float(shadow_threshold) - y) / max(float(shadow_transition), 1.0e-6))
    line_keep = 1.0 - np.clip(float(line_preserve_strength) * line_gate, 0.0, 1.0)
    boost = 1.0 + float(shadow_boost) * shadow_gate
    blend = np.clip(flat_gate * line_keep * dark_gate * boost * float(strength), 0.0, 1.0).astype(
        np.float32, copy=False
    )
    blue_struct = np.zeros_like(blend, dtype=np.float32)
    if blue_structure_inhibit > 0.0:
        blue_struct = blue_structure_mask(
            guide_image,
            blue_threshold=float(blue_structure_threshold),
            blue_transition=float(blue_structure_transition),
            chroma_threshold=float(blue_structure_chroma_threshold),
            chroma_transition=float(blue_structure_chroma_transition),
            luma_min=0.040,
            luma_max=0.86,
        )
        blend *= np.clip(1.0 - float(blue_structure_inhibit) * blue_struct, 0.0, 1.0)
    sky_flat_gate = np.ones_like(blend, dtype=np.float32)
    if sky_flat_strength > 0.0:
        guide_display = np.clip(linear_to_srgb_np(np.clip(guide_image[..., :3].astype(np.float32), 0.0, None)), 0.0, 1.0)
        guide_y = luma(guide_display, LUMA_SRGB)
        guide_chroma = guide_display - guide_y[..., None]
        low_chroma = gaussian_filter(guide_chroma, sigma=(3.0, 3.0, 0.0), mode="reflect")
        low_chroma_mag = np.sqrt(np.sum(low_chroma * low_chroma, axis=2))
        low_blue = low_chroma[..., 2] - 0.5 * (low_chroma[..., 0] + low_chroma[..., 1])
        luma_gate = sigmoid01((guide_y - float(sky_luma_min)) / max(float(sky_luma_transition), 1.0e-6))
        luma_gate *= sigmoid01((float(sky_luma_max) - guide_y) / max(float(sky_luma_transition), 1.0e-6))
        neutral_gate = sigmoid01(
            (float(sky_neutral_threshold) - low_chroma_mag) / max(float(sky_neutral_transition), 1.0e-6)
        )
        blue_abs_gate = sigmoid01(
            (float(sky_blue_abs_threshold) - np.abs(low_blue)) / max(float(sky_blue_abs_transition), 1.0e-6)
        )
        line_open = sigmoid01((float(sky_line_max) - line_gate) / max(float(sky_line_transition), 1.0e-6))
        sky_flat_gate = np.clip(luma_gate * neutral_gate * blue_abs_gate * line_open, 0.0, 1.0).astype(
            np.float32, copy=False
        )
        blend *= np.clip((1.0 - float(sky_flat_strength)) + float(sky_flat_strength) * sky_flat_gate, 0.0, 1.0)

    lift = np.clip(y_med - y, 0.0, float(max_lift))
    out_y = np.clip(y + lift * blend, 0.0, 1.0)

    chroma = display - y[..., None]
    low_chroma = gaussian_filter(chroma, sigma=(float(chroma_sigma), float(chroma_sigma), 0.0), mode="reflect")
    med_chroma = median_filter(chroma, size=(med_size, med_size, 1), mode="reflect")
    target_chroma = 0.60 * med_chroma + 0.40 * low_chroma
    chroma_blend = np.clip(blend * float(chroma_strength), 0.0, 1.0)[..., None]
    out_chroma = chroma * (1.0 - chroma_blend) + target_chroma * chroma_blend
    out_display = np.clip(out_y[..., None] + out_chroma, 0.0, 1.0)
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
        "median_size": int(med_size),
        "dark_threshold": float(dark_threshold),
        "dark_transition": float(dark_transition),
        "local_sigma": float(local_sigma),
        "local_gain": float(local_gain),
        "shadow_boost": float(shadow_boost),
        "shadow_threshold": float(shadow_threshold),
        "shadow_transition": float(shadow_transition),
        "max_lift": float(max_lift),
        "chroma_strength": float(chroma_strength),
        "chroma_sigma": float(chroma_sigma),
        "line_preserve_strength": float(line_preserve_strength),
        "blue_structure_inhibit": float(blue_structure_inhibit),
        "blue_structure_mean": float(np.mean(blue_struct)),
        "blue_structure_p95": float(np.quantile(blue_struct, 0.95)),
        "sky_flat_strength": float(sky_flat_strength),
        "sky_flat_gate_mean": float(np.mean(sky_flat_gate)),
        "sky_flat_gate_p90": float(np.quantile(sky_flat_gate, 0.90)),
        "sky_flat_gate_p99": float(np.quantile(sky_flat_gate, 0.99)),
        "dark_residual_mean": float(np.mean(dark_residual)),
        "dark_residual_p95": float(np.quantile(dark_residual, 0.95)),
        "dark_residual_p99": float(np.quantile(dark_residual, 0.99)),
        "dark_gate_mean": float(np.mean(dark_gate)),
        "dark_gate_p90": float(np.quantile(dark_gate, 0.90)),
        "dark_gate_p99": float(np.quantile(dark_gate, 0.99)),
        "shadow_gate_mean": float(np.mean(shadow_gate)),
        "blend_mean": float(np.mean(blend)),
        "blend_p90": float(np.quantile(blend, 0.90)),
        "blend_p99": float(np.quantile(blend, 0.99)),
        "lift_mean": float(np.mean(lift * blend)),
        "lift_p99": float(np.quantile(lift * blend, 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
        **flat_stats,
        **line_stats,
    }
    return out, stats, blend


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply isolated dark-dot suppression.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--guide-input", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quality")
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--median-size", type=int, default=None)
    parser.add_argument("--dark-threshold", type=float, default=None)
    parser.add_argument("--dark-transition", type=float, default=None)
    parser.add_argument("--local-sigma", type=float, default=None)
    parser.add_argument("--local-gain", type=float, default=None)
    parser.add_argument("--shadow-boost", type=float, default=None)
    parser.add_argument("--shadow-threshold", type=float, default=0.24)
    parser.add_argument("--shadow-transition", type=float, default=0.10)
    parser.add_argument("--max-lift", type=float, default=None)
    parser.add_argument("--chroma-strength", type=float, default=None)
    parser.add_argument("--chroma-sigma", type=float, default=None)
    parser.add_argument("--structure-sigma", type=float, default=1.2)
    parser.add_argument("--detail-sigma", type=float, default=2.8)
    parser.add_argument("--detail-threshold", type=float, default=0.018)
    parser.add_argument("--detail-transition", type=float, default=0.010)
    parser.add_argument("--edge-sigma", type=float, default=1.0)
    parser.add_argument("--edge-threshold", type=float, default=0.030)
    parser.add_argument("--edge-transition", type=float, default=0.015)
    parser.add_argument("--highlight-threshold", type=float, default=1.0)
    parser.add_argument("--highlight-transition", type=float, default=0.25)
    parser.add_argument("--line-sigma", type=float, default=0.70)
    parser.add_argument("--line-smooth-sigma", type=float, default=1.00)
    parser.add_argument("--line-threshold", type=float, default=0.010)
    parser.add_argument("--line-transition", type=float, default=0.006)
    parser.add_argument("--line-coherence-threshold", type=float, default=0.42)
    parser.add_argument("--line-coherence-transition", type=float, default=0.16)
    parser.add_argument("--line-preserve-strength", type=float, default=None)
    parser.add_argument("--blue-structure-inhibit", type=float, default=0.0)
    parser.add_argument("--blue-structure-threshold", type=float, default=0.052)
    parser.add_argument("--blue-structure-transition", type=float, default=0.024)
    parser.add_argument("--blue-structure-chroma-threshold", type=float, default=0.064)
    parser.add_argument("--blue-structure-chroma-transition", type=float, default=0.030)
    parser.add_argument("--sky-flat-strength", type=float, default=0.0)
    parser.add_argument("--sky-luma-min", type=float, default=0.025)
    parser.add_argument("--sky-luma-max", type=float, default=0.46)
    parser.add_argument("--sky-luma-transition", type=float, default=0.075)
    parser.add_argument("--sky-neutral-threshold", type=float, default=0.105)
    parser.add_argument("--sky-neutral-transition", type=float, default=0.045)
    parser.add_argument("--sky-blue-abs-threshold", type=float, default=0.045)
    parser.add_argument("--sky-blue-abs-transition", type=float, default=0.024)
    parser.add_argument("--sky-line-max", type=float, default=0.34)
    parser.add_argument("--sky-line-transition", type=float, default=0.16)
    parser.add_argument("--hdr-restore-peak-threshold", type=float, default=0.95)
    parser.add_argument("--hdr-restore-threshold", type=float, default=0.85)
    parser.add_argument("--hdr-restore-transition", type=float, default=0.25)
    parser.add_argument("--no-tiff", action="store_true")
    args = parser.parse_args()

    params = dict(PRESETS[args.preset])
    for attr, key in (
        ("strength", "strength"),
        ("median_size", "median_size"),
        ("dark_threshold", "dark_threshold"),
        ("dark_transition", "dark_transition"),
        ("local_sigma", "local_sigma"),
        ("local_gain", "local_gain"),
        ("shadow_boost", "shadow_boost"),
        ("max_lift", "max_lift"),
        ("chroma_strength", "chroma_strength"),
        ("chroma_sigma", "chroma_sigma"),
        ("line_preserve_strength", "line_preserve_strength"),
    ):
        value = getattr(args, attr)
        if value is not None:
            params[key] = value

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide_input).expanduser() if args.guide_input else input_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_dark_dot_{args.preset}"

    image = read_image(input_path)
    guide = read_image(guide_path)
    if image.shape[:2] != guide.shape[:2]:
        raise ValueError(f"shape mismatch: input={image.shape}, guide={guide.shape}")

    out, stats, blend = apply_dark_dot_speckle_filter(
        image,
        guide,
        strength=float(params["strength"]),
        median_size=int(params["median_size"]),
        dark_threshold=float(params["dark_threshold"]),
        dark_transition=float(params["dark_transition"]),
        local_sigma=float(params["local_sigma"]),
        local_gain=float(params["local_gain"]),
        shadow_boost=float(params["shadow_boost"]),
        shadow_threshold=args.shadow_threshold,
        shadow_transition=args.shadow_transition,
        max_lift=float(params["max_lift"]),
        chroma_strength=float(params["chroma_strength"]),
        chroma_sigma=float(params["chroma_sigma"]),
        structure_sigma=args.structure_sigma,
        detail_sigma=args.detail_sigma,
        detail_threshold=args.detail_threshold,
        detail_transition=args.detail_transition,
        edge_sigma=args.edge_sigma,
        edge_threshold=args.edge_threshold,
        edge_transition=args.edge_transition,
        highlight_threshold=args.highlight_threshold,
        highlight_transition=args.highlight_transition,
        line_sigma=args.line_sigma,
        line_smooth_sigma=args.line_smooth_sigma,
        line_threshold=args.line_threshold,
        line_transition=args.line_transition,
        line_coherence_threshold=args.line_coherence_threshold,
        line_coherence_transition=args.line_coherence_transition,
        line_preserve_strength=float(params["line_preserve_strength"]),
        blue_structure_inhibit=args.blue_structure_inhibit,
        blue_structure_threshold=args.blue_structure_threshold,
        blue_structure_transition=args.blue_structure_transition,
        blue_structure_chroma_threshold=args.blue_structure_chroma_threshold,
        blue_structure_chroma_transition=args.blue_structure_chroma_transition,
        sky_flat_strength=args.sky_flat_strength,
        sky_luma_min=args.sky_luma_min,
        sky_luma_max=args.sky_luma_max,
        sky_luma_transition=args.sky_luma_transition,
        sky_neutral_threshold=args.sky_neutral_threshold,
        sky_neutral_transition=args.sky_neutral_transition,
        sky_blue_abs_threshold=args.sky_blue_abs_threshold,
        sky_blue_abs_transition=args.sky_blue_abs_transition,
        sky_line_max=args.sky_line_max,
        sky_line_transition=args.sky_line_transition,
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
    Image.fromarray(np.clip(blend * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
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
