"""Suppress signed chroma-axis outliers in dark flat regions.

This is a diagnostic post-filter for the refiner pilot.  It preserves display
luma and only pulls local chroma-axis outliers back toward a robust local axis
surface.  If this helps visible dark purple/green dots without softening real
detail, the same target can become weak supervision for the next refiner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, median_filter, uniform_filter

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from apply_luma_hf_shrink_filter import sigmoid01
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


AXES = {
    "magenta": np.array([0.5, -1.0, 0.5], dtype=np.float32),
    "red": np.array([1.0, -0.5, -0.5], dtype=np.float32),
    "blue": np.array([-0.5, -0.5, 1.0], dtype=np.float32),
}


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def normalize_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    return axis / max(float(np.sqrt(np.sum(axis * axis))), 1.0e-6)


def flat_dark_gate(
    guide_linear: np.ndarray,
    *,
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
) -> tuple[np.ndarray, dict[str, float]]:
    guide = _safe_rgb(guide_linear)
    display = np.clip(linear_to_srgb_np(guide), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)
    y_linear = luma(guide, LUMA_LINEAR)

    structure = gaussian_filter(y, sigma=float(structure_sigma), mode="reflect")
    detail = np.abs(structure - gaussian_filter(structure, sigma=float(detail_sigma), mode="reflect"))
    edge = gaussian_gradient_magnitude(structure, sigma=float(edge_sigma), mode="reflect")
    flat = sigmoid01((float(detail_threshold) - detail) / max(float(detail_transition), 1.0e-6))
    non_edge = sigmoid01((float(edge_threshold) - edge) / max(float(edge_transition), 1.0e-6))
    shadow = sigmoid01((float(shadow_threshold) - y) / max(float(shadow_transition), 1.0e-6))
    highlight = sigmoid01((y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6))
    gate = (flat * non_edge * shadow * (1.0 - highlight)).astype(np.float32, copy=False)
    stats = {
        "flat_mean": float(np.mean(flat)),
        "non_edge_mean": float(np.mean(non_edge)),
        "shadow_mean": float(np.mean(shadow)),
        "gate_mean": float(np.mean(gate)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
    }
    return gate, stats


def robust_axis_target(axis_value: np.ndarray, *, median_size: int, low_sigma: float) -> np.ndarray:
    med = median_filter(axis_value, size=int(median_size), mode="reflect")
    low = gaussian_filter(axis_value, sigma=float(low_sigma), mode="reflect")
    return 0.65 * med + 0.35 * low


def coherent_structure_gate(
    guide_linear: np.ndarray,
    *,
    coherence_threshold: float,
    coherence_transition: float,
    energy_threshold: float,
    energy_transition: float,
) -> tuple[np.ndarray, dict[str, float]]:
    guide = _safe_rgb(guide_linear)
    display = np.clip(linear_to_srgb_np(guide), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)
    structure = gaussian_filter(y, sigma=0.8, mode="reflect")
    gy, gx = np.gradient(structure)
    jxx = gaussian_filter(gx * gx, sigma=2.0, mode="reflect")
    jyy = gaussian_filter(gy * gy, sigma=2.0, mode="reflect")
    jxy = gaussian_filter(gx * gy, sigma=2.0, mode="reflect")
    energy = jxx + jyy
    coherence = np.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy) / np.maximum(energy, 1.0e-8)
    coherence_part = sigmoid01(
        (coherence - float(coherence_threshold)) / max(float(coherence_transition), 1.0e-6)
    )
    energy_part = sigmoid01((energy - float(energy_threshold)) / max(float(energy_transition), 1.0e-6))
    gate = gaussian_filter(
        np.clip(coherence_part * energy_part, 0.0, 1.0).astype(np.float32, copy=False),
        sigma=0.8,
        mode="reflect",
    )
    stats = {
        "coherent_gate_mean": float(np.mean(gate)),
        "coherent_gate_p90": float(np.quantile(gate, 0.90)),
        "coherent_gate_p99": float(np.quantile(gate, 0.99)),
        "coherent_energy_p99": float(np.quantile(energy, 0.99)),
    }
    return gate.astype(np.float32, copy=False), stats


def apply_signed_chroma_outlier_filter(
    image: np.ndarray,
    guide_image: np.ndarray,
    *,
    strength: float,
    median_size: int,
    low_sigma: float,
    outlier_threshold: float,
    outlier_transition: float,
    magenta_weight: float,
    red_weight: float,
    blue_weight: float,
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
    coherent_inhibit_strength: float = 0.0,
    coherent_inhibit_coherence_threshold: float = 0.42,
    coherent_inhibit_energy_threshold: float = 0.0045,
    outlier_density_inhibit_strength: float = 0.0,
    outlier_density_threshold: float = 0.42,
    outlier_density_transition: float = 0.12,
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
    coherent_stats: dict[str, float] = {
        "coherent_inhibit_strength": float(coherent_inhibit_strength),
        "coherent_inhibit_coherence_threshold": float(coherent_inhibit_coherence_threshold),
        "coherent_inhibit_energy_threshold": float(coherent_inhibit_energy_threshold),
        "coherent_inhibit_mean": 0.0,
        "coherent_inhibit_p99": 0.0,
        "outlier_density_inhibit_strength": float(outlier_density_inhibit_strength),
        "outlier_density_threshold": float(outlier_density_threshold),
        "outlier_density_transition": float(outlier_density_transition),
    }
    if float(coherent_inhibit_strength) > 0.0:
        coherent, coherent_gate_stats = coherent_structure_gate(
            guide_image,
            coherence_threshold=coherent_inhibit_coherence_threshold,
            coherence_transition=0.16,
            energy_threshold=coherent_inhibit_energy_threshold,
            energy_transition=max(float(coherent_inhibit_energy_threshold) * 0.85, 0.0025),
        )
        inhibit = np.clip(coherent * float(coherent_inhibit_strength), 0.0, 0.95).astype(np.float32, copy=False)
        gate = gate * (1.0 - inhibit)
        coherent_stats.update(coherent_gate_stats)
        coherent_stats["coherent_inhibit_mean"] = float(np.mean(inhibit))
        coherent_stats["coherent_inhibit_p99"] = float(np.quantile(inhibit, 0.99))

    out_chroma = chroma.copy()
    blend_max = np.zeros_like(y, dtype=np.float32)
    stats: dict[str, float] = {
        "strength": float(strength),
        "median_size": int(median_size),
        "low_sigma": float(low_sigma),
        "outlier_threshold": float(outlier_threshold),
        "outlier_transition": float(outlier_transition),
        "magenta_weight": float(magenta_weight),
        "red_weight": float(red_weight),
        "blue_weight": float(blue_weight),
        **gate_stats,
        **coherent_stats,
    }
    for name, raw_axis, weight in (
        ("magenta", AXES["magenta"], magenta_weight),
        ("red", AXES["red"], red_weight),
        ("blue", AXES["blue"], blue_weight),
    ):
        if float(weight) <= 0.0:
            continue
        axis = normalize_axis(raw_axis)
        axis_value = np.sum(out_chroma * axis.reshape(1, 1, 3), axis=2)
        target = robust_axis_target(axis_value, median_size=median_size, low_sigma=low_sigma)
        outlier = axis_value - target
        outlier_gate = sigmoid01((np.abs(outlier) - float(outlier_threshold)) / max(float(outlier_transition), 1.0e-6))
        if float(outlier_density_inhibit_strength) > 0.0:
            density = uniform_filter(outlier_gate.astype(np.float32, copy=False), size=5, mode="reflect")
            density_inhibit = sigmoid01(
                (density - float(outlier_density_threshold)) / max(float(outlier_density_transition), 1.0e-6)
            )
            outlier_gate = outlier_gate * (1.0 - np.clip(density_inhibit * float(outlier_density_inhibit_strength), 0.0, 0.95))
            stats[f"{name}_density_mean"] = float(np.mean(density))
            stats[f"{name}_density_p99"] = float(np.quantile(density, 0.99))
            stats[f"{name}_density_inhibit_mean"] = float(np.mean(density_inhibit))
        blend = np.clip(gate * outlier_gate * float(strength) * float(weight), 0.0, 1.0).astype(np.float32, copy=False)
        out_chroma -= (outlier * blend)[..., None] * axis.reshape(1, 1, 3)
        blend_max = np.maximum(blend_max, blend)
        stats[f"{name}_outlier_p99"] = float(np.quantile(np.abs(outlier), 0.99))
        stats[f"{name}_blend_mean"] = float(np.mean(blend))
        stats[f"{name}_blend_p99"] = float(np.quantile(blend, 0.99))

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
    stats["blend_mean"] = float(np.mean(blend_max))
    stats["blend_p90"] = float(np.quantile(blend_max, 0.90))
    stats["blend_p99"] = float(np.quantile(blend_max, 0.99))
    stats["hdr_restore_mean"] = float(np.mean(hdr_restore))
    stats["hdr_restore_p99"] = float(np.quantile(hdr_restore, 0.99))
    return out.astype(np.float32, copy=False), stats, blend_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply signed dark chroma-axis outlier suppression.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--guide", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--median-size", type=int, default=7)
    parser.add_argument("--low-sigma", type=float, default=2.2)
    parser.add_argument("--outlier-threshold", type=float, default=0.0065)
    parser.add_argument("--outlier-transition", type=float, default=0.0040)
    parser.add_argument("--magenta-weight", type=float, default=1.0)
    parser.add_argument("--red-weight", type=float, default=0.0)
    parser.add_argument("--blue-weight", type=float, default=0.75)
    parser.add_argument("--structure-sigma", type=float, default=1.2)
    parser.add_argument("--detail-sigma", type=float, default=2.8)
    parser.add_argument("--detail-threshold", type=float, default=0.020)
    parser.add_argument("--detail-transition", type=float, default=0.010)
    parser.add_argument("--edge-sigma", type=float, default=1.0)
    parser.add_argument("--edge-threshold", type=float, default=0.030)
    parser.add_argument("--edge-transition", type=float, default=0.015)
    parser.add_argument("--shadow-threshold", type=float, default=0.55)
    parser.add_argument("--shadow-transition", type=float, default=0.18)
    parser.add_argument("--highlight-threshold", type=float, default=1.0)
    parser.add_argument("--highlight-transition", type=float, default=0.25)
    parser.add_argument("--hdr-restore-peak-threshold", type=float, default=0.95)
    parser.add_argument("--hdr-restore-threshold", type=float, default=0.85)
    parser.add_argument("--hdr-restore-transition", type=float, default=0.25)
    parser.add_argument("--coherent-inhibit-strength", type=float, default=0.0)
    parser.add_argument("--coherent-inhibit-coherence-threshold", type=float, default=0.42)
    parser.add_argument("--coherent-inhibit-energy-threshold", type=float, default=0.0045)
    parser.add_argument("--outlier-density-inhibit-strength", type=float, default=0.0)
    parser.add_argument("--outlier-density-threshold", type=float, default=0.42)
    parser.add_argument("--outlier-density-transition", type=float, default=0.12)
    parser.add_argument("--no-tiff", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide).expanduser() if args.guide else input_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_signed_chroma_outlier"

    image = read_image(input_path)
    guide = read_image(guide_path)
    out, stats, blend = apply_signed_chroma_outlier_filter(
        image,
        guide,
        strength=args.strength,
        median_size=args.median_size,
        low_sigma=args.low_sigma,
        outlier_threshold=args.outlier_threshold,
        outlier_transition=args.outlier_transition,
        magenta_weight=args.magenta_weight,
        red_weight=args.red_weight,
        blue_weight=args.blue_weight,
        structure_sigma=args.structure_sigma,
        detail_sigma=args.detail_sigma,
        detail_threshold=args.detail_threshold,
        detail_transition=args.detail_transition,
        edge_sigma=args.edge_sigma,
        edge_threshold=args.edge_threshold,
        edge_transition=args.edge_transition,
        shadow_threshold=args.shadow_threshold,
        shadow_transition=args.shadow_transition,
        highlight_threshold=args.highlight_threshold,
        highlight_transition=args.highlight_transition,
        hdr_restore_peak_threshold=args.hdr_restore_peak_threshold,
        hdr_restore_threshold=args.hdr_restore_threshold,
        hdr_restore_transition=args.hdr_restore_transition,
        coherent_inhibit_strength=args.coherent_inhibit_strength,
        coherent_inhibit_coherence_threshold=args.coherent_inhibit_coherence_threshold,
        coherent_inhibit_energy_threshold=args.coherent_inhibit_energy_threshold,
        outlier_density_inhibit_strength=args.outlier_density_inhibit_strength,
        outlier_density_threshold=args.outlier_density_threshold,
        outlier_density_transition=args.outlier_density_transition,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    blend_path = out_dir / f"{name}_blend.png"
    meta_path = out_dir / f"{name}_meta.json"
    write_exr(exr_path, out)
    if not args.no_tiff:
        write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out)).save(preview_path)
    Image.fromarray((np.clip(blend, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(blend_path)
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
