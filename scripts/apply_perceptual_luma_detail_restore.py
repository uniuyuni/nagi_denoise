"""Restore coherent perceptual luma detail from a noisy reference.

This is a post-filter experiment for real-photo NR. It borrows only signed luma
detail that is locally coherent in the noisy/original reference, then applies a
small, clipped correction to the denoised base in display space. Chroma remains
from the denoised base to avoid bringing color noise back.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import image_stats, make_preview, read_image


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def coherent_missing_luma_detail(
    reference_display: np.ndarray,
    base_display: np.ndarray,
    *,
    detail_sigma: float,
    coherence_sigma: float,
    coherence_threshold: float,
    coherence_transition: float,
    energy_sigma: float,
    energy_threshold: float,
    energy_transition: float,
    base_detail_saturation: float,
    base_energy_threshold: float = 0.0,
    base_energy_transition: float = 0.006,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    ref_y = luma(reference_display, LUMA_SRGB)
    base_y = luma(base_display, LUMA_SRGB)
    ref_low = gaussian_filter(ref_y, sigma=float(detail_sigma), mode="reflect")
    base_low = gaussian_filter(base_y, sigma=float(detail_sigma), mode="reflect")
    ref_detail = ref_y - ref_low
    base_detail = base_y - base_low
    missing = ref_detail - base_detail

    abs_ref = np.abs(ref_detail)
    coherent = np.abs(gaussian_filter(ref_detail, sigma=float(coherence_sigma), mode="reflect"))
    local_energy = gaussian_filter(abs_ref, sigma=float(coherence_sigma), mode="reflect")
    coherence = coherent / np.maximum(local_energy, 1.0e-6)
    coherence_gate = sigmoid01(
        (coherence - float(coherence_threshold)) / max(float(coherence_transition), 1.0e-6)
    )

    energy = gaussian_filter(abs_ref, sigma=float(energy_sigma), mode="reflect")
    energy_gate = sigmoid01((energy - float(energy_threshold)) / max(float(energy_transition), 1.0e-6))

    base_abs = np.abs(base_detail)
    already_has_detail = sigmoid01((base_abs - abs_ref * float(base_detail_saturation)) / 0.0025)
    gate = np.clip(coherence_gate * energy_gate * (1.0 - 0.65 * already_has_detail), 0.0, 1.0).astype(
        np.float32, copy=False
    )
    base_energy_gate = np.ones_like(gate, dtype=np.float32)
    if float(base_energy_threshold) > 0.0:
        base_energy = gaussian_filter(base_abs, sigma=float(energy_sigma), mode="reflect")
        base_energy_gate = sigmoid01(
            (base_energy - float(base_energy_threshold)) / max(float(base_energy_transition), 1.0e-6)
        )
        gate *= base_energy_gate
    stats = {
        "ref_detail_abs_p95": float(np.quantile(abs_ref, 0.95)),
        "ref_detail_abs_p99": float(np.quantile(abs_ref, 0.99)),
        "base_detail_abs_p95": float(np.quantile(base_abs, 0.95)),
        "base_detail_abs_p99": float(np.quantile(base_abs, 0.99)),
        "coherence_mean": float(np.mean(coherence)),
        "coherence_gate_mean": float(np.mean(coherence_gate)),
        "energy_gate_mean": float(np.mean(energy_gate)),
        "base_energy_gate_mean": float(np.mean(base_energy_gate)),
        "already_has_detail_mean": float(np.mean(already_has_detail)),
        "restore_gate_mean": float(np.mean(gate)),
        "restore_gate_p90": float(np.quantile(gate, 0.90)),
        "restore_gate_p99": float(np.quantile(gate, 0.99)),
    }
    return missing.astype(np.float32, copy=False), gate, stats


def apply_perceptual_luma_detail_restore(
    reference: np.ndarray,
    base: np.ndarray,
    *,
    strength: float,
    detail_sigma: float,
    coherence_sigma: float,
    coherence_threshold: float,
    coherence_transition: float,
    energy_sigma: float,
    energy_threshold: float,
    energy_transition: float,
    base_detail_saturation: float,
    max_detail_frac: float,
    min_detail_limit: float,
    correction_limit: float,
    zero_mean_sigma: float,
    base_energy_threshold: float = 0.0,
    base_energy_transition: float = 0.006,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    ref_rgb = np.clip(_safe_rgb(reference), 0.0, None)
    base_rgb = np.clip(_safe_rgb(base), 0.0, None)
    if ref_rgb.shape[:2] != base_rgb.shape[:2]:
        raise ValueError(f"shape mismatch: reference={ref_rgb.shape}, base={base_rgb.shape}")

    ref_display = np.clip(linear_to_srgb_np(ref_rgb), 0.0, 1.0)
    base_display = np.clip(linear_to_srgb_np(base_rgb), 0.0, 1.0)
    base_y = luma(base_display, LUMA_SRGB)
    missing, gate, detail_stats = coherent_missing_luma_detail(
        ref_display,
        base_display,
        detail_sigma=detail_sigma,
        coherence_sigma=coherence_sigma,
        coherence_threshold=coherence_threshold,
        coherence_transition=coherence_transition,
        energy_sigma=energy_sigma,
        energy_threshold=energy_threshold,
        energy_transition=energy_transition,
        base_detail_saturation=base_detail_saturation,
        base_energy_threshold=base_energy_threshold,
        base_energy_transition=base_energy_transition,
    )

    if zero_mean_sigma > 0:
        weighted = gaussian_filter(missing * gate, sigma=float(zero_mean_sigma), mode="reflect")
        weight = gaussian_filter(gate, sigma=float(zero_mean_sigma), mode="reflect")
        missing = missing - weighted / np.maximum(weight, 1.0e-6)

    limit = np.maximum(float(min_detail_limit), np.maximum(base_y, 0.0) * float(max_detail_frac))
    limit = np.minimum(limit, float(correction_limit))
    correction = np.clip(missing, -limit, limit) * gate * float(strength)
    out_y = np.clip(base_y + correction, 0.0, 1.0)

    base_chroma = base_display / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(base_chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)

    stats = {
        **detail_stats,
        "strength": float(strength),
        "detail_sigma": float(detail_sigma),
        "coherence_sigma": float(coherence_sigma),
        "coherence_threshold": float(coherence_threshold),
        "energy_threshold": float(energy_threshold),
        "base_energy_threshold": float(base_energy_threshold),
        "base_energy_transition": float(base_energy_transition),
        "max_detail_frac": float(max_detail_frac),
        "min_detail_limit": float(min_detail_limit),
        "correction_limit": float(correction_limit),
        "correction_abs_mean": float(np.mean(np.abs(correction))),
        "correction_abs_p95": float(np.quantile(np.abs(correction), 0.95)),
        "correction_abs_p99": float(np.quantile(np.abs(correction), 0.99)),
    }
    return out, stats, gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore coherent perceptual luma detail from a noisy reference.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--strength", type=float, default=0.28)
    parser.add_argument("--detail-sigma", type=float, default=1.0)
    parser.add_argument("--coherence-sigma", type=float, default=1.2)
    parser.add_argument("--coherence-threshold", type=float, default=0.36)
    parser.add_argument("--coherence-transition", type=float, default=0.16)
    parser.add_argument("--energy-sigma", type=float, default=1.6)
    parser.add_argument("--energy-threshold", type=float, default=0.010)
    parser.add_argument("--energy-transition", type=float, default=0.006)
    parser.add_argument("--base-detail-saturation", type=float, default=0.70)
    parser.add_argument("--base-energy-threshold", type=float, default=0.0)
    parser.add_argument("--base-energy-transition", type=float, default=0.006)
    parser.add_argument("--max-detail-frac", type=float, default=0.055)
    parser.add_argument("--min-detail-limit", type=float, default=0.004)
    parser.add_argument("--correction-limit", type=float, default=0.020)
    parser.add_argument("--zero-mean-sigma", type=float, default=8.0)
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_perceptual_detail"

    reference = read_image(reference_path)
    base = read_image(base_path)
    out, stats, gate = apply_perceptual_luma_detail_restore(
        reference,
        base,
        strength=args.strength,
        detail_sigma=args.detail_sigma,
        coherence_sigma=args.coherence_sigma,
        coherence_threshold=args.coherence_threshold,
        coherence_transition=args.coherence_transition,
        energy_sigma=args.energy_sigma,
        energy_threshold=args.energy_threshold,
        energy_transition=args.energy_transition,
        base_detail_saturation=args.base_detail_saturation,
        max_detail_frac=args.max_detail_frac,
        min_detail_limit=args.min_detail_limit,
        correction_limit=args.correction_limit,
        zero_mean_sigma=args.zero_mean_sigma,
        base_energy_threshold=args.base_energy_threshold,
        base_energy_transition=args.base_energy_transition,
    )

    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    meta = {
        "reference": str(reference_path),
        "base": str(base_path),
        "outputs": {"exr": str(exr_path), "preview": str(preview_path), "gate": str(gate_path)},
        "params": vars(args),
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


if __name__ == "__main__":
    main()
