"""Highlight luma detail guard for Perfect NR experiments.

This is a deliberately small postprocess prototype. It starts from a denoised
base image that already has acceptable color/luma, then restores only coherent
high-frequency luma detail from the reference/input inside a smooth highlight
mask. Chroma is kept from the base image to avoid reintroducing color noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from perfect_nr_compare import high_frequency_luma, luma
from perfect_nr_probe import image_stats, make_preview, read_image


LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    x = x[..., :3]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_smooth_luma_mask(
    reference: np.ndarray,
    mode: str,
    threshold: float,
    top_percent: float,
    transition: float,
) -> tuple[np.ndarray, dict]:
    ref_luma = luma(reference)
    if mode == "top-luma":
        q = 1.0 - float(top_percent) / 100.0
        hard_threshold = float(np.quantile(ref_luma, q))
    elif mode == "luma-threshold":
        hard_threshold = float(threshold)
    elif mode == "peak-threshold":
        peak = np.max(np.clip(_safe_rgb(reference), 0.0, None), axis=2)
        hard_threshold = float(threshold)
        mask = smoothstep((peak - hard_threshold) / max(float(transition), 1.0e-6))
        return mask.astype(np.float32), {
            "mode": mode,
            "threshold": hard_threshold,
            "top_percent": top_percent,
            "transition": transition,
            "mean": float(np.mean(mask)),
            "hard_fraction": float(np.mean(peak > hard_threshold)),
        }
    else:
        raise ValueError(f"unknown mask mode: {mode}")

    mask = smoothstep((ref_luma - hard_threshold) / max(float(transition), 1.0e-6))
    return mask.astype(np.float32), {
        "mode": mode,
        "threshold": hard_threshold,
        "top_percent": top_percent,
        "transition": transition,
        "mean": float(np.mean(mask)),
        "hard_fraction": float(np.mean(ref_luma > hard_threshold)),
    }


def coherent_luma_detail(
    reference: np.ndarray,
    detail_sigma: float,
    coherence_sigma: float,
    coherence_threshold: float,
    coherence_transition: float,
) -> tuple[np.ndarray, np.ndarray]:
    ref_luma = luma(reference)
    raw_detail = ref_luma - gaussian_filter(ref_luma, sigma=float(detail_sigma), mode="reflect")

    abs_detail = np.abs(raw_detail)
    coherent = np.abs(gaussian_filter(raw_detail, sigma=float(coherence_sigma), mode="reflect"))
    local_energy = gaussian_filter(abs_detail, sigma=float(coherence_sigma), mode="reflect")
    coherence = coherent / np.maximum(local_energy, 1.0e-6)
    gate = smoothstep(
        (coherence - float(coherence_threshold)) / max(float(coherence_transition), 1.0e-6)
    )
    return raw_detail * gate, gate.astype(np.float32)


def zero_mean_under_mask(detail: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return detail
    weighted = gaussian_filter(detail * mask, sigma=float(sigma), mode="reflect")
    weight = gaussian_filter(mask, sigma=float(sigma), mode="reflect")
    local_mean = weighted / np.maximum(weight, 1.0e-6)
    return detail - local_mean * mask


def apply_detail_guard(
    reference: np.ndarray,
    base: np.ndarray,
    mask: np.ndarray,
    detail: np.ndarray,
    strength: float,
    max_detail_frac: float,
    min_detail_limit: float,
    zero_mean_sigma: float,
) -> tuple[np.ndarray, dict]:
    ref = _safe_rgb(reference)
    base_rgb = np.clip(_safe_rgb(base), 0.0, None)
    if ref.shape != base_rgb.shape:
        raise ValueError(f"shape mismatch: reference={ref.shape}, base={base_rgb.shape}")

    base_luma = luma(base_rgb)
    detail = zero_mean_under_mask(detail, mask, sigma=float(zero_mean_sigma))
    limit = np.maximum(float(min_detail_limit), np.maximum(base_luma, 0.0) * float(max_detail_frac))
    limited_detail = np.clip(detail, -limit, limit)
    applied = mask * float(strength) * limited_detail

    target_luma = np.maximum(base_luma + applied, 0.0)
    base_chroma = base_rgb / np.maximum(base_luma[..., None], 1.0e-6)
    out = base_chroma * target_luma[..., None]

    # Avoid unstable chroma in extremely dark pixels.
    dark = base_luma < 1.0e-5
    out[dark] = base_rgb[dark]

    hard_mask = mask > 0.5
    stats = {
        "mask_fraction_gt_0_5": float(np.mean(hard_mask)),
        "mask_mean": float(np.mean(mask)),
        "raw_detail_abs_mean": float(np.mean(np.abs(detail[hard_mask]))) if np.any(hard_mask) else 0.0,
        "applied_luma_abs_mean": float(np.mean(np.abs(applied[hard_mask]))) if np.any(hard_mask) else 0.0,
        "applied_luma_mean": float(np.mean(applied[hard_mask])) if np.any(hard_mask) else 0.0,
        "base_luma_mean": float(np.mean(base_luma[hard_mask])) if np.any(hard_mask) else 0.0,
        "output_luma_mean": float(np.mean(luma(out)[hard_mask])) if np.any(hard_mask) else 0.0,
        "reference_hf_abs_mean": float(np.mean(np.abs(high_frequency_luma(ref, 2.0)[hard_mask])))
        if np.any(hard_mask)
        else 0.0,
        "output_hf_abs_mean": float(np.mean(np.abs(high_frequency_luma(out, 2.0)[hard_mask])))
        if np.any(hard_mask)
        else 0.0,
    }
    return out.astype(np.float32, copy=False), stats


def write_exr(path: Path, image: np.ndarray) -> None:
    import OpenEXR

    rgb = np.ascontiguousarray(_safe_rgb(image).astype(np.float32, copy=False))
    height, width, _ = rgb.shape
    header = OpenEXR.Header(width, height)
    out = OpenEXR.OutputFile(str(path), header)
    try:
        out.writePixels(
            {
                "R": np.ascontiguousarray(rgb[..., 0]).tobytes(),
                "G": np.ascontiguousarray(rgb[..., 1]).tobytes(),
                "B": np.ascontiguousarray(rgb[..., 2]).tobytes(),
            }
        )
    finally:
        out.close()


def write_tiff(path: Path, image: np.ndarray) -> None:
    import tifffile

    tifffile.imwrite(path, _safe_rgb(image).astype(np.float32, copy=False), photometric="rgb")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore coherent highlight luma detail without chroma transfer.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", default="runs/perfect_nr/detail_guard")
    parser.add_argument("--name", default="detail_guard")
    parser.add_argument("--mask-mode", choices=["peak-threshold", "luma-threshold", "top-luma"], default="luma-threshold")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--top-percent", type=float, default=8.0)
    parser.add_argument("--transition", type=float, default=0.80)
    parser.add_argument("--detail-sigma", type=float, default=1.2)
    parser.add_argument("--coherence-sigma", type=float, default=0.9)
    parser.add_argument("--coherence-threshold", type=float, default=0.22)
    parser.add_argument("--coherence-transition", type=float, default=0.28)
    parser.add_argument("--strength", type=float, default=0.50)
    parser.add_argument("--max-detail-frac", type=float, default=0.18)
    parser.add_argument("--min-detail-limit", type=float, default=0.02)
    parser.add_argument("--zero-mean-sigma", type=float, default=8.0)
    parser.add_argument("--preview-exposure", type=float, default=1.0)
    parser.add_argument("--preview-tone", choices=["reinhard", "clip"], default="reinhard")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    reference = read_image(reference_path)
    base = read_image(base_path)

    mask, mask_meta = make_smooth_luma_mask(
        reference,
        mode=args.mask_mode,
        threshold=args.threshold,
        top_percent=args.top_percent,
        transition=args.transition,
    )
    detail, coherence_gate = coherent_luma_detail(
        reference,
        detail_sigma=args.detail_sigma,
        coherence_sigma=args.coherence_sigma,
        coherence_threshold=args.coherence_threshold,
        coherence_transition=args.coherence_transition,
    )
    output, guard_stats = apply_detail_guard(
        reference,
        base,
        mask=mask,
        detail=detail,
        strength=args.strength,
        max_detail_frac=args.max_detail_frac,
        min_detail_limit=args.min_detail_limit,
        zero_mean_sigma=args.zero_mean_sigma,
    )

    exr_path = out_dir / f"{args.name}.exr"
    tiff_path = out_dir / f"{args.name}.tiff"
    preview_path = out_dir / f"{args.name}_preview.png"
    mask_path = out_dir / f"{args.name}_mask.png"
    gate_path = out_dir / f"{args.name}_coherence_gate.png"
    json_path = out_dir / f"{args.name}.json"

    write_exr(exr_path, output)
    write_tiff(tiff_path, output)
    Image.fromarray(make_preview(output, exposure=args.preview_exposure, tone=args.preview_tone)).save(preview_path)
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(mask_path)
    Image.fromarray((np.clip(coherence_gate, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(gate_path)

    metadata = {
        "reference": str(reference_path),
        "base": str(base_path),
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "mask": str(mask_path),
            "coherence_gate": str(gate_path),
        },
        "parameters": {
            "mask_mode": args.mask_mode,
            "threshold": args.threshold,
            "top_percent": args.top_percent,
            "transition": args.transition,
            "detail_sigma": args.detail_sigma,
            "coherence_sigma": args.coherence_sigma,
            "coherence_threshold": args.coherence_threshold,
            "coherence_transition": args.coherence_transition,
            "strength": args.strength,
            "max_detail_frac": args.max_detail_frac,
            "min_detail_limit": args.min_detail_limit,
            "zero_mean_sigma": args.zero_mean_sigma,
        },
        "mask": mask_meta,
        "guard_stats": guard_stats,
        "output_stats": image_stats(output),
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata["guard_stats"], indent=2))
    print(f"wrote {exr_path}")
    print(f"wrote {preview_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
