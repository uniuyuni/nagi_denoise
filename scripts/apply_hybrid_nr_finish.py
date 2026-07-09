"""Apply the current best hybrid NR finishing stack.

This script is intentionally small glue around the validated post filters:

1. signed opponent-chroma outlier suppression
2. balanced luma-tail speckle suppression

The default is the quality path because the current real-photo set benefits
from stronger chroma-dot cleanup. Use ``--mode hdr_safe`` for HDR stress images
where highlight microstructure is more important than maximum cleanup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from apply_flat_chroma_smoother import LUMA_LINEAR, luma
from apply_luma_tail_speckle_filter import PRESETS as LUMA_PRESETS
from apply_luma_tail_speckle_filter import apply_luma_tail_speckle_filter
from apply_perceptual_luma_detail_restore import apply_perceptual_luma_detail_restore
from apply_signed_chroma_outlier_filter import apply_signed_chroma_outlier_filter
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


CHROMA_QUALITY = {
    "strength": 0.82,
    "median_size": 7,
    "low_sigma": 2.4,
    "outlier_threshold": 0.0032,
    "outlier_transition": 0.0022,
    "magenta_weight": 1.20,
    "red_weight": 1.15,
    "blue_weight": 1.20,
}

CHROMA_HDR_SAFE = {
    "strength": 0.50,
    "median_size": 7,
    "low_sigma": 2.2,
    "outlier_threshold": 0.0052,
    "outlier_transition": 0.0032,
    "magenta_weight": 1.00,
    "red_weight": 0.0,
    "blue_weight": 0.85,
}

COMMON_CHROMA = {
    "structure_sigma": 1.2,
    "detail_sigma": 2.8,
    "detail_threshold": 0.018,
    "detail_transition": 0.009,
    "edge_sigma": 1.0,
    "edge_threshold": 0.027,
    "edge_transition": 0.013,
    "shadow_threshold": 0.55,
    "shadow_transition": 0.18,
    "highlight_threshold": 1.0,
    "highlight_transition": 0.25,
    "hdr_restore_peak_threshold": 0.95,
    "hdr_restore_threshold": 0.85,
    "hdr_restore_transition": 0.25,
}

LUMA_COMMON = {
    "highpass_sigma": 0.9,
    "local_sigma": 3.0,
    "structure_sigma": 1.2,
    "detail_sigma": 2.8,
    "detail_threshold": 0.018,
    "detail_transition": 0.010,
    "edge_sigma": 1.0,
    "edge_threshold": 0.030,
    "edge_transition": 0.015,
    "highlight_threshold": 1.0,
    "highlight_transition": 0.25,
    "hdr_restore_peak_threshold": 0.95,
    "hdr_restore_threshold": 0.85,
    "hdr_restore_transition": 0.25,
    "correction_limit": 0.025,
}

DETAIL_RESTORE_QUALITY = {
    "strength": 0.18,
    "detail_sigma": 1.0,
    "coherence_sigma": 1.2,
    "coherence_threshold": 0.36,
    "coherence_transition": 0.16,
    "energy_sigma": 1.6,
    "energy_threshold": 0.010,
    "energy_transition": 0.006,
    "base_detail_saturation": 0.70,
    "max_detail_frac": 0.055,
    "min_detail_limit": 0.004,
    "correction_limit": 0.020,
    "zero_mean_sigma": 8.0,
}


def safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def hdr_stats(img: np.ndarray) -> dict[str, float]:
    rgb = np.clip(safe_rgb(img), 0.0, None)
    peak = np.max(rgb, axis=2)
    y = luma(rgb, LUMA_LINEAR)
    return {
        "rgb_max": float(np.max(rgb)),
        "luma_max": float(np.max(y)),
        "peak_gt_1_fraction": float(np.mean(peak > 1.0)),
        "peak_gt_2_fraction": float(np.mean(peak > 2.0)),
        "peak_gt_4_fraction": float(np.mean(peak > 4.0)),
    }


def choose_mode(mode: str, stats: dict[str, float], hdr_fraction_threshold: float, hdr_peak_threshold: float) -> str:
    if mode in {"quality", "hdr_safe"}:
        return mode
    sparse_peak_fraction = max(hdr_fraction_threshold * 0.10, 1.0e-4)
    is_hdr = stats["peak_gt_1_fraction"] >= hdr_fraction_threshold or (
        stats["rgb_max"] >= hdr_peak_threshold and stats["peak_gt_1_fraction"] >= sparse_peak_fraction
    )
    return "hdr_safe" if is_hdr else "quality"


def save_outputs(out_dir: Path, name: str, suffix: str, img: np.ndarray, blend: np.ndarray | None = None) -> dict[str, str]:
    exr_path = out_dir / f"{name}_{suffix}.exr"
    tiff_path = out_dir / f"{name}_{suffix}.tiff"
    preview_path = out_dir / f"{name}_{suffix}_preview.png"
    write_exr(exr_path, img)
    write_tiff(tiff_path, img)
    Image.fromarray(make_preview(img, exposure=1.0, tone="reinhard")).save(preview_path)
    paths = {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path)}
    if blend is not None:
        blend_path = out_dir / f"{name}_{suffix}_blend.png"
        Image.fromarray(np.clip(blend * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(blend_path)
        paths["blend"] = str(blend_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the current best hybrid NR finishing stack.")
    parser.add_argument("--input", required=True, help="Base denoised EXR/TIFF.")
    parser.add_argument("--guide-input", default=None, help="Structure guide. Defaults to --input.")
    parser.add_argument("--hdr-reference", default=None, help="Original linear/HDR source for auto HDR detection.")
    parser.add_argument(
        "--detail-reference",
        default=None,
        help="Original/noisy source for optional coherent luma-detail restoration.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--mode",
        choices=["auto", "quality", "hdr_safe"],
        default="quality",
        help="quality is the production default; hdr_safe is manual HDR protection; auto is diagnostic.",
    )
    parser.add_argument("--hdr-fraction-threshold", type=float, default=0.01, help=argparse.SUPPRESS)
    parser.add_argument("--hdr-peak-threshold", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--luma-preset", choices=sorted(LUMA_PRESETS), default="balanced")
    parser.add_argument("--skip-luma-tail", action="store_true")
    parser.add_argument("--restore-detail", action="store_true", help="Enable coherent luma-detail restoration.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide_input).expanduser() if args.guide_input else input_path
    hdr_ref_path = Path(args.hdr_reference).expanduser() if args.hdr_reference else input_path
    detail_ref_path = Path(args.detail_reference).expanduser() if args.detail_reference else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or input_path.stem

    image = read_image(input_path)
    guide = read_image(guide_path)
    hdr_ref = read_image(hdr_ref_path)
    detail_ref = read_image(detail_ref_path) if detail_ref_path is not None else None
    if image.shape[:2] != guide.shape[:2]:
        raise ValueError(f"shape mismatch: input={image.shape}, guide={guide.shape}")

    stats = hdr_stats(hdr_ref)
    selected_mode = choose_mode(args.mode, stats, args.hdr_fraction_threshold, args.hdr_peak_threshold)
    chroma_params = dict(CHROMA_HDR_SAFE if selected_mode == "hdr_safe" else CHROMA_QUALITY)
    chroma_params.update(COMMON_CHROMA)

    chroma_out, chroma_stats, chroma_blend = apply_signed_chroma_outlier_filter(image, guide, **chroma_params)
    chroma_paths = save_outputs(out_dir, name, "chroma", chroma_out, chroma_blend)

    luma_params = {**LUMA_PRESETS[args.luma_preset], **LUMA_COMMON}

    if args.skip_luma_tail:
        luma_out = chroma_out
        luma_stats = None
        luma_blend = None
    else:
        luma_out, luma_stats, luma_blend = apply_luma_tail_speckle_filter(chroma_out, guide, **luma_params)

    detail_stats = None
    detail_gate = None
    if args.restore_detail:
        if detail_ref is None:
            raise ValueError("--restore-detail requires --detail-reference")
        if detail_ref.shape[:2] != image.shape[:2]:
            raise ValueError(f"shape mismatch: detail_reference={detail_ref.shape}, input={image.shape}")
        final_out, detail_stats, detail_gate = apply_perceptual_luma_detail_restore(
            detail_ref,
            luma_out,
            **DETAIL_RESTORE_QUALITY,
        )
    else:
        final_out = luma_out
    final_paths = save_outputs(out_dir, name, "final", final_out, detail_gate if args.restore_detail else luma_blend)

    meta = {
        "input": str(input_path),
        "guide_input": str(guide_path),
        "hdr_reference": str(hdr_ref_path),
        "detail_reference": None if detail_ref_path is None else str(detail_ref_path),
        "requested_mode": args.mode,
        "selected_mode": selected_mode,
        "hdr_detection": stats,
        "params": {
            "chroma": chroma_params,
            "luma_tail": None if args.skip_luma_tail else luma_params,
            "detail_restore": DETAIL_RESTORE_QUALITY if args.restore_detail else None,
        },
        "outputs": {"chroma": chroma_paths, "final": final_paths},
        "filter": {"chroma": chroma_stats, "luma_tail": luma_stats, "detail_restore": detail_stats},
        "input_stats": image_stats(image),
        "output_stats": image_stats(final_out),
    }
    meta_path = out_dir / f"{name}_hybrid_finish.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
