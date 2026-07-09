"""Compare Perfect NR diagnostic outputs against a reference crop.

The goal is not to replace visual judgment. The goal is to make every visual
judgment reproducible: same preview tone curve, same highlight mask, same
chroma/luma/detail summaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .probe import image_stats, make_preview, read_image


LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    x = x[..., :3]
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def luma(img: np.ndarray) -> np.ndarray:
    return np.sum(_safe_rgb(img) * LUMA_WEIGHTS, axis=2)


def chroma_ratio(img: np.ndarray) -> np.ndarray:
    rgb = np.clip(_safe_rgb(img), 0.0, None)
    lum = np.sum(rgb * LUMA_WEIGHTS, axis=2, keepdims=True)
    return rgb / np.maximum(lum, 1.0e-6)


def high_frequency_luma(img: np.ndarray, sigma: float) -> np.ndarray:
    y = luma(img)
    return y - gaussian_filter(y, sigma=float(sigma), mode="reflect")


def mask_stats(mask: np.ndarray) -> dict:
    return {
        "fraction": float(np.mean(mask)),
        "pixels": int(np.sum(mask)),
    }


def compare_one(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray, hf_sigma: float) -> dict:
    ref = _safe_rgb(reference)
    out = _safe_rgb(candidate)
    if ref.shape != out.shape:
        raise ValueError(f"shape mismatch: reference={ref.shape}, candidate={out.shape}")

    ref_luma = luma(ref)
    out_luma = luma(out)
    ref_chroma = chroma_ratio(ref)
    out_chroma = chroma_ratio(out)
    ref_hf = high_frequency_luma(ref, sigma=hf_sigma)
    out_hf = high_frequency_luma(out, sigma=hf_sigma)

    def mean_abs(a: np.ndarray) -> float:
        return float(np.mean(np.abs(a[mask]))) if np.any(mask) else 0.0

    def mean_value(a: np.ndarray) -> float:
        return float(np.mean(a[mask])) if np.any(mask) else 0.0

    ref_hf_abs = mean_value(np.abs(ref_hf))
    out_hf_abs = mean_value(np.abs(out_hf))
    return {
        "rgb_min": float(np.nanmin(out)),
        "rgb_max": float(np.nanmax(out)),
        "luma_mean_in_mask": mean_value(out_luma),
        "reference_luma_mean_in_mask": mean_value(ref_luma),
        "luma_mean_delta_in_mask": mean_value(out_luma - ref_luma),
        "chroma_abs_mean_in_mask": mean_abs(out_chroma - ref_chroma),
        "hf_luma_abs_mean_in_mask": out_hf_abs,
        "reference_hf_luma_abs_mean_in_mask": ref_hf_abs,
        "hf_luma_retention_ratio_in_mask": out_hf_abs / max(ref_hf_abs, 1.0e-8),
        "global_luma_mae": float(np.mean(np.abs(out_luma - ref_luma))),
        "global_chroma_abs_mean": float(np.mean(np.abs(out_chroma - ref_chroma))),
    }


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser()


def make_mask(reference: np.ndarray, mode: str, threshold: float, top_percent: float) -> np.ndarray:
    rgb = np.clip(_safe_rgb(reference), 0.0, None)
    if mode == "peak-threshold":
        return np.max(rgb, axis=2) > float(threshold)
    if mode == "luma-threshold":
        return luma(rgb) > float(threshold)
    if mode == "top-luma":
        q = 1.0 - float(top_percent) / 100.0
        return luma(rgb) > float(np.quantile(luma(rgb), q))
    raise ValueError(f"unknown mask mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NR outputs on a diagnostic crop.")
    parser.add_argument("--reference", required=True, help="Reference/input image path.")
    parser.add_argument("--candidate", action="append", default=[], help="NAME=PATH or PATH. Repeatable.")
    parser.add_argument("--output-dir", default="runs/nagi_v2/compare")
    parser.add_argument("--mask-mode", choices=["peak-threshold", "luma-threshold", "top-luma"], default="top-luma")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--hf-sigma", type=float, default=2.0)
    parser.add_argument("--preview-exposure", type=float, default=1.0)
    parser.add_argument("--preview-tone", choices=["reinhard", "clip"], default="reinhard")
    args = parser.parse_args()

    if not args.candidate:
        raise SystemExit("at least one --candidate is required")

    out_dir = Path(args.output_dir)
    preview_dir = out_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference).expanduser()
    reference = read_image(reference_path)
    mask = make_mask(reference, args.mask_mode, args.threshold, args.top_percent)

    reference_preview = make_preview(reference, exposure=args.preview_exposure, tone=args.preview_tone)
    Image.fromarray(reference_preview).save(preview_dir / "reference_preview.png")

    report = {
        "reference": str(reference_path),
        "reference_stats": image_stats(reference),
        "mask": {
            "mode": args.mask_mode,
            "threshold": args.threshold,
            "top_percent": args.top_percent,
            **mask_stats(mask),
        },
        "hf_sigma": args.hf_sigma,
        "preview": {
            "exposure": args.preview_exposure,
            "tone": args.preview_tone,
        },
        "candidates": [],
    }

    for spec in args.candidate:
        name, path = parse_candidate(spec)
        candidate = read_image(path)
        metrics = compare_one(reference, candidate, mask=mask, hf_sigma=args.hf_sigma)
        preview = make_preview(candidate, exposure=args.preview_exposure, tone=args.preview_tone)
        preview_path = preview_dir / f"{name}_preview.png"
        Image.fromarray(preview).save(preview_path)
        report["candidates"].append(
            {
                "name": name,
                "path": str(path),
                "preview": str(preview_path),
                "metrics": metrics,
            }
        )

    json_path = out_dir / "comparison.json"
    md_path = out_dir / "comparison.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Perfect NR Comparison",
        "",
        f"Reference: `{reference_path}`",
        "",
        f"Mask: `{args.mask_mode}`, fraction={report['mask']['fraction']:.6f}, pixels={report['mask']['pixels']}",
        "",
        "| candidate | chroma drift | luma delta | HF retention | rgb max | preview |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["candidates"]:
        m = item["metrics"]
        lines.append(
            "| {name} | {chroma:.6f} | {luma_delta:.6f} | {hf:.3f} | {rgb_max:.3f} | `{preview}` |".format(
                name=item["name"],
                chroma=m["chroma_abs_mean_in_mask"],
                luma_delta=m["luma_mean_delta_in_mask"],
                hf=m["hf_luma_retention_ratio_in_mask"],
                rgb_max=m["rgb_max"],
                preview=item["preview"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

