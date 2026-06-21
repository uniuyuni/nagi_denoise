"""Break residual real-photo noise into luma and opponent-color directions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perfect_nr_probe import read_image
from real_photo_noise_eval import LUMA_SRGB, display_rgb, luma, make_masks, parse_candidate, safe_ratio


def highpass(x: np.ndarray, sigma: float) -> np.ndarray:
    return x - gaussian_filter(x, sigma=float(sigma), mode="reflect")


def axis_energy(display: np.ndarray, sigma: float) -> dict[str, np.ndarray]:
    rgb = np.nan_to_num(display[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = luma(rgb, LUMA_SRGB)
    magenta = highpass(0.5 * (r + b) - g, sigma)
    red = highpass(r - 0.5 * (g + b), sigma)
    blue = highpass(b - 0.5 * (r + g), sigma)
    return {
        "luma_abs": np.abs(highpass(y, sigma)),
        "magenta_pos": np.maximum(magenta, 0.0),
        "green_pos": np.maximum(-magenta, 0.0),
        "red_pos": np.maximum(red, 0.0),
        "cyan_pos": np.maximum(-red, 0.0),
        "blue_pos": np.maximum(blue, 0.0),
        "yellow_pos": np.maximum(-blue, 0.0),
        "rg_abs": np.abs(highpass(r - g, sigma)),
        "by_abs": np.abs(highpass(b - 0.5 * (r + g), sigma)),
    }


def masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0}
    v = values[mask]
    return {
        "mean": float(np.mean(v)),
        "p95": float(np.quantile(v, 0.95)),
        "p99": float(np.quantile(v, 0.99)),
    }


def candidate_axis_metrics(reference_axes: dict[str, np.ndarray], candidate_axes: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict:
    result: dict[str, dict] = {}
    for region in ("flat", "shadow_flat"):
        mask = masks[region]
        region_metrics = {}
        for key, candidate_values in candidate_axes.items():
            ref = masked_stats(reference_axes[key], mask)
            cand = masked_stats(candidate_values, mask)
            region_metrics[key] = {
                **cand,
                "mean_ratio": safe_ratio(cand["mean"], ref["mean"]),
                "p95_ratio": safe_ratio(cand["p95"], ref["p95"]),
                "p99_ratio": safe_ratio(cand["p99"], ref["p99"]),
            }
        chroma_keys = ("magenta_pos", "green_pos", "red_pos", "cyan_pos", "blue_pos", "yellow_pos")
        chroma_mean_sum = sum(region_metrics[key]["mean"] for key in chroma_keys)
        region_metrics["chroma_share_mean"] = {
            key: safe_ratio(region_metrics[key]["mean"], chroma_mean_sum) for key in chroma_keys
        }
        result[region] = region_metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real-photo residual noise by color direction.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", default=[], help="NAME=PATH or PATH. Repeatable.")
    parser.add_argument("--output-dir", default="runs/perfect_nr/axis_noise_eval")
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--tone", choices=["reinhard", "clip"], default="reinhard")
    parser.add_argument("--hf-sigma", type=float, default=1.2)
    parser.add_argument("--flat-hf-threshold", type=float, default=0.018)
    parser.add_argument("--flat-edge-threshold", type=float, default=0.030)
    parser.add_argument("--edge-threshold", type=float, default=0.050)
    parser.add_argument("--min-display-luma", type=float, default=0.035)
    parser.add_argument("--max-display-luma", type=float, default=0.92)
    parser.add_argument("--shadow-display-luma", type=float, default=0.22)
    parser.add_argument("--highlight-linear-luma", type=float, default=1.0)
    parser.add_argument("--top-luma-percent", type=float, default=1.0)
    args = parser.parse_args()

    if not args.candidate:
        raise SystemExit("at least one --candidate is required")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_path = Path(args.reference).expanduser()
    reference = read_image(reference_path)
    reference_display = display_rgb(reference, exposure=args.exposure, tone=args.tone)
    masks = make_masks(
        reference,
        reference_display,
        hf_sigma=args.hf_sigma,
        edge_sigma=1.0,
        flat_hf_threshold=args.flat_hf_threshold,
        flat_edge_threshold=args.flat_edge_threshold,
        min_display_luma=args.min_display_luma,
        max_display_luma=args.max_display_luma,
        shadow_display_luma=args.shadow_display_luma,
        highlight_linear_luma=args.highlight_linear_luma,
        edge_threshold=args.edge_threshold,
        top_luma_percent=args.top_luma_percent,
    )
    reference_axes = axis_energy(reference_display, args.hf_sigma)
    report = {
        "reference": str(reference_path),
        "masks": {name: {"fraction": float(np.mean(mask)), "pixels": int(np.sum(mask))} for name, mask in masks.items()},
        "candidates": [],
    }

    for spec in args.candidate:
        name, path = parse_candidate(spec)
        candidate = read_image(path)
        if candidate.shape[:2] != reference.shape[:2]:
            raise ValueError(f"shape mismatch for {name}: reference={reference.shape}, candidate={candidate.shape}")
        candidate_display = display_rgb(candidate, exposure=args.exposure, tone=args.tone)
        metrics = candidate_axis_metrics(reference_axes, axis_energy(candidate_display, args.hf_sigma), masks)
        report["candidates"].append({"name": name, "path": str(path), "metrics": metrics})

    json_path = out_dir / "axis_noise_eval.json"
    md_path = out_dir / "axis_noise_eval.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    axes = ("luma_abs", "magenta_pos", "green_pos", "red_pos", "cyan_pos", "blue_pos", "yellow_pos", "rg_abs", "by_abs")
    lines = ["# Axis Noise Evaluation", "", f"Reference: `{reference_path}`", ""]
    for region in ("flat", "shadow_flat"):
        lines += [f"## {region}", "", "| candidate | axis | mean ratio | p95 ratio | p99 ratio | share mean |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for item in report["candidates"]:
            metrics = item["metrics"][region]
            shares = metrics["chroma_share_mean"]
            for axis in axes:
                m = metrics[axis]
                share = shares.get(axis)
                share_text = "" if share is None else f"{share:.3f}"
                lines.append(
                    f"| {item['name']} | {axis} | {m['mean_ratio']:.3f} | {m['p95_ratio']:.3f} | {m['p99_ratio']:.3f} | {share_text} |"
                )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
