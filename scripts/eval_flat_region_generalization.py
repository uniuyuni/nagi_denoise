"""Probe flat/dark-region cleanup eligibility on arbitrary EXR photos.

This does not judge final denoise quality. It only checks whether the
region-aware masks would strongly open the flat cleanup branch on a broader
set of real photos, so we can avoid tuning only to Dance/Ice/Occi.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_region_aware_flat_gate import build_strength_map
from apply_scunet_preset_chooser import REGION_AWARE_FLAT_GATE_PRESETS
from perfect_nr_probe import read_image


def summarize(mask: np.ndarray) -> dict[str, float]:
    x = np.asarray(mask, dtype=np.float32)
    return {
        "mean": float(np.mean(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "gt_025": float(np.mean(x > 0.25)),
        "gt_040": float(np.mean(x > 0.40)),
        "gt_055": float(np.mean(x > 0.55)),
    }


def probe_image(
    path: Path,
    preset: str,
    shadow_threshold: float,
    structure_suppress: float,
    min_candidate_gt040: float,
    max_structure_mean: float,
    max_side: int,
) -> dict[str, object]:
    image = read_image(path)
    original_shape = list(image.shape)
    if max_side > 0:
        h, w = image.shape[:2]
        stride = int(np.ceil(max(h, w) / float(max_side)))
        if stride > 1:
            image = image[::stride, ::stride].copy()
    params = dict(REGION_AWARE_FLAT_GATE_PRESETS[preset])
    params.pop("smooth_params", None)
    for key in list(params):
        if key.startswith("reopen_") or key.startswith("limiter_"):
            params.pop(key)
    _, stats, masks = build_strength_map(image, image, **params)
    shadow_gate = np.clip((masks["shadow_flat"] - shadow_threshold) / 0.12, 0.0, 1.0)
    safe = np.clip(1.0 - masks["structure_protect"] * structure_suppress, 0.0, 1.0)
    candidate = np.clip(masks["flat"] * shadow_gate * safe, 0.0, 1.0)
    candidate_stats = summarize(candidate)
    structure_stats = summarize(masks["structure_protect"])
    guard_pass = (
        candidate_stats["gt_040"] >= float(min_candidate_gt040)
        and structure_stats["mean"] <= float(max_structure_mean)
    )
    return {
        "name": path.stem,
        "path": str(path),
        "original_shape": original_shape,
        "shape": list(image.shape),
        "preset": preset,
        "guard_pass": bool(guard_pass),
        "candidate": candidate_stats,
        "shadow_flat": summarize(masks["shadow_flat"]),
        "structure_protect": structure_stats,
        "flat": summarize(masks["flat"]),
        "highlight": summarize(masks["highlight"]),
        "build_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe flat-cleanup region generalization.")
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--preset", default="dark_sky_strict")
    parser.add_argument("--shadow-threshold", type=float, default=0.40)
    parser.add_argument("--structure-suppress", type=float, default=1.0)
    parser.add_argument("--min-candidate-gt040", type=float, default=0.05)
    parser.add_argument("--max-structure-mean", type=float, default=0.62)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = [
        probe_image(
            Path(item).expanduser(),
            args.preset,
            args.shadow_threshold,
            args.structure_suppress,
            args.min_candidate_gt040,
            args.max_structure_mean,
            args.max_side,
        )
        for item in args.image
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in report:
        c = item["candidate"]
        s = item["structure_protect"]
        print(
            f"{item['name']} guard={item['guard_pass']} "
            f"candidate_mean={c['mean']:.5f} gt_040={c['gt_040']:.5f} "
            f"structure_mean={s['mean']:.5f} structure_p95={s['p95']:.5f}"
        )


if __name__ == "__main__":
    main()
