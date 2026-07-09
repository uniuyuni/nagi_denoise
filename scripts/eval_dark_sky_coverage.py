"""Evaluate global dark-sky coverage for guarded reopen decisions."""
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

ROOT = Path("runs/refiner_pilot_stage11_hybrid_best")
DATA = Path("/Users/uniuyuni/ProjectData/test_photos")
CURRENT_ROOT = ROOT / "scunet_preset_chooser_v12_flat_cleanup_auto_outputs"
SCENES = {
    "k5_dance": (
        DATA / "K-5 Dance noisy.EXR",
        CURRENT_ROOT / "k5_dance_scunet_preset_chooser_v12_auto.exr",
    ),
    "xt5_occi": (
        DATA / "X-T5 Occi noisy.EXR",
        CURRENT_ROOT / "xt5_occi_scunet_preset_chooser_v12_auto.exr",
    ),
    "k5_ice": (
        DATA / "K-5 Ice noisy.EXR",
        CURRENT_ROOT / "k5_ice_scunet_preset_chooser_v12_auto.exr",
    ),
}


def summarize_mask(mask: np.ndarray) -> dict[str, float]:
    m = np.asarray(mask, dtype=np.float32)
    return {
        "mean": float(np.mean(m)),
        "p50": float(np.quantile(m, 0.50)),
        "p90": float(np.quantile(m, 0.90)),
        "p95": float(np.quantile(m, 0.95)),
        "p99": float(np.quantile(m, 0.99)),
        "gt_025": float(np.mean(m > 0.25)),
        "gt_040": float(np.mean(m > 0.40)),
        "gt_055": float(np.mean(m > 0.55)),
    }


def parse_pair(value: str) -> tuple[str, Path, Path]:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise ValueError(f"pair must be name,reference,current: {value!r}")
    name, reference, current = parts
    return name.strip(), Path(reference).expanduser(), Path(current).expanduser()


def downsample_stride(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image
    h, w = image.shape[:2]
    stride = int(np.ceil(max(h, w) / float(max_side)))
    if stride <= 1:
        return image
    return image[::stride, ::stride].copy()


def evaluate_paths(
    name: str,
    ref_path: Path,
    cur_path: Path,
    preset: str,
    shadow_threshold: float,
    structure_suppress: float,
    min_candidate_gt040: float,
    max_structure_mean: float,
    max_side: int,
) -> dict[str, object]:
    reference_full = read_image(ref_path)
    current_full = read_image(cur_path)
    reference = downsample_stride(reference_full, max_side)
    current = downsample_stride(current_full, max_side)
    params = dict(REGION_AWARE_FLAT_GATE_PRESETS[preset])
    params.pop("smooth_params", None)
    for key in list(params):
        if key.startswith("reopen_") or key.startswith("limiter_"):
            params.pop(key)
    _, stats, masks = build_strength_map(reference, current, **params)
    shadow_gate = np.clip((masks["shadow_flat"] - shadow_threshold) / 0.12, 0.0, 1.0)
    safe = np.clip(1.0 - masks["structure_protect"] * structure_suppress, 0.0, 1.0)
    candidate = np.clip(masks["flat"] * shadow_gate * safe, 0.0, 1.0)
    strong = candidate > 0.12
    candidate_stats = summarize_mask(candidate)
    structure_stats = summarize_mask(masks["structure_protect"])
    guard_pass = candidate_stats["gt_040"] >= float(min_candidate_gt040) and structure_stats["mean"] <= float(max_structure_mean)
    return {
        "scene": name,
        "preset": preset,
        "paths": {"reference": str(ref_path), "current": str(cur_path)},
        "original_shape": list(reference_full.shape),
        "shape": list(reference.shape),
        "params": {
            "shadow_threshold": shadow_threshold,
            "structure_suppress": structure_suppress,
            "min_candidate_gt040": min_candidate_gt040,
            "max_structure_mean": max_structure_mean,
        },
        "guard_pass": bool(guard_pass),
        "candidate": candidate_stats,
        "shadow_flat": summarize_mask(masks["shadow_flat"]),
        "structure_protect": structure_stats,
        "flat": summarize_mask(masks["flat"]),
        "strong_candidate_fraction": float(np.mean(strong)),
        "build_stats": stats,
    }


def evaluate(
    scene: str,
    preset: str,
    shadow_threshold: float,
    structure_suppress: float,
    min_candidate_gt040: float,
    max_structure_mean: float,
    max_side: int,
) -> dict[str, object]:
    ref_path, cur_path = SCENES[scene]
    return evaluate_paths(
        scene,
        ref_path,
        cur_path,
        preset,
        shadow_threshold,
        structure_suppress,
        min_candidate_gt040,
        max_structure_mean,
        max_side,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dark-sky coverage masks.")
    parser.add_argument("--scene", action="append", choices=sorted(SCENES), default=[])
    parser.add_argument("--pair", action="append", default=[], help="name,reference,current. Repeatable.")
    parser.add_argument("--preset", default="dark_sky_strict")
    parser.add_argument("--shadow-threshold", type=float, default=0.40)
    parser.add_argument("--structure-suppress", type=float, default=1.0)
    parser.add_argument("--min-candidate-gt040", type=float, default=0.05)
    parser.add_argument("--max-structure-mean", type=float, default=0.62)
    parser.add_argument("--max-side", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.scene and not args.pair:
        raise SystemExit("at least one --scene or --pair is required")
    report = [
        evaluate(
            scene,
            args.preset,
            args.shadow_threshold,
            args.structure_suppress,
            args.min_candidate_gt040,
            args.max_structure_mean,
            args.max_side,
        )
        for scene in args.scene
    ]
    report.extend(
        evaluate_paths(
            name,
            ref_path,
            cur_path,
            args.preset,
            args.shadow_threshold,
            args.structure_suppress,
            args.min_candidate_gt040,
            args.max_structure_mean,
            args.max_side,
        )
        for name, ref_path, cur_path in (parse_pair(item) for item in args.pair)
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in report:
        c = item["candidate"]
        print(
            f"{item['scene']} guard={item['guard_pass']} candidate_mean={c['mean']:.5f} "
            f"gt_025={c['gt_025']:.5f} gt_040={c['gt_040']:.5f} "
            f"structure_mean={item['structure_protect']['mean']:.5f} strong={item['strong_candidate_fraction']:.5f}"
        )


if __name__ == "__main__":
    main()
