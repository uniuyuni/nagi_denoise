"""Scan luma-detail discriminator parameters without writing candidate EXRs."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_luma_detail_discriminator import apply_luma_delta, build_gate
from perfect_nr_probe import read_image
from real_photo_noise_eval import candidate_metrics, display_rgb, make_masks


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/refiner_pilot_stage11_hybrid_best"
TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")


@dataclass(frozen=True)
class Scene:
    name: str
    reference: Path
    current: Path
    base: Path
    rebuild: Path
    result: Path


SCENES = (
    Scene(
        "occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        RUN_ROOT / "blend_selector_pilot_v2_outputs/xt5_occi_blend_selector_v2.exr",
        RUN_ROOT / "final_v4_red115_blue120_detailguard_mild/xt5_occi_final_v4_red115_blue120_detailguard_mild.exr",
        RUN_ROOT / "final_v7_luma_rebuild/xt5_occi_luma_rebuild.exr",
        RUN_ROOT / "luma_rebuilder_pilot_v3_outputs/xt5_occi_luma_rebuilder_v3_s1.exr",
    ),
    Scene(
        "dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        RUN_ROOT / "blend_selector_pilot_v2_outputs/k5_dance_blend_selector_v2.exr",
        RUN_ROOT / "final_v4_red115_blue120_detailguard_mild/k5_dance_final_v4_red115_blue120_detailguard_mild.exr",
        RUN_ROOT / "final_v7_luma_rebuild/k5_dance_luma_rebuild.exr",
        RUN_ROOT / "luma_rebuilder_pilot_v3_outputs/k5_dance_luma_rebuilder_v3_s1.exr",
    ),
)

ROI_TOP_LEFT: dict[str, list[tuple[str, int, int]]] = {
    "occi": [
        ("face_center", 2120, 1260),
        ("hair_detail", 2420, 1040),
        ("cheek_hair", 2280, 1420),
        ("root", 512, 5632),
        ("noise_dark", 3072, 3600),
    ],
    "dance": [
        ("sky_existing", 4096, 0),
        ("sky_center", 2300, 320),
        ("dancer_center", 2800, 1200),
        ("right_dancer", 3820, 1200),
        ("snow_ground", 2100, 2500),
        ("house_detail", 260, 1180),
    ],
}


PRESETS: dict[str, dict[str, float]] = {
    "v1": {
        "floor": 0.18,
        "structure_gain": 1.18,
        "coherent_weight": 1.00,
        "base_texture_weight": 0.95,
        "rebuild_texture_weight": 0.75,
        "dark_flat_suppress": 0.92,
        "skin_suppress": 0.32,
        "grain_suppress": 0.82,
        "gate_blur": 0.65,
    },
    "detail_push": {
        "floor": 0.24,
        "structure_gain": 1.34,
        "coherent_weight": 1.08,
        "base_texture_weight": 1.05,
        "rebuild_texture_weight": 0.84,
        "dark_flat_suppress": 0.88,
        "skin_suppress": 0.30,
        "grain_suppress": 0.78,
        "gate_blur": 0.60,
    },
    "balanced": {
        "floor": 0.20,
        "structure_gain": 1.28,
        "coherent_weight": 1.05,
        "base_texture_weight": 1.03,
        "rebuild_texture_weight": 0.78,
        "dark_flat_suppress": 1.00,
        "skin_suppress": 0.34,
        "grain_suppress": 0.92,
        "gate_blur": 0.70,
    },
    "strict_noise": {
        "floor": 0.12,
        "structure_gain": 1.22,
        "coherent_weight": 1.12,
        "base_texture_weight": 1.05,
        "rebuild_texture_weight": 0.62,
        "dark_flat_suppress": 1.15,
        "skin_suppress": 0.38,
        "grain_suppress": 1.18,
        "gate_blur": 0.75,
    },
    "base_texture": {
        "floor": 0.18,
        "structure_gain": 1.30,
        "coherent_weight": 1.02,
        "base_texture_weight": 1.30,
        "rebuild_texture_weight": 0.50,
        "dark_flat_suppress": 0.96,
        "skin_suppress": 0.34,
        "grain_suppress": 0.95,
        "gate_blur": 0.70,
    },
    "coherent_only": {
        "floor": 0.10,
        "structure_gain": 1.55,
        "coherent_weight": 1.55,
        "base_texture_weight": 0.45,
        "rebuild_texture_weight": 0.30,
        "dark_flat_suppress": 1.05,
        "skin_suppress": 0.36,
        "grain_suppress": 1.05,
        "gate_blur": 0.80,
    },
    "soft_open": {
        "floor": 0.32,
        "structure_gain": 1.08,
        "coherent_weight": 0.94,
        "base_texture_weight": 0.88,
        "rebuild_texture_weight": 0.75,
        "dark_flat_suppress": 0.72,
        "skin_suppress": 0.24,
        "grain_suppress": 0.62,
        "gate_blur": 0.60,
    },
}


def score_metrics(metrics: dict, baseline: dict) -> float:
    detail_gain = metrics["edge_luma_hf_retention"] - baseline["edge_luma_hf_retention"]
    visible_inc = metrics["flat_luma_visible_ratio"] - baseline["flat_luma_visible_ratio"]
    p99_inc = metrics["flat_luma_hf_p99_ratio"] - baseline["flat_luma_hf_p99_ratio"]
    mag_inc = metrics["flat_magenta_dot_visible_ratio"] - baseline["flat_magenta_dot_visible_ratio"]
    return float(detail_gain - 1.05 * max(visible_inc, 0.0) - 0.35 * max(p99_inc, 0.0) - 0.50 * max(mag_inc, 0.0))


def crop(arr: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    x = int(np.clip(x, 0, max(0, w - size)))
    y = int(np.clip(y, 0, max(0, h - size)))
    return arr[y : y + size, x : x + size]


def average_metrics(items: list[dict]) -> dict:
    keys = items[0].keys()
    out = {}
    for key in keys:
        values = [item[key] for item in items if isinstance(item.get(key), (int, float))]
        if values:
            out[key] = float(np.mean(values))
    return out


def evaluate_scene(scene: Scene, presets: dict[str, dict[str, float]], args: argparse.Namespace) -> dict:
    reference = read_image(scene.reference)
    current = read_image(scene.current)
    base = read_image(scene.base)
    rebuild = read_image(scene.rebuild)
    result = read_image(scene.result)
    baseline_parts: list[dict] = []
    preset_parts: dict[str, list[dict]] = {name: [] for name in presets}
    gate_parts: dict[str, list[dict]] = {name: [] for name in presets}
    rois = ROI_TOP_LEFT[scene.name]
    size = int(args.crop_size)
    for _, x, y in rois:
        ref_c = crop(reference, x, y, size)
        cur_c = crop(current, x, y, size)
        base_c = crop(base, x, y, size)
        rebuild_c = crop(rebuild, x, y, size)
        result_c = crop(result, x, y, size)
        reference_display = display_rgb(ref_c, exposure=args.exposure, tone=args.tone)
        masks = make_masks(
            ref_c,
            reference_display,
            hf_sigma=args.hf_sigma,
            edge_sigma=args.edge_sigma,
            flat_hf_threshold=args.flat_hf_threshold,
            flat_edge_threshold=args.flat_edge_threshold,
            min_display_luma=args.min_display_luma,
            max_display_luma=args.max_display_luma,
            shadow_display_luma=args.shadow_display_luma,
            highlight_linear_luma=args.highlight_linear_luma,
            edge_threshold=args.edge_threshold,
            top_luma_percent=args.top_luma_percent,
        )
        baseline_display = display_rgb(cur_c, exposure=args.exposure, tone=args.tone)
        baseline_parts.append(
            candidate_metrics(
                ref_c,
                cur_c,
                reference_display,
                baseline_display,
                masks,
                hf_sigma=args.hf_sigma,
            )
        )
        current_y = np.sum(baseline_display * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=2)
        result_display = display_rgb(result_c, exposure=args.exposure, tone=args.tone)
        result_y = np.sum(result_display * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=2)
        raw_delta = result_y - current_y
        for name, params in presets.items():
            gate, gate_stats, _ = build_gate(
                ref_c,
                cur_c,
                base_c,
                rebuild_c,
                structure_gain=params["structure_gain"],
                coherent_weight=params["coherent_weight"],
                base_texture_weight=params["base_texture_weight"],
                rebuild_texture_weight=params["rebuild_texture_weight"],
                dark_flat_suppress=params["dark_flat_suppress"],
                skin_suppress=params["skin_suppress"],
                grain_suppress=params["grain_suppress"],
                gate_blur=params["gate_blur"],
            )
            gated_delta = raw_delta * (params["floor"] + (1.0 - params["floor"]) * gate)
            candidate = apply_luma_delta(cur_c, gated_delta)
            candidate_display = display_rgb(candidate, exposure=args.exposure, tone=args.tone)
            preset_parts[name].append(
                candidate_metrics(
                    ref_c,
                    candidate,
                    reference_display,
                    candidate_display,
                    masks,
                    hf_sigma=args.hf_sigma,
                )
            )
            gate_parts[name].append(
                {
                    "mean": gate_stats["gate_mean"],
                    "p95": gate_stats["gate_p95"],
                    "gated_delta_abs_mean": float(np.mean(np.abs(gated_delta))),
                    "gated_delta_abs_p95": float(np.quantile(np.abs(gated_delta), 0.95)),
                }
            )

    baseline_metrics = average_metrics(baseline_parts)
    rows = []
    for name, params in presets.items():
        metrics = average_metrics(preset_parts[name])
        gate_avg = average_metrics(gate_parts[name])
        rows.append(
            {
                "name": name,
                "params": params,
                "score": score_metrics(metrics, baseline_metrics),
                "metrics": metrics,
                "gate": gate_avg,
            }
        )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return {"scene": scene.name, "baseline": baseline_metrics, "rows": rows}


def markdown_report(results: list[dict]) -> str:
    lines = ["# Luma Detail Discriminator Scan", ""]
    combined: dict[str, float] = {}
    for result in results:
        lines += [
            f"## {result['scene']}",
            "",
            "| preset | score | edge retention | luma visible | luma p99 | mag visible | gate mean | gate p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in result["rows"]:
            m = row["metrics"]
            g = row["gate"]
            combined[row["name"]] = combined.get(row["name"], 0.0) + float(row["score"])
            lines.append(
                f"| {row['name']} | {row['score']:.4f} | {m['edge_luma_hf_retention']:.3f} | "
                f"{m['flat_luma_visible_ratio']:.3f} | {m['flat_luma_hf_p99_ratio']:.3f} | "
                f"{m['flat_magenta_dot_visible_ratio']:.3f} | {g['mean']:.3f} | {g['p95']:.3f} |"
            )
        lines.append("")
    lines += ["## Combined", "", "| preset | combined score |", "| --- | ---: |"]
    for name, value in sorted(combined.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {name} | {value:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan luma detail discriminator presets.")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "luma_detail_discriminator_scan_v1"))
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--tone", choices=["reinhard", "clip"], default="reinhard")
    parser.add_argument("--hf-sigma", type=float, default=1.2)
    parser.add_argument("--edge-sigma", type=float, default=1.0)
    parser.add_argument("--flat-hf-threshold", type=float, default=0.018)
    parser.add_argument("--flat-edge-threshold", type=float, default=0.030)
    parser.add_argument("--edge-threshold", type=float, default=0.050)
    parser.add_argument("--min-display-luma", type=float, default=0.035)
    parser.add_argument("--max-display-luma", type=float, default=0.92)
    parser.add_argument("--shadow-display-luma", type=float, default=0.22)
    parser.add_argument("--highlight-linear-luma", type=float, default=1.0)
    parser.add_argument("--top-luma-percent", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_scene(scene, PRESETS, args) for scene in SCENES]
    payload = {"presets": PRESETS, "results": results}
    (out_dir / "scan.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "scan.md").write_text(markdown_report(results), encoding="utf-8")
    print(markdown_report(results))


if __name__ == "__main__":
    main()
