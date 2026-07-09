"""Scan detail-protected flat cleanup presets on ROI crops.

The full-frame cleanup is expensive and visual tradeoffs are local. This scans
small diagnostic crops and scores how much extra flat noise can be removed while
preserving edge/detail retention from the learned luma detail gate baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_detail_protected_flat_cleanup import apply_cleanup
from perfect_nr_probe import make_preview, read_image
from real_photo_noise_eval import candidate_metrics, display_rgb, make_masks


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/refiner_pilot_stage11_hybrid_best"
TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")


@dataclass(frozen=True)
class Scene:
    name: str
    reference: Path
    current: Path
    detail_gate: Path
    pl: Path | None = None


SCENES = (
    Scene(
        "occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        RUN_ROOT / "luma_detail_gate_pilot_v2_strict_outputs/xt5_occi_luma_detail_gate_v2_strict.exr",
        RUN_ROOT / "luma_detail_gate_pilot_v2_strict_outputs/xt5_occi_luma_detail_gate_v2_strict_gate.png",
        TEST_PHOTOS / "X-T5 Occi PL deepprimeXD.tif",
    ),
    Scene(
        "dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        RUN_ROOT / "luma_detail_gate_pilot_v2_strict_outputs/k5_dance_luma_detail_gate_v2_strict.exr",
        RUN_ROOT / "luma_detail_gate_pilot_v2_strict_outputs/k5_dance_luma_detail_gate_v2_strict_gate.png",
        TEST_PHOTOS / "K-5 Dance PL deepprimeXD.tif",
    ),
)


ROI_TOP_LEFT: dict[str, list[tuple[str, int, int, str]]] = {
    "occi": [
        ("hair_detail", 2420, 1040, "detail"),
        ("root", 512, 5632, "detail"),
        ("face_center", 2120, 1260, "skin"),
        ("noise_dark", 3072, 3600, "flat"),
    ],
    "dance": [
        ("sky_existing", 4096, 0, "flat"),
        ("sky_center", 2300, 320, "flat"),
        ("dancer_center", 2800, 1200, "detail"),
        ("house_detail", 260, 1180, "detail"),
        ("snow_ground", 2100, 2500, "flat"),
    ],
}


BASE_PARAMS = {
    "luma_strength": 0.84,
    "chroma_strength": 0.92,
    "luma_sigma": 2.20,
    "chroma_sigma": 2.70,
    "flat_threshold": 0.024,
    "flat_transition": 0.010,
    "edge_threshold": 0.024,
    "edge_transition": 0.012,
    "coherent_protect": 1.05,
    "texture_protect": 0.55,
    "detail_gate_protect": 1.35,
    "skin_protect": 0.60,
    "highlight_threshold": 1.0,
    "highlight_transition": 0.25,
    "gate_blur": 0.90,
}


PRESETS: dict[str, dict[str, float]] = {
    "v2_sky": {},
    "v3_more_flat": {
        "luma_strength": 0.92,
        "chroma_strength": 0.96,
        "luma_sigma": 2.60,
        "chroma_sigma": 3.20,
        "flat_threshold": 0.027,
        "edge_threshold": 0.022,
        "texture_protect": 0.45,
        "detail_gate_protect": 1.45,
    },
    "v3_soft_flat": {
        "luma_strength": 0.88,
        "chroma_strength": 0.94,
        "luma_sigma": 2.35,
        "chroma_sigma": 3.00,
        "flat_threshold": 0.026,
        "edge_threshold": 0.023,
        "texture_protect": 0.50,
        "detail_gate_protect": 1.42,
        "gate_blur": 1.15,
    },
    "v3_skin_safe": {
        "luma_strength": 0.90,
        "chroma_strength": 0.95,
        "luma_sigma": 2.55,
        "chroma_sigma": 3.10,
        "flat_threshold": 0.028,
        "edge_threshold": 0.022,
        "coherent_protect": 1.15,
        "texture_protect": 0.50,
        "detail_gate_protect": 1.60,
        "skin_protect": 0.85,
    },
    "v3_aggressive_sky": {
        "luma_strength": 1.00,
        "chroma_strength": 1.00,
        "luma_sigma": 3.00,
        "chroma_sigma": 3.70,
        "flat_threshold": 0.030,
        "edge_threshold": 0.020,
        "coherent_protect": 1.18,
        "texture_protect": 0.38,
        "detail_gate_protect": 1.70,
        "skin_protect": 0.90,
        "gate_blur": 1.20,
    },
    "v3_texture_safe": {
        "luma_strength": 0.92,
        "chroma_strength": 0.96,
        "luma_sigma": 2.70,
        "chroma_sigma": 3.30,
        "flat_threshold": 0.029,
        "edge_threshold": 0.021,
        "coherent_protect": 1.22,
        "texture_protect": 0.68,
        "detail_gate_protect": 1.55,
        "skin_protect": 0.80,
    },
    "v4_pl_flat": {
        "luma_strength": 1.00,
        "chroma_strength": 1.00,
        "luma_sigma": 3.20,
        "chroma_sigma": 4.10,
        "flat_threshold": 0.033,
        "edge_threshold": 0.019,
        "coherent_protect": 1.18,
        "texture_protect": 0.32,
        "detail_gate_protect": 1.90,
        "skin_protect": 1.00,
        "gate_blur": 1.35,
    },
    "v4_pl_soft": {
        "luma_strength": 0.96,
        "chroma_strength": 0.99,
        "luma_sigma": 2.95,
        "chroma_sigma": 3.80,
        "flat_threshold": 0.031,
        "edge_threshold": 0.020,
        "coherent_protect": 1.14,
        "texture_protect": 0.38,
        "detail_gate_protect": 1.75,
        "skin_protect": 0.92,
        "gate_blur": 1.20,
    },
    "v4_flat_open": {
        "luma_strength": 0.96,
        "chroma_strength": 0.98,
        "luma_sigma": 2.85,
        "chroma_sigma": 3.60,
        "flat_threshold": 0.034,
        "edge_threshold": 0.021,
        "coherent_protect": 1.08,
        "texture_protect": 0.30,
        "detail_gate_protect": 1.65,
        "skin_protect": 0.82,
        "gate_blur": 1.00,
    },
}


def crop(arr: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    x = int(np.clip(x, 0, max(0, w - size)))
    y = int(np.clip(y, 0, max(0, h - size)))
    return arr[y : y + size, x : x + size]


def read_gate(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def score_candidate(metrics: dict, baseline: dict, roi_kind: str, pl_metrics: dict | None) -> float:
    luma_gain = baseline["flat_luma_visible_ratio"] - metrics["flat_luma_visible_ratio"]
    chroma_gain = baseline["flat_chroma_visible_ratio"] - metrics["flat_chroma_visible_ratio"]
    mag_gain = baseline["flat_magenta_dot_visible_ratio"] - metrics["flat_magenta_dot_visible_ratio"]
    edge_loss = max(baseline["edge_luma_hf_retention"] - metrics["edge_luma_hf_retention"], 0.0)
    p99_inc = max(metrics["flat_luma_hf_p99_ratio"] - baseline["flat_luma_hf_p99_ratio"], 0.0)
    pl_bonus = 0.0
    if pl_metrics is not None:
        # PL is not a target image teacher here. It only defines a useful flat
        # noise floor; candidates get credit for closing that residual gap.
        for key, weight in (
            ("flat_luma_visible_ratio", 0.40),
            ("flat_chroma_visible_ratio", 0.28),
            ("flat_magenta_dot_visible_ratio", 0.24),
        ):
            base_gap = abs(baseline[key] - pl_metrics[key])
            candidate_gap = abs(metrics[key] - pl_metrics[key])
            pl_bonus += weight * max(base_gap - candidate_gap, 0.0)
    if roi_kind == "flat":
        return float(
            luma_gain * 1.35
            + chroma_gain * 0.85
            + mag_gain * 0.65
            + pl_bonus
            - edge_loss * 1.10
            - p99_inc * 0.35
        )
    if roi_kind == "skin":
        return float(
            luma_gain * 0.75
            + chroma_gain * 0.60
            + mag_gain * 0.45
            + pl_bonus * 0.45
            - edge_loss * 2.20
            - p99_inc * 0.45
        )
    return float(
        luma_gain * 0.45
        + chroma_gain * 0.45
        + mag_gain * 0.35
        + pl_bonus * 0.25
        - edge_loss * 3.20
        - p99_inc * 0.55
    )


def evaluate(args: argparse.Namespace) -> dict:
    report = {"settings": vars(args), "scenes": {}, "combined": {}}
    combined_scores = {name: 0.0 for name in PRESETS}
    combined_counts = {name: 0 for name in PRESETS}
    for scene in SCENES:
        reference = read_image(scene.reference)
        current = read_image(scene.current)
        detail_gate_full = read_gate(scene.detail_gate)
        pl = read_image(scene.pl) if scene.pl is not None and scene.pl.exists() else None
        scene_rows = []
        for roi_name, x, y, roi_kind in ROI_TOP_LEFT[scene.name]:
            ref_c = crop(reference, x, y, args.crop_size)
            cur_c = crop(current, x, y, args.crop_size)
            gate_c = crop(detail_gate_full, x, y, args.crop_size)
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
            baseline = candidate_metrics(ref_c, cur_c, reference_display, baseline_display, masks, hf_sigma=args.hf_sigma)
            pl_metrics = None
            if pl is not None:
                pl_c = crop(pl, x, y, args.crop_size)
                if pl_c.shape[:2] == ref_c.shape[:2]:
                    pl_display = display_rgb(pl_c, exposure=args.exposure, tone=args.tone)
                    pl_metrics = candidate_metrics(ref_c, pl_c, reference_display, pl_display, masks, hf_sigma=args.hf_sigma)
            roi_report = {
                "roi": roi_name,
                "kind": roi_kind,
                "baseline": baseline,
                "pl": pl_metrics,
                "presets": {},
            }
            for preset_name, overrides in PRESETS.items():
                params = dict(BASE_PARAMS)
                params.update(overrides)
                out, stats, _ = apply_cleanup(ref_c, cur_c, gate_c, **params)
                out_display = display_rgb(out, exposure=args.exposure, tone=args.tone)
                metrics = candidate_metrics(ref_c, out, reference_display, out_display, masks, hf_sigma=args.hf_sigma)
                score = score_candidate(metrics, baseline, roi_kind, pl_metrics)
                roi_report["presets"][preset_name] = {"score": score, "metrics": metrics, "filter": stats}
                combined_scores[preset_name] += score
                combined_counts[preset_name] += 1
            scene_rows.append(roi_report)
        report["scenes"][scene.name] = scene_rows
    report["combined"] = {
        name: combined_scores[name] / max(combined_counts[name], 1)
        for name in sorted(combined_scores, key=lambda n: combined_scores[n] / max(combined_counts[n], 1), reverse=True)
    }
    return report


def write_markdown(report: dict, out_dir: Path) -> None:
    lines = ["# Detail-Protected Flat Cleanup Scan", ""]
    lines += ["## Combined", "", "| preset | score |", "| --- | ---: |"]
    for name, score in report["combined"].items():
        lines.append(f"| {name} | {score:.5f} |")
    for scene_name, rois in report["scenes"].items():
        lines += ["", f"## {scene_name}", ""]
        for roi in rois:
            lines += [f"### {roi['roi']} ({roi['kind']})", ""]
            if roi.get("pl") is not None:
                pl = roi["pl"]
                lines += [
                    "PL flat floor:",
                    "",
                    f"- luma visible `{pl['flat_luma_visible_ratio']:.3f}`",
                    f"- chroma visible `{pl['flat_chroma_visible_ratio']:.3f}`",
                    f"- magenta visible `{pl['flat_magenta_dot_visible_ratio']:.3f}`",
                    f"- edge retention `{pl['edge_luma_hf_retention']:.3f}`",
                    "",
                ]
            rows = sorted(roi["presets"].items(), key=lambda item: item[1]["score"], reverse=True)
            lines += [
                "| preset | score | luma visible | chroma visible | magenta visible | edge retention | gate mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for name, data in rows:
                m = data["metrics"]
                f = data["filter"]
                lines.append(
                    "| {name} | {score:.5f} | {lv:.3f} | {cv:.3f} | {mv:.3f} | {edge:.3f} | {gate:.3f} |".format(
                        name=name,
                        score=data["score"],
                        lv=m["flat_luma_visible_ratio"],
                        cv=m["flat_chroma_visible_ratio"],
                        mv=m["flat_magenta_dot_visible_ratio"],
                        edge=m["edge_luma_hf_retention"],
                        gate=f["gate_mean"],
                    )
                )
            lines.append("")
    (out_dir / "scan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan flat cleanup presets on ROI crops.")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "detail_protected_flat_cleanup_scan_v1"))
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
    report = evaluate(args)
    (out_dir / "scan.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, out_dir)
    print(json.dumps(report["combined"], indent=2))
    print(f"wrote {out_dir / 'scan.md'}")


if __name__ == "__main__":
    main()
