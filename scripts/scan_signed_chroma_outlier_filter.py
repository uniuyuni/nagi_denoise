"""Scan signed chroma outlier filter settings against real-photo crops.

The direct PL-distillation pilots were unstable because they had to predict a
color correction surface. This scan keeps the stable hand-built correction
target from ``apply_signed_chroma_outlier_filter`` and only searches for safer
axis/gate settings, using PL frequency-split pseudo teachers as a soft target.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from apply_signed_chroma_outlier_filter import apply_signed_chroma_outlier_filter
from apply_region_aware_luma_cleanup import make_coherent_structure_mask
from build_frequency_split_pseudo_teacher import RUN_ROOT, SCENES
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import make_preview, read_image
from real_photo_noise_eval import (
    LUMA_SRGB,
    candidate_metrics,
    display_rgb,
    luma,
    make_masks,
)


TEACHER_DIR = RUN_ROOT / "frequency_split_pseudo_teacher_v3_chroma_safe"


@dataclass(frozen=True)
class Candidate:
    name: str
    strength: float
    median_size: int
    low_sigma: float
    outlier_threshold: float
    outlier_transition: float
    magenta_weight: float
    red_weight: float
    blue_weight: float
    detail_threshold: float
    edge_threshold: float
    shadow_threshold: float
    line_restore_strength: float
    line_restore_threshold: float
    coherent_restore_strength: float
    coherent_energy_threshold: float
    coherent_inhibit_strength: float
    coherent_inhibit_energy_threshold: float
    density_inhibit_strength: float
    density_threshold: float


def teacher_path(scene_name: str) -> Path:
    return TEACHER_DIR / f"{scene_name}_freqsplit_teacher_v3_chroma_safe.exr"


def clamp_crop(x: int, y: int, size: int, width: int, height: int) -> tuple[slice, slice]:
    half = size // 2
    x0 = max(0, min(width - size, int(x) - half))
    y0 = max(0, min(height - size, int(y) - half))
    return slice(y0, y0 + size), slice(x0, x0 + size)


def crop_image(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    ys, xs = clamp_crop(x, y, size, width, height)
    return image[ys, xs].copy()


def axis_hf_chroma_mae(a_linear: np.ndarray, b_linear: np.ndarray) -> float:
    a = display_rgb(a_linear, exposure=1.0, tone="reinhard")
    b = display_rgb(b_linear, exposure=1.0, tone="reinhard")
    ay = luma(a, LUMA_SRGB)
    by = luma(b, LUMA_SRGB)
    ach = a - ay[..., None]
    bch = b - by[..., None]
    return float(np.mean(np.abs(ach - bch)))


def display_luma_mae(a_linear: np.ndarray, b_linear: np.ndarray) -> float:
    a = display_rgb(a_linear, exposure=1.0, tone="reinhard")
    b = display_rgb(b_linear, exposure=1.0, tone="reinhard")
    return float(np.mean(np.abs(luma(a, LUMA_SRGB) - luma(b, LUMA_SRGB))))


def sigmoid01(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32, copy=False)


def restore_fine_lines(
    base_linear: np.ndarray,
    out_linear: np.ndarray,
    *,
    strength: float,
    threshold: float,
    transition: float = 0.010,
    sigma: float = 0.65,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    if strength <= 0.0:
        gate = np.zeros(base_linear.shape[:2], dtype=np.float32)
        return out_linear, {"line_restore_mean": 0.0, "line_restore_p99": 0.0}, gate
    base_display = display_rgb(base_linear, exposure=1.0, tone="reinhard")
    y = luma(base_display, LUMA_SRGB)
    fine = np.abs(y - gaussian_filter(y, sigma=float(sigma), mode="reflect"))
    gate = sigmoid01((fine - float(threshold)) / max(float(transition), 1.0e-6))
    gate = np.clip(gate * float(strength), 0.0, 1.0).astype(np.float32, copy=False)
    out = out_linear * (1.0 - gate[..., None]) + base_linear * gate[..., None]
    stats = {
        "line_restore_strength": float(strength),
        "line_restore_threshold": float(threshold),
        "line_restore_mean": float(np.mean(gate)),
        "line_restore_p99": float(np.quantile(gate, 0.99)),
    }
    return out.astype(np.float32, copy=False), stats, gate


def restore_coherent_structure(
    base_linear: np.ndarray,
    out_linear: np.ndarray,
    *,
    strength: float,
    energy_threshold: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    if strength <= 0.0:
        gate = np.zeros(base_linear.shape[:2], dtype=np.float32)
        return out_linear, {"coherent_restore_mean": 0.0, "coherent_restore_p99": 0.0}, gate
    base_display = display_rgb(base_linear, exposure=1.0, tone="reinhard")
    coherent = make_coherent_structure_mask(
        base_display,
        coherence_threshold=0.42,
        coherence_transition=0.16,
        energy_threshold=float(energy_threshold),
        energy_transition=max(float(energy_threshold) * 0.85, 0.0025),
    )
    gate = np.clip(coherent * float(strength), 0.0, 1.0).astype(np.float32, copy=False)
    out = out_linear * (1.0 - gate[..., None]) + base_linear * gate[..., None]
    stats = {
        "coherent_restore_strength": float(strength),
        "coherent_energy_threshold": float(energy_threshold),
        "coherent_restore_mean": float(np.mean(gate)),
        "coherent_restore_p99": float(np.quantile(gate, 0.99)),
    }
    return out.astype(np.float32, copy=False), stats, gate


def prepare_crop_metrics(noisy: np.ndarray, base: np.ndarray, teacher: np.ndarray) -> dict[str, object]:
    ref_display = display_rgb(noisy, exposure=1.0, tone="reinhard")
    masks = make_masks(
        noisy,
        ref_display,
        hf_sigma=1.2,
        edge_sigma=1.0,
        flat_hf_threshold=0.018,
        flat_edge_threshold=0.030,
        min_display_luma=0.035,
        max_display_luma=0.92,
        shadow_display_luma=0.22,
        highlight_linear_luma=1.0,
        edge_threshold=0.050,
        top_luma_percent=1.0,
    )
    base_display = display_rgb(base, exposure=1.0, tone="reinhard")
    teacher_display = display_rgb(teacher, exposure=1.0, tone="reinhard")
    return {
        "ref_display": ref_display,
        "masks": masks,
        "base_display": base_display,
        "teacher_display": teacher_display,
        "base_metrics": candidate_metrics(noisy, base, ref_display, base_display, masks, hf_sigma=1.2),
        "teacher_metrics": candidate_metrics(noisy, teacher, ref_display, teacher_display, masks, hf_sigma=1.2),
    }


def eval_crop(
    noisy: np.ndarray,
    cand: np.ndarray,
    prepared: dict[str, object],
) -> dict[str, dict[str, float]]:
    cand_metrics = candidate_metrics(
        noisy,
        cand,
        prepared["ref_display"],
        display_rgb(cand, exposure=1.0, tone="reinhard"),
        prepared["masks"],
        hf_sigma=1.2,
    )
    return {
        "base": prepared["base_metrics"],
        "teacher": prepared["teacher_metrics"],
        "candidate": cand_metrics,
    }


def score_metrics(metrics: list[dict[str, dict[str, float]]], teacher_dist: list[float], luma_dist: list[float]) -> dict[str, float]:
    rel_chroma = []
    rel_magenta = []
    rel_shadow = []
    rel_luma = []
    edge_loss = []
    worse_count = 0
    for item in metrics:
        base = item["base"]
        cand = item["candidate"]
        teacher = item["teacher"]
        rel_chroma.append(cand["flat_chroma_visible_ratio"] / max(base["flat_chroma_visible_ratio"], 1.0e-6))
        rel_magenta.append(cand["flat_magenta_dot_visible_ratio"] / max(base["flat_magenta_dot_visible_ratio"], 1.0e-6))
        rel_shadow.append(cand["shadow_magenta_dot_visible_ratio"] / max(base["shadow_magenta_dot_visible_ratio"], 1.0e-6))
        rel_luma.append(cand["flat_luma_visible_ratio"] / max(base["flat_luma_visible_ratio"], 1.0e-6))
        edge_loss.append(max(0.0, base["edge_luma_hf_retention"] - cand["edge_luma_hf_retention"]))
        if cand["flat_chroma_visible_ratio"] > base["flat_chroma_visible_ratio"] * 1.015:
            worse_count += 1
        if cand["flat_magenta_dot_visible_ratio"] > base["flat_magenta_dot_visible_ratio"] * 1.015:
            worse_count += 1
        if cand["flat_luma_visible_ratio"] > base["flat_luma_visible_ratio"] * 1.020:
            worse_count += 1
        if cand["edge_luma_hf_retention"] < base["edge_luma_hf_retention"] * 0.990:
            worse_count += 1
        if cand["flat_chroma_visible_ratio"] > teacher["flat_chroma_visible_ratio"] * 1.20:
            worse_count += 1

    avg_rel_chroma = float(np.mean(rel_chroma))
    avg_rel_magenta = float(np.mean(rel_magenta))
    avg_rel_shadow = float(np.mean(rel_shadow))
    avg_rel_luma = float(np.mean(rel_luma))
    avg_edge_loss = float(np.mean(edge_loss))
    avg_teacher_chroma_mae = float(np.mean(teacher_dist))
    avg_teacher_luma_mae = float(np.mean(luma_dist))
    score = (
        1.35 * avg_rel_chroma
        + 1.15 * avg_rel_magenta
        + 0.75 * avg_rel_shadow
        + 0.45 * max(0.0, avg_rel_luma - 1.0)
        + 12.0 * avg_edge_loss
        + 18.0 * avg_teacher_chroma_mae
        + 4.0 * avg_teacher_luma_mae
        + 0.22 * worse_count
    )
    return {
        "score": float(score),
        "avg_rel_chroma": avg_rel_chroma,
        "avg_rel_magenta": avg_rel_magenta,
        "avg_rel_shadow_magenta": avg_rel_shadow,
        "avg_rel_luma": avg_rel_luma,
        "avg_edge_loss": avg_edge_loss,
        "avg_teacher_chroma_mae": avg_teacher_chroma_mae,
        "avg_teacher_luma_mae": avg_teacher_luma_mae,
        "worse_count": int(worse_count),
    }


def make_candidates(*, wide: bool) -> list[Candidate]:
    candidates: list[Candidate] = []
    idx = 0
    strengths = (0.70, 0.82, 0.92) if wide else (0.82, 0.88)
    thresholds = (0.0024, 0.0032, 0.0042) if wide else (0.0028, 0.0036)
    axis_weights = ((0.25, 0.85), (0.45, 1.00), (0.65, 1.15))
    detail_edges = ((0.018, 0.028), (0.022, 0.034)) if wide else ((0.020, 0.030),)
    line_restores = ((0.0, 0.030), (0.35, 0.022), (0.55, 0.026)) if wide else ((0.0, 0.030),)
    coherent_restores = ((0.0, 0.0060),) if not wide else ((0.0, 0.0060), (0.35, 0.0040), (0.55, 0.0060))
    coherent_inhibits = ((0.0, 0.0060),) if not wide else ((0.0, 0.0060), (0.25, 0.0035), (0.45, 0.0045), (0.65, 0.0060))
    density_inhibits = ((0.0, 0.42), (0.35, 0.34), (0.55, 0.42)) if not wide else ((0.0, 0.42), (0.25, 0.30), (0.45, 0.36), (0.65, 0.44))
    for strength in strengths:
        for threshold in thresholds:
            for red_weight, blue_weight in axis_weights:
                for detail_threshold, edge_threshold in detail_edges:
                    for line_strength, line_threshold in line_restores:
                        for coherent_strength, coherent_energy in coherent_restores:
                            for inhibit_strength, inhibit_energy in coherent_inhibits:
                                for density_strength, density_threshold in density_inhibits:
                                    idx += 1
                                    candidates.append(
                                        Candidate(
                                            name=(
                                                f"s{strength:.2f}_t{threshold:.4f}_r{red_weight:.2f}_b{blue_weight:.2f}_"
                                                f"d{detail_threshold:.3f}_e{edge_threshold:.3f}_lr{line_strength:.2f}_lt{line_threshold:.3f}_"
                                                f"cr{coherent_strength:.2f}_ce{coherent_energy:.4f}_"
                                                f"ci{inhibit_strength:.2f}_cie{inhibit_energy:.4f}_"
                                                f"di{density_strength:.2f}_dt{density_threshold:.2f}"
                                            ).replace(".", "p"),
                                            strength=strength,
                                            median_size=7,
                                            low_sigma=2.4,
                                            outlier_threshold=threshold,
                                            outlier_transition=max(0.0018, threshold * 0.70),
                                            magenta_weight=1.20,
                                            red_weight=red_weight,
                                            blue_weight=blue_weight,
                                            detail_threshold=detail_threshold,
                                            edge_threshold=edge_threshold,
                                            shadow_threshold=0.58,
                                            line_restore_strength=line_strength,
                                            line_restore_threshold=line_threshold,
                                            coherent_restore_strength=coherent_strength,
                                            coherent_energy_threshold=coherent_energy,
                                            coherent_inhibit_strength=inhibit_strength,
                                            coherent_inhibit_energy_threshold=inhibit_energy,
                                            density_inhibit_strength=density_strength,
                                            density_threshold=density_threshold,
                                        )
                                    )
    return candidates


def apply_candidate(image: np.ndarray, guide: np.ndarray, cand: Candidate) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    out, stats, blend = apply_signed_chroma_outlier_filter(
        image,
        guide,
        strength=cand.strength,
        median_size=cand.median_size,
        low_sigma=cand.low_sigma,
        outlier_threshold=cand.outlier_threshold,
        outlier_transition=cand.outlier_transition,
        magenta_weight=cand.magenta_weight,
        red_weight=cand.red_weight,
        blue_weight=cand.blue_weight,
        structure_sigma=1.2,
        detail_sigma=2.8,
        detail_threshold=cand.detail_threshold,
        detail_transition=0.010,
        edge_sigma=1.0,
        edge_threshold=cand.edge_threshold,
        edge_transition=0.015,
        shadow_threshold=cand.shadow_threshold,
        shadow_transition=0.18,
        highlight_threshold=1.0,
        highlight_transition=0.25,
        hdr_restore_peak_threshold=0.95,
        hdr_restore_threshold=0.85,
        hdr_restore_transition=0.25,
        coherent_inhibit_strength=cand.coherent_inhibit_strength,
        coherent_inhibit_energy_threshold=cand.coherent_inhibit_energy_threshold,
        outlier_density_inhibit_strength=cand.density_inhibit_strength,
        outlier_density_threshold=cand.density_threshold,
    )
    out, line_stats, line_gate = restore_fine_lines(
        image,
        out,
        strength=cand.line_restore_strength,
        threshold=cand.line_restore_threshold,
    )
    stats.update(line_stats)
    out, coherent_stats, coherent_gate = restore_coherent_structure(
        image,
        out,
        strength=cand.coherent_restore_strength,
        energy_threshold=cand.coherent_energy_threshold,
    )
    stats.update(coherent_stats)
    return out, stats, np.maximum(np.maximum(blend, line_gate), coherent_gate)


def render_crop_compare(path: Path, noisy: np.ndarray, base: np.ndarray, teacher: np.ndarray, cand: np.ndarray, labels: tuple[str, ...]) -> None:
    panels = [make_preview(x) for x in (noisy, base, teacher, cand)]
    h, w = panels[0].shape[:2]
    label_h = 24
    canvas = Image.new("RGB", (w * len(panels), h + label_h), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    for i, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (i * w, label_h))
        draw.text((i * w + 8, 5), labels[i], fill=(235, 235, 235))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan signed chroma outlier filter settings on real-photo crops.")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "signed_chroma_outlier_scan_v9_densityinhibit"))
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--apply-full", action="store_true")
    parser.add_argument("--wide-grid", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "best_crops"
    full_dir = out_dir / "full_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    scene_data = {}
    for scene_name, scene in SCENES.items():
        if not scene.base.exists() or not teacher_path(scene_name).exists() or not scene.noisy.exists():
            raise FileNotFoundError(f"missing inputs for {scene_name}")
        scene_data[scene_name] = {
            "scene": scene,
            "noisy": read_image(scene.noisy),
            "base": read_image(scene.base),
            "teacher": read_image(teacher_path(scene_name)),
        }

    crop_data = []
    for scene_name, data in scene_data.items():
        scene = data["scene"]
        for roi_name, x, y in scene.rois:
            crop_data.append(
                {
                    "scene": scene_name,
                    "roi": roi_name,
                    "noisy": crop_image(data["noisy"], x, y, args.crop_size),
                    "base": crop_image(data["base"], x, y, args.crop_size),
                    "teacher": crop_image(data["teacher"], x, y, args.crop_size),
                }
            )
    for crop in crop_data:
        crop["prepared"] = prepare_crop_metrics(crop["noisy"], crop["base"], crop["teacher"])

    rows = []
    candidates = make_candidates(wide=args.wide_grid)
    for index, cand in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {cand.name}", flush=True)
        metrics = []
        chroma_dist = []
        luma_dist = []
        for crop in crop_data:
            out, _, _ = apply_candidate(crop["base"], crop["base"], cand)
            metrics.append(eval_crop(crop["noisy"], out, crop["prepared"]))
            chroma_dist.append(axis_hf_chroma_mae(out, crop["teacher"]))
            luma_dist.append(display_luma_mae(out, crop["teacher"]))
        score = score_metrics(metrics, chroma_dist, luma_dist)
        rows.append({"candidate": asdict(cand), **score})

    rows.sort(key=lambda item: item["score"])
    report = {
        "crop_size": int(args.crop_size),
        "teacher_dir": str(TEACHER_DIR),
        "scenes": {name: {"noisy": str(data["scene"].noisy), "base": str(data["scene"].base), "teacher": str(teacher_path(name))} for name, data in scene_data.items()},
        "top": rows[: args.top_k],
        "all": rows,
    }
    (out_dir / "scan.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Signed Chroma Outlier Scan v9",
        "",
        "| rank | candidate | score | chroma | magenta | shadow magenta | luma | edge loss | teacher chroma MAE | worse |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows[: args.top_k], start=1):
        cand = row["candidate"]
        lines.append(
            "| {rank} | {name} | {score:.4f} | {chroma:.3f} | {magenta:.3f} | {shadow:.3f} | {luma:.3f} | {edge:.5f} | {teacher:.5f} | {worse} |".format(
                rank=rank,
                name=cand["name"],
                score=row["score"],
                chroma=row["avg_rel_chroma"],
                magenta=row["avg_rel_magenta"],
                shadow=row["avg_rel_shadow_magenta"],
                luma=row["avg_rel_luma"],
                edge=row["avg_edge_loss"],
                teacher=row["avg_teacher_chroma_mae"],
                worse=row["worse_count"],
            )
        )
    (out_dir / "scan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = Candidate(**rows[0]["candidate"])
    for crop in crop_data:
        out, _, _ = apply_candidate(crop["base"], crop["base"], best)
        render_crop_compare(
            crop_dir / f"{crop['scene']}_{crop['roi']}_best_compare.png",
            crop["noisy"],
            crop["base"],
            crop["teacher"],
            out,
            ("noisy", "base", "PL-safe teacher", "signed outlier"),
        )

    if args.apply_full:
        full_dir.mkdir(parents=True, exist_ok=True)
        full_report = {}
        for scene_name, data in scene_data.items():
            out, stats, blend = apply_candidate(data["base"], data["base"], best)
            stem = f"{scene_name}_signed_chroma_outlier_v9_densityinhibit"
            exr_path = full_dir / f"{stem}.exr"
            tiff_path = full_dir / f"{stem}.tiff"
            preview_path = full_dir / f"{stem}_preview.png"
            blend_path = full_dir / f"{stem}_blend.png"
            write_exr(exr_path, out)
            write_tiff(tiff_path, out)
            Image.fromarray(make_preview(out)).save(preview_path)
            Image.fromarray((np.clip(blend, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(blend_path)
            full_report[scene_name] = {
                "output": str(exr_path),
                "preview": str(preview_path),
                "blend": str(blend_path),
                "stats": stats,
            }
        (full_dir / "full_report.json").write_text(json.dumps(full_report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out_dir / 'scan.md'}")
    print(f"best {best.name}")
    print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
