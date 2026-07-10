"""Acceptance harness: candidate vs frozen baseline v12 vs PhotoLab reference.

Evaluates full-image denoiser outputs on the three frozen acceptance scenes
using the ROI noise/detail machinery from ``roi_noise_eval``. For every scene
the ratio base is the frozen baseline EXR (runs/baseline_v12), the candidate(s)
follow, and the PhotoLab DeepPRIME XD render is included as a reference
candidate. Scores are aggregated across all scenes/ROIs into one markdown
summary with a PASS/FAIL verdict per candidate:

    PASS  <=>  mean score <= 1.0  AND  no detail/skin ROI score > 1.02

Usage:
    python -m nagi_denoise.pipeline.acceptance \
        --candidate myrun=runs/myrun/{scene}_denoised.exr \
        --output-dir runs/acceptance/myrun

The candidate path may be:
  * a template containing ``{scene}`` (replaced with k5_dance / k5_ice / xt5_occi),
  * a directory (searched for a file whose name starts with the scene key),
  * a single file (only valid when exactly one scene is evaluated).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .probe import read_image
from .roi_noise_eval import DEFAULT_ROIS, ROI_KIND, display_crop, roi_metrics, roi_score, safe_ratio

# Scene key -> PhotoLab reference filename stem in --pl-dir.
SCENE_PL_NAMES = {
    "k5_dance": "K-5 Dance",
    "k5_ice": "K-5 Ice",
    "xt5_occi": "X-T5 Occi",
}
BASELINE_NAME = "baseline_v12"
PL_NAME = "pl_deepprime_xd"
IMAGE_SUFFIXES = (".exr", ".tif", ".tiff", ".png")


def parse_candidate(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--candidate must be NAME=path_template_or_dir, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if name in (BASELINE_NAME, PL_NAME):
        raise ValueError(f"candidate name {name!r} is reserved")
    return name, path.strip()


def resolve_candidate_path(template: str, scene: str, n_scenes: int) -> Path:
    if "{scene}" in template:
        return Path(template.format(scene=scene)).expanduser()
    p = Path(template).expanduser()
    if p.is_dir():
        matches = sorted(
            f for f in p.iterdir() if f.name.startswith(scene) and f.suffix.lower() in IMAGE_SUFFIXES
        )
        if not matches:
            raise FileNotFoundError(f"no file starting with {scene!r} under {p}")
        return matches[0]
    if n_scenes != 1:
        raise ValueError(
            f"candidate path {template!r} is a single file but {n_scenes} scenes are evaluated; "
            "use a {scene} template or a directory"
        )
    return p


def evaluate(
    scenes: list[str],
    candidates: list[tuple[str, str]],
    baseline_dir: Path,
    pl_dir: Path,
    median_size: int,
) -> dict:
    results: list[dict] = []
    for scene in scenes:
        baseline_path = baseline_dir / f"{scene}_scunet_preset_chooser_v12_auto.exr"
        pl_path = pl_dir / f"{SCENE_PL_NAMES[scene]} PL deepprimeXD.tif"
        entries: list[tuple[str, Path]] = [(BASELINE_NAME, baseline_path)]
        entries += [
            (name, resolve_candidate_path(template, scene, len(scenes))) for name, template in candidates
        ]
        entries.append((PL_NAME, pl_path))

        images = []
        for name, path in entries:
            if not path.exists():
                raise FileNotFoundError(f"{name}: {path}")
            print(f"[{scene}] reading {name}: {path}")
            images.append((name, read_image(path), str(path)))

        for roi_name, x, y, size in DEFAULT_ROIS[scene]:
            base_metrics = None
            kind = ROI_KIND.get(roi_name, "mixed")
            for cand_name, image, path in images:
                metrics = roi_metrics(display_crop(image, x, y, size), median_size)
                if base_metrics is None:
                    base_metrics = metrics
                ratios = {f"{k}_ratio": safe_ratio(v, base_metrics[k]) for k, v in metrics.items()}
                results.append(
                    {
                        "scene": scene,
                        "roi": roi_name,
                        "kind": kind,
                        "candidate": cand_name,
                        "path": path,
                        "score": roi_score(kind, ratios),
                        "metrics": metrics,
                        "ratios": ratios,
                    }
                )
    return {"results": results}


def aggregate(report: dict, candidates: list[tuple[str, str]]) -> list[dict]:
    summary = []
    names = [BASELINE_NAME] + [name for name, _ in candidates] + [PL_NAME]
    for name in names:
        items = [r for r in report["results"] if r["candidate"] == name]
        scores = [float(r["score"]) for r in items]
        detail_bad = [
            f"{r['scene']}/{r['roi']}={r['score']:.3f}"
            for r in items
            if r["kind"] in ("detail", "skin") and float(r["score"]) > 1.02
        ]
        mean_score = float(np.mean(scores)) if scores else float("nan")
        is_reference = name in (BASELINE_NAME, PL_NAME)
        entry = {
            "candidate": name,
            "mean_score": mean_score,
            "roi_count": len(scores),
            "detail_skin_violations": detail_bad,
            "reference": is_reference,
        }
        if not is_reference:
            entry["verdict"] = "PASS" if (mean_score <= 1.0 and not detail_bad) else "FAIL"
        summary.append(entry)
    return summary


def write_markdown(path: Path, report: dict, summary: list[dict], median_size: int) -> None:
    lines = [
        "# Acceptance Evaluation",
        "",
        f"Ratio base: `{BASELINE_NAME}` (frozen). Reference: `{PL_NAME}`. Median size: {median_size}.",
        "PASS = mean score <= 1.000 and no detail/skin ROI score > 1.02 (lower is better).",
        "",
        "## Summary",
        "",
        "| candidate | mean score | ROI count | detail/skin violations | verdict |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for item in summary:
        verdict = item.get("verdict", "(reference)")
        violations = ", ".join(item["detail_skin_violations"]) or "-"
        lines.append(
            f"| {item['candidate']} | {item['mean_score']:.3f} | {item['roi_count']} | {violations} | {verdict} |"
        )
    lines.extend(["", "## Per-ROI scores (ratio vs baseline)", ""])
    lines.append("| scene | ROI | kind | candidate | score | luma p99 | chroma p99 | magenta p999 | blue p999 | contrast |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in report["results"]:
        rr = r["ratios"]
        lines.append(
            "| {scene} | {roi} | {kind} | {cand} | {score:.3f} | {lp:.3f} | {cp:.3f} | {mp:.3f} | {bp:.3f} | {lc:.3f} |".format(
                scene=r["scene"],
                roi=r["roi"],
                kind=r["kind"],
                cand=r["candidate"],
                score=float(r["score"]),
                lp=rr["luma_imp_p99_ratio"],
                cp=rr["chroma_imp_p99_ratio"],
                mp=rr["magenta_pos_p999_ratio"],
                bp=rr["blue_pos_p999_ratio"],
                lc=rr["local_contrast_ratio"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Acceptance evaluation vs frozen baseline v12 and PhotoLab reference.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME=path_template_or_dir; template may contain {scene}. Repeatable.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scenes",
        default=",".join(sorted(SCENE_PL_NAMES)),
        help=f"comma-separated subset of {sorted(SCENE_PL_NAMES)}",
    )
    parser.add_argument("--baseline-dir", default="runs/baseline_v12")
    parser.add_argument("--pl-dir", default="~/ProjectData/test_photos")
    parser.add_argument("--median-size", type=int, default=3)
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    unknown = set(scenes) - set(SCENE_PL_NAMES)
    if unknown:
        raise SystemExit(f"unknown scenes: {sorted(unknown)}; choose from {sorted(SCENE_PL_NAMES)}")
    candidates = [parse_candidate(v) for v in args.candidate]

    report = evaluate(
        scenes,
        candidates,
        baseline_dir=Path(args.baseline_dir).expanduser(),
        pl_dir=Path(args.pl_dir).expanduser(),
        median_size=int(args.median_size),
    )
    summary = aggregate(report, candidates)
    report["summary"] = summary

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "acceptance.json"
    md_path = out_dir / "acceptance.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, report, summary, int(args.median_size))

    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))
    for item in summary:
        verdict = item.get("verdict", "(reference)")
        print(f"{item['candidate']:>24s}  mean={item['mean_score']:.3f}  {verdict}")


if __name__ == "__main__":
    main()
