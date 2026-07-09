"""Evaluate local residual noise/detail metrics on fixed real-photo ROIs.

The full-frame residual speckle evaluator is useful but expensive on large EXR
files. This script is intentionally ROI-only so we can iterate on sky, hair,
branch, and flat-shadow failures while other tasks are running.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma
from perfect_nr_probe import read_image


DEFAULT_ROIS: dict[str, list[tuple[str, int, int, int]]] = {
    "xt5_occi": [
        ("hair_detail", 2420, 1040, 384),
        ("root", 512, 5632, 384),
        ("face_center", 2120, 1260, 384),
        ("noise_dark", 3072, 3600, 384),
    ],
    "k5_dance": [
        ("sky_existing", 4096, 0, 384),
        ("sky_center", 2300, 320, 384),
        ("dancer_center", 2800, 1200, 384),
        ("house_detail", 260, 1180, 384),
        ("snow_ground", 2100, 2500, 384),
    ],
    "k5_ice": [
        ("blue_shadow", 1900, 900, 384),
        ("ice_branch", 2420, 1200, 384),
        ("dark_edge", 1400, 1520, 384),
        ("flat_blue", 3000, 780, 384),
    ],
}


ROI_KIND: dict[str, str] = {
    "hair_detail": "detail",
    "root": "detail",
    "face_center": "skin",
    "noise_dark": "flat",
    "sky_existing": "flat",
    "sky_center": "flat",
    "dancer_center": "detail",
    "house_detail": "detail",
    "snow_ground": "flat",
    "blue_shadow": "mixed",
    "ice_branch": "detail",
    "dark_edge": "detail",
    "flat_blue": "flat",
}


def roi_score(kind: str, ratios: dict[str, float]) -> float:
    noise = (
        0.35 * ratios["luma_imp_p99_ratio"]
        + 0.25 * ratios["chroma_imp_p99_ratio"]
        + 0.20 * ratios["magenta_pos_p999_ratio"]
        + 0.20 * ratios["blue_pos_p999_ratio"]
    )
    contrast_loss = max(0.0, 1.0 - ratios["local_contrast_ratio"])
    std_loss = max(0.0, 1.0 - ratios["luma_std_ratio"])
    if kind == "flat":
        return float(noise + 0.18 * contrast_loss + 0.04 * std_loss)
    if kind == "skin":
        return float(noise + 1.50 * contrast_loss + 0.24 * std_loss)
    if kind == "mixed":
        return float(noise + 0.80 * contrast_loss + 0.14 * std_loss)
    return float(noise + 1.65 * contrast_loss + 0.22 * std_loss)


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value).expanduser()
        return path.stem, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser()


def parse_roi(value: str) -> tuple[str, int, int, int]:
    parts = value.split(",")
    if len(parts) not in (3, 4):
        raise ValueError(f"roi must be name,x,y[,size]: {value!r}")
    name, x, y = parts[:3]
    size = parts[3] if len(parts) == 4 else "384"
    return name.strip(), int(x), int(y), int(size)


def crop_center(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, min(w - size, int(x) - size // 2))
    y0 = max(0, min(h - size, int(y) - size // 2))
    return image[y0 : y0 + size, x0 : x0 + size]


def display_crop(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    crop = crop_center(image, x, y, size)
    return linear_to_srgb_np(np.clip(crop[..., :3], 0.0, None)).astype(np.float32, copy=False)


def impulse(x: np.ndarray, median_size: int) -> np.ndarray:
    base = median_filter(x, size=int(median_size), mode="reflect")
    return (x - base).astype(np.float32, copy=False)


def roi_metrics(display: np.ndarray, median_size: int) -> dict[str, float]:
    y = luma(display, LUMA_SRGB)
    chroma = display - y[..., None]
    rg = chroma[..., 0] - chroma[..., 1]
    by = chroma[..., 2] - 0.5 * (chroma[..., 0] + chroma[..., 1])
    magenta = 0.5 * (display[..., 0] + display[..., 2]) - display[..., 1]
    blue = display[..., 2] - 0.5 * (display[..., 0] + display[..., 1])

    luma_imp = np.abs(impulse(y, median_size))
    rg_imp = impulse(rg, median_size)
    by_imp = impulse(by, median_size)
    chroma_imp = np.sqrt(0.5 * (rg_imp * rg_imp + by_imp * by_imp))
    magenta_pos = np.maximum(impulse(magenta, median_size), 0.0)
    blue_pos = np.maximum(impulse(blue, median_size), 0.0)

    y_low = gaussian_filter(y, sigma=2.0, mode="reflect")
    return {
        "luma_imp_p99": float(np.quantile(luma_imp, 0.99)),
        "luma_imp_p999": float(np.quantile(luma_imp, 0.999)),
        "chroma_imp_p99": float(np.quantile(chroma_imp, 0.99)),
        "chroma_imp_p999": float(np.quantile(chroma_imp, 0.999)),
        "magenta_pos_p999": float(np.quantile(magenta_pos, 0.999)),
        "blue_pos_p999": float(np.quantile(blue_pos, 0.999)),
        "luma_std": float(np.std(y)),
        "chroma_std": float(np.std(chroma)),
        "mean_y": float(np.mean(y)),
        "local_contrast": float(np.mean(np.abs(y - y_low))),
    }


def safe_ratio(value: float, base: float) -> float:
    if abs(base) < 1.0e-12:
        return 1.0
    return float(value / base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local ROI noise metrics.")
    parser.add_argument("--scene", choices=sorted(DEFAULT_ROIS), default=None)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=PATH. First candidate is ratio base.")
    parser.add_argument("--roi", action="append", default=[], help="name,x,y[,size]. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--median-size", type=int, default=3)
    args = parser.parse_args()

    rois = [parse_roi(v) for v in args.roi]
    if args.scene:
        rois = [*DEFAULT_ROIS[args.scene], *rois]
    if not rois:
        raise SystemExit("--scene or at least one --roi is required")

    candidates = [(name, read_image(path), str(path)) for name, path in (parse_candidate(v) for v in args.candidate)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "scene": args.scene,
        "median_size": args.median_size,
        "rois": [{"name": name, "x": x, "y": y, "size": size} for name, x, y, size in rois],
        "candidates": [{"name": name, "path": path} for name, _, path in candidates],
        "results": [],
    }

    for roi_name, x, y, size in rois:
        base_metrics = None
        for cand_name, image, path in candidates:
            metrics = roi_metrics(display_crop(image, x, y, size), args.median_size)
            if base_metrics is None:
                base_metrics = metrics
            ratios = {f"{key}_ratio": safe_ratio(value, base_metrics[key]) for key, value in metrics.items()}
            kind = ROI_KIND.get(roi_name, "mixed")
            score = roi_score(kind, ratios)
            report["results"].append(
                {
                    "roi": roi_name,
                    "kind": kind,
                    "candidate": cand_name,
                    "path": path,
                    "score": score,
                    "metrics": metrics,
                    "ratios": ratios,
                }
            )


    by_candidate: dict[str, list[float]] = {}
    for item in report["results"]:
        by_candidate.setdefault(item["candidate"], []).append(float(item.get("score", 1.0)))
    summary = [
        {"candidate": name, "mean_score": float(np.mean(scores)), "roi_count": len(scores)}
        for name, scores in by_candidate.items()
    ]
    summary.sort(key=lambda x: x["mean_score"])
    report["summary"] = summary

    json_path = out_dir / "roi_noise_eval.json"
    md_path = out_dir / "roi_noise_eval.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# ROI Noise Evaluation",
        "",
        f"Scene: `{args.scene}`",
        "",
        "| ROI | kind | candidate | score | luma p99 | chroma p99 | magenta p999 | blue p999 | luma std | chroma std | contrast | mean y |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["results"]:
        m = item["metrics"]
        lines.append(
            "| {roi} | {kind} | {candidate} | {score:.3f} | {lp99:.6f} | {cp99:.6f} | {mp999:.6f} | {bp999:.6f} | {ls:.6f} | {cs:.6f} | {lc:.6f} | {my:.6f} |".format(
                roi=item["roi"],
                kind=item.get("kind", "mixed"),
                candidate=item["candidate"],
                score=float(item.get("score", 1.0)),
                lp99=m["luma_imp_p99"],
                cp99=m["chroma_imp_p99"],
                mp999=m["magenta_pos_p999"],
                bp999=m["blue_pos_p999"],
                ls=m["luma_std"],
                cs=m["chroma_std"],
                lc=m["local_contrast"],
                my=m["mean_y"],
            )
        )
    lines.extend(
        [
            "",
            "Ratios vs first candidate:",
            "",
            "| ROI | kind | candidate | score | luma p99 | chroma p99 | magenta p999 | blue p999 | luma std | chroma std | contrast |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["results"]:
        r = item["ratios"]
        lines.append(
            "| {roi} | {kind} | {candidate} | {score:.3f} | {lp99:.3f} | {cp99:.3f} | {mp999:.3f} | {bp999:.3f} | {ls:.3f} | {cs:.3f} | {lc:.3f} |".format(
                roi=item["roi"],
                kind=item.get("kind", "mixed"),
                candidate=item["candidate"],
                score=float(item.get("score", 1.0)),
                lp99=r["luma_imp_p99_ratio"],
                cp99=r["chroma_imp_p99_ratio"],
                mp999=r["magenta_pos_p999_ratio"],
                bp999=r["blue_pos_p999_ratio"],
                ls=r["luma_std_ratio"],
                cs=r["chroma_std_ratio"],
                lc=r["local_contrast_ratio"],
            )
        )

    lines.extend(["", "Summary score (lower is better):", "", "| candidate | mean score | ROI count |", "| --- | ---: | ---: |"])
    for item in report["summary"]:
        lines.append(f"| {item['candidate']} | {item['mean_score']:.3f} | {item['roi_count']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
