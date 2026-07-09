"""Summarize saved mask PNG values on named ROIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_mask(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value).expanduser()
        return path.stem, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser()


def parse_roi(value: str) -> tuple[str, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError(f"roi must be name,x,y,size: {value!r}")
    name, x, y, size = parts
    return name.strip(), int(x), int(y), int(size)


def read_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def crop_center(mask: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = mask.shape[:2]
    x0 = max(0, min(w - size, int(x) - size // 2))
    y0 = max(0, min(h - size, int(y) - size // 2))
    return mask[y0 : y0 + size, x0 : x0 + size]


def summarize(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "gt_025": float(np.mean(x > 0.25)),
        "gt_050": float(np.mean(x > 0.50)),
        "gt_075": float(np.mean(x > 0.75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved mask PNG values on ROIs.")
    parser.add_argument("--mask", action="append", required=True, help="name=path. Repeatable.")
    parser.add_argument("--roi", action="append", required=True, help="name,x,y,size. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    masks = [(name, read_mask(path), str(path)) for name, path in (parse_mask(item) for item in args.mask)]
    rois = [parse_roi(item) for item in args.roi]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"masks": [{"name": name, "path": path} for name, _, path in masks], "rois": [], "results": []}
    for roi_name, x, y, size in rois:
        report["rois"].append({"name": roi_name, "x": x, "y": y, "size": size})
        for mask_name, mask, path in masks:
            report["results"].append({
                "roi": roi_name,
                "mask": mask_name,
                "path": path,
                "summary": summarize(crop_center(mask, x, y, size)),
            })

    json_path = out_dir / "mask_roi_eval.json"
    md_path = out_dir / "mask_roi_eval.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Mask ROI Evaluation",
        "",
        "| ROI | mask | mean | p90 | p95 | p99 | >0.25 | >0.50 | >0.75 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["results"]:
        s = item["summary"]
        lines.append(
            "| {roi} | {mask} | {mean:.4f} | {p90:.4f} | {p95:.4f} | {p99:.4f} | {g25:.4f} | {g50:.4f} | {g75:.4f} |".format(
                roi=item["roi"],
                mask=item["mask"],
                mean=s["mean"],
                p90=s["p90"],
                p95=s["p95"],
                p99=s["p99"],
                g25=s["gt_025"],
                g50=s["gt_050"],
                g75=s["gt_075"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
