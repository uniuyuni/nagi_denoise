"""Phase 4B evaluation harness: structure-vs-noise selectivity, HDR highlight
retention, and 5-panel artifact-hunt sheets for the texture-statistics
fine-tune (nagi_v2_l_ft5) vs production (nagi_v2_l_ft2).

Not a unit test -- a driver script for the honest-comparison pass described
in the Phase 4B task notes. Heavy (full-resolution tiled inference over
several hundred-MB EXRs); run it after training has stopped so it doesn't
contend with the trainer for MPS/CPU/memory.

Usage:
    pixi run python scripts/eval_phase4b.py \\
        --ft5-weights runs/nagi_v2_l_ft5/nagi_v2_l_ft5_final.pt \\
        --strengths 0.25 1 2 4 \\
        --device mps --output-dir runs/phase4b_texture
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nagi_denoise.pipeline import eval_selectivity as sel
from nagi_denoise.pipeline.denoise import PRODUCTION_WEIGHTS, _linear_luma_np, denoise
from nagi_denoise.pipeline.detail_guard import write_exr
from nagi_denoise.pipeline.make_crop_compare import crop, render
from nagi_denoise.pipeline.probe import read_image

TEST_PHOTOS = Path("~/ProjectData/test_photos").expanduser()

# name -> (noisy EXR, roi x, roi y, roi size)
SCENES: dict[str, tuple[Path, int, int, int]] = {
    "occi_hair": (TEST_PHOTOS / "X-T5 Occi noisy.EXR", 2420, 1040, 256),
    "ice_detail": (TEST_PHOTOS / "K-5 Ice noisy.EXR", 2420, 1200, 256),
    "z7_bird": (TEST_PHOTOS / "Z7 bird noisy.EXR", 4000, 2480, 256),  # not emphasized in training
    "xt5_cat": (TEST_PHOTOS / "X-T5 Cat noisy.EXR", 1200, 520, 256),  # not emphasized in training
}
ROOM_PATH = TEST_PHOTOS / "X-T5 Room.EXR"


def top1pct_retention(in_img: np.ndarray, out_img: np.ndarray) -> float:
    """Mean output luma / mean input luma over the top 1% brightest input
    pixels, over the WHOLE image (not a hand-picked crop, to avoid
    cherry-picking the HDR-safety check)."""
    luma_in = _linear_luma_np(in_img)
    luma_out = _linear_luma_np(out_img)
    thr = np.percentile(luma_in, 99.0)
    mask = luma_in >= thr
    return float(luma_out[mask].mean() / luma_in[mask].mean())


def run_hdr_check(weights: str | None, name: str, device: str, detail_strength: float | None, out_dir: Path) -> dict:
    img = read_image(ROOM_PATH)
    out = denoise(
        img,
        weights=weights,
        device=device,
        chroma_cleanup=True,
        input_blend=0.0,  # clean comparison, per task notes
        detail_strength=detail_strength,
        highlight_guard=True,
    )
    retention = top1pct_retention(img, out)
    result = {
        "candidate": name,
        "detail_strength": detail_strength,
        "top1pct_luma_retention": retention,
        "highlight_guard": denoise.last_highlight_guard,
    }
    print(json.dumps(result, indent=2))
    return result


def run_selectivity(
    weights: Path, name: str, device: str, detail_scale: float | None, out_dir: Path
) -> tuple[dict, dict[str, Path]]:
    reports = {}
    render_paths: dict[str, Path] = {}
    for scene_name, (noisy_path, x, y, size) in SCENES.items():
        noisy_full = read_image(noisy_path)
        candidate_full = sel.run_candidate(
            noisy_full, weights, device=device, detail_scale=detail_scale,
        )
        noisy_crop = sel.crop_center(noisy_full, x, y, size)
        cand_crop = sel.crop_center(candidate_full, x, y, size)
        metrics = sel.selectivity_metrics(noisy_crop, cand_crop)
        metrics.update({"scene": scene_name, "candidate": name, "detail_scale": detail_scale})
        print(json.dumps(metrics, indent=2))
        reports[scene_name] = metrics
        # Cache the full-res candidate render per (candidate, scene) so the
        # visual-sheet pass below doesn't need to re-run inference.
        cache_path = out_dir / "renders" / f"{name}_{scene_name}_full.exr"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_exr(cache_path, candidate_full)
        render_paths[scene_name] = cache_path
    return reports, render_paths


def make_sheets(
    candidates: list[str],
    renders: dict[str, dict[str, Path]],
    out_dir: Path,
) -> None:
    """5-panel sheets: NOISY / each candidate, one per scene ROI, 2x NEAREST."""
    for scene_name, (noisy_path, x, y, size) in SCENES.items():
        panels = [("NOISY", read_image(noisy_path))]
        for name in candidates:
            panels.append((name, read_image(renders[name][scene_name])))
        out_path = out_dir / f"phase4b_{scene_name}_compare.png"
        render(
            out_path,
            [(label, crop(img, x, y, size)) for label, img in panels],
            scale=2,
            exposure=1.0,
            tone="reinhard",
        )
        print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4B evaluation harness.")
    parser.add_argument("--ft5-weights", required=True)
    parser.add_argument("--production-weights", default=str(PRODUCTION_WEIGHTS))
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.25, 1.0, 2.0, 4.0])
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--output-dir", default="runs/phase4b_texture")
    parser.add_argument("--skip-sheets", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ft5_weights = Path(args.ft5_weights)

    report: dict = {"selectivity": {}, "hdr_retention": {}}
    candidate_order: list[str] = []
    renders: dict[str, dict[str, Path]] = {}

    # Production (detail gate closed -> detail_scale is a no-op; run once).
    prod_name = "production_ft2"
    report["selectivity"][prod_name], renders[prod_name] = run_selectivity(
        Path(args.production_weights), prod_name, args.device, None, out_dir
    )
    report["hdr_retention"][prod_name] = run_hdr_check(
        args.production_weights, prod_name, args.device, None, out_dir
    )
    candidate_order.append(prod_name)

    for strength in args.strengths:
        cand_name = f"ft5_s{strength:g}"
        report["selectivity"][cand_name], renders[cand_name] = run_selectivity(
            ft5_weights, cand_name, args.device, strength, out_dir
        )
        report["hdr_retention"][cand_name] = run_hdr_check(
            str(ft5_weights), cand_name, args.device, strength, out_dir
        )
        candidate_order.append(cand_name)

    json_path = out_dir / "phase4b_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")

    if not args.skip_sheets:
        make_sheets(candidate_order, renders, out_dir)


if __name__ == "__main__":
    main()
