"""Phase 5 Stage B (Core ML) production quality gates.

Runs the SAME four gates as `scripts/phase5_gates.py`, but through the
exported Core ML NagiV2-L graph (`scripts/coreml_infer.py`) instead of
PyTorch, so Core ML's real-world quality can be measured with identical
methodology/mask definitions to the PyTorch numbers -- directly comparable,
not compared against historical numbers from a differently-defined mask.

Reproduces `nagi_denoise.pipeline.denoise.denoise()`'s semantics exactly:
  1. Core ML tiled model forward (Hann-window stitched, guard disarmed
     in-graph).
  2. `apply_highlight_guard_np` -- the post-hoc equivalent of the in-graph
     HDR highlight guard (see `scripts/coreml_infer.py` docstring for the
     equivalence proof), using the SAME adaptive-threshold policy as
     `nagi_denoise.pipeline.denoise._resolve_highlight_guard`.
  3. `flat_chroma_smoother.smooth_chroma` with the production `CH` params.

Also times each stage so the Core ML numbers can be placed against the
237.1s PyTorch-MPS model-only baseline AND the full end-to-end production
pipeline cost.

Usage:
    pixi run python scripts/gates_coreml_nagi_v2.py --precision fp16 --compute-units cpu_and_gpu
    pixi run python scripts/gates_coreml_nagi_v2.py --precision fp32 --compute-units cpu_and_gpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import coremltools as ct  # noqa: E402

from coreml_infer import CoreMLTiledDenoiser, apply_highlight_guard_np, linear_luma_np  # noqa: E402
from nagi_denoise.pipeline.denoise import (  # noqa: E402
    CH,
    HIGHLIGHT_GUARD_DEFAULT_STRENGTH,
    HIGHLIGHT_GUARD_DEFAULT_THRESHOLD,
    HIGHLIGHT_GUARD_DEFAULT_TRANSITION,
    HIGHLIGHT_GUARD_MIN_P999,
)
from nagi_denoise.pipeline.eval_selectivity import DEFAULT_MASK_KWARGS, crop_center, selectivity_metrics  # noqa: E402
from nagi_denoise.pipeline.flat_chroma_smoother import LUMA_LINEAR, luma, smooth_chroma  # noqa: E402
from nagi_denoise.pipeline.probe import read_image  # noqa: E402

TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
OCCI = TEST_PHOTOS / "X-T5 Occi noisy.EXR"
ROOM = TEST_PHOTOS / "X-T5 Room.EXR"
COREML_DIR = REPO_ROOT / "runs" / "phase5_speed" / "coreml"

REFERENCE = {
    "selectivity_pt": None,  # filled from a same-methodology PT run by the caller for a fair diff
    "hdr_highlight_retention_min": 0.99,
}

CU_MAP = {"cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU, "all": ct.ComputeUnit.ALL}


def run_full_pipeline(dn: CoreMLTiledDenoiser, img: np.ndarray, overlap: int, apply_guard: bool, timings: dict) -> np.ndarray:
    per_tile_times: list = []
    t0 = time.perf_counter()
    out = dn.denoise_array(img, overlap=overlap, per_tile_times=per_tile_times)
    timings["model_seconds"] = time.perf_counter() - t0
    timings["n_tiles"] = len(per_tile_times)

    t0 = time.perf_counter()
    if apply_guard:
        y_in = linear_luma_np(np.clip(img, 0.0, None))
        p999 = float(np.percentile(y_in, 99.9))
        if p999 > HIGHLIGHT_GUARD_MIN_P999:
            out = apply_highlight_guard_np(
                out, img,
                threshold=HIGHLIGHT_GUARD_DEFAULT_THRESHOLD,
                transition=HIGHLIGHT_GUARD_DEFAULT_TRANSITION,
                strength=HIGHLIGHT_GUARD_DEFAULT_STRENGTH,
            )
            timings["guard_mode"] = "conditional-armed"
        else:
            timings["guard_mode"] = "conditional-disarmed"
        timings["guard_p999_input_luma"] = p999
    timings["guard_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out, _stats, _gate = smooth_chroma(out, **CH)
    timings["chroma_seconds"] = time.perf_counter() - t0
    return out


def gate_selectivity(dn: CoreMLTiledDenoiser, overlap: int) -> dict:
    img = read_image(OCCI)
    timings: dict = {}
    # Selectivity gate is measured with input_blend=0, highlight_guard=False
    # in phase5_gates.py -- match that exactly.
    out = run_full_pipeline(dn, img, overlap, apply_guard=False, timings=timings)
    noisy_crop = crop_center(img, 2420, 1040, 256)
    cand_crop = crop_center(out, 2420, 1040, 256)
    metrics = selectivity_metrics(noisy_crop, cand_crop, mask_kwargs=DEFAULT_MASK_KWARGS)
    metrics["_timings"] = timings
    return metrics


def gate_hdr_highlight(dn: CoreMLTiledDenoiser, overlap: int) -> dict:
    img = read_image(ROOM)
    timings: dict = {}
    out = run_full_pipeline(dn, img, overlap, apply_guard=True, timings=timings)
    y_in = luma(np.clip(img, 0.0, None), LUMA_LINEAR)
    y_out = luma(np.clip(out, 0.0, None), LUMA_LINEAR)
    thr = np.percentile(y_in, 99.0)
    mask = y_in >= thr
    mean_in = float(y_in[mask].mean())
    mean_out = float(y_out[mask].mean())
    retention = mean_out / mean_in if mean_in > 1e-8 else float("nan")
    return {"mean_in_top1pct": mean_in, "mean_out_top1pct": mean_out, "retention": retention, "_timings": timings}


def _tile_boundary_lines(length: int, tile: int, overlap: int, size_multiple: int = 8) -> list[int]:
    tile = int(tile)
    if tile % size_multiple != 0:
        tile = (tile // size_multiple) * size_multiple
    if tile <= 0 or length <= tile:
        return []
    stride = max(1, tile - overlap)
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    if not starts or starts[-1] + tile < length:
        starts.append(max(0, length - tile))
    lines: set[int] = set()
    for s in starts:
        lines.add(s)
        lines.add(min(s + tile, length))
    lines.discard(0)
    lines.discard(length)
    return sorted(lines)


def gate_seam(dn: CoreMLTiledDenoiser, overlap: int, scene_path: Path, tile: int) -> dict:
    img = read_image(scene_path)
    timings: dict = {}
    out = run_full_pipeline(dn, img, overlap, apply_guard=True, timings=timings)
    y = luma(np.clip(out, 0.0, None), LUMA_LINEAR)
    h, w = y.shape
    row_diff = np.abs(np.diff(y, axis=0)).mean(axis=1)

    y_lines = _tile_boundary_lines(h, tile, overlap)
    x_lines = _tile_boundary_lines(w, tile, overlap)

    band = 2
    boundary_row_idx: set[int] = set()
    for yl in y_lines:
        for r in range(max(0, yl - band), min(h - 1, yl + band)):
            boundary_row_idx.add(r)

    ordinary_p999 = float(np.percentile(row_diff, 99.9))
    worst_boundary = float(row_diff[list(boundary_row_idx)].max()) if boundary_row_idx else None

    return {
        "n_y_lines": len(y_lines),
        "n_x_lines": len(x_lines),
        "ordinary_row_diff_p999": ordinary_p999,
        "worst_boundary_row_diff": worst_boundary,
        "pass": bool(worst_boundary is not None and worst_boundary <= ordinary_p999),
        "_timings": timings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--compute-units", choices=list(CU_MAP), default="cpu_and_gpu")
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--skip-seam", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pkg = COREML_DIR / f"nagi_v2_l_ft2_t{args.tile}_{args.precision}.mlpackage"
    dn = CoreMLTiledDenoiser(str(pkg), tile=args.tile, compute_units=CU_MAP[args.compute_units])

    report: dict = {
        "backend": "coreml", "precision": args.precision, "compute_units": args.compute_units,
        "tile": args.tile, "overlap": args.overlap,
    }

    print("[1/3] selectivity (X-T5 Occi hair)...")
    report["selectivity"] = gate_selectivity(dn, args.overlap)
    print(json.dumps(report["selectivity"], indent=2))

    print("[2/3] HDR highlight retention (X-T5 Room)...")
    report["hdr_highlight"] = gate_hdr_highlight(dn, args.overlap)
    print(json.dumps(report["hdr_highlight"], indent=2))

    if not args.skip_seam:
        print("[3/3] seam invisibility (X-T5 Occi)...")
        report["seam"] = gate_seam(dn, args.overlap, OCCI, args.tile)
        print(json.dumps(report["seam"], indent=2))

    report["reference"] = REFERENCE
    print(json.dumps(report, indent=2))
    out_path = Path(args.out) if args.out else COREML_DIR / f"gates_{args.precision}_{args.compute_units}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
