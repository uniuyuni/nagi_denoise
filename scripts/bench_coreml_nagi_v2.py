"""Phase 5 Stage B (Core ML) speed benchmark.

Runs the exported Core ML NagiV2-L graph (`scripts/export_coreml_nagi_v2.py`)
through the SAME Hann-window tiling scheme as production
(`scripts/coreml_infer.py::CoreMLTiledDenoiser`, geometry imported directly
from `nagi_denoise.infer`) over a full 39.8MP image, so the resulting
wall-clock number is directly comparable to the 237.1s PyTorch-MPS baseline
measured in Stage A.

Also reports:
  * per-tile ms (mean/median/min/max)
  * a thermal-throttling check: mean per-tile time over the first quarter of
    tiles processed vs the last quarter, since the previous model line found
    ANE compute units thermally unstable on sustained full-image runs.

Usage:
    pixi run python scripts/bench_coreml_nagi_v2.py --precision fp16 --compute-units cpu_and_gpu
    pixi run python scripts/bench_coreml_nagi_v2.py --precision fp16 --compute-units all
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

from coreml_infer import CoreMLTiledDenoiser  # noqa: E402
from nagi_denoise.pipeline.probe import read_image  # noqa: E402

TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
OCCI = TEST_PHOTOS / "X-T5 Occi noisy.EXR"
COREML_DIR = REPO_ROOT / "runs" / "phase5_speed" / "coreml"

CU_MAP = {
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "all": ct.ComputeUnit.ALL,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--compute-units", choices=list(CU_MAP), default="cpu_and_gpu")
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--scene", default=str(OCCI))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pkg = COREML_DIR / f"nagi_v2_l_ft2_t{args.tile}_{args.precision}.mlpackage"
    if not pkg.exists():
        raise FileNotFoundError(f"missing export: {pkg} (run scripts/export_coreml_nagi_v2.py first)")

    print(f"reading {args.scene} ...")
    img = read_image(args.scene)
    mp = img.shape[0] * img.shape[1] / 1e6
    print(f"shape={img.shape} ({mp:.2f} MP)")

    print(f"loading {pkg.name} compute_units={args.compute_units} (includes warm-up call) ...")
    t0 = time.perf_counter()
    dn = CoreMLTiledDenoiser(str(pkg), tile=args.tile, compute_units=CU_MAP[args.compute_units])
    t_load = time.perf_counter() - t0
    print(f"load+warmup: {t_load:.2f}s")

    per_tile_times: list[float] = []
    t0 = time.perf_counter()
    out = dn.denoise_array(img, overlap=args.overlap, per_tile_times=per_tile_times)
    t_total = time.perf_counter() - t0

    n = len(per_tile_times)
    q = max(1, n // 4)
    first_q = per_tile_times[:q]
    last_q = per_tile_times[-q:]
    arr = np.array(per_tile_times)

    report = {
        "precision": args.precision,
        "compute_units": args.compute_units,
        "tile": args.tile,
        "overlap": args.overlap,
        "scene": str(args.scene),
        "megapixels": mp,
        "load_warmup_seconds": t_load,
        "total_seconds": t_total,
        "n_tiles": n,
        "per_tile_ms_mean": float(arr.mean() * 1000),
        "per_tile_ms_median": float(np.median(arr) * 1000),
        "per_tile_ms_min": float(arr.min() * 1000),
        "per_tile_ms_max": float(arr.max() * 1000),
        "first_quarter_mean_ms": float(np.mean(first_q) * 1000),
        "last_quarter_mean_ms": float(np.mean(last_q) * 1000),
        "thermal_slowdown_ratio": float(np.mean(last_q) / np.mean(first_q)) if np.mean(first_q) > 0 else None,
        "output_finite": bool(np.isfinite(out).all()),
        "output_max": float(out.max()),
        "output_min": float(out.min()),
    }
    print(json.dumps(report, indent=2))

    out_path = Path(args.out) if args.out else COREML_DIR / f"speed_{args.precision}_{args.compute_units}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
