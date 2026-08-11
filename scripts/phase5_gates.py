"""Phase 5 quality-gate harness.

Runs the four gates specified in the Phase 5 task against a given
denoise()-compatible configuration (weights / tile / overlap / batch_size /
amp_dtype), so Stage A/B changes can be checked for regressions against the
fixed reference numbers:

  - Selectivity, X-T5 Occi hair crop (256px @ x=2420,y=1040): reference
    +11.2pt at ret_on 90.5% / off 79.3%. Measured with input_blend=0,
    highlight_guard=False.
  - HDR highlight retention, X-T5 Room: mean of top-1% input luma, out/in,
    must stay >= 0.99.
  - Seam invisibility: worst tile-boundary row's row-to-row luma delta vs the
    99.9th percentile of ordinary rows -- must stay below it.
  - (SIDD Val is run separately via the `nagi-eval-sidd` pixi task; it does
    not go through the tiled path at all with tile=256/overlap=0 patches, so
    it is not duplicated here.)

Usage:
    pixi run python scripts/phase5_gates.py --tile 768 --overlap 64
    pixi run python scripts/phase5_gates.py --tile 1024 --overlap 64 --batch 2 --amp fp16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from nagi_denoise.devices import resolve_device
from nagi_denoise.infer import Denoiser
from nagi_denoise.pipeline.denoise import CH, PRODUCTION_WEIGHTS
from nagi_denoise.pipeline.eval_selectivity import (
    DEFAULT_MASK_KWARGS,
    crop_center,
    selectivity_metrics,
)
from nagi_denoise.pipeline.flat_chroma_smoother import LUMA_LINEAR, luma, smooth_chroma
from nagi_denoise.pipeline.probe import read_image

TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
OCCI = TEST_PHOTOS / "X-T5 Occi noisy.EXR"
ROOM = TEST_PHOTOS / "X-T5 Room.EXR"

REFERENCE = {
    "selectivity_pt": 11.2,
    "retention_on": 0.905,
    "retention_off": 0.793,
    "hdr_highlight_retention_min": 0.99,
}


def run_model(
    img: np.ndarray,
    dn: Denoiser,
    tile: int,
    overlap: int,
    batch_size: int,
    amp_dtype,
    chroma_cleanup: bool,
) -> np.ndarray:
    out = dn.denoise_array(
        img, input_space="linear", tile=tile, overlap=overlap,
        batch_size=batch_size, amp_dtype=amp_dtype,
    )
    if chroma_cleanup:
        out, _stats, _gate = smooth_chroma(out, **CH)
    return out


def gate_selectivity(dn: Denoiser, tile: int, overlap: int, batch_size: int, amp_dtype) -> dict:
    img = read_image(OCCI)
    out = run_model(img, dn, tile, overlap, batch_size, amp_dtype, chroma_cleanup=True)
    noisy_crop = crop_center(img, 2420, 1040, 256)
    cand_crop = crop_center(out, 2420, 1040, 256)
    metrics = selectivity_metrics(noisy_crop, cand_crop, mask_kwargs=DEFAULT_MASK_KWARGS)
    return metrics


def gate_hdr_highlight(dn: Denoiser, tile: int, overlap: int, batch_size: int, amp_dtype) -> dict:
    img = read_image(ROOM)
    out = run_model(img, dn, tile, overlap, batch_size, amp_dtype, chroma_cleanup=True)
    y_in = luma(np.clip(img, 0.0, None), LUMA_LINEAR)
    y_out = luma(np.clip(out, 0.0, None), LUMA_LINEAR)
    thr = np.percentile(y_in, 99.0)
    mask = y_in >= thr
    mean_in = float(y_in[mask].mean())
    mean_out = float(y_out[mask].mean())
    retention = mean_out / mean_in if mean_in > 1e-8 else float("nan")
    return {"mean_in_top1pct": mean_in, "mean_out_top1pct": mean_out, "retention": retention}


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


def gate_seam(dn: Denoiser, tile: int, overlap: int, batch_size: int, amp_dtype, scene_path: Path) -> dict:
    """Row-to-row luma discontinuity check on a SINGLE tiled output.

    For each tile boundary row y, compute mean |luma[y+1] - luma[y]| in a
    +-2px band around y. Compare the worst such boundary row against the
    99.9th percentile of the same per-row statistic computed over ALL rows
    (i.e. "ordinary" row-to-row variation). A seam-free tiler should have its
    worst boundary row look like ordinary image content, not an outlier.
    """
    img = read_image(scene_path)
    out = run_model(img, dn, tile, overlap, batch_size, amp_dtype, chroma_cleanup=True)
    y = luma(np.clip(out, 0.0, None), LUMA_LINEAR)
    h, w = y.shape
    row_diff = np.abs(np.diff(y, axis=0)).mean(axis=1)  # length h-1, row_diff[i] = |row i+1 - row i|

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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--amp", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--skip-seam", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    weights = args.weights or str(PRODUCTION_WEIGHTS)
    device = resolve_device(args.device)
    amp_dtype = torch.float16 if args.amp == "fp16" else None

    dn = Denoiser.load(weights, device=device)

    report: dict = {
        "weights": weights, "tile": args.tile, "overlap": args.overlap,
        "batch": args.batch, "amp": args.amp,
    }

    print("[1/3] selectivity (X-T5 Occi hair)...")
    report["selectivity"] = gate_selectivity(dn, args.tile, args.overlap, args.batch, amp_dtype)
    print(json.dumps(report["selectivity"], indent=2))

    print("[2/3] HDR highlight retention (X-T5 Room)...")
    report["hdr_highlight"] = gate_hdr_highlight(dn, args.tile, args.overlap, args.batch, amp_dtype)
    print(json.dumps(report["hdr_highlight"], indent=2))

    if not args.skip_seam:
        print("[3/3] seam invisibility (X-T5 Occi)...")
        report["seam"] = gate_seam(dn, args.tile, args.overlap, args.batch, amp_dtype, OCCI)
        print(json.dumps(report["seam"], indent=2))

    report["reference"] = REFERENCE
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
