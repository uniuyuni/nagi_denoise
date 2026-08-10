"""Seam verification for the tiled NagiV2 inference path (Phase 3, task 2).

The tiler in ``nagi_denoise.infer.Denoiser._tiled_forward`` blends overlapping
tiles with a 2D Hann window. If that blending is seam-free, denoising the
same image with two *different* tile grids (different tile size / overlap)
must produce near-identical output everywhere -- including right at the tile
boundary lines of either grid, where a seam bug would show up as a thin band
of elevated error tracing the grid.

This module runs exactly that comparison plus, where memory allows, a
stronger one: tiled vs. fully non-tiled inference on a small crop (the
non-tiled path has no grid at all, so any elevated error at the tiled run's
grid lines is unambiguous evidence of a seam).

Usage:
    pixi run python -m nagi_denoise.pipeline.seam_check \\
        --input "/Users/uniuyuni/ProjectData/test_photos/X-T5 Occi noisy.EXR" \\
        --output-dir runs/phase3_full/seam_check --device mps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .denoise import denoise
from .probe import read_image


def _tile_boundary_lines(length: int, tile: int, overlap: int, size_multiple: int = 8) -> list[int]:
    """Replicate the start/stop coordinates used by Denoiser._tiled_forward.

    Returns the sorted set of *interior* coordinates (excluding the image
    edges 0 and length) where a tile edge falls -- i.e. every place a seam
    could appear.
    """
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


def _boundary_band_mask(h: int, w: int, y_lines: list[int], x_lines: list[int], band: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for y in y_lines:
        mask[max(0, y - band) : min(h, y + band), :] = True
    for x in x_lines:
        mask[:, max(0, x - band) : min(w, x + band)] = True
    return mask


def _heatmap_png(diff2d: np.ndarray, scale_p: float = 99.9) -> np.ndarray:
    """Auto-scaled black -> blue -> green -> yellow -> red heatmap of a 2D error map."""
    d = np.nan_to_num(diff2d.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    hi = float(np.percentile(d, scale_p))
    hi = max(hi, 1e-8)
    t = np.clip(d / hi, 0.0, 1.0)

    # 5-stop gradient: black, blue, green, yellow, red.
    stops = np.array(
        [
            [0, 0, 0],
            [0, 0, 255],
            [0, 200, 0],
            [255, 230, 0],
            [255, 0, 0],
        ],
        dtype=np.float32,
    )
    n = stops.shape[0] - 1
    pos = t * n
    idx = np.clip(pos.astype(np.int32), 0, n - 1)
    frac = (pos - idx)[..., None]
    lo = stops[idx]
    hi_c = stops[idx + 1]
    rgb = lo + (hi_c - lo) * frac
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def compare_runs(
    a: np.ndarray,
    b: np.ndarray,
    *,
    y_lines: list[int],
    x_lines: list[int],
    band: int = 3,
) -> dict:
    """Compute seam metrics between two denoise outputs of the same image."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    diff2d = diff.max(axis=-1)  # worst-case across channels per pixel

    h, w = diff2d.shape
    band_mask = _boundary_band_mask(h, w, y_lines, x_lines, band)

    flat = diff2d.reshape(-1)
    stats = {
        "max": float(diff2d.max()),
        "mean": float(diff2d.mean()),
        "p99_9": float(np.percentile(flat, 99.9)),
        "p99": float(np.percentile(flat, 99.0)),
        "median": float(np.median(flat)),
    }
    if band_mask.any() and (~band_mask).any():
        band_mean = float(diff2d[band_mask].mean())
        other_mean = float(diff2d[~band_mask].mean())
        stats["boundary_band_mean"] = band_mean
        stats["elsewhere_mean"] = other_mean
        stats["boundary_ratio"] = band_mean / max(other_mean, 1e-12)
        stats["boundary_band_pixels"] = int(band_mask.sum())
        stats["boundary_band_fraction"] = float(band_mask.mean())
    else:
        stats["boundary_band_mean"] = None
        stats["elsewhere_mean"] = None
        stats["boundary_ratio"] = None
        stats["boundary_band_pixels"] = 0
        stats["boundary_band_fraction"] = 0.0
    stats["n_y_lines"] = len(y_lines)
    stats["n_x_lines"] = len(x_lines)
    return stats, diff2d


def run_cross_grid_check(
    img: np.ndarray,
    *,
    weights,
    device: str,
    tile_a: int,
    overlap_a: int,
    tile_b: int,
    overlap_b: int,
    band: int,
    chroma_cleanup: bool = True,
) -> tuple[dict, np.ndarray]:
    h, w = img.shape[:2]
    out_a = denoise(img, weights=weights, device=device, tile=tile_a, overlap=overlap_a, chroma_cleanup=chroma_cleanup)
    out_b = denoise(img, weights=weights, device=device, tile=tile_b, overlap=overlap_b, chroma_cleanup=chroma_cleanup)
    y_lines = sorted(set(_tile_boundary_lines(h, tile_a, overlap_a)) | set(_tile_boundary_lines(h, tile_b, overlap_b)))
    x_lines = sorted(set(_tile_boundary_lines(w, tile_a, overlap_a)) | set(_tile_boundary_lines(w, tile_b, overlap_b)))
    stats, diff2d = compare_runs(out_a, out_b, y_lines=y_lines, x_lines=x_lines, band=band)
    stats["mode"] = "cross_tile_grid"
    stats["tile_a"] = tile_a
    stats["overlap_a"] = overlap_a
    stats["tile_b"] = tile_b
    stats["overlap_b"] = overlap_b
    return stats, diff2d


def run_tiled_vs_nontiled_check(
    img: np.ndarray,
    *,
    weights,
    device: str,
    tile: int,
    overlap: int,
    band: int,
    chroma_cleanup: bool = True,
) -> tuple[dict, np.ndarray]:
    h, w = img.shape[:2]
    out_tiled = denoise(img, weights=weights, device=device, tile=tile, overlap=overlap, chroma_cleanup=chroma_cleanup)
    out_full = denoise(img, weights=weights, device=device, tile=0, overlap=0, chroma_cleanup=chroma_cleanup)
    y_lines = _tile_boundary_lines(h, tile, overlap)
    x_lines = _tile_boundary_lines(w, tile, overlap)
    stats, diff2d = compare_runs(out_tiled, out_full, y_lines=y_lines, x_lines=x_lines, band=band)
    stats["mode"] = "tiled_vs_nontiled"
    stats["tile"] = tile
    stats["overlap"] = overlap
    return stats, diff2d


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    size = min(size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0 : y0 + size, x0 : x0 + size]


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify the NagiV2 tiler is seam-free.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="runs/phase3_full/seam_check")
    ap.add_argument("--name", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tile-a", type=int, default=768)
    ap.add_argument("--overlap-a", type=int, default=64)
    ap.add_argument("--tile-b", type=int, default=512)
    ap.add_argument("--overlap-b", type=int, default=64)
    ap.add_argument("--band", type=int, default=3, help="Half-width in pixels of the boundary band.")
    ap.add_argument("--crop-size", type=int, default=1024, help="Crop size for the non-tiled reference check. 0 to skip.")
    ap.add_argument("--no-chroma-cleanup", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or input_path.stem.replace(" ", "_")
    chroma_cleanup = not args.no_chroma_cleanup

    img = read_image(input_path)
    print(f"loaded {input_path} shape={img.shape}")

    report: dict = {"input": str(input_path), "shape": list(img.shape), "chroma_cleanup": chroma_cleanup}

    print(f"[1/2] cross-tile-grid check: tile={args.tile_a}/{args.overlap_a} vs tile={args.tile_b}/{args.overlap_b} (full resolution)")
    grid_stats, grid_diff2d = run_cross_grid_check(
        img,
        weights=args.weights,
        device=args.device,
        tile_a=args.tile_a,
        overlap_a=args.overlap_a,
        tile_b=args.tile_b,
        overlap_b=args.overlap_b,
        band=args.band,
        chroma_cleanup=chroma_cleanup,
    )
    report["cross_tile_grid"] = grid_stats
    heatmap_path = out_dir / f"{name}_seam_heatmap_grid.png"
    Image.fromarray(_heatmap_png(grid_diff2d)).save(heatmap_path)
    report["cross_tile_grid"]["heatmap"] = str(heatmap_path)
    print(json.dumps(grid_stats, indent=2))

    if args.crop_size > 0:
        print(f"[2/2] tiled-vs-non-tiled check on a {args.crop_size}x{args.crop_size} center crop")
        crop = _center_crop(img, args.crop_size)
        nontiled_stats, nontiled_diff2d = run_tiled_vs_nontiled_check(
            crop,
            weights=args.weights,
            device=args.device,
            tile=args.tile_a,
            overlap=args.overlap_a,
            band=args.band,
            chroma_cleanup=chroma_cleanup,
        )
        report["tiled_vs_nontiled"] = nontiled_stats
        heatmap_path2 = out_dir / f"{name}_seam_heatmap_nontiled.png"
        Image.fromarray(_heatmap_png(nontiled_diff2d)).save(heatmap_path2)
        report["tiled_vs_nontiled"]["heatmap"] = str(heatmap_path2)
        print(json.dumps(nontiled_stats, indent=2))
    else:
        report["tiled_vs_nontiled"] = None

    # Verdict: seams would show up as boundary_ratio >> 1. In a seam-free
    # tiler the boundary band is just ordinary image content, so the ratio
    # should sit close to 1 (mild deviations are expected from natural image
    # structure that happens to cross a grid line).
    ratios = [
        report["cross_tile_grid"].get("boundary_ratio"),
    ]
    if report.get("tiled_vs_nontiled"):
        ratios.append(report["tiled_vs_nontiled"].get("boundary_ratio"))
    ratios = [r for r in ratios if r is not None]
    seam_detected = any(r > 3.0 for r in ratios)
    report["verdict"] = {
        "seam_detected": bool(seam_detected),
        "boundary_ratios": ratios,
        "note": "boundary_ratio = mean(|diff|) in a band around tile edges / mean(|diff|) elsewhere. "
        "A seam-free tiler should show a ratio close to 1; >3x indicates a visible seam.",
    }

    json_path = out_dir / f"{name}_seam_check.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
