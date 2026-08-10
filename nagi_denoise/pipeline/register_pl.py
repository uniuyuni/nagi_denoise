"""Non-rigid registration of PhotoLab DeepPrime XD output onto our EXR geometry.

Background (see task notes): PL's fine detail is real (it is not synthesizing
texture) but is geometrically displaced from our pipeline's output by up to a
few px locally, on top of a much larger *global* translation. A single global
translation cannot reconcile the two because the local displacement varies
across the frame (residual lens-correction differences). This module:

  1. Applies a known global integer offset (dx, dy) established empirically
     per scene (see SCENE_OFFSETS below). Convention, verified by local patch
     correlation on both scenes (full-band NCC 0.70-0.96 at this offset vs
     <0.2 at nearby alternatives): the PL pixel corresponding to noisy pixel
     (x, y) is PL[y + dy, x + dx] (row, col indexing).
  2. Refines a per-block local flow field by exhaustive block matching (64px
     blocks by default, +/-6px search by default) using the *signed* fine
     luma band (y - gaussian(y, 1.2)) in display space as the matching
     criterion (this is what the task's reference table was measured with:
     64px blocks, +/-6px search -> 0.901 fine-band correlation on the Occi
     hair ROI).
  3. Regularises the flow: blocks whose best correlation is below
     ``--min-corr`` are marked invalid and filled from their nearest valid
     neighbour, then the whole field is median-filtered and gaussian-blurred
     so warping cannot tear structure.
  4. Resamples PL (bilinear, via ``scipy.ndimage.map_coordinates``) onto the
     noisy image's grid using the smoothed flow field, per channel.

Whole-image band correlation (fine/mid/coarse/low, all vs the *display-space*
luma of the noisy input's own... no: vs OUR output's fine/mid/coarse/low luma,
consistent with the task's evidence table) is reported before vs after, along
with flow-magnitude statistics and the fraction of rejected blocks.

Usage:
    pixi run python -m nagi_denoise.pipeline.register_pl --scene xt5_occi \\
        --output-dir runs/phase4_detail/register
    pixi run python -m nagi_denoise.pipeline.register_pl --scene xt5_cat2 \\
        --output-dir runs/phase4_detail/register
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
    median_filter,
    zoom,
)

from .probe import read_image, srgb_oetf


ROOT = Path(__file__).resolve().parents[2]
TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")

# Display-luma weights used throughout the pipeline's deterministic passes
# (region_aware_luma_cleanup, eval_selectivity's hf_map). Kept consistent so
# fine-band numbers here are directly comparable to those elsewhere.
LUMA_SRGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Established global integer offsets (see module docstring for sign
# convention and how these were verified).
SCENE_OFFSETS: dict[str, dict] = {
    "xt5_occi": dict(
        noisy=TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        pl=TEST_PHOTOS / "X-T5 Occi PL deepprimeXD3.tif",
        dx=-20,
        dy=90,
    ),
    "xt5_cat2": dict(
        noisy=TEST_PHOTOS / "X-T5 Cat2 noisy.EXR",
        pl=TEST_PHOTOS / "X-T5 Cat2 PL deepprimeXD3.tif",
        dx=-2,
        dy=-12,
    ),
}


def _luma(display_rgb: np.ndarray) -> np.ndarray:
    return (display_rgb[..., :3] * LUMA_SRGB).sum(axis=-1).astype(np.float32, copy=False)


def noisy_to_display(noisy_linear: np.ndarray) -> np.ndarray:
    """Linear HDR EXR -> reinhard-tonemapped sRGB display, matching the
    domain PL's TIFF is already encoded in."""
    x = np.clip(noisy_linear[..., :3].astype(np.float32, copy=False), 0.0, None)
    x = x / (1.0 + x)
    return srgb_oetf(x).astype(np.float32, copy=False)


def pl_to_display(pl_srgb: np.ndarray) -> np.ndarray:
    return np.clip(pl_srgb[..., :3].astype(np.float32, copy=False), 0.0, 1.0)


def band_split(y: np.ndarray) -> dict[str, np.ndarray]:
    """Difference-of-Gaussians frequency bands, matching the task's evidence
    table naming: fine (<1.2), mid (1.2-3), coarse (3-8), low (8-20)."""
    g12 = gaussian_filter(y, sigma=1.2, mode="reflect")
    g3 = gaussian_filter(y, sigma=3.0, mode="reflect")
    g8 = gaussian_filter(y, sigma=8.0, mode="reflect")
    g20 = gaussian_filter(y, sigma=20.0, mode="reflect")
    return {
        "fine": y - g12,
        "mid": g12 - g3,
        "coarse": g3 - g8,
        "low": g8 - g20,
    }


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a.reshape(-1).astype(np.float64) - a.mean()
    b0 = b.reshape(-1).astype(np.float64) - b.mean()
    denom = np.sqrt((a0 * a0).sum() * (b0 * b0).sum())
    if denom < 1e-12:
        return 0.0
    return float((a0 * b0).sum() / denom)


@dataclass
class BlockMatchResult:
    flow_dy: np.ndarray  # (nby, nbx) int-ish local offsets, additional to global
    flow_dx: np.ndarray
    corr: np.ndarray
    valid: np.ndarray
    block: int
    radius: int


def block_match(
    noisy_fine: np.ndarray,
    pl_fine: np.ndarray,
    global_dx: int,
    global_dy: int,
    block: int = 64,
    radius: int = 6,
    min_corr: float = 0.20,
) -> BlockMatchResult:
    """Exhaustive block matching, vectorised over the whole image per
    candidate shift (loop over the ~(2R+1)^2 candidate shifts, not over
    blocks -- keeps this to a few hundred numpy passes instead of tens of
    thousands of Python-level block loops)."""
    H, W = noisy_fine.shape
    nby = (H + block - 1) // block
    nbx = (W + block - 1) // block
    H_pad, W_pad = nby * block, nbx * block

    noisy_padded = np.pad(
        noisy_fine, ((0, H_pad - H), (0, W_pad - W)), mode="reflect"
    ).astype(np.float32, copy=False)

    margin_y = radius + abs(global_dy) + block + 16
    margin_x = radius + abs(global_dx) + block + 16
    pl_padded = np.pad(
        pl_fine, ((margin_y, margin_y), (margin_x, margin_x)), mode="reflect"
    ).astype(np.float32, copy=False)

    noisy_tiles = noisy_padded.reshape(nby, block, nbx, block)
    noisy_mean = noisy_tiles.mean(axis=(1, 3), keepdims=True)
    noisy0 = noisy_tiles - noisy_mean
    noisy_ss = (noisy0 * noisy0).sum(axis=(1, 3))  # (nby, nbx)

    best_corr = np.full((nby, nbx), -2.0, dtype=np.float64)
    best_dy = np.zeros((nby, nbx), dtype=np.int32)
    best_dx = np.zeros((nby, nbx), dtype=np.int32)

    for ddy in range(-radius, radius + 1):
        for ddx in range(-radius, radius + 1):
            y0 = global_dy + ddy + margin_y
            x0 = global_dx + ddx + margin_x
            assert 0 <= y0 and y0 + H_pad <= pl_padded.shape[0], (y0, H_pad, pl_padded.shape)
            assert 0 <= x0 and x0 + W_pad <= pl_padded.shape[1], (x0, W_pad, pl_padded.shape)
            pl_slice = pl_padded[y0 : y0 + H_pad, x0 : x0 + W_pad]
            pl_tiles = pl_slice.reshape(nby, block, nbx, block)
            pl_mean = pl_tiles.mean(axis=(1, 3), keepdims=True)
            pl0 = pl_tiles - pl_mean
            num = (pl0 * noisy0).sum(axis=(1, 3))
            pl_ss = (pl0 * pl0).sum(axis=(1, 3))
            denom = np.sqrt(np.maximum(pl_ss * noisy_ss, 1e-12))
            corr = num / denom
            better = corr > best_corr
            best_corr = np.where(better, corr, best_corr)
            best_dy = np.where(better, ddy, best_dy)
            best_dx = np.where(better, ddx, best_dx)

    valid = best_corr >= min_corr
    return BlockMatchResult(
        flow_dy=best_dy.astype(np.float32),
        flow_dx=best_dx.astype(np.float32),
        corr=best_corr.astype(np.float32),
        valid=valid,
        block=block,
        radius=radius,
    )


def regularize_flow(
    result: BlockMatchResult, median_size: int = 3, gaussian_sigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Fill rejected blocks from nearest valid neighbour, then median +
    gaussian smooth so the warp cannot tear structure."""
    valid = result.valid
    if not np.any(valid):
        # Degenerate: nothing passed the correlation gate. Fall back to a
        # zero local flow (pure global alignment) rather than fabricating one.
        fy = np.zeros_like(result.flow_dy)
        fx = np.zeros_like(result.flow_dx)
        return fy, fx

    if not np.all(valid):
        _, idx = distance_transform_edt(~valid, return_distances=True, return_indices=True)
        fy = result.flow_dy[tuple(idx)]
        fx = result.flow_dx[tuple(idx)]
    else:
        fy = result.flow_dy.copy()
        fx = result.flow_dx.copy()

    if median_size > 1:
        fy = median_filter(fy, size=median_size, mode="nearest")
        fx = median_filter(fx, size=median_size, mode="nearest")
    if gaussian_sigma > 0:
        fy = gaussian_filter(fy, sigma=gaussian_sigma, mode="nearest")
        fx = gaussian_filter(fx, sigma=gaussian_sigma, mode="nearest")
    return fy.astype(np.float32, copy=False), fx.astype(np.float32, copy=False)


def upsample_flow(flow_block: np.ndarray, block: int, out_shape: tuple[int, int]) -> np.ndarray:
    """Upsample a (nby, nbx) block-grid flow to full resolution, aligning
    block centers to their pixel centers (order=1 bilinear)."""
    nby, nbx = flow_block.shape
    H, W = out_shape
    # zoom factor chosen so block index i (covering pixel rows [i*block,
    # (i+1)*block)) maps its value to the block's pixel-center region; using
    # plain zoom with the ratio of output/input size is a good enough
    # approximation for a smoothed, low-frequency field like this one.
    zy = H / nby
    zx = W / nbx
    up = zoom(flow_block, (zy, zx), order=1, mode="nearest")
    up = up[:H, :W]
    if up.shape != out_shape:
        pad_h = max(0, H - up.shape[0])
        pad_w = max(0, W - up.shape[1])
        up = np.pad(up, ((0, pad_h), (0, pad_w)), mode="edge")[:H, :W]
    return up.astype(np.float32, copy=False)


def warp_pl_rgb(pl_rgb: np.ndarray, total_dx: np.ndarray, total_dy: np.ndarray) -> np.ndarray:
    """Resample PL (H, W, 3) onto the noisy grid using per-pixel (total_dx,
    total_dy) displacement: out(y, x) = PL(y + total_dy(y, x), x + total_dx(y, x))."""
    H, W = total_dx.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    src_y = yy + total_dy
    src_x = xx + total_dx
    coords = np.stack([src_y, src_x], axis=0)
    out = np.empty((H, W, pl_rgb.shape[2]), dtype=np.float32)
    for c in range(pl_rgb.shape[2]):
        out[..., c] = map_coordinates(
            pl_rgb[..., c].astype(np.float32, copy=False), coords, order=1, mode="reflect"
        )
    return out


def flow_visualization(flow_dy: np.ndarray, flow_dx: np.ndarray) -> np.ndarray:
    mag = np.sqrt(flow_dx.astype(np.float64) ** 2 + flow_dy.astype(np.float64) ** 2)
    vmax = max(float(np.quantile(mag, 0.99)), 1e-6)
    norm = np.clip(mag / vmax, 0.0, 1.0)
    # Simple "hot" colormap without matplotlib: black -> red -> yellow -> white.
    r = np.clip(norm * 3.0, 0.0, 1.0)
    g = np.clip(norm * 3.0 - 1.0, 0.0, 1.0)
    b = np.clip(norm * 3.0 - 2.0, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def run_registration(
    scene: str,
    block: int,
    radius: int,
    min_corr: float,
    median_size: int,
    gaussian_sigma: float,
    output_dir: Path,
) -> dict:
    cfg = SCENE_OFFSETS[scene]
    noisy_path, pl_path, dx, dy = cfg["noisy"], cfg["pl"], int(cfg["dx"]), int(cfg["dy"])
    t0 = time.time()

    noisy_linear = read_image(noisy_path)
    pl_srgb = read_image(pl_path)
    if noisy_linear.shape != pl_srgb.shape:
        raise ValueError(f"shape mismatch: noisy {noisy_linear.shape} vs pl {pl_srgb.shape}")

    noisy_disp = noisy_to_display(noisy_linear)
    pl_disp = pl_to_display(pl_srgb)
    noisy_y = _luma(noisy_disp)
    pl_y = _luma(pl_disp)

    noisy_bands = band_split(noisy_y)
    pl_bands_native = band_split(pl_y)  # PL's own bands, not yet warped

    # --- "before" correlation: global-offset-only alignment ---
    H, W = noisy_y.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    global_src_y = yy + dy
    global_src_x = xx + dx
    global_coords = np.stack([global_src_y, global_src_x], axis=0)
    pl_y_global = map_coordinates(pl_y, global_coords, order=1, mode="reflect")
    pl_bands_global = band_split(pl_y_global)
    corr_before = {name: pearson_corr(noisy_bands[name], pl_bands_global[name]) for name in noisy_bands}

    # --- block matching on the signed fine band ---
    result = block_match(
        noisy_bands["fine"], pl_bands_native["fine"], dx, dy, block=block, radius=radius, min_corr=min_corr
    )
    rejected_fraction = float(1.0 - result.valid.mean())

    flow_dy_block, flow_dx_block = regularize_flow(result, median_size=median_size, gaussian_sigma=gaussian_sigma)
    flow_dy_full = upsample_flow(flow_dy_block, block, (H, W))
    flow_dx_full = upsample_flow(flow_dx_block, block, (H, W))

    total_dx = dx + flow_dx_full
    total_dy = dy + flow_dy_full

    # --- warp full RGB PL and re-derive luma bands from the warped result ---
    registered_pl_disp = warp_pl_rgb(pl_disp, total_dx, total_dy)
    registered_pl_y = _luma(registered_pl_disp)
    registered_bands = band_split(registered_pl_y)
    corr_after = {name: pearson_corr(noisy_bands[name], registered_bands[name]) for name in noisy_bands}

    mag = np.sqrt(flow_dx_full.astype(np.float64) ** 2 + flow_dy_full.astype(np.float64) ** 2)
    flow_stats = {
        "mean_px": float(mag.mean()),
        "p50_px": float(np.quantile(mag, 0.50)),
        "p95_px": float(np.quantile(mag, 0.95)),
        "p99_px": float(np.quantile(mag, 0.99)),
        "max_px": float(mag.max()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    reg_exr_path = output_dir / f"{scene}_registered_pl.exr"
    reg_preview_path = output_dir / f"{scene}_registered_pl_preview.png"
    flow_vis_path = output_dir / f"{scene}_flow_magnitude.png"
    meta_path = output_dir / f"{scene}_register.json"

    from .detail_guard import write_exr  # local import to avoid unused warnings if not needed

    # Store the registered PL in *linear* space (inverse sRGB OETF) so it is
    # directly comparable/compositable with our own linear pipeline outputs.
    from .flat_chroma_smoother import srgb_to_linear_np

    registered_pl_linear = srgb_to_linear_np(np.clip(registered_pl_disp, 0.0, 1.0))
    write_exr(reg_exr_path, registered_pl_linear)
    Image.fromarray((np.clip(registered_pl_disp, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(reg_preview_path)
    Image.fromarray(flow_visualization(flow_dy_full, flow_dx_full)).save(flow_vis_path)

    elapsed = time.time() - t0
    gate_pass = corr_after["fine"] >= 0.75
    meta = {
        "scene": scene,
        "inputs": {"noisy": str(noisy_path), "pl": str(pl_path)},
        "global_offset": {"dx": dx, "dy": dy},
        "block_match": {"block": block, "radius": radius, "min_corr": min_corr},
        "regularize": {"median_size": median_size, "gaussian_sigma": gaussian_sigma},
        "rejected_block_fraction": rejected_fraction,
        "flow_magnitude_px": flow_stats,
        "band_correlation_whole_image": {"before_global_only": corr_before, "after_registration": corr_after},
        "gate_fine_band_corr_after": corr_after["fine"],
        "gate_threshold": 0.75,
        "gate_pass": bool(gate_pass),
        "outputs": {
            "registered_pl_exr": str(reg_exr_path),
            "registered_pl_preview": str(reg_preview_path),
            "flow_magnitude_png": str(flow_vis_path),
        },
        "elapsed_sec": elapsed,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-rigid registration of PL onto our EXR geometry.")
    parser.add_argument("--scene", required=True, choices=sorted(SCENE_OFFSETS))
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--min-corr", type=float, default=0.20)
    parser.add_argument("--median-size", type=int, default=3)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(ROOT / "runs/phase4_detail/register"))
    args = parser.parse_args()

    run_registration(
        scene=args.scene,
        block=args.block,
        radius=args.radius,
        min_corr=args.min_corr,
        median_size=args.median_size,
        gaussian_sigma=args.gaussian_sigma,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
