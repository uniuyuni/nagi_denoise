"""Prototype flat-region chroma smoothing for real-photo NR diagnostics.

This is not meant as the final pipeline. It is a falsification tool: if direct
flat chroma smoothing cannot reduce the measured/visible color grain without
hurting luma edges, then asking the model to learn the same behavior is the
wrong next move.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .detail_guard import write_exr, write_tiff
from .probe import image_stats, make_preview, read_image


LUMA_SRGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)
LUMA_LINEAR = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# ---------------------------------------------------------------------------
# Phase 5 speed: row-block thread-parallel helpers.
#
# On a 40MP frame, profiling found `smooth_chroma`'s two `gaussian_filter`
# calls and its two sRGB<->linear power-law conversions (`np.power`) plus its
# `sigmoid01` calls (`np.exp`) are genuinely CPU-bound -- benchmarked at
# 2.6-3.9x faster across an 8-thread pool on this M1, because numpy/scipy
# release the GIL for these ops. Simple elementwise arithmetic (add/multiply)
# is memory-bandwidth-bound instead and barely benefits from threading (this
# machine's single core already saturates a large fraction of unified-memory
# bandwidth) -- confirmed by benchmark, which is why only these specific ops
# are parallelized below, not the whole function.
#
# All of this is bit-exact vs the single whole-array call it replaces:
#   - the pow/exp helpers are pure per-pixel elementwise maps, so splitting
#     rows across threads cannot change any individual output value.
#   - the gaussian helpers hand each row-block a real (not reflected) halo of
#     `radius = int(truncate*sigma + 0.5)` rows pulled from the source array
#     on each side (clamped to the true array bounds) -- exactly the pixel
#     neighborhood scipy's separable correlation reads for those rows in a
#     single whole-array call. `mode="reflect"` only ever triggers where the
#     halo clamps to the true top/bottom image edge, i.e. exactly where the
#     unblocked call would also apply it. Verified empirically: max abs diff
#     vs the whole-array call is exactly 0.0 in float32 (see
#     scripts/bench_chroma_speed.py).
# ---------------------------------------------------------------------------

_WORKERS = max(1, min(8, os.cpu_count() or 4))
_POOL: ThreadPoolExecutor | None = None

# Block counts tuned by benchmark on a 7728x5152 (39.8MP) frame on an 8-core
# (4P+4E) M1. More blocks than workers is deliberate -- it reduces per-block
# working-set size (better cache behavior) and smooths out load imbalance
# between the P and E cores; going further gave diminishing/negative returns.
_GAUSS_CHROMA_BLOCKS_PER_CHANNEL = 3  # -> 9 tasks for the 3-channel chroma blur
_GAUSS_DETAIL_BLOCKS = 8  # 1-channel, small sigma -> tiny halo, scales further
_POW_BLOCKS = 16  # linear_to_srgb_np / srgb_to_linear_np
_SIGMOID_BLOCKS = 16  # sigmoid01


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="chroma-pass")
    return _POOL


def _row_blocks(height: int, nblocks: int) -> list[tuple[int, int]]:
    nblocks = max(1, min(int(nblocks), int(height)))
    edges = np.linspace(0, height, nblocks + 1).astype(np.int64)
    blocks = []
    for i in range(nblocks):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi > lo:
            blocks.append((lo, hi))
    return blocks


def _parallel_map(fn, blocks: list) -> None:
    """Run ``fn(*block)`` for each block, in parallel when there is more
    than one. Falls back to a plain loop for 0/1 blocks to avoid pool
    dispatch overhead (and to keep tiny/test-sized images correct and fast).
    """
    if len(blocks) <= 1:
        for b in blocks:
            fn(*b)
        return
    list(_pool().map(lambda b: fn(*b), blocks))


def _parallel_elementwise(func, arr: np.ndarray, nblocks: int, out: np.ndarray | None = None) -> np.ndarray:
    """Row-block-parallel ``out[:] = func(arr)`` for a pure per-pixel
    elementwise map. Bit-exact vs ``func(arr)`` -- every row is independent.
    """
    if out is None:
        out = np.empty_like(arr)
    blocks = _row_blocks(arr.shape[0], nblocks)

    def work(lo, hi):
        out[lo:hi] = func(arr[lo:hi])

    _parallel_map(work, blocks)
    return out


def _gaussian_radius(sigma: float, truncate: float = 4.0) -> int:
    return int(float(truncate) * float(sigma) + 0.5)


def _parallel_gaussian_2d(
    arr2d: np.ndarray, sigma: float, nblocks: int, mode: str = "reflect", truncate: float = 4.0
) -> np.ndarray:
    """Row-block-parallel ``gaussian_filter(arr2d, sigma=sigma, mode=mode)``
    for a single-channel (H, W) array. See module note above for the
    bit-exactness argument.
    """
    h = arr2d.shape[0]
    out = np.empty_like(arr2d)
    radius = _gaussian_radius(sigma, truncate)
    blocks = _row_blocks(h, nblocks)

    def work(lo, hi):
        pad_lo = max(0, lo - radius)
        pad_hi = min(h, hi + radius)
        filtered = gaussian_filter(arr2d[pad_lo:pad_hi], sigma=sigma, mode=mode, truncate=truncate)
        out[lo:hi] = filtered[lo - pad_lo : lo - pad_lo + (hi - lo)]

    _parallel_map(work, blocks)
    return out


def _parallel_gaussian_chroma(
    chroma3d: np.ndarray, sigma: float, nblocks_per_channel: int, mode: str = "reflect", truncate: float = 4.0
) -> np.ndarray:
    """Row+channel-block-parallel version of
    ``gaussian_filter(chroma3d, sigma=(sigma, sigma, 0.0), mode=mode)`` for a
    3-channel (H, W, 3) array. The zero-sigma channel axis makes each channel
    an independent 2D blur, so this dispatches 3x
    ``_parallel_gaussian_2d``'s worth of row-blocks onto the shared pool at
    once. See module note above for the bit-exactness argument.
    """
    h = chroma3d.shape[0]
    out = np.empty_like(chroma3d)
    radius = _gaussian_radius(sigma, truncate)
    tasks = [(c, lo, hi) for c in range(3) for lo, hi in _row_blocks(h, nblocks_per_channel)]

    def work(c, lo, hi):
        pad_lo = max(0, lo - radius)
        pad_hi = min(h, hi + radius)
        filtered = gaussian_filter(chroma3d[pad_lo:pad_hi, :, c], sigma=sigma, mode=mode, truncate=truncate)
        out[lo:hi, :, c] = filtered[lo - pad_lo : lo - pad_lo + (hi - lo)]

    _parallel_map(work, tasks)
    return out


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    x = x[..., :3]
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def linear_to_srgb_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None).astype(np.float32, copy=False)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_to_linear_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def luma(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(_safe_rgb(rgb) * weights.reshape(1, 1, 3), axis=2)


def _luma_presafe(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Same as ``luma`` but skips the ``_safe_rgb`` (nan_to_num) pass.

    Only safe to call on an array that is already known finite -- used
    internally by ``smooth_chroma``, which sanitizes its input exactly once
    up front. ``nan_to_num`` over a full-resolution image is not free (it
    was showing up as ~5s/call on a 40MP frame in profiling); calling it
    again on data that is already finite is pure waste.
    """
    return np.sum(rgb * weights.reshape(1, 1, 3), axis=2)


def sigmoid01(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False)


def smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def flat_gate(
    base_linear: np.ndarray,
    *,
    detail_sigma: float,
    threshold: float,
    transition: float,
    highlight_threshold: float,
    highlight_transition: float,
) -> np.ndarray:
    base_srgb = linear_to_srgb_np(_safe_rgb(base_linear))
    y_display = luma(base_srgb, LUMA_SRGB)
    detail = np.abs(y_display - gaussian_filter(y_display, sigma=float(detail_sigma), mode="reflect"))
    flat = sigmoid01((float(threshold) - detail) / max(float(transition), 1.0e-6))

    y_linear = luma(base_linear, LUMA_LINEAR)
    highlight = sigmoid01(
        (y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6)
    )
    return (flat * (1.0 - highlight)).astype(np.float32, copy=False)


def smooth_chroma(
    base_linear: np.ndarray,
    *,
    strength: float,
    chroma_sigma: float,
    detail_sigma: float,
    threshold: float,
    transition: float,
    highlight_threshold: float,
    highlight_transition: float,
    hdr_restore_peak_threshold: float,
    hdr_restore_threshold: float,
    hdr_restore_transition: float,
    compute_stats: bool = True,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Phase 5 note: this is numerically IDENTICAL to the original
    implementation (same op-for-op math), just with the redundant work
    removed. Profiling on a 40MP frame found the two gaussian_filter calls
    -- the actual "work" -- cost ~3s of a ~22-29s total; the rest was
    re-sanitizing (nan_to_num) and re-deriving (linear<->sRGB, luma) the
    same full-resolution arrays multiple times across this function and the
    (now inlined) ``flat_gate`` helper. Each quantity below is computed once
    and reused. ``compute_stats=False`` additionally skips the diagnostic
    percentile stats, which every production/eval caller discards anyway.
    """
    base = _safe_rgb(base_linear)  # sanitize ONCE; everything below is finite.

    # `flat_gate`'s internal `base_srgb` is deliberately *unclipped* (it can
    # exceed 1.0 on HDR highlights) -- that is what its `detail`/`y_display`
    # heuristic is computed from. `smooth_chroma`'s own chroma split instead
    # uses the *clipped* [0,1] sRGB. Both need computing; do the expensive
    # asinh-free power-law conversion once and derive the clipped copy from
    # it with a cheap clip, instead of calling `linear_to_srgb_np` twice on
    # the same `base`.
    # linear_to_srgb_np / srgb_to_linear_np are dominated by np.power, which
    # is CPU-bound and thread-parallelizes almost linearly (see module note
    # above) -- threading these two calls alone accounts for ~2s of the
    # Phase 5 speedup on a 40MP frame.
    base_srgb_full = _parallel_elementwise(linear_to_srgb_np, base, nblocks=_POW_BLOCKS)
    base_srgb = np.clip(base_srgb_full, 0.0, 1.0)

    y = _luma_presafe(base_srgb, LUMA_SRGB)[..., None]
    y_display = _luma_presafe(base_srgb_full, LUMA_SRGB)  # matches flat_gate's y_display exactly
    chroma = base_srgb - y
    # gaussian_filter is likewise CPU-bound (separable correlation), not
    # memory-bandwidth-bound -- row+channel-block threading is bit-exact
    # here (see module note) and is the single largest Phase 5 win.
    low_chroma = _parallel_gaussian_chroma(
        chroma, float(chroma_sigma), nblocks_per_channel=_GAUSS_CHROMA_BLOCKS_PER_CHANNEL, mode="reflect"
    )

    y_linear = _luma_presafe(base, LUMA_LINEAR)  # used by both the gate's highlight term and HDR restore below

    detail_blur = _parallel_gaussian_2d(y_display, float(detail_sigma), nblocks=_GAUSS_DETAIL_BLOCKS, mode="reflect")
    detail = np.abs(y_display - detail_blur)
    flat = _parallel_elementwise(
        sigmoid01, (float(threshold) - detail) / max(float(transition), 1.0e-6), nblocks=_SIGMOID_BLOCKS
    )
    highlight = _parallel_elementwise(
        sigmoid01,
        (y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6),
        nblocks=_SIGMOID_BLOCKS,
    )
    gate = (flat * (1.0 - highlight)).astype(np.float32, copy=False)

    blend = np.clip(gate * float(strength), 0.0, 1.0)[..., None]
    out_srgb = y + chroma * (1.0 - blend) + low_chroma * blend
    out = _parallel_elementwise(srgb_to_linear_np, out_srgb, nblocks=_POW_BLOCKS)
    peak_linear = np.max(base, axis=2)
    hdr_signal = np.maximum(
        y_linear - float(hdr_restore_threshold),
        peak_linear - float(hdr_restore_peak_threshold),
    )
    hdr_restore = smoothstep(
        hdr_signal / max(float(hdr_restore_transition), 1.0e-6)
    )
    out = out * (1.0 - hdr_restore[..., None]) + base * hdr_restore[..., None]
    if compute_stats:
        stats = {
            "strength": float(strength),
            "chroma_sigma": float(chroma_sigma),
            "detail_sigma": float(detail_sigma),
            "threshold": float(threshold),
            "transition": float(transition),
            "highlight_threshold": float(highlight_threshold),
            "highlight_transition": float(highlight_transition),
            "hdr_restore_peak_threshold": float(hdr_restore_peak_threshold),
            "gate_mean": float(np.mean(gate)),
            "gate_p50": float(np.quantile(gate, 0.50)),
            "gate_p90": float(np.quantile(gate, 0.90)),
            "gate_p99": float(np.quantile(gate, 0.99)),
            "hdr_restore_mean": float(np.mean(hdr_restore)),
            "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
        }
    else:
        stats = {
            "strength": float(strength),
            "chroma_sigma": float(chroma_sigma),
            "detail_sigma": float(detail_sigma),
            "threshold": float(threshold),
            "transition": float(transition),
            "highlight_threshold": float(highlight_threshold),
            "highlight_transition": float(highlight_transition),
            "hdr_restore_peak_threshold": float(hdr_restore_peak_threshold),
        }
    return out.astype(np.float32, copy=False), stats, gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply diagnostic flat chroma smoothing.")
    parser.add_argument("--input", required=True, help="Denoised linear EXR/TIFF input.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--chroma-sigma", type=float, default=1.4)
    parser.add_argument("--detail-sigma", type=float, default=1.2)
    parser.add_argument("--threshold", type=float, default=0.010)
    parser.add_argument("--transition", type=float, default=0.006)
    parser.add_argument("--highlight-threshold", type=float, default=1.0)
    parser.add_argument("--highlight-transition", type=float, default=0.2)
    parser.add_argument("--hdr-restore-peak-threshold", type=float, default=0.95)
    parser.add_argument("--hdr-restore-threshold", type=float, default=0.85)
    parser.add_argument("--hdr-restore-transition", type=float, default=0.25)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_flat_chroma_smooth"

    base = read_image(input_path)
    out, stats, gate = smooth_chroma(
        base,
        strength=args.strength,
        chroma_sigma=args.chroma_sigma,
        detail_sigma=args.detail_sigma,
        threshold=args.threshold,
        transition=args.transition,
        highlight_threshold=args.highlight_threshold,
        highlight_transition=args.highlight_transition,
        hdr_restore_peak_threshold=args.hdr_restore_peak_threshold,
        hdr_restore_threshold=args.hdr_restore_threshold,
        hdr_restore_transition=args.hdr_restore_transition,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}_meta.json"

    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)

    meta = {
        "input": str(input_path),
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
        },
        "smoother": stats,
        "input_stats": image_stats(base),
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
