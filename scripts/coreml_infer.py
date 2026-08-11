"""Shared Core ML tiled-inference wrapper for Phase 5 Stage B benchmarking/gates.

Mirrors `nagi_denoise.infer.Denoiser._tiled_forward` (Hann-window tile
stitching) exactly, but calls an exported Core ML `.mlpackage`
(`scripts/export_coreml_nagi_v2.py`, fixed 768x768 batch-1 tile) instead of
the PyTorch model for each tile. Tile coordinate generation, stride, and the
Hann window itself are the *same code path* (imported from
`nagi_denoise.infer`) so the stitching geometry cannot drift from
production.

The exported Core ML graph has the in-graph HDR highlight guard permanently
disarmed (`highlight_protect_strength=0.0` at export time, see the export
script's docstring for the proof that a pointwise post-hoc guard on the
stitched image is exactly equivalent to a per-tile in-graph guard when the
threshold is shared across all tiles, which production always does). This
module reproduces that guard as an explicit, separate step
(`apply_highlight_guard_np`) so callers can opt into the exact same
`nagi_denoise.pipeline.denoise.denoise()` semantics.

Only fixed batch=1 tiles are supported (Core ML export is a static graph),
matching the Phase 5 task's speed-benchmark spec (batch 1, tile 768).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import coremltools as ct  # noqa: E402

from nagi_denoise.infer import _cosine_window2d, _pad_to_multiple  # noqa: E402


class CoreMLTiledDenoiser:
    """Tiled Core ML inference matching `Denoiser._tiled_forward`'s geometry."""

    def __init__(self, mlpackage_path: str, tile: int, compute_units: "ct.ComputeUnit", size_multiple: int = 8):
        self.tile = int(tile)
        self.size_multiple = int(size_multiple)
        self.compute_units = compute_units
        self.model = ct.models.MLModel(str(mlpackage_path), compute_units=compute_units)
        # Warm up (first Core ML call pays a one-off graph-compile/load cost
        # that must not be attributed to per-tile timing).
        dummy = np.zeros((1, 3, self.tile, self.tile), dtype=np.float32)
        self.model.predict({"tile_in": dummy})

    def predict_tile(self, tile_chw: np.ndarray) -> np.ndarray:
        """tile_chw: (3, tile, tile) float32 -> (3, tile, tile) float32."""
        x = tile_chw[None].astype(np.float32, copy=False)
        y = self.model.predict({"tile_in": x})["tile_out"]
        return y[0]

    def tiled_forward(
        self,
        img_chw: torch.Tensor,
        overlap: int,
        per_tile_times: Optional[list] = None,
    ) -> torch.Tensor:
        """img_chw: (3, H, W) float32 torch tensor (CPU). Returns (3, H, W) float32 CPU tensor.

        Structurally identical to `Denoiser._tiled_forward` for batch_size=1:
        same coordinate grid, same stride, same Hann window, same
        weighted-accumulate-then-normalize. Only the per-tile forward call is
        swapped for a Core ML `predict`.
        """
        C, H, W = img_chw.shape
        tile = self.tile
        if tile % self.size_multiple != 0:
            tile = (tile // self.size_multiple) * self.size_multiple
        stride = max(1, tile - overlap)

        ys = list(range(0, max(H - tile, 0) + 1, stride))
        xs = list(range(0, max(W - tile, 0) + 1, stride))
        if not ys or ys[-1] + tile < H:
            ys.append(max(0, H - tile))
        if not xs or xs[-1] + tile < W:
            xs.append(max(0, W - tile))

        out = torch.zeros_like(img_chw)
        weight = torch.zeros((1, H, W), dtype=img_chw.dtype)

        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for ty in ys:
            th = min(tile, H - ty)
            for tx in xs:
                tw = min(tile, W - tx)
                groups.setdefault((th, tw), []).append((ty, tx))

        win_cache: dict[tuple[int, int], torch.Tensor] = {}

        for (th, tw), coords in groups.items():
            win = win_cache.get((th, tw))
            if win is None:
                win = _cosine_window2d(th, tw, device="cpu", dtype=img_chw.dtype)[0]  # (1, th, tw)
                win_cache[(th, tw)] = win

            for (ty, tx) in coords:
                patch = img_chw[..., ty : ty + th, tx : tx + tw]
                padded, _ = _pad_to_multiple(patch.unsqueeze(0), self.size_multiple)
                padded_np = padded[0].numpy()

                t0 = time.perf_counter()
                y_np = self.predict_tile(padded_np)
                dt = time.perf_counter() - t0
                if per_tile_times is not None:
                    per_tile_times.append(dt)

                y = torch.from_numpy(y_np)[..., :th, :tw].float()
                out[..., ty : ty + th, tx : tx + tw] += y * win
                weight[..., ty : ty + th, tx : tx + tw] += win

        return out / weight.clamp_min(1e-8)

    def denoise_array(self, img_hwc: np.ndarray, overlap: int, per_tile_times: Optional[list] = None) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(img_hwc[..., :3])).permute(2, 0, 1).float()
        H, W = t.shape[-2:]
        if H > self.tile or W > self.tile:
            out = self.tiled_forward(t, overlap=overlap, per_tile_times=per_tile_times)
        else:
            padded, _ = _pad_to_multiple(t.unsqueeze(0), self.size_multiple)
            t0 = time.perf_counter()
            y_np = self.predict_tile(padded[0].numpy())
            dt = time.perf_counter() - t0
            if per_tile_times is not None:
                per_tile_times.append(dt)
            out = torch.from_numpy(y_np)[..., :H, :W].float()
        return np.ascontiguousarray(out.permute(1, 2, 0).numpy().astype(np.float32, copy=False))


def linear_luma_np(x: np.ndarray) -> np.ndarray:
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def apply_highlight_guard_np(denoised: np.ndarray, inp: np.ndarray, threshold: float, transition: float, strength: float) -> np.ndarray:
    """Numpy re-implementation of `nagi_v2.input_highlight_guard`, applied
    once on the assembled (post-stitch) image -- see module docstring for
    why this is exactly equivalent to the in-graph per-tile guard when the
    threshold/transition/strength are shared across all tiles of an image,
    which `nagi_denoise.pipeline.denoise.denoise()` always does."""
    if strength <= 0.0:
        return denoised
    inp_nonneg = np.clip(inp, 0.0, None)
    y = linear_luma_np(inp_nonneg)
    if transition <= 0.0:
        mask = (y > threshold).astype(np.float32)
    else:
        mask = 1.0 / (1.0 + np.exp(-(y - threshold) / max(transition, 1e-6)))
    mask = (mask * min(max(strength, 0.0), 1.0))[..., None]
    return denoised * (1.0 - mask) + inp_nonneg * mask
