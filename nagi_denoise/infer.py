"""Inference API for Nagi Denoise.

Supports arbitrary input sizes via reflection padding and overlap-tiled forward
with a cosine window to avoid seam artifacts. Works with both NagiNR and NagiV2
checkpoints; the model architecture is auto-detected from the checkpoint config.
"""
from __future__ import annotations
import contextlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .models.nagi_nr import NagiNR
from .models.nagi_v2 import NagiV2, build_nagi_v2
from .transforms import srgb_to_linear, linear_to_srgb
from .devices import resolve_device


def _size_multiple(model: torch.nn.Module) -> int:
    return int(getattr(model, "SIZE_MULTIPLE", None) or getattr(model, "size_multiple", 8))


def _pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, tuple[int, int]]:
    H, W = x.shape[-2:]
    pH = (m - H % m) % m
    pW = (m - W % m) % m
    if pH == 0 and pW == 0:
        return x, (0, 0)
    return F.pad(x, (0, pW, 0, pH), mode="reflect"), (pH, pW)


def _autocast_ctx(device: torch.device, amp_dtype: Optional[torch.dtype]):
    """Return an autocast context manager for `amp_dtype` on `device`, or a
    no-op context when `amp_dtype` is None (the default, full fp32 path).

    Model weights stay fp32; only intermediate math (convs, norms) runs in
    the lower-precision dtype where the backend allows it. This is the
    "quality-lossless" speed lever from Phase 5 Stage A -- it must be opted
    into explicitly by the caller, production defaults are unaffected.
    """
    if amp_dtype is None:
        return contextlib.nullcontext()
    device_type = device.type if device.type in ("mps", "cuda", "cpu") else "cpu"
    return torch.autocast(device_type=device_type, dtype=amp_dtype)


def _cosine_window2d(h: int, w: int, device, dtype) -> torch.Tensor:
    wh = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    ww = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    win = (wh[:, None] * ww[None, :]).clamp_min_(1e-4)
    return win.view(1, 1, h, w)


_NAGI_V2_MARKER_KEYS = (
    "enc_blk_nums",
    "dec_blk_nums",
    "middle_blk_num",
    "detail_scale",
    "compressed_output_clamp",
    "chroma_branch",
    "chroma_smooth_branch",
    "chroma_cleanup_branch",
    "luma_smooth_branch",
)


def _detect_arch(model_cfg: dict) -> str:
    """Infer which model architecture a checkpoint's config dict describes.

    NagiV2 trainers stamp an explicit ``arch: "nagi_v2"`` field into the saved
    config. Older NagiPerfect-era checkpoints instead carry a
    ``kind: "nagiperfect"`` marker or NagiV2-specific constructor keys. Absent
    all of that, default to NagiNR for backward compatibility.
    """
    arch = model_cfg.get("arch")
    if arch:
        return str(arch)
    kind = str(model_cfg.get("kind") or "")
    if kind in ("nagi_v2", "nagiperfect"):
        return "nagi_v2"
    preset = str(model_cfg.get("preset") or "")
    if preset.startswith("v2-") or preset.startswith("perfect-"):
        return "nagi_v2"
    if any(key in model_cfg for key in _NAGI_V2_MARKER_KEYS):
        return "nagi_v2"
    return "nagi_nr"


class Denoiser:
    """High-level inference wrapper around a trained NagiV2 (or legacy NagiNR) model.

    This is the *model stage only*: tiled forward with Hann-window blending. It
    does not run the chroma cleanup pass, the highlight guard, or
    ``input_blend``. For production output call
    ``nagi_denoise.denoise()`` instead.

    Example:
        dn = Denoiser.load("runs/nagi_v2_l_ft2/nagi_v2_l_ft2_final.pt", device="mps")
        out = dn(img_linear, input_space="linear")   # img_linear: (3,H,W) float32
    """

    def __init__(self, model: Union[NagiNR, NagiV2], device: Union[str, torch.device] = "auto"):
        self.device = resolve_device(device)
        self.model = model.to(self.device, dtype=torch.float32).eval()
        self.size_multiple = _size_multiple(self.model)

    # ---- Loaders ----
    @classmethod
    def load(
        cls,
        weights_path: str | Path,
        device: Union[str, torch.device] = "auto",
        strict: bool = True,
    ) -> "Denoiser":
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        model_cfg = (ckpt.get("config") or {}).get("model", {}) if isinstance(ckpt, dict) else {}
        arch = _detect_arch(model_cfg)
        if arch == "nagi_v2":
            model = build_nagi_v2(model_cfg)
        else:
            nagi_nr_cfg = {k: v for k, v in model_cfg.items() if k not in ("arch", "kind", "preset")}
            model = NagiNR(**nagi_nr_cfg)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=strict)
        return cls(model, device=device)

    # ---- Forward ----
    @torch.inference_mode()
    def __call__(
        self,
        img: torch.Tensor,
        input_space: str = "linear",
        tile: int = 512,
        overlap: int = 32,
        batch_size: int = 1,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Denoise an image.

        Args:
            img: (3, H, W) or (B, 3, H, W) float32.
            input_space: "linear" (default) or "srgb". sRGB is converted in/out.
            tile: tile size (square). Set to 0 or None to disable tiling.
            overlap: pixels of overlap between adjacent tiles.
            batch_size: number of tiles forwarded together in one model call
                during tiled inference (>=1). Ignored on the non-tiled path.
            amp_dtype: if set (e.g. ``torch.float16``), runs the model forward
                under ``torch.autocast`` at this dtype. Weights stay fp32;
                only intermediate math is lower precision. ``None`` (default)
                keeps the exact fp32 path used by every existing caller.

        Returns: tensor on CPU with the same shape and color space as input.
        """
        if img.dtype != torch.float32:
            img = img.float()
        added_batch = False
        if img.ndim == 3:
            img = img.unsqueeze(0)
            added_batch = True
        elif img.ndim != 4:
            raise ValueError(f"Expected (3,H,W) or (B,3,H,W), got {tuple(img.shape)}")

        if input_space == "srgb":
            img = srgb_to_linear(img)
        elif input_space != "linear":
            raise ValueError("input_space must be 'linear' or 'srgb'")

        img = img.to(self.device, dtype=torch.float32, non_blocking=True)

        B, C, H, W = img.shape
        if tile and (H > tile or W > tile):
            out = self._tiled_forward(
                img, tile=int(tile), overlap=int(overlap),
                batch_size=int(batch_size), amp_dtype=amp_dtype,
            )
        else:
            x, (pH, pW) = _pad_to_multiple(img, self.size_multiple)
            with _autocast_ctx(self.device, amp_dtype):
                y = self.model(x)
            out = y[..., :H, :W].float()

        if input_space == "srgb":
            out = linear_to_srgb(out.clamp_(0.0, 1.0))

        out = out.detach().to("cpu")
        if added_batch:
            out = out[0]
        return out

    # ---- Tiling ----
    def _tiled_forward(
        self,
        img: torch.Tensor,
        tile: int,
        overlap: int,
        batch_size: int = 1,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        B, C, H, W = img.shape
        if tile % self.size_multiple != 0:
            tile = (tile // self.size_multiple) * self.size_multiple
        stride = max(1, tile - overlap)

        ys = list(range(0, max(H - tile, 0) + 1, stride))
        xs = list(range(0, max(W - tile, 0) + 1, stride))
        if not ys or ys[-1] + tile < H:
            ys.append(max(0, H - tile))
        if not xs or xs[-1] + tile < W:
            xs.append(max(0, W - tile))

        out = torch.zeros_like(img)
        weight = torch.zeros((1, 1, H, W), device=img.device, dtype=img.dtype)

        # In the common case (both H and W >= tile, true for every
        # full-resolution photo this pipeline runs on) every tile is exactly
        # `tile x tile`: the last row/col start is pulled back to
        # `length - tile`, never clipped. That makes every patch the same
        # shape, so they can be stacked into one batched forward call. The
        # rarer case (image thinner than `tile` on one axis) still produces
        # uniform shapes *within* each row or column of the grid, so grouping
        # by (th, tw) keeps batching correct there too, just with a smaller
        # effective batch.
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for ty in ys:
            th = min(tile, H - ty)
            for tx in xs:
                tw = min(tile, W - tx)
                groups.setdefault((th, tw), []).append((ty, tx))

        bs = max(1, int(batch_size))
        win_cache: dict[tuple[int, int], torch.Tensor] = {}

        for (th, tw), coords in groups.items():
            win = win_cache.get((th, tw))
            if win is None:
                win = _cosine_window2d(th, tw, device=img.device, dtype=img.dtype)
                win_cache[(th, tw)] = win

            for start in range(0, len(coords), bs):
                chunk = coords[start : start + bs]
                patches = [img[..., ty : ty + th, tx : tx + tw] for ty, tx in chunk]
                batch = torch.cat(patches, dim=0) if len(patches) > 1 else patches[0]
                padded, _ = _pad_to_multiple(batch, self.size_multiple)
                with _autocast_ctx(self.device, amp_dtype):
                    y = self.model(padded)
                y = y[..., :th, :tw].float()

                for i, (ty, tx) in enumerate(chunk):
                    out[..., ty : ty + th, tx : tx + tw] += y[i : i + 1] * win
                    weight[..., ty : ty + th, tx : tx + tw] += win

        return out / weight.clamp_min(1e-8)

    # ---- Numpy convenience ----
    def denoise_array(
        self,
        img: np.ndarray,
        input_space: str = "linear",
        tile: int = 512,
        overlap: int = 64,
        batch_size: int = 1,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> np.ndarray:
        """Denoise a float32 HWC RGB numpy array using the tiled forward path.

        Args:
            img: (H, W, 3) float32 array.
            input_space: "linear" (default) or "srgb".
            tile: tile size (square). Set to 0 or None to disable tiling.
            overlap: pixels of overlap between adjacent tiles.
            batch_size: tiles per batched forward call (Phase 5 speed lever).
            amp_dtype: optional autocast dtype (Phase 5 speed lever), e.g.
                ``torch.float16``. ``None`` keeps the exact fp32 path.

        Returns: (H, W, 3) float32 numpy array in the same color space as input.
        """
        if img.ndim != 3 or img.shape[-1] < 3:
            raise ValueError(f"Expected (H, W, 3) array, got shape {img.shape}")
        t = torch.from_numpy(np.ascontiguousarray(img[..., :3])).permute(2, 0, 1).float()
        out = self(
            t, input_space=input_space, tile=tile, overlap=overlap,
            batch_size=batch_size, amp_dtype=amp_dtype,
        )
        return np.ascontiguousarray(out.permute(1, 2, 0).numpy().astype(np.float32, copy=False))
