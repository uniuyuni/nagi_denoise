"""Inference API for Nagi NR.

Supports arbitrary input sizes via reflection padding and overlap-tiled forward
with a cosine window to avoid seam artifacts.
"""
from __future__ import annotations
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F

from .model import NagiNR
from .transforms import srgb_to_linear, linear_to_srgb
from .devices import resolve_device


def _pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, tuple[int, int]]:
    H, W = x.shape[-2:]
    pH = (m - H % m) % m
    pW = (m - W % m) % m
    if pH == 0 and pW == 0:
        return x, (0, 0)
    return F.pad(x, (0, pW, 0, pH), mode="reflect"), (pH, pW)


def _cosine_window2d(h: int, w: int, device, dtype) -> torch.Tensor:
    wh = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    ww = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    win = (wh[:, None] * ww[None, :]).clamp_min_(1e-4)
    return win.view(1, 1, h, w)


class Denoiser:
    """High-level inference wrapper around a trained NagiNR model.

    Example:
        dn = Denoiser.load("runs/nagi_nr_s/nagi_nr_s_final.pt", device="mps")
        out = dn(img_linear, input_space="linear")   # img_linear: (3,H,W) float32
    """

    def __init__(self, model: NagiNR, device: Union[str, torch.device] = "auto"):
        self.device = resolve_device(device)
        self.model = model.to(self.device, dtype=torch.float32).eval()

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
        model = NagiNR(**model_cfg)
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
    ) -> torch.Tensor:
        """Denoise an image.

        Args:
            img: (3, H, W) or (B, 3, H, W) float32.
            input_space: "linear" (default) or "srgb". sRGB is converted in/out.
            tile: tile size (square). Set to 0 or None to disable tiling.
            overlap: pixels of overlap between adjacent tiles.

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
            out = self._tiled_forward(img, tile=int(tile), overlap=int(overlap))
        else:
            x, (pH, pW) = _pad_to_multiple(img, NagiNR.SIZE_MULTIPLE)
            y = self.model(x)
            out = y[..., :H, :W]

        if input_space == "srgb":
            out = linear_to_srgb(out.clamp_(0.0, 1.0))

        out = out.detach().to("cpu")
        if added_batch:
            out = out[0]
        return out

    # ---- Tiling ----
    def _tiled_forward(self, img: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
        B, C, H, W = img.shape
        if tile % NagiNR.SIZE_MULTIPLE != 0:
            tile = (tile // NagiNR.SIZE_MULTIPLE) * NagiNR.SIZE_MULTIPLE
        stride = max(1, tile - overlap)

        ys = list(range(0, max(H - tile, 0) + 1, stride))
        xs = list(range(0, max(W - tile, 0) + 1, stride))
        if not ys or ys[-1] + tile < H:
            ys.append(max(0, H - tile))
        if not xs or xs[-1] + tile < W:
            xs.append(max(0, W - tile))

        out = torch.zeros_like(img)
        weight = torch.zeros((1, 1, H, W), device=img.device, dtype=img.dtype)

        for ty in ys:
            for tx in xs:
                th = min(tile, H - ty)
                tw = min(tile, W - tx)
                patch = img[..., ty : ty + th, tx : tx + tw]
                padded, _ = _pad_to_multiple(patch, NagiNR.SIZE_MULTIPLE)
                y = self.model(padded)[..., :th, :tw]
                win = _cosine_window2d(th, tw, device=img.device, dtype=img.dtype)
                out[..., ty : ty + th, tx : tx + tw] += y * win
                weight[..., ty : ty + th, tx : tx + tw] += win

        return out / weight.clamp_min(1e-8)
