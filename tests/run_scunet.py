"""Tiled SCUNet inference on a single image."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Resolve package path without requiring an editable install already active
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "nagi_nr_bench" / "src"))

from nagi_nr_bench.third_party.scunet import SCUNet

# ---- config ----
WEIGHTS = REPO / "benchmarks/scunet/scunet_color_real_psnr.pth"
INPUT   = REPO / "tests/sample.jpg"
OUTPUT  = REPO / "tests/scunet.jpg"
DEVICE  = "mps"
TILE    = 512      # tile size (px)
OVERLAP = 64       # overlap between tiles (px); SCUNet has larger receptive field


def _pad(x: torch.Tensor, m: int = 8):
    H, W = x.shape[-2:]
    pH = (m - H % m) % m
    pW = (m - W % m) % m
    if pH or pW:
        x = F.pad(x, (0, pW, 0, pH), mode="reflect")
    return x, pH, pW


def _hann2d(h, w, device, dtype):
    wh = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    ww = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    return (wh[:, None] * ww[None, :]).clamp_min(1e-4).view(1, 1, h, w)


@torch.inference_mode()
def tiled_forward(model, img_t: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    """img_t: (1, 3, H, W) float32 [0,1] on device."""
    _, _, H, W = img_t.shape
    stride = max(1, tile - overlap)

    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    if not xs or xs[-1] + tile < W:
        xs.append(max(0, W - tile))

    out    = torch.zeros_like(img_t)
    weight = torch.zeros(1, 1, H, W, device=img_t.device, dtype=img_t.dtype)

    total = len(ys) * len(xs)
    done  = 0
    for ty in ys:
        for tx in xs:
            th = min(tile, H - ty)
            tw = min(tile, W - tx)
            patch = img_t[..., ty:ty+th, tx:tx+tw]
            padded, pH, pW = _pad(patch, m=8)
            y = model(padded)[..., :th, :tw]
            win = _hann2d(th, tw, device=img_t.device, dtype=img_t.dtype)
            out   [..., ty:ty+th, tx:tx+tw] += y * win
            weight[..., ty:ty+th, tx:tx+tw] += win
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] tiles processed", flush=True)

    return out / weight.clamp_min(1e-8)


def main():
    device = torch.device(DEVICE)

    print("loading model …")
    model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    state = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SCUNet: {n_params:.2f}M params")

    print(f"loading {INPUT} …")
    img_pil = Image.open(INPUT).convert("RGB")
    img_np  = np.array(img_pil).astype(np.float32) / 255.0          # (H,W,3)
    img_t   = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    H, W    = img_t.shape[-2:]
    print(f"image size: {W} x {H}")

    print(f"running tiled inference (tile={TILE}, overlap={OVERLAP}) …")
    out_t = tiled_forward(model, img_t, tile=TILE, overlap=OVERLAP)

    out_np = (out_t.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(out_np).save(OUTPUT, quality=95)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
