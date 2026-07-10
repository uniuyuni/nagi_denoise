"""Pre-compute NAFNet teacher denoised outputs for SIDD Medium pairs.

For each <scene>/<id>_NOISY_SRGB_*.PNG, runs tiled NAFNet inference and writes
<scene>/<id>_TEACHER_SRGB_*.PNG (sRGB uint8) next to it. Phase 3 distillation
then loads these as soft targets.

One-time cost ~1.5-2h on MPS, ~8GB on disk for the full 320-image set.

Usage:
    python3 -m nagi_denoise.pipeline.precompute_teacher \
        --sidd-root SIDD_Medium_Srgb \
        --weights benchmarks/nafnet/NAFNet-SIDD-width64.pth \
        --device mps
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..data import find_sidd_pairs
from ..devices import resolve_device


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
    _, _, H, W = img_t.shape
    stride = max(1, tile - overlap)
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    if not xs or xs[-1] + tile < W:
        xs.append(max(0, W - tile))

    out = torch.zeros_like(img_t)
    weight = torch.zeros(1, 1, H, W, device=img_t.device, dtype=img_t.dtype)
    for ty in ys:
        for tx in xs:
            th = min(tile, H - ty)
            tw = min(tile, W - tx)
            patch = img_t[..., ty:ty + th, tx:tx + tw]
            padded, _, _ = _pad(patch, m=8)
            y = model(padded)[..., :th, :tw]
            win = _hann2d(th, tw, device=img_t.device, dtype=img_t.dtype)
            out[..., ty:ty + th, tx:tx + tw] += y * win
            weight[..., ty:ty + th, tx:tx + tw] += win
    return out / weight.clamp_min(1e-8)


def teacher_path_for(noisy_path: str, out_root: str | None = None) -> Path:
    """Output path for a teacher PNG.

    Default: next to the noisy file, NOISY -> TEACHER (what the training-time
    loader expects). With ``out_root``, mirrors the ``<scene>/<name>`` layout
    under that root instead (pass the same path as ``data.teacher_root`` in the
    training config).
    """
    p = Path(noisy_path)
    name = p.name.replace("NOISY", "TEACHER")
    if out_root:
        return Path(out_root) / p.parent.name / name
    return p.with_name(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidd-root", required=True)
    ap.add_argument("--weights", default="benchmarks/nafnet/NAFNet-SIDD-width64.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true", help="recompute even if output exists")
    ap.add_argument("--limit", type=int, default=0, help="process only first N pairs (0 = all)")
    ap.add_argument(
        "--out-root",
        default=None,
        help="write TEACHER PNGs under this root (mirroring <scene>/<name>) instead of "
        "next to the noisy files; match data.teacher_root in the training config",
    )
    args = ap.parse_args()

    from ..bench.third_party.nafnet import NAFNet

    device = resolve_device(args.device)
    print("loading NAFNet-width64 teacher ...")
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    state = ckpt.get("params", ckpt.get("params_ema", ckpt))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"teacher params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    pairs = find_sidd_pairs(args.sidd_root)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"SIDD pairs: {len(pairs)}")

    t_start = time.time()
    n_done = 0
    n_skip = 0
    for i, (noisy_path, _gt_path) in enumerate(pairs):
        out_path = teacher_path_for(noisy_path, args.out_root)
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img_np = np.array(Image.open(noisy_path).convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        t0 = time.time()
        out_t = tiled_forward(model, img_t, tile=args.tile, overlap=args.overlap)
        out_np = (out_t.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255 + 0.5).astype(np.uint8)
        Image.fromarray(out_np).save(out_path)
        n_done += 1

        dt = time.time() - t0
        elapsed = time.time() - t_start
        eta = elapsed / n_done * (len(pairs) - n_skip - n_done) if n_done else 0
        print(f"[{i+1:3d}/{len(pairs)}] {out_path.name}  ({dt:.1f}s, "
              f"elapsed {elapsed/60:.1f}min, eta {eta/60:.1f}min)", flush=True)

    print(f"done. computed {n_done}, skipped {n_skip} (existing). "
          f"total {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
