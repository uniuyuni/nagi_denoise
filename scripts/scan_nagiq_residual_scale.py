"""Scan output residual scales for a NagiQ checkpoint on SIDD validation patches."""
from __future__ import annotations

import argparse

import numpy as np
import scipy.io as sio
import torch

from nagi_nr.devices import resolve_device
from nagi_nr.nagiq import NagiQ
from nagi_nr_bench.eval_sidd_val import psnr_srgb


def load_model(weights: str, device: torch.device) -> NagiQ:
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = dict(cfg.get("model", {}))
    model_cfg.pop("preset", None)
    model = NagiQ(**model_cfg)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    return model


def load_validation(noisy_mat: str, gt_mat: str) -> tuple[np.ndarray, np.ndarray]:
    noisy = sio.loadmat(noisy_mat)
    nkey = next(k for k in noisy if not k.startswith("__"))
    gt = sio.loadmat(gt_mat)
    gkey = next(k for k in gt if not k.startswith("__"))
    return noisy[nkey], gt[gkey]


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description="Scan NagiQ output residual scales.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--max-patches", type=int, default=16)
    ap.add_argument("--scales", default="0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,1")
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    args = ap.parse_args()

    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    device = resolve_device(args.device)
    model = load_model(args.weights, device)
    noisy, gt = load_validation(args.noisy_mat, args.gt_mat)
    scores = {scale: [] for scale in scales}
    noisy_scores = []
    done = 0
    for i in range(noisy.shape[0]):
        for j in range(noisy.shape[1]):
            if done >= args.max_patches:
                break
            n_patch = noisy[i, j]
            g_patch = gt[i, j]
            x = torch.from_numpy(n_patch).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            x = x.to(device=device, dtype=torch.float32)
            y = model(x)
            residual = y - x
            noisy_scores.append(psnr_srgb(n_patch, g_patch))
            for scale in scales:
                out = (x + float(scale) * residual).clamp(0, 1)
                out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                out_u8 = (out_np * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
                scores[scale].append(psnr_srgb(out_u8, g_patch))
            done += 1
        if done >= args.max_patches:
            break

    print(f"patches={done} noisy={np.mean(noisy_scores):.3f}")
    for scale in scales:
        print(f"scale={scale:g} psnr={np.mean(scores[scale]):.3f}")


if __name__ == "__main__":
    main()
