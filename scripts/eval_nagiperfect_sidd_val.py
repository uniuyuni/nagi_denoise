"""Evaluate a NagiPerfect checkpoint on SIDD validation blocks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "nagi_nr" / "src"))

from nagi_nr.transforms import linear_to_srgb, srgb_to_linear
from nagi_nr_bench.eval_sidd_val import psnr_srgb
from denoise_exr_nagiperfect import load_model


def _mat_array(path: str) -> np.ndarray:
    data = sio.loadmat(path)
    key = next(k for k in data if not k.startswith("__"))
    return data[key]


def iter_patches(noisy: np.ndarray, gt: np.ndarray, max_patches: int):
    done = 0
    for i in range(noisy.shape[0]):
        for j in range(noisy.shape[1]):
            if max_patches > 0 and done >= max_patches:
                return
            yield noisy[i, j], gt[i, j]
            done += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NagiPerfect on SIDD val blocks.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    parser.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--state-key", default="state_dict")
    parser.add_argument("--max-patches", type=int, default=32)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--highlight-protect-threshold", type=float, default=None)
    parser.add_argument("--highlight-protect-transition", type=float, default=None)
    parser.add_argument("--highlight-protect-strength", type=float, default=None)
    parser.add_argument("--chroma-smooth-strength", type=float, default=None)
    parser.add_argument("--luma-smooth-strength", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(Path(args.weights), device=device, state_key=args.state_key)
    if args.highlight_protect_threshold is not None:
        model.highlight_protect_threshold = float(args.highlight_protect_threshold)
    if args.highlight_protect_transition is not None:
        model.highlight_protect_transition = float(args.highlight_protect_transition)
    if args.highlight_protect_strength is not None:
        model.highlight_protect_strength = float(args.highlight_protect_strength)
    if args.chroma_smooth_strength is not None and hasattr(model, "chroma_smooth_strength"):
        model.chroma_smooth_strength = float(args.chroma_smooth_strength)
    if args.luma_smooth_strength is not None and hasattr(model, "luma_smooth_strength"):
        model.luma_smooth_strength = float(args.luma_smooth_strength)
    noisy = _mat_array(args.noisy_mat)
    gt = _mat_array(args.gt_mat)
    if noisy.shape != gt.shape:
        raise RuntimeError(f"shape mismatch: noisy={noisy.shape}, gt={gt.shape}")

    psnr_in = []
    psnr_out = []
    start = time.time()
    done = 0
    with torch.inference_mode():
        for noisy_patch, gt_patch in iter_patches(noisy, gt, max_patches=args.max_patches):
            x_srgb = torch.from_numpy(noisy_patch).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            x_linear = srgb_to_linear(x_srgb).to(device=device)
            out_linear = model(x_linear)
            out_srgb = linear_to_srgb(out_linear).clamp(0.0, 1.0)
            out_np = out_srgb.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_u8 = (out_np * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
            psnr_in.append(psnr_srgb(noisy_patch, gt_patch))
            psnr_out.append(psnr_srgb(out_u8, gt_patch))
            done += 1

    elapsed = time.time() - start
    result = {
        "weights": args.weights,
        "patches": done,
        "psnr_in": float(np.mean(psnr_in)),
        "psnr_out": float(np.mean(psnr_out)),
        "delta": float(np.mean(psnr_out) - np.mean(psnr_in)),
        "seconds": elapsed,
        "ms_patch": elapsed / max(done, 1) * 1000.0,
        "highlight_protect": {
            "threshold": model.highlight_protect_threshold,
            "transition": model.highlight_protect_transition,
            "strength": model.highlight_protect_strength,
        },
        "chroma_smooth": {
            "enabled": bool(getattr(model, "chroma_smooth_branch", False)),
            "strength": float(getattr(model, "chroma_smooth_strength", 0.0)),
        },
        "luma_smooth": {
            "enabled": bool(getattr(model, "luma_smooth_branch", False)),
            "strength": float(getattr(model, "luma_smooth_strength", 0.0)),
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
