"""Phase 4B early-check probe: confidence / detail_applied / output-HF on a
real-photo crop, at one or more model.detail_scale values.

This is the abort-criterion tool for the Phase 4B texture-statistics
fine-tune (nagi_v2_l_ft5): at iter 1000, if the output HF has not moved
materially above the noisy-crop baseline as detail_scale increases, that is
Phase 2C's null result again -- stop the run.

NOTE (v1.0 cleanup): the ft5 checkpoints under `runs/nagi_v2_l_ft5/` were
moved to the Trash (negative result). Point `--weights` at any NagiV2
checkpoint; the production one is `runs/nagi_v2_l_ft2/nagi_v2_l_ft2_final.pt`.

Usage:
    pixi run python scripts/probe_detail_head.py \\
        --weights runs/nagi_v2_l_ft5/nagi_v2_l_ft5_0001000.pt \\
        --device mps --detail-scales 0.25 1 2 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from nagi_denoise.pipeline.denoise_exr import load_model, run_full_image
from nagi_denoise.pipeline.eval_selectivity import crop_center, hf_map, to_display
from nagi_denoise.pipeline.probe import read_image

DEFAULT_NOISY = "~/ProjectData/test_photos/X-T5 Occi noisy.EXR"


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the detail/confidence heads on a crop.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--noisy", default=DEFAULT_NOISY)
    parser.add_argument("--x", type=int, default=2420)
    parser.add_argument("--y", type=int, default=1040)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--state-key", default="state_dict", help="'state_dict' = EMA weights (default).")
    parser.add_argument("--detail-scales", type=float, nargs="+", default=[0.25])
    args = parser.parse_args()

    device = torch.device(args.device)
    noisy_path = Path(args.noisy).expanduser()
    noisy_full = read_image(noisy_path)
    noisy_crop = crop_center(noisy_full, args.x, args.y, args.size)
    noisy_hf_mean = float(hf_map(to_display(noisy_crop)).mean())

    model = load_model(Path(args.weights), device=device, state_key=args.state_key)

    print(f"weights={args.weights} state_key={args.state_key} noisy_hf_mean={noisy_hf_mean:.6f}")
    for scale in args.detail_scales:
        model.detail_scale = float(scale)
        with torch.inference_mode():
            out, extras = run_full_image(model, noisy_crop, device=device)
        conf_mean = float(extras["detail_confidence"].mean())
        detail_applied_absmean = float(np.abs(extras["detail_applied"]).mean())
        out_hf_mean = float(hf_map(to_display(out)).mean())
        ratio = out_hf_mean / noisy_hf_mean if noisy_hf_mean > 1e-8 else float("nan")
        print(
            f"detail_scale={scale:6.3f} "
            f"confidence_mean={conf_mean:.6f} "
            f"detail_applied_absmean={detail_applied_absmean:.6e} "
            f"output_hf_mean={out_hf_mean:.6f} "
            f"hf_ratio_vs_noisy={ratio:.4f}"
        )


if __name__ == "__main__":
    main()
