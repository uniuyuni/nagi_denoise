"""Evaluate a NagiQ checkpoint on deterministic SIDD training crops.

Used for micro-overfit diagnostics:
  * noisy vs GT
  * teacher vs GT
  * student vs GT
  * student vs teacher
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "nagi_nr" / "src"))

from nagi_nr.data import SIDDPatchDataset, find_sidd_pairs
from nagi_nr.devices import resolve_device
from nagi_nr.train_q import build_model


def psnr_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate NagiQ on deterministic training crops.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sidd-root", default="SIDD_Medium_Srgb")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--state", default="ema", choices=["ema", "live"])
    ap.add_argument("--max-items", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = dict(cfg["data"])
    data_cfg["randomize_each_access"] = False
    pairs = None
    max_pairs = int(data_cfg.get("max_pairs", 0))
    if max_pairs > 0:
        pairs = find_sidd_pairs(args.sidd_root)[:max_pairs]

    ds = SIDDPatchDataset(
        root=args.sidd_root,
        patch_size=data_cfg["patch_size"],
        patches_per_image=data_cfg["patches_per_image"],
        exposure_jitter=tuple(data_cfg["exposure_jitter"]) if data_cfg.get("exposure_jitter") else None,
        flip_rot=data_cfg.get("flip_rot", True),
        seed=0,
        return_teacher=True,
        output_space="srgb",
        randomize_each_access=False,
        pairs=pairs,
    )

    limit = len(ds) if args.max_items <= 0 else min(args.max_items, len(ds))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    device = resolve_device(args.device)
    model = build_model(cfg).to(device=device, dtype=torch.float32).eval()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    key = "model_state_dict" if args.state == "live" else "state_dict"
    model.load_state_dict(ckpt[key], strict=True)

    sums = {
        "noisy_gt": 0.0,
        "teacher_gt": 0.0,
        "student_gt": 0.0,
        "student_teacher": 0.0,
    }
    n = 0
    for noisy, gt, teacher, has_teacher in loader:
        if n >= limit:
            break
        noisy_d = noisy.to(device)
        out = model(noisy_d).clamp(0.0, 1.0).cpu()
        sums["noisy_gt"] += psnr_tensor(noisy, gt)
        sums["teacher_gt"] += psnr_tensor(teacher, gt)
        sums["student_gt"] += psnr_tensor(out, gt)
        sums["student_teacher"] += psnr_tensor(out, teacher)
        n += 1

    print(f"weights: {args.weights}")
    print(f"state: {args.state}")
    print(f"items: {n}/{len(ds)}")
    for key, value in sums.items():
        print(f"{key}: {value / max(1, n):.3f} dB")


if __name__ == "__main__":
    main()
