"""Write an inference-only checkpoint from a training checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Export EMA-only inference checkpoint.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--state-key", default="state_dict")
    args = ap.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("input must be a checkpoint dict")
    if args.state_key not in ckpt:
        raise KeyError(f"checkpoint does not contain {args.state_key!r}")

    out = {
        "state_dict": ckpt[args.state_key],
        "config": ckpt.get("config"),
        "model_kind": ckpt.get("model_kind", "nagiq"),
        "step": ckpt.get("step", 0),
        "metrics": ckpt.get("metrics", {}),
        "best_val_psnr": ckpt.get("best_val_psnr"),
        "source_checkpoint": args.input,
        "inference_only": True,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(path))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
