"""Export NAFNet/NagiQ-style checkpoints to Core ML for the NAFNet-Fast track."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from nagi_nr.nagiq import NagiQ


def _model_cfg_from_checkpoint(ckpt) -> dict | None:
    if not isinstance(ckpt, dict):
        return None
    cfg = ckpt.get("config")
    if isinstance(cfg, dict) and isinstance(cfg.get("model"), dict):
        model_cfg = dict(cfg["model"])
        model_cfg.pop("preset", None)
        return model_cfg
    return None


def _state_from_checkpoint(ckpt, state_key: str) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if state_key in ckpt:
            return ckpt[state_key]
        if "params" in ckpt:
            return ckpt["params"]
        if "params_ema" in ckpt:
            return ckpt["params_ema"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def load_checkpoint_as_nagiq(weights: str, state_key: str) -> NagiQ:
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = _state_from_checkpoint(ckpt, state_key)
    model_cfg = _model_cfg_from_checkpoint(ckpt) or dict(
        width=64,
        enc_blk_nums=(2, 2, 4, 8),
        middle_blk_num=12,
        dec_blk_nums=(2, 2, 2, 2),
    )
    model = NagiQ(**model_cfg)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(
        "loaded model: width={} enc={} middle={} dec={} params={:.2f}M".format(
            model.width,
            model.enc_blk_nums,
            model.middle_blk_num,
            model.dec_blk_nums,
            model.param_count() / 1e6,
        )
    )
    return model


def convert_precision(name: str, fp16_skip_ops: set[str] | None = None):
    import coremltools as ct

    if name == "float32":
        return ct.precision.FLOAT32
    if name == "float16":
        if fp16_skip_ops:
            def op_selector(op) -> bool:
                return getattr(op, "op_type", None) not in fp16_skip_ops

            return ct.transform.FP16ComputePrecision(op_selector=op_selector)
        return ct.precision.FLOAT16
    raise ValueError(f"unknown precision: {name}")


def deployment_target(name: str):
    import coremltools as ct

    targets = {
        "macos11": ct.target.macOS11,
        "macos12": ct.target.macOS12,
        "macos13": ct.target.macOS13,
        "macos14": ct.target.macOS14,
        "macos15": ct.target.macOS15,
    }
    return targets[name]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export NAFNet/NagiQ-style checkpoints to Core ML.")
    ap.add_argument("--weights", default="benchmarks/nafnet/NAFNet-SIDD-width64.pth")
    ap.add_argument("--state-key", default="state_dict")
    ap.add_argument("--output", default="runs/nafnet_fast_coreml/nafnet_width64_fp32.mlpackage")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--precision", choices=["float32", "float16"], default="float32")
    ap.add_argument("--convert-to", choices=["mlprogram", "neuralnetwork"], default="mlprogram")
    ap.add_argument(
        "--deployment-target",
        choices=["macos11", "macos12", "macos13", "macos14", "macos15"],
        default="macos13",
    )
    ap.add_argument(
        "--fp16-skip-ops",
        default="",
        help="Comma-separated MIL op types to keep in fp32 when --precision=float16.",
    )
    ap.add_argument(
        "--compute-units",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default="all",
    )
    args = ap.parse_args()

    import coremltools as ct

    compute_units = {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[args.compute_units]
    fp16_skip_ops = {op.strip() for op in args.fp16_skip_ops.split(",") if op.strip()}
    if fp16_skip_ops:
        print(f"keeping these op types in fp32: {sorted(fp16_skip_ops)}")

    model = load_checkpoint_as_nagiq(args.weights, args.state_key)
    example = torch.rand(args.batch, 3, args.height, args.width, dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=True)
        traced = torch.jit.freeze(traced.eval())
        ref = model(example)
        traced_ref = traced(example)
    max_diff = float((ref - traced_ref).abs().max().item())
    print(f"trace max abs diff: {max_diff:.8f}")

    convert_kwargs = {
        "convert_to": args.convert_to,
        "inputs": [ct.TensorType(name="input", shape=example.shape, dtype=np.float32)],
        "outputs": [ct.TensorType(name="output", dtype=np.float32)],
        "compute_units": compute_units,
        "minimum_deployment_target": deployment_target(args.deployment_target),
    }
    if args.convert_to == "mlprogram":
        convert_kwargs["compute_precision"] = convert_precision(args.precision, fp16_skip_ops)
    mlmodel = ct.convert(traced, **convert_kwargs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    print(f"wrote {out}")
    print(f"convert_to: {args.convert_to}")
    print(f"deployment_target: {args.deployment_target}")
    print(f"precision: {args.precision}")
    if fp16_skip_ops:
        print(f"fp16_skip_ops: {','.join(sorted(fp16_skip_ops))}")
    print(f"compute_units: {args.compute_units}")


if __name__ == "__main__":
    main()
