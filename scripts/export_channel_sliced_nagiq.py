"""Export a narrower NagiQ checkpoint by channel-slicing a wider checkpoint.

This is a surgery initializer, not an exact function-preserving transform. It is
meant to give W48/W56 students a strong NAFNet-like starting point before
distillation fine-tuning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nagi_nr.nagiq import NagiQ


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def load_state(path: str, state_key: str) -> tuple[dict[str, torch.Tensor], dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if state_key in ckpt:
            state = ckpt[state_key]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "params" in ckpt:
            state = ckpt["params"]
        elif "params_ema" in ckpt:
            state = ckpt["params_ema"]
        else:
            raise KeyError(f"no usable state found in {path}")
        cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
        return state, cfg
    return ckpt, {}


def slice_tensor(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src.clone()
    if src.ndim != len(target_shape):
        raise ValueError(f"rank mismatch: src={tuple(src.shape)} target={tuple(target_shape)}")
    slices = []
    for src_dim, dst_dim in zip(src.shape, target_shape):
        if src_dim < dst_dim:
            raise ValueError(f"source dim too small: src={tuple(src.shape)} target={tuple(target_shape)}")
        slices.append(slice(0, int(dst_dim)))
    return src[tuple(slices)].clone()


def build_sliced_state(
    source_state: dict[str, torch.Tensor],
    target_model: NagiQ,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    target_state = target_model.state_dict()
    out: dict[str, torch.Tensor] = {}
    stats = {"exact": 0, "sliced": 0, "missing": 0}

    for key, target_value in target_state.items():
        if key not in source_state:
            out[key] = target_value.clone()
            stats["missing"] += 1
            continue
        source_value = source_state[key]
        try:
            copied = slice_tensor(source_value, target_value.shape)
        except ValueError as exc:
            raise ValueError(f"cannot slice {key}: {exc}") from exc
        out[key] = copied
        if tuple(source_value.shape) == tuple(target_value.shape):
            stats["exact"] += 1
        else:
            stats["sliced"] += 1

    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Channel-slice a NagiQ/NAFNet checkpoint.")
    ap.add_argument("--source", required=True)
    ap.add_argument("--source-state", default="state_dict")
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--enc", default="2,2,4,6")
    ap.add_argument("--middle", type=int, default=10)
    ap.add_argument("--dec", default="2,2,2,2")
    ap.add_argument(
        "--ending-scale",
        type=float,
        default=1.0,
        help="Scale ending.* after slicing. Use 0 for identity-safe initialization.",
    )
    args = ap.parse_args()

    source_state, source_cfg = load_state(args.source, args.source_state)
    model_cfg = {
        "width": int(args.width),
        "enc_blk_nums": parse_ints(args.enc),
        "middle_blk_num": int(args.middle),
        "dec_blk_nums": parse_ints(args.dec),
    }
    model = NagiQ(**model_cfg)
    state, stats = build_sliced_state(source_state, model)
    if args.ending_scale != 1.0:
        for key in ("ending.weight", "ending.bias"):
            if key in state:
                state[key] = state[key] * float(args.ending_scale)
    model.load_state_dict(state, strict=True)

    cfg = {
        "model": {
            "width": model.width,
            "enc_blk_nums": model.enc_blk_nums,
            "middle_blk_num": model.middle_blk_num,
            "dec_blk_nums": model.dec_blk_nums,
            "dw_expand": model.dw_expand,
            "ffn_expand": model.ffn_expand,
            "drop_out_rate": model.drop_out_rate,
        },
        "source": {
            "kind": "channel_sliced_nagiq",
            "source_checkpoint": args.source,
            "source_state": args.source_state,
            "source_config": source_cfg,
            "ending_scale": float(args.ending_scale),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": 0,
            "state_dict": state,
            "model_state_dict": state,
            "ema_state_dict": state,
            "config": cfg,
            "model_kind": "nagiq",
        },
        str(out),
    )
    print(f"wrote {out}")
    print(
        "model: width={} enc={} middle={} dec={} params={:.2f}M".format(
            model.width,
            model.enc_blk_nums,
            model.middle_blk_num,
            model.dec_blk_nums,
            model.param_count() / 1e6,
        )
    )
    print(f"copy stats: {stats}")


if __name__ == "__main__":
    main()
