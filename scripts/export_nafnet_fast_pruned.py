"""Export a physically pruned NAFNet-Fast checkpoint.

The output is NagiQ-compatible because the local NagiQ module has the same
NAFBlock/state_dict layout for 256x256 SIDD validation patches. Conceptually the
checkpoint is still NAFNet-Fast: it starts from the official NAFNet-width64
teacher and removes selected blocks while preserving the order of the rest.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from nagi_nr.nagiq import NagiQ
from nagi_nr_bench.third_party.nafnet import NAFNet


TEACHER_ENC = [2, 2, 4, 8]
TEACHER_MID = 12
TEACHER_DEC = [2, 2, 2, 2]


BLOCK_PATTERNS = (
    re.compile(r"^encoders\.(\d+)\.(\d+)\."),
    re.compile(r"^middle_blks\.(\d+)\."),
    re.compile(r"^decoders\.(\d+)\.(\d+)\."),
)


def load_teacher_state(weights: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    return ckpt["params"] if "params" in ckpt else (
        ckpt["params_ema"] if "params_ema" in ckpt else ckpt
    )


def parse_skip(skip: list[str]) -> tuple[list[set[int]], set[int], list[set[int]]]:
    enc_skip = [set() for _ in TEACHER_ENC]
    mid_skip: set[int] = set()
    dec_skip = [set() for _ in TEACHER_DEC]
    for item in skip:
        m = re.fullmatch(r"enc(\d+)\.(\d+)", item)
        if m:
            stage, block = int(m.group(1)), int(m.group(2))
            enc_skip[stage].add(block)
            continue
        m = re.fullmatch(r"middle\.(\d+)", item)
        if m:
            mid_skip.add(int(m.group(1)))
            continue
        m = re.fullmatch(r"dec(\d+)\.(\d+)", item)
        if m:
            stage, block = int(m.group(1)), int(m.group(2))
            dec_skip[stage].add(block)
            continue
        raise ValueError(f"unknown skip id {item!r}; expected enc3.4, middle.2, or dec1.0")

    for stage, skipped in enumerate(enc_skip):
        bad = [idx for idx in skipped if idx < 0 or idx >= TEACHER_ENC[stage]]
        if bad:
            raise ValueError(f"invalid encoder skip indices for enc{stage}: {bad}")
    bad_mid = [idx for idx in mid_skip if idx < 0 or idx >= TEACHER_MID]
    if bad_mid:
        raise ValueError(f"invalid middle skip indices: {bad_mid}")
    for stage, skipped in enumerate(dec_skip):
        bad = [idx for idx in skipped if idx < 0 or idx >= TEACHER_DEC[stage]]
        if bad:
            raise ValueError(f"invalid decoder skip indices for dec{stage}: {bad}")
    return enc_skip, mid_skip, dec_skip


def keep_indices(counts: list[int], skipped: list[set[int]]) -> list[list[int]]:
    return [
        [idx for idx in range(count) if idx not in skipped[stage]]
        for stage, count in enumerate(counts)
    ]


def is_block_key(key: str) -> bool:
    return any(pat.match(key) for pat in BLOCK_PATTERNS)


def copy_group(
    out_state: dict[str, torch.Tensor],
    teacher_state: dict[str, torch.Tensor],
    group: str,
    keep: list[int],
) -> None:
    for new_idx, old_idx in enumerate(keep):
        old_prefix = f"{group}.{old_idx}."
        new_prefix = f"{group}.{new_idx}."
        for key, value in teacher_state.items():
            if key.startswith(old_prefix):
                out_state[new_prefix + key[len(old_prefix):]] = value.clone()


def build_pruned_state(
    teacher_state: dict[str, torch.Tensor],
    enc_keep: list[list[int]],
    mid_keep: list[int],
    dec_keep: list[list[int]],
) -> tuple[NagiQ, dict[str, torch.Tensor]]:
    model = NagiQ(
        width=64,
        enc_blk_nums=tuple(len(x) for x in enc_keep),
        middle_blk_num=len(mid_keep),
        dec_blk_nums=tuple(len(x) for x in dec_keep),
    )
    target_state = model.state_dict()
    out_state: dict[str, torch.Tensor] = {}

    for key, value in teacher_state.items():
        if is_block_key(key):
            continue
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            out_state[key] = value.clone()

    for stage, keep in enumerate(enc_keep):
        copy_group(out_state, teacher_state, f"encoders.{stage}", keep)
    copy_group(out_state, teacher_state, "middle_blks", mid_keep)
    for stage, keep in enumerate(dec_keep):
        copy_group(out_state, teacher_state, f"decoders.{stage}", keep)

    missing = [key for key, value in target_state.items() if key not in out_state or out_state[key].shape != value.shape]
    if missing:
        raise RuntimeError(f"failed to populate target state keys: {missing[:10]}")
    model.load_state_dict(out_state, strict=True)
    return model, out_state


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a pruned NAFNet-Fast checkpoint.")
    ap.add_argument("--weights", default="benchmarks/nafnet/NAFNet-SIDD-width64.pth")
    ap.add_argument("--skip", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    enc_skip, mid_skip, dec_skip = parse_skip(args.skip)
    enc_keep = keep_indices(TEACHER_ENC, enc_skip)
    mid_keep = [idx for idx in range(TEACHER_MID) if idx not in mid_skip]
    dec_keep = keep_indices(TEACHER_DEC, dec_skip)
    teacher_state = load_teacher_state(args.weights)
    model, state = build_pruned_state(teacher_state, enc_keep, mid_keep, dec_keep)

    cfg = {
        "model": {
            "width": 64,
            "enc_blk_nums": tuple(len(x) for x in enc_keep),
            "middle_blk_num": len(mid_keep),
            "dec_blk_nums": tuple(len(x) for x in dec_keep),
        },
        "source": {
            "kind": "nafnet_fast_pruned",
            "teacher_weights": args.weights,
            "skip": list(args.skip),
            "enc_keep": enc_keep,
            "middle_keep": mid_keep,
            "dec_keep": dec_keep,
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
            "model_kind": "nafnet_fast",
        },
        str(out),
    )
    print(f"wrote {out}")
    print(f"skip: {', '.join(args.skip)}")
    print(
        "blocks: enc={} middle={} dec={} params={:.2f}M".format(
            tuple(len(x) for x in enc_keep),
            len(mid_keep),
            tuple(len(x) for x in dec_keep),
            model.param_count() / 1e6,
        )
    )


if __name__ == "__main__":
    main()
