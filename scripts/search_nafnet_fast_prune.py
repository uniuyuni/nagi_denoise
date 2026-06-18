"""Greedy block-pruning search for the NAFNet-Fast-P track.

This is intentionally conservative: it keeps the full NAFNet teacher in memory,
turns candidate blocks into identity with gates, and adds one skip at a time only
when the cumulative PSNR drop stays within a configured guard.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from nagi_nr.devices import resolve_device
from nagi_nr_bench.eval_sidd_val import psnr_srgb
from nagi_nr_bench.third_party.nafnet import NAFNet


DEFAULT_CANDIDATES = [f"enc3.{idx}" for idx in range(8)] + [
    f"middle.{idx}" for idx in range(1, 12)
]

# Measured 256x256 MPS stage-profile costs from docs/nagiq_redesign_from_first_principles.md.
# These are only a tie-breaker/proxy; the actual gated model time is recorded for every trial.
STAGE_MS_PER_BLOCK = {
    "enc0": 82.3 / 2,
    "enc1": 46.7 / 2,
    "enc2": 54.5 / 4,
    "enc3": 72.0 / 8,
    "middle": 92.0 / 12,
    "dec0": 18.6 / 2,
    "dec1": 26.9 / 2,
    "dec2": 45.8 / 2,
    "dec3": 82.4 / 2,
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


class GatedBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.gate = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate == 1.0:
            return self.block(x)
        if self.gate == 0.0:
            return x
        return x + float(self.gate) * (self.block(x) - x)


@dataclass(frozen=True)
class Patch:
    noisy: np.ndarray
    gt: np.ndarray


@dataclass
class BlockRef:
    name: str
    module: GatedBlock


def load_nafnet(weights: str, device: torch.device) -> NAFNet:
    model = NAFNet(
        img_channel=3,
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    state = ckpt["params"] if "params" in ckpt else (
        ckpt["params_ema"] if "params_ema" in ckpt else ckpt
    )
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    return model


def install_gates(model: NAFNet) -> list[BlockRef]:
    blocks: list[BlockRef] = []
    for stage_idx, seq in enumerate(model.encoders):
        for block_idx, block in enumerate(seq):
            gated = GatedBlock(block)
            seq[block_idx] = gated
            blocks.append(BlockRef(f"enc{stage_idx}.{block_idx}", gated))
    for block_idx, block in enumerate(model.middle_blks):
        gated = GatedBlock(block)
        model.middle_blks[block_idx] = gated
        blocks.append(BlockRef(f"middle.{block_idx}", gated))
    for stage_idx, seq in enumerate(model.decoders):
        for block_idx, block in enumerate(seq):
            gated = GatedBlock(block)
            seq[block_idx] = gated
            blocks.append(BlockRef(f"dec{stage_idx}.{block_idx}", gated))
    return blocks


def load_validation(noisy_mat: str, gt_mat: str, max_patches: int) -> list[Patch]:
    noisy = sio.loadmat(noisy_mat)
    nkey = next(k for k in noisy if not k.startswith("__"))
    noisy_arr = noisy[nkey]
    gt = sio.loadmat(gt_mat)
    gkey = next(k for k in gt if not k.startswith("__"))
    gt_arr = gt[gkey]
    if noisy_arr.shape != gt_arr.shape:
        raise RuntimeError(f"shape mismatch: noisy={noisy_arr.shape}, gt={gt_arr.shape}")

    total = noisy_arr.shape[0] * noisy_arr.shape[1]
    limit = total if max_patches <= 0 else min(max_patches, total)
    patches: list[Patch] = []
    for i in range(noisy_arr.shape[0]):
        for j in range(noisy_arr.shape[1]):
            if len(patches) >= limit:
                return patches
            patches.append(Patch(noisy=noisy_arr[i, j], gt=gt_arr[i, j]))
    return patches


def input_tensor(np_noisy_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np_noisy_uint8)
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )


@torch.inference_mode()
def evaluate(
    model: NAFNet,
    device: torch.device,
    patches: list[Patch],
    teacher_outputs: list[np.ndarray] | None = None,
    collect_outputs: bool = False,
) -> dict[str, Any]:
    psnr_in: list[float] = []
    psnr_out: list[float] = []
    teacher_mse: list[float] = []
    outputs: list[np.ndarray] = []
    start = time.perf_counter()

    for idx, patch in enumerate(patches):
        t = input_tensor(patch.noisy, device)
        out = model(t)
        out_np = out.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out_u8 = (out_np * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

        psnr_in.append(psnr_srgb(patch.noisy, patch.gt))
        psnr_out.append(psnr_srgb(out_u8, patch.gt))
        if teacher_outputs is not None:
            teacher_mse.append(float(np.mean((out_np - teacher_outputs[idx]) ** 2)))
        if collect_outputs:
            outputs.append(out_np.astype(np.float32, copy=True))

    _sync(device)
    elapsed = time.perf_counter() - start
    result: dict[str, Any] = {
        "patches": len(patches),
        "psnr_in": float(np.mean(psnr_in)),
        "psnr_out": float(np.mean(psnr_out)),
        "ms_patch": elapsed / max(1, len(patches)) * 1000.0,
    }
    if teacher_mse:
        result["teacher_mse"] = float(np.mean(teacher_mse))
    if collect_outputs:
        result["outputs"] = outputs
    return result


def reset_gates(blocks: list[BlockRef]) -> None:
    for ref in blocks:
        ref.module.gate = 1.0


def set_skip(block_by_name: dict[str, BlockRef], skip: set[str]) -> None:
    for name, ref in block_by_name.items():
        ref.module.gate = 0.0 if name in skip else 1.0


def block_cost_ms(block_name: str) -> float:
    m = re.fullmatch(r"(enc\d+|dec\d+|middle)\.\d+", block_name)
    if not m:
        return 0.0
    return STAGE_MS_PER_BLOCK.get(m.group(1), 0.0)


def estimated_saved_ms(skip: set[str]) -> float:
    return sum(block_cost_ms(name) for name in skip)


def validate_names(names: list[str], block_by_name: dict[str, BlockRef], label: str) -> None:
    unknown = [name for name in names if name not in block_by_name]
    if unknown:
        examples = ", ".join(list(block_by_name)[:8])
        raise ValueError(f"unknown {label}: {unknown}; known examples: {examples}")


def format_skip(skip: set[str]) -> str:
    return ", ".join(sorted(skip)) if skip else "(none)"


def row_from_result(
    *,
    step: int,
    candidate: str,
    skip: set[str],
    result: dict[str, Any],
    baseline: dict[str, Any],
    score: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "candidate": candidate,
        "skip": sorted(skip),
        "skip_text": format_skip(skip),
        "psnr": float(result["psnr_out"]),
        "drop": float(result["psnr_out"] - baseline["psnr_out"]),
        "ms_patch": float(result["ms_patch"]),
        "teacher_mse": float(result.get("teacher_mse", 0.0)),
        "estimated_saved_ms": float(estimated_saved_ms(skip)),
        "candidate_cost_ms": float(block_cost_ms(candidate)),
        "score": float(score),
    }


def select_candidate(
    rows: list[dict[str, Any]],
    max_drop: float,
) -> dict[str, Any] | None:
    viable = [row for row in rows if row["drop"] >= -max_drop]
    if not viable:
        return None
    return max(viable, key=lambda row: (row["score"], row["psnr"], row["estimated_saved_ms"]))


def markdown_report(
    args: argparse.Namespace,
    device: torch.device,
    baseline: dict[str, Any],
    initial: dict[str, Any] | None,
    selected: list[dict[str, Any]],
    trials: list[dict[str, Any]],
) -> str:
    lines = [
        "# NAFNet-Fast-P Greedy Prune Search",
        "",
        f"Device: `{device}`",
        f"Weights: `{args.weights}`",
        f"Patches: `{baseline['patches']}`",
        f"Max cumulative drop: `{args.max_drop:.3f} dB`",
        f"Cost bias: `{args.cost_bias:.4f}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| baseline PSNR | {baseline['psnr_out']:.3f} dB |",
        f"| baseline speed | {baseline['ms_patch']:.1f} ms/patch |",
        f"| noisy PSNR | {baseline['psnr_in']:.3f} dB |",
    ]
    if initial is not None:
        lines.extend(
            [
                "",
                "## Initial Skip",
                "",
                "| skip | PSNR | drop | est saved | ms/patch | teacher MSE |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                "| {skip_text} | {psnr:.3f} | {drop:+.3f} | {estimated_saved_ms:.1f} | {ms_patch:.1f} | {teacher_mse:.6f} |".format(
                    **initial
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Selected Sequence",
            "",
            "| step | added | skip set | PSNR | drop | est saved | ms/patch | teacher MSE |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if selected:
        for row in selected:
            lines.append(
                "| {step} | {candidate} | {skip_text} | {psnr:.3f} | {drop:+.3f} | {estimated_saved_ms:.1f} | {ms_patch:.1f} | {teacher_mse:.6f} |".format(
                    **row
                )
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Trial Rows",
            "",
            "| step | candidate | PSNR | drop | candidate cost | est saved | score | ms/patch | teacher MSE |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(trials, key=lambda r: (r["step"], -r["score"])):
        lines.append(
            "| {step} | {candidate} | {psnr:.3f} | {drop:+.3f} | {candidate_cost_ms:.1f} | {estimated_saved_ms:.1f} | {score:.3f} | {ms_patch:.1f} | {teacher_mse:.6f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy search for NAFNet-Fast-P block pruning.")
    ap.add_argument("--weights", default="benchmarks/nafnet/NAFNet-SIDD-width64.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    ap.add_argument("--max-patches", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-drop", type=float, default=0.35)
    ap.add_argument(
        "--cost-bias",
        type=float,
        default=0.002,
        help="Small PSNR-score bonus per estimated saved ms. Keep small because quality is primary.",
    )
    ap.add_argument("--initial-skip", nargs="*", default=[])
    ap.add_argument("--candidates", nargs="*", default=None)
    ap.add_argument("--output-md", default="runs/nafnet_fast_prune_search/greedy_search.md")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model = load_nafnet(args.weights, device)
    blocks = install_gates(model)
    block_by_name = {ref.name: ref for ref in blocks}

    candidates = args.candidates if args.candidates is not None else DEFAULT_CANDIDATES
    validate_names(args.initial_skip, block_by_name, "initial skip ids")
    validate_names(candidates, block_by_name, "candidate ids")
    current_skip = set(args.initial_skip)
    candidates = [name for name in candidates if name not in current_skip]

    patches = load_validation(args.noisy_mat, args.gt_mat, args.max_patches)
    print(f"device: {device}")
    print(f"patches: {len(patches)}")
    print(f"initial skip: {format_skip(current_skip)}")
    print(f"candidates: {', '.join(candidates)}")

    reset_gates(blocks)
    baseline_eval = evaluate(model, device, patches, collect_outputs=True)
    teacher_outputs = baseline_eval.pop("outputs")
    print(
        "baseline: psnr={psnr_out:.3f} ms={ms_patch:.1f} noisy={psnr_in:.3f}".format(
            **baseline_eval
        )
    )

    initial_row: dict[str, Any] | None = None
    if current_skip:
        set_skip(block_by_name, current_skip)
        current_eval = evaluate(model, device, patches, teacher_outputs=teacher_outputs)
        initial_row = row_from_result(
            step=0,
            candidate="initial",
            skip=current_skip,
            result=current_eval,
            baseline=baseline_eval,
            score=float(current_eval["psnr_out"]) + args.cost_bias * estimated_saved_ms(current_skip),
        )
        print(
            "initial: psnr={psnr:.3f} drop={drop:+.3f} est_saved={estimated_saved_ms:.1f} ms={ms_patch:.1f}".format(
                **initial_row
            )
        )

    selected_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    remaining = list(candidates)

    for step in range(1, args.max_steps + 1):
        if not remaining:
            break
        step_rows: list[dict[str, Any]] = []
        print(f"step {step}: evaluating {len(remaining)} candidates", flush=True)
        for candidate in remaining:
            trial_skip = set(current_skip)
            trial_skip.add(candidate)
            set_skip(block_by_name, trial_skip)
            result = evaluate(model, device, patches, teacher_outputs=teacher_outputs)
            score = float(result["psnr_out"]) + args.cost_bias * estimated_saved_ms(trial_skip)
            row = row_from_result(
                step=step,
                candidate=candidate,
                skip=trial_skip,
                result=result,
                baseline=baseline_eval,
                score=score,
            )
            step_rows.append(row)
            trial_rows.append(row)
            print(
                "  {candidate:10s} psnr={psnr:.3f} drop={drop:+.3f} "
                "est_saved={estimated_saved_ms:.1f} score={score:.3f} ms={ms_patch:.1f}".format(
                    **row
                ),
                flush=True,
            )

        chosen = select_candidate(step_rows, args.max_drop)
        if chosen is None:
            print(f"stop: no candidate kept cumulative drop within {args.max_drop:.3f} dB")
            break

        current_skip = set(chosen["skip"])
        remaining = [name for name in remaining if name != chosen["candidate"]]
        selected_rows.append(chosen)
        print(
            "selected step {step}: {candidate} -> drop={drop:+.3f} "
            "est_saved={estimated_saved_ms:.1f}".format(**chosen),
            flush=True,
        )

    reset_gates(blocks)

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        markdown_report(args, device, baseline_eval, initial_row, selected_rows, trial_rows),
        encoding="utf-8",
    )
    out_json = Path(args.output_json) if args.output_json else out_md.with_suffix(".json")
    out_json.write_text(
        json.dumps(
            {
                "args": vars(args),
                "device": str(device),
                "baseline": baseline_eval,
                "initial": initial_row,
                "selected": selected_rows,
                "trials": trial_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    if selected_rows:
        print(f"recommended skip: {' '.join(selected_rows[-1]['skip'])}")


if __name__ == "__main__":
    main()
