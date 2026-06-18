"""Single-block ablation probe for the NAFNet-Fast pruning track."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from nagi_nr.devices import resolve_device
from nagi_nr_bench.eval_sidd_val import psnr_srgb
from nagi_nr_bench.third_party.nafnet import NAFNet


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


def load_validation(noisy_mat: str, gt_mat: str) -> tuple[np.ndarray, np.ndarray]:
    noisy = sio.loadmat(noisy_mat)
    nkey = next(k for k in noisy if not k.startswith("__"))
    noisy_arr = noisy[nkey]
    gt = sio.loadmat(gt_mat)
    gkey = next(k for k in gt if not k.startswith("__"))
    gt_arr = gt[gkey]
    if noisy_arr.shape != gt_arr.shape:
        raise RuntimeError(f"shape mismatch: noisy={noisy_arr.shape}, gt={gt_arr.shape}")
    return noisy_arr, gt_arr


def iter_patches(noisy: np.ndarray, gt: np.ndarray, limit: int):
    total = noisy.shape[0] * noisy.shape[1]
    limit = total if limit <= 0 else min(limit, total)
    done = 0
    for i in range(noisy.shape[0]):
        for j in range(noisy.shape[1]):
            if done >= limit:
                return
            yield noisy[i, j], gt[i, j]
            done += 1


@torch.inference_mode()
def evaluate(model: NAFNet, device: torch.device, noisy: np.ndarray, gt: np.ndarray, max_patches: int) -> dict[str, float]:
    psnr_in: list[float] = []
    psnr_out: list[float] = []
    start = time.perf_counter()
    done = 0
    for n_patch, g_patch in iter_patches(noisy, gt, max_patches):
        t = torch.from_numpy(n_patch).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        t = t.to(device=device, dtype=torch.float32)
        out = model(t)
        out_np = out.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out_u8 = (out_np * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        psnr_in.append(psnr_srgb(n_patch, g_patch))
        psnr_out.append(psnr_srgb(out_u8, g_patch))
        done += 1
    _sync(device)
    elapsed = time.perf_counter() - start
    return {
        "patches": float(done),
        "psnr_in": float(np.mean(psnr_in)),
        "psnr_out": float(np.mean(psnr_out)),
        "ms_patch": elapsed / max(1, done) * 1000.0,
    }


def _markdown(rows: list[dict[str, float | str]], baseline: dict[str, float], max_patches: int) -> str:
    lines = [
        "# NAFNet-Fast Block Ablation",
        "",
        f"Patches: `{max_patches}`",
        f"Baseline PSNR: `{baseline['psnr_out']:.3f} dB`",
        f"Baseline speed: `{baseline['ms_patch']:.1f} ms/patch`",
        "",
        "| block | PSNR | drop | ms/patch | speed delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {block} | {psnr:.3f} | {drop:+.3f} | {ms_patch:.1f} | {speed_delta:+.1f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe NAFNet single-block ablations.")
    ap.add_argument("--weights", default="benchmarks/nafnet/NAFNet-SIDD-width64.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    ap.add_argument("--max-patches", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None, help="Specific block ids, e.g. enc0.0 middle.3 dec3.1")
    ap.add_argument(
        "--skip-set",
        action="append",
        nargs="+",
        default=None,
        help="Evaluate one multi-block skip set. Can be passed more than once.",
    )
    ap.add_argument("--max-blocks", type=int, default=0, help="Limit after --only filtering; 0 means all.")
    ap.add_argument("--output-md", default="runs/nafnet_fast_ablation.md")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model = load_nafnet(args.weights, device)
    blocks = install_gates(model)
    block_by_name = {b.name: b for b in blocks}
    if args.only:
        unknown = [name for name in args.only if name not in block_by_name]
        if unknown:
            raise ValueError(f"unknown block ids: {unknown}; known examples: {[b.name for b in blocks[:5]]}")
        selected = [block_by_name[name] for name in args.only]
    else:
        selected = blocks
    if args.skip_set:
        unknown = [name for group in args.skip_set for name in group if name not in block_by_name]
        if unknown:
            raise ValueError(f"unknown block ids: {unknown}; known examples: {[b.name for b in blocks[:5]]}")
    if args.max_blocks > 0:
        selected = selected[: args.max_blocks]

    noisy, gt = load_validation(args.noisy_mat, args.gt_mat)
    baseline = evaluate(model, device, noisy, gt, args.max_patches)
    print(
        "baseline: patches={patches:.0f} psnr={psnr_out:.3f} ms={ms_patch:.1f}".format(
            **baseline
        )
    )

    def reset_gates() -> None:
        for ref in blocks:
            ref.module.gate = 1.0

    rows: list[dict[str, float | str]] = []
    probes: list[tuple[str, list[BlockRef]]]
    if args.skip_set:
        probes = [
            ("+".join(group), [block_by_name[name] for name in group])
            for group in args.skip_set
        ]
    else:
        probes = [(ref.name, [ref]) for ref in selected]

    for label, refs in probes:
        reset_gates()
        for ref in refs:
            ref.module.gate = 0.0
        result = evaluate(model, device, noisy, gt, args.max_patches)
        reset_gates()
        row = {
            "block": label,
            "psnr": result["psnr_out"],
            "drop": result["psnr_out"] - baseline["psnr_out"],
            "ms_patch": result["ms_patch"],
            "speed_delta": result["ms_patch"] - baseline["ms_patch"],
        }
        rows.append(row)
        print(
            "{block:10s} psnr={psnr:.3f} drop={drop:+.3f} "
            "ms={ms_patch:.1f} speed_delta={speed_delta:+.1f}".format(**row),
            flush=True,
        )

    rows.sort(key=lambda row: (float(row["drop"]), float(row["speed_delta"])))
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_markdown(rows, baseline, args.max_patches), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
