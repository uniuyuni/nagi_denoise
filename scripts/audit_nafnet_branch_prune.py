"""Audit NAFBlock branch-level pruning for teacher-compatible speedups."""
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


TEACHER_ENC = [2, 2, 4, 8]
TEACHER_MID = 12
TEACHER_DEC = [2, 2, 2, 2]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


class BranchGatedNAFBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.attn_gate = 1.0
        self.ffn_gate = 1.0

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        block = self.block
        x = block.norm1(inp)
        x = block.conv1(x)
        x = block.conv2(x)
        x = block.sg(x)
        x = x * block.sca(x)
        x = block.conv3(x)
        x = block.dropout1(x)
        y = inp + x * block.beta * float(self.attn_gate)

        x = block.conv4(block.norm2(y))
        x = block.sg(x)
        x = block.conv5(x)
        x = block.dropout2(x)
        return y + x * block.gamma * float(self.ffn_gate)


@dataclass(frozen=True)
class Patch:
    noisy: np.ndarray
    gt: np.ndarray


@dataclass(frozen=True)
class BranchRef:
    block_name: str
    branch: str
    module: BranchGatedNAFBlock

    @property
    def name(self) -> str:
        return f"{self.block_name}.{self.branch}"


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


def install_branch_gates(model: NAFNet) -> list[BranchRef]:
    refs: list[BranchRef] = []
    for stage_idx, seq in enumerate(model.encoders):
        for block_idx, block in enumerate(seq):
            gated = BranchGatedNAFBlock(block)
            seq[block_idx] = gated
            block_name = f"enc{stage_idx}.{block_idx}"
            refs.append(BranchRef(block_name, "attn", gated))
            refs.append(BranchRef(block_name, "ffn", gated))
    for block_idx, block in enumerate(model.middle_blks):
        gated = BranchGatedNAFBlock(block)
        model.middle_blks[block_idx] = gated
        block_name = f"middle.{block_idx}"
        refs.append(BranchRef(block_name, "attn", gated))
        refs.append(BranchRef(block_name, "ffn", gated))
    for stage_idx, seq in enumerate(model.decoders):
        for block_idx, block in enumerate(seq):
            gated = BranchGatedNAFBlock(block)
            seq[block_idx] = gated
            block_name = f"dec{stage_idx}.{block_idx}"
            refs.append(BranchRef(block_name, "attn", gated))
            refs.append(BranchRef(block_name, "ffn", gated))
    return refs


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


def reset_gates(refs: list[BranchRef]) -> None:
    seen: set[int] = set()
    for ref in refs:
        ident = id(ref.module)
        if ident in seen:
            continue
        seen.add(ident)
        ref.module.attn_gate = 1.0
        ref.module.ffn_gate = 1.0


def set_branch(ref_by_name: dict[str, BranchRef], name: str, gate: float) -> None:
    ref = ref_by_name[name]
    if ref.branch == "attn":
        ref.module.attn_gate = gate
    elif ref.branch == "ffn":
        ref.module.ffn_gate = gate
    else:
        raise ValueError(f"unknown branch {ref.branch}")


def stage_shape(block_name: str, height: int, width: int) -> tuple[int, int, int]:
    m = re.fullmatch(r"(enc|dec)(\d+)\.(\d+)", block_name)
    if m:
        kind = m.group(1)
        stage = int(m.group(2))
        if kind == "enc":
            channels = 64 * (2 ** stage)
            return channels, height // (2 ** stage), width // (2 ** stage)
        channels = 64 * (2 ** (3 - stage))
        spatial_div = 2 ** (3 - stage)
        return channels, height // spatial_div, width // spatial_div
    m = re.fullmatch(r"middle\.(\d+)", block_name)
    if m:
        return 1024, height // 16, width // 16
    raise ValueError(f"bad block name: {block_name}")


def conv_macs(in_ch: int, out_ch: int, kernel: int, h: int, w: int, groups: int = 1) -> int:
    return out_ch * (in_ch // groups) * kernel * kernel * h * w


def branch_gmac(branch_name: str, height: int, width: int) -> float:
    block_name, branch = branch_name.rsplit(".", 1)
    channels, h, w = stage_shape(block_name, height, width)
    dw = channels * 2
    ffn = channels * 2
    macs = 0
    if branch == "attn":
        macs += conv_macs(channels, dw, 1, h, w)
        macs += conv_macs(dw, dw, 3, h, w, groups=dw)
        macs += conv_macs(dw // 2, dw // 2, 1, 1, 1)
        macs += conv_macs(dw // 2, channels, 1, h, w)
    elif branch == "ffn":
        macs += conv_macs(channels, ffn, 1, h, w)
        macs += conv_macs(ffn // 2, channels, 1, h, w)
    else:
        raise ValueError(f"bad branch name: {branch_name}")
    return macs / 1e9


def default_candidates() -> list[str]:
    names: list[str] = []
    for idx in range(8):
        names.extend([f"enc3.{idx}.attn", f"enc3.{idx}.ffn"])
    for idx in range(12):
        names.extend([f"middle.{idx}.attn", f"middle.{idx}.ffn"])
    for idx in range(2):
        names.extend([f"dec0.{idx}.attn", f"dec0.{idx}.ffn"])
    return names


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
    result: dict[str, Any] = {
        "patches": len(patches),
        "psnr_in": float(np.mean(psnr_in)),
        "psnr_out": float(np.mean(psnr_out)),
        "ms_patch": (time.perf_counter() - start) / max(1, len(patches)) * 1000.0,
    }
    if teacher_mse:
        result["teacher_mse"] = float(np.mean(teacher_mse))
    if collect_outputs:
        result["outputs"] = outputs
    return result


def markdown(args: argparse.Namespace, device: torch.device, baseline: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# NAFNet Branch-Prune Audit",
        "",
        f"Device: `{device}`",
        f"Patches: `{baseline['patches']}`",
        f"Baseline PSNR: `{baseline['psnr_out']:.3f} dB`",
        f"Noisy PSNR: `{baseline['psnr_in']:.3f} dB`",
        "",
        "| rank | branch | PSNR | drop | saved GMAC | dB/GMAC | teacher MSE | ms/patch |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = sorted(rows, key=lambda r: (r["drop"], r["saved_gmac"]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            "| {rank} | {branch} | {psnr:.3f} | {drop:+.3f} | {saved_gmac:.3f} | {db_per_gmac:.3f} | {teacher_mse:.8f} | {ms_patch:.1f} |".format(
                rank=rank,
                **row,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit branch-level NAFNet pruning.")
    ap.add_argument("--weights", default="benchmarks/nafnet/NAFNet-SIDD-width64.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    ap.add_argument("--max-patches", type=int, default=8)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--candidates", nargs="*", default=None)
    ap.add_argument("--output-md", default="runs/nafnet_branch_prune_audit/audit.md")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    patches = load_validation(args.noisy_mat, args.gt_mat, args.max_patches)
    model = load_nafnet(args.weights, device)
    refs = install_branch_gates(model)
    ref_by_name = {ref.name: ref for ref in refs}
    candidates = args.candidates if args.candidates else default_candidates()
    unknown = [name for name in candidates if name not in ref_by_name]
    if unknown:
        raise ValueError(f"unknown candidates: {unknown[:8]}")

    print(f"device: {device}")
    print(f"patches: {len(patches)}")
    print(f"candidates: {len(candidates)}")
    baseline = evaluate(model, device, patches, collect_outputs=True)
    teacher_outputs = baseline.pop("outputs")
    print(
        "baseline: psnr={psnr_out:.3f} noisy={psnr_in:.3f} ms={ms_patch:.1f}".format(
            **baseline
        )
    )

    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(candidates, start=1):
        reset_gates(refs)
        set_branch(ref_by_name, name, 0.0)
        result = evaluate(model, device, patches, teacher_outputs=teacher_outputs)
        saved = branch_gmac(name, args.height, args.width)
        drop = float(result["psnr_out"] - baseline["psnr_out"])
        row = {
            "branch": name,
            "psnr": float(result["psnr_out"]),
            "drop": drop,
            "saved_gmac": saved,
            "db_per_gmac": drop / max(saved, 1e-9),
            "teacher_mse": float(result.get("teacher_mse", 0.0)),
            "ms_patch": float(result["ms_patch"]),
        }
        rows.append(row)
        print(
            f"[{idx:02d}/{len(candidates)}] {name:14s} psnr={row['psnr']:.3f} "
            f"drop={row['drop']:+.3f} saved={saved:.3f} teacher_mse={row['teacher_mse']:.8f}",
            flush=True,
        )
    reset_gates(refs)

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output_json) if args.output_json else out_md.with_suffix(".json")
    payload = {
        "args": vars(args),
        "device": str(device),
        "baseline": baseline,
        "rows": rows,
    }
    out_md.write_text(markdown(args, device, baseline, rows), encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
