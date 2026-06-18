"""Audit whether half-res NAFNet residual output is cleanup-model friendly."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image

from nagi_nr.devices import resolve_device
from nagi_nr_bench.eval_sidd_val import psnr_srgb
from nagi_nr_bench.third_party.nafnet import NAFNet


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


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


def load_validation(noisy_mat: str, gt_mat: str, max_patches: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
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
    patches: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for i in range(noisy_arr.shape[0]):
        for j in range(noisy_arr.shape[1]):
            if len(patches) >= limit:
                return patches
            patches.append((i, j, noisy_arr[i, j], gt_arr[i, j]))
    return patches


def to_tensor(x_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(x_uint8)
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )


def to_uint8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def rgb_to_ycbcr(x: np.ndarray) -> np.ndarray:
    mat = np.array(
        [
            [0.299000, 0.587000, 0.114000],
            [-0.168736, -0.331264, 0.500000],
            [0.500000, -0.418688, -0.081312],
        ],
        dtype=np.float32,
    )
    return x @ mat.T


def band_energy_2d(x: np.ndarray) -> dict[str, float]:
    h, w = x.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    power = np.abs(np.fft.fft2(x)) ** 2
    total = float(power.sum()) + 1e-18
    low = float(power[radius <= 0.10].sum()) / total
    mid = float(power[(radius > 0.10) & (radius <= 0.25)].sum()) / total
    high = float(power[radius > 0.25].sum()) / total
    return {"low": low, "mid": mid, "high": high}


def summarize_delta(delta_rgb: np.ndarray, teacher_residual: np.ndarray) -> dict[str, float]:
    delta_ycc = rgb_to_ycbcr(delta_rgb)
    abs_mean = np.mean(np.abs(delta_ycc), axis=(0, 1))
    rms = np.sqrt(np.mean(delta_ycc * delta_ycc, axis=(0, 1)))
    teacher_energy = float(np.mean(teacher_residual * teacher_residual)) + 1e-18
    delta_energy = float(np.mean(delta_rgb * delta_rgb))

    bands_y = band_energy_2d(delta_ycc[..., 0])
    bands_cb = band_energy_2d(delta_ycc[..., 1])
    bands_cr = band_energy_2d(delta_ycc[..., 2])
    chroma_energy = float(np.mean(delta_ycc[..., 1:] * delta_ycc[..., 1:]))
    luma_energy = float(np.mean(delta_ycc[..., :1] * delta_ycc[..., :1])) + 1e-18
    return {
        "delta_rms_rgb": float(np.sqrt(delta_energy)),
        "delta_energy_vs_teacher_residual": delta_energy / teacher_energy,
        "mean_abs_y": float(abs_mean[0]),
        "mean_abs_cb": float(abs_mean[1]),
        "mean_abs_cr": float(abs_mean[2]),
        "rms_y": float(rms[0]),
        "rms_cb": float(rms[1]),
        "rms_cr": float(rms[2]),
        "chroma_to_luma_energy": chroma_energy / luma_energy,
        "band_y_low": bands_y["low"],
        "band_y_mid": bands_y["mid"],
        "band_y_high": bands_y["high"],
        "band_cb_low": bands_cb["low"],
        "band_cb_mid": bands_cb["mid"],
        "band_cb_high": bands_cb["high"],
        "band_cr_low": bands_cr["low"],
        "band_cr_mid": bands_cr["mid"],
        "band_cr_high": bands_cr["high"],
    }


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def save_example(
    path: Path,
    noisy: np.ndarray,
    coarse: np.ndarray,
    teacher: np.ndarray,
    gt: np.ndarray,
    delta: np.ndarray,
) -> None:
    delta_vis = np.clip(delta * 8.0 + 0.5, 0.0, 1.0)
    panels = [
        noisy,
        to_uint8(coarse),
        to_uint8(teacher),
        gt,
        to_uint8(delta_vis),
    ]
    strip = np.concatenate(panels, axis=1)
    Image.fromarray(strip, mode="RGB").save(path)


@torch.inference_mode()
def forward_full_and_coarse(model: NAFNet, x: torch.Tensor, scale: float) -> tuple[np.ndarray, np.ndarray]:
    teacher = model(x).clamp(0, 1)
    low = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    low_out = model(low).clamp(0, 1)
    low_residual = low_out - low
    residual_up = F.interpolate(low_residual, size=x.shape[-2:], mode="bilinear", align_corners=False)
    coarse = (x + residual_up).clamp(0, 1)
    teacher_np = teacher.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32, copy=False)
    coarse_np = coarse.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32, copy=False)
    return teacher_np, coarse_np


def markdown_report(args: argparse.Namespace, device: torch.device, summary: dict[str, Any], examples: list[str]) -> str:
    lines = [
        "# NAFNet-Fast-C Residual Audit",
        "",
        f"Device: `{device}`",
        f"Weights: `{args.weights}`",
        f"Patches: `{summary['patches']}`",
        f"Scale: `{args.scale}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| noisy PSNR | {summary['psnr_noisy']:.3f} dB |",
        f"| coarse PSNR | {summary['psnr_coarse']:.3f} dB |",
        f"| teacher PSNR | {summary['psnr_teacher']:.3f} dB |",
        f"| cleanup gap | {summary['cleanup_gap_db']:.3f} dB |",
        f"| coarse vs teacher PSNR | {summary['psnr_coarse_teacher']:.3f} dB |",
        f"| ms/patch | {summary['ms_patch']:.1f} |",
        "",
        "## Delta Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    delta = summary["delta"]
    for key in [
        "delta_rms_rgb",
        "delta_energy_vs_teacher_residual",
        "mean_abs_y",
        "mean_abs_cb",
        "mean_abs_cr",
        "chroma_to_luma_energy",
        "band_y_low",
        "band_y_mid",
        "band_y_high",
        "band_cb_low",
        "band_cb_mid",
        "band_cb_high",
        "band_cr_low",
        "band_cr_mid",
        "band_cr_high",
    ]:
        lines.append(f"| {key} | {delta[key]:.6f} |")

    verdict = summary["verdict"]
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| check | result |",
            "| --- | --- |",
            f"| coarse >= {args.min_coarse_psnr:.2f} dB | {verdict['coarse_psnr']} |",
            f"| cleanup gap <= {args.max_cleanup_gap:.2f} dB | {verdict['cleanup_gap']} |",
            f"| luma high+mid >= {args.min_luma_mid_high:.2f} | {verdict['luma_mid_high']} |",
            f"| chroma/luma <= {args.max_chroma_luma:.2f} | {verdict['chroma_luma']} |",
            f"| overall | {verdict['overall']} |",
        ]
    )
    if examples:
        lines.extend(["", "## Examples", ""])
        for item in examples:
            lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit half-res NAFNet residual for cleanup feasibility.")
    ap.add_argument("--weights", default="benchmarks/nafnet/NAFNet-SIDD-width64.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    ap.add_argument("--max-patches", type=int, default=128)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--min-coarse-psnr", type=float, default=37.5)
    ap.add_argument("--max-cleanup-gap", type=float, default=2.0)
    ap.add_argument("--min-luma-mid-high", type=float, default=0.55)
    ap.add_argument("--max-chroma-luma", type=float, default=1.25)
    ap.add_argument("--output-dir", default="runs/nafnet_fast_cascade_audit")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"device: {device}")
    print(f"loading patches: {args.noisy_mat}")
    patches = load_validation(args.noisy_mat, args.gt_mat, args.max_patches)
    print(f"patches: {len(patches)}")
    print("loading NAFNet teacher")
    model = load_nafnet(args.weights, device)

    psnr_noisy: list[float] = []
    psnr_coarse: list[float] = []
    psnr_teacher: list[float] = []
    psnr_coarse_teacher: list[float] = []
    delta_rows: list[dict[str, float]] = []
    records: list[dict[str, Any]] = []

    start = time.perf_counter()
    for idx, (image_idx, block_idx, noisy, gt) in enumerate(patches, start=1):
        x = to_tensor(noisy, device)
        teacher, coarse = forward_full_and_coarse(model, x, args.scale)
        _sync(device)

        noisy_f = noisy.astype(np.float32) / 255.0
        teacher_u8 = to_uint8(teacher)
        coarse_u8 = to_uint8(coarse)
        delta = teacher - coarse
        teacher_residual = teacher - noisy_f
        row = summarize_delta(delta, teacher_residual)
        delta_rows.append(row)

        p_noisy = psnr_srgb(noisy, gt)
        p_coarse = psnr_srgb(coarse_u8, gt)
        p_teacher = psnr_srgb(teacher_u8, gt)
        p_ct = psnr_srgb(coarse_u8, teacher_u8)
        psnr_noisy.append(p_noisy)
        psnr_coarse.append(p_coarse)
        psnr_teacher.append(p_teacher)
        psnr_coarse_teacher.append(p_ct)
        records.append(
            {
                "idx": idx - 1,
                "image": image_idx,
                "block": block_idx,
                "psnr_noisy": p_noisy,
                "psnr_coarse": p_coarse,
                "psnr_teacher": p_teacher,
                "cleanup_gap_db": p_teacher - p_coarse,
                "psnr_coarse_teacher": p_ct,
                "delta": row,
            }
        )
        if idx % 16 == 0 or idx == len(patches):
            elapsed = time.perf_counter() - start
            print(
                f"[{idx:4d}/{len(patches)}] coarse={np.mean(psnr_coarse):.3f} "
                f"teacher={np.mean(psnr_teacher):.3f} ms/patch={elapsed / idx * 1000.0:.1f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    delta_summary = mean_dict(delta_rows)
    summary: dict[str, Any] = {
        "patches": len(patches),
        "scale": args.scale,
        "psnr_noisy": float(np.mean(psnr_noisy)),
        "psnr_coarse": float(np.mean(psnr_coarse)),
        "psnr_teacher": float(np.mean(psnr_teacher)),
        "cleanup_gap_db": float(np.mean(psnr_teacher) - np.mean(psnr_coarse)),
        "psnr_coarse_teacher": float(np.mean(psnr_coarse_teacher)),
        "elapsed_sec": elapsed,
        "ms_patch": elapsed / max(1, len(patches)) * 1000.0,
        "delta": delta_summary,
        "records": records,
    }

    luma_mid_high = delta_summary["band_y_mid"] + delta_summary["band_y_high"]
    verdict = {
        "coarse_psnr": "pass" if summary["psnr_coarse"] >= args.min_coarse_psnr else "fail",
        "cleanup_gap": "pass" if summary["cleanup_gap_db"] <= args.max_cleanup_gap else "fail",
        "luma_mid_high": "pass" if luma_mid_high >= args.min_luma_mid_high else "fail",
        "chroma_luma": "pass" if delta_summary["chroma_to_luma_energy"] <= args.max_chroma_luma else "fail",
    }
    verdict["overall"] = "pass" if all(value == "pass" for value in verdict.values()) else "fail"
    summary["verdict"] = verdict

    worst = sorted(records, key=lambda r: r["cleanup_gap_db"], reverse=True)[: max(0, args.examples)]
    example_paths: list[str] = []
    patch_by_id = {(i, j): (noisy, gt) for i, j, noisy, gt in patches}
    for rank, rec in enumerate(worst, start=1):
        noisy, gt = patch_by_id[(rec["image"], rec["block"])]
        x = to_tensor(noisy, device)
        teacher, coarse = forward_full_and_coarse(model, x, args.scale)
        _sync(device)
        path = ex_dir / f"rank{rank:02d}_img{rec['image']:02d}_blk{rec['block']:02d}.png"
        save_example(path, noisy, coarse, teacher, gt, teacher - coarse)
        example_paths.append(str(path))

    json_path = out_dir / "audit.json"
    md_path = out_dir / "audit.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_report(args, device, summary, example_paths), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"overall: {verdict['overall']}")


if __name__ == "__main__":
    main()
