"""Train NagiV2 HDR-first pilot models."""
from __future__ import annotations

import argparse
import glob
import math
import re
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..data import (
    ChunkedShuffleSampler,
    MIXTURE_SOURCE_NAMES,
    MixturePatchDataset,
    SIDDPatchDataset,
    WeakTeacherPatchDataset,
    find_polyu_pairs,
    find_sidd_pairs,
)
from ..devices import resolve_device
from ..losses import NagiV2Loss, masked_teacher_charbonnier
from ..models.nagi_v2 import NagiV2, build_nagi_v2
from ..transforms import linear_to_srgb


def lr_at(step: int, total: int, warmup: int, lr: float, lr_min: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


def build_model(cfg: dict) -> NagiV2:
    model_cfg = dict(cfg["model"])
    kind = str(model_cfg.get("kind", "nagi_v2"))
    if kind not in ("nagi_v2", "nagiperfect"):
        raise ValueError(f"train_v2 only supports kind='nagi_v2', got {kind!r}")
    return build_nagi_v2(model_cfg)


def _srgb_luma(x: torch.Tensor) -> torch.Tensor:
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x[:, :3] * weights).sum(dim=1, keepdim=True)


def _local_mean(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    k = max(1, int(kernel_size))
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x_pad, kernel_size=k, stride=1)


def _masked_charbonnier(
    diff: torch.Tensor,
    mask: torch.Tensor,
    *,
    charbonnier_eps: float,
) -> torch.Tensor:
    mask = mask.to(dtype=diff.dtype).clamp(0.0, 1.0)
    if diff.shape[1] != mask.shape[1]:
        mask = mask.expand(-1, diff.shape[1], -1, -1)
    mask_mean = mask.mean().clamp_min(1.0e-6)
    eps2 = float(charbonnier_eps) ** 2
    loss = (torch.sqrt(diff.pow(2) + eps2) * mask).sum()
    return loss / (mask_mean * mask.numel())


def _chroma_axes(x: torch.Tensor) -> torch.Tensor:
    rg = x[:, 0:1] - x[:, 1:2]
    by = x[:, 2:3] - 0.5 * (x[:, 0:1] + x[:, 1:2])
    return torch.cat([rg, by], dim=1)


def weak_teacher_loss(
    pred: torch.Tensor | dict[str, torch.Tensor],
    teacher: torch.Tensor,
    mask: torch.Tensor,
    *,
    luma_weight: float,
    chroma_weight: float,
    luma_hf_weight: float,
    luma_hf_tail_weight: float,
    luma_hf_tail_fraction: float,
    chroma_hf_weight: float,
    chroma_hf_tail_weight: float,
    chroma_hf_tail_fraction: float,
    chroma_axis_hf_weight: float,
    chroma_axis_hf_tail_weight: float,
    chroma_axis_hf_tail_fraction: float,
    magenta_axis_hf_weight: float,
    magenta_axis_hf_tail_weight: float,
    magenta_axis_hf_tail_fraction: float,
    baseline_luma_weight: float,
    baseline_chroma_weight: float,
    baseline_chroma_hf_weight: float,
    hf_kernel_size: int,
    charbonnier_eps: float,
) -> dict[str, torch.Tensor]:
    output = pred["output"] if isinstance(pred, dict) else pred
    output_srgb = linear_to_srgb(output.clamp_min(0.0))
    teacher_srgb = linear_to_srgb(teacher.clamp_min(0.0))
    mask = mask.to(dtype=output.dtype).clamp(0.0, 1.0)

    out_y = _srgb_luma(output_srgb)
    teacher_y = _srgb_luma(teacher_srgb)
    loss_luma = _masked_charbonnier(
        out_y - teacher_y,
        mask,
        charbonnier_eps=charbonnier_eps,
    )

    out_chroma = output_srgb - out_y
    teacher_chroma = teacher_srgb - teacher_y
    loss_chroma = _masked_charbonnier(
        out_chroma - teacher_chroma,
        mask,
        charbonnier_eps=charbonnier_eps,
    )

    out_y_hf = out_y - _local_mean(out_y, hf_kernel_size)
    teacher_y_hf = teacher_y - _local_mean(teacher_y, hf_kernel_size)
    loss_luma_hf = _masked_charbonnier(
        out_y_hf - teacher_y_hf,
        mask,
        charbonnier_eps=charbonnier_eps,
    )
    luma_hf_tail = torch.zeros((), dtype=output.dtype, device=output.device)
    if luma_hf_tail_weight > 0.0:
        tail = torch.sqrt((out_y_hf - teacher_y_hf).pow(2) + float(charbonnier_eps) ** 2)
        tail = tail * mask.expand_as(tail)
        tail = tail.flatten(1)
        fraction = max(min(float(luma_hf_tail_fraction), 1.0), 1.0e-6)
        k = max(1, int(tail.shape[1] * fraction))
        luma_hf_tail = tail.topk(k, dim=1, largest=True).values.mean()

    out_chroma_hf = out_chroma - _local_mean(out_chroma, hf_kernel_size)
    teacher_chroma_hf = teacher_chroma - _local_mean(teacher_chroma, hf_kernel_size)
    loss_chroma_hf = _masked_charbonnier(
        out_chroma_hf - teacher_chroma_hf,
        mask,
        charbonnier_eps=charbonnier_eps,
    )
    chroma_hf_tail = torch.zeros((), dtype=output.dtype, device=output.device)
    if chroma_hf_tail_weight > 0.0:
        tail = torch.sqrt((out_chroma_hf - teacher_chroma_hf).pow(2) + float(charbonnier_eps) ** 2)
        tail = tail * mask.expand_as(tail)
        tail = tail.flatten(1)
        fraction = max(min(float(chroma_hf_tail_fraction), 1.0), 1.0e-6)
        k = max(1, int(tail.shape[1] * fraction))
        chroma_hf_tail = tail.topk(k, dim=1, largest=True).values.mean()

    out_axis = _chroma_axes(output_srgb)
    teacher_axis = _chroma_axes(teacher_srgb)
    out_axis_hf = out_axis - _local_mean(out_axis, hf_kernel_size)
    teacher_axis_hf = teacher_axis - _local_mean(teacher_axis, hf_kernel_size)
    loss_chroma_axis_hf = _masked_charbonnier(
        out_axis_hf - teacher_axis_hf,
        mask,
        charbonnier_eps=charbonnier_eps,
    )
    chroma_axis_hf_tail = torch.zeros((), dtype=output.dtype, device=output.device)
    if chroma_axis_hf_tail_weight > 0.0:
        axis_diff = out_axis_hf - teacher_axis_hf
        rg = axis_diff[:, 0:1]
        by = axis_diff[:, 1:2]
        tail = torch.sqrt(0.5 * (rg.pow(2) + by.pow(2)) + float(charbonnier_eps) ** 2)
        tail = tail * mask
        tail = tail.flatten(1)
        fraction = max(min(float(chroma_axis_hf_tail_fraction), 1.0), 1.0e-6)
        k = max(1, int(tail.shape[1] * fraction))
        chroma_axis_hf_tail = tail.topk(k, dim=1, largest=True).values.mean()

    out_magenta = 0.5 * (output_srgb[:, 0:1] + output_srgb[:, 2:3]) - output_srgb[:, 1:2]
    teacher_magenta = 0.5 * (teacher_srgb[:, 0:1] + teacher_srgb[:, 2:3]) - teacher_srgb[:, 1:2]
    out_magenta_hf = out_magenta - _local_mean(out_magenta, hf_kernel_size)
    teacher_magenta_hf = teacher_magenta - _local_mean(teacher_magenta, hf_kernel_size)
    loss_magenta_axis_hf = _masked_charbonnier(
        out_magenta_hf - teacher_magenta_hf,
        mask,
        charbonnier_eps=charbonnier_eps,
    )
    magenta_axis_hf_tail = torch.zeros((), dtype=output.dtype, device=output.device)
    if magenta_axis_hf_tail_weight > 0.0:
        tail = torch.sqrt((out_magenta_hf - teacher_magenta_hf).pow(2) + float(charbonnier_eps) ** 2)
        tail = tail * mask
        tail = tail.flatten(1)
        fraction = max(min(float(magenta_axis_hf_tail_fraction), 1.0), 1.0e-6)
        k = max(1, int(tail.shape[1] * fraction))
        magenta_axis_hf_tail = tail.topk(k, dim=1, largest=True).values.mean()

    baseline_luma = torch.zeros((), dtype=output.dtype, device=output.device)
    baseline_chroma = torch.zeros((), dtype=output.dtype, device=output.device)
    baseline_chroma_hf = torch.zeros((), dtype=output.dtype, device=output.device)
    if isinstance(pred, dict) and "output_pre_cleanup" in pred:
        baseline = pred["output_pre_cleanup"].detach()
        baseline_srgb = linear_to_srgb(baseline.clamp_min(0.0))
        baseline_y = _srgb_luma(baseline_srgb)
        baseline_luma = _masked_charbonnier(
            out_y - baseline_y,
            mask,
            charbonnier_eps=charbonnier_eps,
        )
        baseline_chroma_value = baseline_srgb - baseline_y
        baseline_chroma = _masked_charbonnier(
            out_chroma - baseline_chroma_value,
            mask,
            charbonnier_eps=charbonnier_eps,
        )
        baseline_chroma_hf_value = baseline_chroma_value - _local_mean(baseline_chroma_value, hf_kernel_size)
        baseline_chroma_hf = _masked_charbonnier(
            out_chroma_hf - baseline_chroma_hf_value,
            mask,
            charbonnier_eps=charbonnier_eps,
        )

    total = (
        float(luma_weight) * loss_luma
        + float(chroma_weight) * loss_chroma
        + float(luma_hf_weight) * loss_luma_hf
        + float(luma_hf_tail_weight) * luma_hf_tail
        + float(chroma_hf_weight) * loss_chroma_hf
        + float(chroma_hf_tail_weight) * chroma_hf_tail
        + float(chroma_axis_hf_weight) * loss_chroma_axis_hf
        + float(chroma_axis_hf_tail_weight) * chroma_axis_hf_tail
        + float(magenta_axis_hf_weight) * loss_magenta_axis_hf
        + float(magenta_axis_hf_tail_weight) * magenta_axis_hf_tail
        + float(baseline_luma_weight) * baseline_luma
        + float(baseline_chroma_weight) * baseline_chroma
        + float(baseline_chroma_hf_weight) * baseline_chroma_hf
    )
    return {
        "total": total,
        "weak_luma": loss_luma.detach(),
        "weak_chroma": loss_chroma.detach(),
        "weak_luma_hf": loss_luma_hf.detach(),
        "weak_luma_hf_tail": luma_hf_tail.detach(),
        "weak_chroma_hf": loss_chroma_hf.detach(),
        "weak_chroma_hf_tail": chroma_hf_tail.detach(),
        "weak_chroma_axis_hf": loss_chroma_axis_hf.detach(),
        "weak_chroma_axis_hf_tail": chroma_axis_hf_tail.detach(),
        "weak_magenta_axis_hf": loss_magenta_axis_hf.detach(),
        "weak_magenta_axis_hf_tail": magenta_axis_hf_tail.detach(),
        "weak_baseline_luma": baseline_luma.detach(),
        "weak_baseline_chroma": baseline_chroma.detach(),
        "weak_baseline_chroma_hf": baseline_chroma_hf.detach(),
        "weak_mask": mask.mean().detach(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NagiV2 pilot.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sidd-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--max-iters", type=int, default=0, help="Debug override for train.total_iters.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt-prefix", default="nagi_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    loss_cfg = cfg["loss"]
    distill_weight = float(loss_cfg.get("distill_weight", 0.0))
    mixture_cfg = data_cfg.get("mixture")

    if mixture_cfg:
        # ---- Mixture path: infinite stream over {SIDD, PolyU, synthetic}. ----
        sidd_pairs = find_sidd_pairs(args.sidd_root)
        max_pairs = int(data_cfg.get("max_pairs", 0))
        if max_pairs > 0:
            sidd_pairs = sidd_pairs[:max_pairs]
        polyu_pairs: list[tuple[str, str]] = []
        polyu_root = data_cfg.get("polyu_root")
        if polyu_root:
            polyu_pairs = find_polyu_pairs(
                str(polyu_root), split=str(data_cfg.get("polyu_split", "CroppedImages"))
            )
            max_polyu_pairs = int(data_cfg.get("polyu_max_pairs", 0))
            if max_polyu_pairs > 0:
                polyu_pairs = polyu_pairs[:max_polyu_pairs]
        synth_cfg = dict(data_cfg.get("synthetic") or {})
        teacher_root = data_cfg.get("teacher_root")
        ds = MixturePatchDataset(
            sidd_pairs=sidd_pairs,
            polyu_pairs=polyu_pairs,
            weights=dict(mixture_cfg),
            patch_size=int(data_cfg["patch_size"]),
            exposure_jitter=tuple(data_cfg["exposure_jitter"]) if data_cfg.get("exposure_jitter") else None,
            flip_rot=bool(data_cfg.get("flip_rot", True)),
            seed=args.seed,
            burst_length=int(data_cfg.get("burst_length", 8)),
            teacher_root=str(teacher_root) if teacher_root else None,
            with_teacher=distill_weight > 0.0,
            synth_a_range=tuple(synth_cfg.get("a_range", (3.0e-4, 2.0e-2))),
            synth_b_range=tuple(synth_cfg.get("b_range", (1.0e-6, 1.0e-3))),
            synth_chroma_prob=float(synth_cfg.get("chroma_noise_prob", 0.5)),
            synth_chroma_scale=float(synth_cfg.get("chroma_noise_scale", 0.35)),
        )
        num_workers = int(data_cfg.get("num_workers", 1))
        dl_kwargs = dict(
            batch_size=int(cfg["train"]["batch_size"]),
            num_workers=num_workers,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        if num_workers > 0 and "prefetch_factor" in data_cfg:
            dl_kwargs["prefetch_factor"] = int(data_cfg["prefetch_factor"])
        dl = DataLoader(ds, **dl_kwargs)
        print(
            f"mixture weights: {ds.weights} | sidd pairs: {len(sidd_pairs)} "
            f"polyu pairs: {len(polyu_pairs)} clean images (synth): {len(ds._clean_paths)}"
        )
        if distill_weight > 0.0:
            if ds.num_teacher_files == 0:
                print(
                    "WARNING: loss.distill_weight > 0 but no precomputed teacher outputs "
                    "were found (run nagi_denoise.pipeline.precompute_teacher). "
                    "The distillation term will be zero for every sample."
                )
            else:
                print(f"teacher outputs available for {ds.num_teacher_files}/{len(sidd_pairs)} SIDD pairs")
    else:
        pairs = None
        max_pairs = int(data_cfg.get("max_pairs", 0))
        if max_pairs > 0:
            pairs = find_sidd_pairs(args.sidd_root)[:max_pairs]

        polyu_root = data_cfg.get("polyu_root")
        if polyu_root:
            sidd_pairs = pairs if pairs is not None else find_sidd_pairs(args.sidd_root)
            polyu_pairs = find_polyu_pairs(
                str(polyu_root),
                split=str(data_cfg.get("polyu_split", "CroppedImages")),
            )
            max_polyu_pairs = int(data_cfg.get("polyu_max_pairs", 0))
            if max_polyu_pairs > 0:
                polyu_pairs = polyu_pairs[:max_polyu_pairs]
            if not polyu_pairs:
                raise FileNotFoundError(f"No PolyU real/mean pairs discovered under {polyu_root}")
            pairs = list(sidd_pairs) + list(polyu_pairs)
            print(f"extra PolyU pairs: {len(polyu_pairs)}")

        ds = SIDDPatchDataset(
            root=args.sidd_root,
            patch_size=int(data_cfg["patch_size"]),
            patches_per_image=int(data_cfg["patches_per_image"]),
            exposure_jitter=tuple(data_cfg["exposure_jitter"]) if data_cfg.get("exposure_jitter") else None,
            flip_rot=bool(data_cfg.get("flip_rot", True)),
            seed=args.seed,
            pairs=pairs,
            synth_prob=float(data_cfg.get("synth_prob", 0.0)),
            gauss_sigma_max=float(data_cfg.get("gauss_sigma_max", 30.0)),
            poisson_lambda_range=tuple(data_cfg.get("poisson_lambda_range", (10.0, 100.0))),
            jpeg_q_range=tuple(data_cfg.get("jpeg_q_range", (60, 100))),
            return_teacher=False,
            output_space="linear",
            randomize_each_access=bool(data_cfg.get("randomize_each_access", True)),
        )
        num_workers = int(data_cfg.get("num_workers", 1))
        sampler = ChunkedShuffleSampler(
            num_pairs=len(ds.pairs),
            patches_per_image=ds.patches_per_image,
            chunk_size=int(data_cfg.get("chunk_size", ds.patches_per_image)),
            seed=args.seed,
        )
        dl_kwargs = dict(
            batch_size=int(cfg["train"]["batch_size"]),
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        if num_workers > 0 and "prefetch_factor" in data_cfg:
            dl_kwargs["prefetch_factor"] = int(data_cfg["prefetch_factor"])
        dl = DataLoader(ds, **dl_kwargs)

    weak_dl = None
    weak_cfg = dict(cfg.get("weak_teacher") or {})
    if weak_cfg.get("enabled", False):
        weak_pairs_cfg = weak_cfg.get("pairs") or []
        weak_pairs: list[tuple[str, str]] = []
        for item in weak_pairs_cfg:
            if isinstance(item, dict):
                weak_pairs.append((str(item["noisy"]), str(item["teacher"])))
            else:
                weak_pairs.append((str(item[0]), str(item[1])))
        weak_ds = WeakTeacherPatchDataset(
            pairs=weak_pairs,
            patch_size=int(weak_cfg.get("patch_size", data_cfg["patch_size"])),
            patches_per_image=int(weak_cfg.get("patches_per_image", 64)),
            exposure_jitter=tuple(weak_cfg["exposure_jitter"]) if weak_cfg.get("exposure_jitter") else None,
            flip_rot=bool(weak_cfg.get("flip_rot", True)),
            seed=args.seed + int(weak_cfg.get("seed_offset", 100000)),
            randomize_each_access=bool(weak_cfg.get("randomize_each_access", True)),
            structure_sigma=float(weak_cfg.get("structure_sigma", 1.2)),
            detail_sigma=float(weak_cfg.get("detail_sigma", 2.8)),
            detail_threshold=float(weak_cfg.get("detail_threshold", 0.018)),
            detail_transition=float(weak_cfg.get("detail_transition", 0.010)),
            edge_sigma=float(weak_cfg.get("edge_sigma", 1.0)),
            edge_threshold=float(weak_cfg.get("edge_threshold", 0.030)),
            edge_transition=float(weak_cfg.get("edge_transition", 0.015)),
            highlight_threshold=float(weak_cfg.get("highlight_threshold", 1.0)),
            highlight_transition=float(weak_cfg.get("highlight_transition", 0.25)),
            delta_threshold=float(weak_cfg.get("delta_threshold", 0.002)),
            delta_transition=float(weak_cfg.get("delta_transition", 0.010)),
            min_mask_mean=float(weak_cfg.get("min_mask_mean", 0.02)),
        )
        weak_sampler = ChunkedShuffleSampler(
            num_pairs=len(weak_ds.pairs),
            patches_per_image=weak_ds.patches_per_image,
            chunk_size=int(weak_cfg.get("chunk_size", weak_ds.patches_per_image)),
            seed=args.seed + int(weak_cfg.get("seed_offset", 100000)),
        )
        weak_dl = DataLoader(
            weak_ds,
            batch_size=int(weak_cfg.get("batch_size", cfg["train"]["batch_size"])),
            sampler=weak_sampler,
            num_workers=int(weak_cfg.get("num_workers", 0)),
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )
        print(f"weak teacher pairs: {len(weak_ds.pairs)}, items: {len(weak_ds)}")

    model = build_model(cfg).to(device=device, dtype=torch.float32)
    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)

    train_cfg = cfg["train"]
    total_iters = int(args.max_iters or train_cfg["total_iters"])
    warmup = int(train_cfg["warmup_iters"])
    lr = float(train_cfg["lr"])
    lr_min = float(train_cfg["lr_min"])
    grad_clip = float(train_cfg["grad_clip"])
    log_every = int(train_cfg["log_every"])
    save_every = int(train_cfg["save_every"])
    ema_decay = float(train_cfg["ema_decay"])
    accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    keep_last = int(train_cfg.get("keep_last_ckpts", 0))
    skip_nonfinite = bool(train_cfg.get("skip_nonfinite", False))
    prefix = args.ckpt_prefix

    opt = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    loss_cfg = cfg["loss"]
    criterion = NagiV2Loss(
        compress_fn=model.compress,
        final_weight=float(loss_cfg.get("final_weight", 1.0)),
        base_weight=float(loss_cfg.get("base_weight", 0.35)),
        fft_weight=float(loss_cfg.get("fft_weight", 0.0)),
        srgb_weight=float(loss_cfg.get("srgb_weight", 0.0)),
        srgb_base_weight=float(loss_cfg.get("srgb_base_weight", 0.0)),
        body_srgb_weight=float(loss_cfg.get("body_srgb_weight", 0.0)),
        body_srgb_base_weight=float(loss_cfg.get("body_srgb_base_weight", 0.0)),
        flat_srgb_hf_weight=float(loss_cfg.get("flat_srgb_hf_weight", 0.0)),
        flat_luma_hf_weight=float(loss_cfg.get("flat_luma_hf_weight", 0.0)),
        flat_luma_tail_weight=float(loss_cfg.get("flat_luma_tail_weight", 0.0)),
        flat_luma_tail_fraction=float(loss_cfg.get("flat_luma_tail_fraction", 0.02)),
        flat_chroma_hf_weight=float(loss_cfg.get("flat_chroma_hf_weight", 0.0)),
        flat_chroma_damp_weight=float(loss_cfg.get("flat_chroma_damp_weight", 0.0)),
        flat_chroma_distill_weight=float(loss_cfg.get("flat_chroma_distill_weight", 0.0)),
        flat_chroma_distill_strength=float(loss_cfg.get("flat_chroma_distill_strength", 0.5)),
        flat_chroma_distill_kernel_size=int(loss_cfg.get("flat_chroma_distill_kernel_size", 9)),
        flat_chroma_delta_distill_weight=float(loss_cfg.get("flat_chroma_delta_distill_weight", 0.0)),
        chroma_smooth_gate_l1_weight=float(loss_cfg.get("chroma_smooth_gate_l1_weight", 0.0)),
        edge_luma_hf_weight=float(loss_cfg.get("edge_luma_hf_weight", 0.0)),
        edge_threshold=float(loss_cfg.get("edge_threshold", 0.040)),
        edge_transition=float(loss_cfg.get("edge_transition", 0.015)),
        flat_threshold=float(loss_cfg.get("flat_threshold", 0.015)),
        flat_transition=float(loss_cfg.get("flat_transition", 0.010)),
        flat_kernel_size=int(loss_cfg.get("flat_kernel_size", 9)),
        linear_weight=float(loss_cfg.get("linear_weight", 0.08)),
        highlight_luma_weight=float(loss_cfg.get("highlight_luma_weight", 0.20)),
        highlight_chroma_weight=float(loss_cfg.get("highlight_chroma_weight", 0.08)),
        highlight_ratio_weight=float(loss_cfg.get("highlight_ratio_weight", 0.0)),
        highlight_luma_weight_end=loss_cfg.get("highlight_luma_weight_end"),
        highlight_chroma_weight_end=loss_cfg.get("highlight_chroma_weight_end"),
        highlight_ratio_weight_end=loss_cfg.get("highlight_ratio_weight_end"),
        highlight_ramp_start=float(loss_cfg.get("highlight_ramp_start", 0.0)),
        highlight_ramp_end=float(loss_cfg.get("highlight_ramp_end", 1.0)),
        detail_weight=float(loss_cfg.get("detail_weight", 0.12)),
        confidence_l1_weight=float(loss_cfg.get("confidence_l1_weight", 0.002)),
        highlight_threshold=float(loss_cfg.get("highlight_threshold", 1.0)),
        highlight_transition=float(loss_cfg.get("highlight_transition", 0.5)),
        detail_kernel_size=int(loss_cfg.get("detail_kernel_size", 9)),
        linear_clamp=float(loss_cfg.get("linear_clamp", 16.0)),
        charbonnier_eps=float(loss_cfg.get("charbonnier_eps", 1.0e-3)),
    ).to(device)

    print(f"device: {device}")
    if not mixture_cfg:
        print(f"pairs: {len(ds.pairs)}, dataset items: {len(ds)}")
    print(f"params: {model.param_count() / 1e6:.3f}M")
    print(f"effective batch: {cfg['train']['batch_size']} x {accum_steps}")
    print(f"output: {out_dir}")

    start_step = 0
    if args.resume_latest:
        final_ckpt = out_dir / f"{prefix}_final.pt"
        if final_ckpt.exists():
            ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=False)
            if int(ckpt.get("step", -1)) >= total_iters:
                args.resume = str(final_ckpt)
                print(f"--resume-latest: found completed {args.resume}")
        if not args.resume:
            pattern = str(out_dir / f"{prefix}_[0-9]*.pt")
            candidates = sorted(
                glob.glob(pattern),
                key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)),
            )
            if candidates:
                args.resume = candidates[-1]
                print(f"--resume-latest: found {args.resume}")
            else:
                print("--resume-latest: no checkpoint found, starting from scratch")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt["state_dict"]))
        ema.load_state_dict(ckpt.get("ema_state_dict", ckpt["state_dict"]))
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        init_ckpt = train_cfg.get("init_checkpoint")
        if init_ckpt:
            state_key = str(train_cfg.get("init_state", "state_dict"))
            init_strict = bool(train_cfg.get("init_strict", True))
            ckpt = torch.load(str(init_ckpt), map_location="cpu", weights_only=False)
            state = ckpt.get(state_key)
            if state is None:
                raise KeyError(f"init checkpoint has no state key {state_key!r}: {init_ckpt}")
            model_result = model.load_state_dict(state, strict=init_strict)
            ema_result = ema.load_state_dict(state, strict=init_strict)
            print(f"initialized from {init_ckpt} ({state_key}, strict={init_strict}); fresh optimizer")
            if not init_strict:
                print(f"missing keys: {model_result.missing_keys}")
                print(f"unexpected keys: {model_result.unexpected_keys}")

    if bool(train_cfg.get("freeze_for_chroma_branch", False)):
        if (
            getattr(model, "chroma_head", None) is None
            and getattr(model, "chroma_smooth_head", None) is None
            and getattr(model, "chroma_cleanup_head", None) is None
            and getattr(model, "luma_smooth_head", None) is None
        ):
            raise ValueError("freeze_for_chroma_branch requires a chroma branch")
        trainable_prefixes = train_cfg.get("trainable_prefixes")
        if trainable_prefixes is None:
            trainable_prefixes = ["chroma_head.", "chroma_smooth_head.", "chroma_cleanup_head.", "luma_smooth_head."]
        trainable_prefixes = tuple(str(prefix) for prefix in trainable_prefixes)
        if not trainable_prefixes:
            raise ValueError("trainable_prefixes must not be empty when freeze_for_chroma_branch is enabled")
        for name, param in model.named_parameters():
            param.requires_grad_(any(name.startswith(prefix) for prefix in trainable_prefixes))
        for name, param in ema.named_parameters():
            param.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"freeze_for_chroma_branch: trainable prefixes={trainable_prefixes}, params={trainable}")

    log_path = out_dir / "train.log"
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== train-v2 {prefix} ===\n")

    data_iter = iter(dl)
    weak_iter = iter(weak_dl) if weak_dl is not None else None
    step = start_step
    t0 = time.time()
    source_counts: dict[str, int] = {}
    model.train()
    while step < total_iters:
        cur_lr = lr_at(step, total_iters, warmup, lr, lr_min)
        criterion.set_step(step, total_iters)
        for group in opt.param_groups:
            group["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        accum_log: dict[str, float] = {}
        bad_step = False

        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)

            teacher = None
            has_teacher = None
            if len(batch) == 5:  # mixture: (noisy, clean, teacher, has_teacher, source_id)
                noisy, clean, teacher, has_teacher, source_id = batch
                for sid in source_id.tolist():
                    name = MIXTURE_SOURCE_NAMES.get(int(sid), str(int(sid)))
                    source_counts[name] = source_counts.get(name, 0) + 1
            else:
                noisy, clean = batch

            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(noisy, return_aux=True)
            losses = criterion(pred, clean)
            if teacher is not None and distill_weight > 0.0:
                teacher = teacher.to(device, non_blocking=True)
                has_teacher = has_teacher.to(device, non_blocking=True)
                loss_distill = masked_teacher_charbonnier(
                    model.compress(pred["output"]),
                    model.compress(teacher),
                    has_teacher,
                    eps=float(loss_cfg.get("charbonnier_eps", 1.0e-3)),
                )
                losses["total"] = losses["total"] + distill_weight * loss_distill
                losses["distill"] = loss_distill.detach()
                losses["teacher_frac"] = has_teacher.mean().detach()
            if weak_dl is not None and weak_iter is not None:
                try:
                    weak_noisy, weak_teacher, weak_mask = next(weak_iter)
                except StopIteration:
                    weak_iter = iter(weak_dl)
                    weak_noisy, weak_teacher, weak_mask = next(weak_iter)
                weak_noisy = weak_noisy.to(device, non_blocking=True)
                weak_teacher = weak_teacher.to(device, non_blocking=True)
                weak_mask = weak_mask.to(device, non_blocking=True)
                weak_pred = model(weak_noisy, return_aux=True)
                weak_losses = weak_teacher_loss(
                    weak_pred,
                    weak_teacher,
                    weak_mask,
                    luma_weight=float(weak_cfg.get("luma_weight", 0.0)),
                    chroma_weight=float(weak_cfg.get("chroma_weight", 0.0)),
                    luma_hf_weight=float(weak_cfg.get("luma_hf_weight", 0.0)),
                    luma_hf_tail_weight=float(weak_cfg.get("luma_hf_tail_weight", 0.0)),
                    luma_hf_tail_fraction=float(weak_cfg.get("luma_hf_tail_fraction", 0.02)),
                    chroma_hf_weight=float(weak_cfg.get("chroma_hf_weight", 0.0)),
                    chroma_hf_tail_weight=float(weak_cfg.get("chroma_hf_tail_weight", 0.0)),
                    chroma_hf_tail_fraction=float(weak_cfg.get("chroma_hf_tail_fraction", 0.02)),
                    chroma_axis_hf_weight=float(weak_cfg.get("chroma_axis_hf_weight", 0.0)),
                    chroma_axis_hf_tail_weight=float(weak_cfg.get("chroma_axis_hf_tail_weight", 0.0)),
                    chroma_axis_hf_tail_fraction=float(weak_cfg.get("chroma_axis_hf_tail_fraction", 0.02)),
                    magenta_axis_hf_weight=float(weak_cfg.get("magenta_axis_hf_weight", 0.0)),
                    magenta_axis_hf_tail_weight=float(weak_cfg.get("magenta_axis_hf_tail_weight", 0.0)),
                    magenta_axis_hf_tail_fraction=float(weak_cfg.get("magenta_axis_hf_tail_fraction", 0.02)),
                    baseline_luma_weight=float(weak_cfg.get("baseline_luma_weight", 0.0)),
                    baseline_chroma_weight=float(weak_cfg.get("baseline_chroma_weight", 0.0)),
                    baseline_chroma_hf_weight=float(weak_cfg.get("baseline_chroma_hf_weight", 0.0)),
                    hf_kernel_size=int(weak_cfg.get("hf_kernel_size", 9)),
                    charbonnier_eps=float(loss_cfg.get("charbonnier_eps", 1.0e-3)),
                )
                weak_total = weak_losses["total"] * float(weak_cfg.get("weight", 1.0))
                losses["total"] = losses["total"] + weak_total
                for key in (
                    "weak_luma",
                    "weak_chroma",
                    "weak_luma_hf",
                    "weak_luma_hf_tail",
                    "weak_chroma_hf",
                    "weak_chroma_hf_tail",
                    "weak_chroma_axis_hf",
                    "weak_chroma_axis_hf_tail",
                    "weak_magenta_axis_hf",
                    "weak_magenta_axis_hf_tail",
                    "weak_baseline_luma",
                    "weak_baseline_chroma",
                    "weak_baseline_chroma_hf",
                    "weak_mask",
                ):
                    losses[key] = weak_losses[key]
            if not torch.isfinite(losses["total"]).item():
                bad_terms = []
                for key, value in losses.items():
                    finite = torch.isfinite(value).all().item()
                    if not finite:
                        bad_terms.append(key)
                term_summary = " ".join(
                    f"{key}={float(value.detach().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).mean().item()):.6g}"
                    for key, value in losses.items()
                )
                msg = f"[fatal {step:7d}] non-finite loss detected; stopping before optimizer step/checkpoint"
                if bad_terms:
                    msg += f"; bad_terms={','.join(bad_terms)}"
                print(msg)
                print(f"[fatal {step:7d}] terms {term_summary}")
                log_f.write(msg + "\n")
                log_f.write(f"[fatal {step:7d}] terms {term_summary}\n")
                if skip_nonfinite:
                    opt.zero_grad(set_to_none=True)
                    bad_step = True
                    break
                log_f.close()
                raise FloatingPointError(msg)
            (losses["total"] / accum_steps).backward()
            for key, value in losses.items():
                accum_log[key] = accum_log.get(key, 0.0) + float(value.detach().item()) / accum_steps

        if bad_step:
            step += 1
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        with torch.no_grad():
            for p, ep in zip(model.parameters(), ema.parameters()):
                ep.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

        if step % log_every == 0:
            dt = time.time() - t0
            ips = (step - start_step + 1) / max(dt, 1.0e-6)
            msg = (
                f"[{step:7d}/{total_iters}] loss={accum_log['total']:.4f} "
                f"final={accum_log['final']:.4f} base={accum_log['base']:.4f} "
            )
            for key in (
                "fft",
                "distill",
                "teacher_frac",
                "srgb",
                "srgb_base",
                "body_srgb",
                "body_srgb_base",
                "flat_srgb_hf",
                "flat_luma_hf",
                "flat_luma_tail",
                "flat_chroma_hf",
                "flat_chroma_damp",
                "flat_chroma_distill",
                "flat_chroma_delta_distill",
                "flat_chroma_distill_mask_mean",
                "chroma_smooth_gate_l1",
                "edge_luma_hf",
                "edge_mask_mean",
                "flat_mask_mean",
                "linear",
                "highlight_luma",
                "highlight_chroma",
                "highlight_ratio",
                "detail_luma",
                "confidence_l1",
                "highlight_mask_mean",
                "highlight_luma_weight",
                "highlight_chroma_weight",
                "highlight_ratio_weight",
                "weak_luma",
                "weak_chroma",
                "weak_luma_hf",
                "weak_luma_hf_tail",
                "weak_chroma_hf",
                "weak_chroma_hf_tail",
                "weak_chroma_axis_hf",
                "weak_chroma_axis_hf_tail",
                "weak_magenta_axis_hf",
                "weak_magenta_axis_hf_tail",
                "weak_baseline_luma",
                "weak_baseline_chroma",
                "weak_baseline_chroma_hf",
                "weak_mask",
            ):
                if key in accum_log:
                    msg += f"{key}={accum_log[key]:.4f} "
            if source_counts:
                counts = " ".join(f"src_{name}={source_counts.get(name, 0)}" for name in ("sidd", "polyu", "synthetic"))
                msg += counts + " "
            msg += f"lr={cur_lr:.2e} ips={ips:.2f}"
            print(msg)
            log_f.write(msg + "\n")

        if save_every > 0 and step > 0 and step % save_every == 0:
            save_ckpt(out_dir / f"{prefix}_{step:07d}.pt", step, model, ema, opt, cfg)
            if keep_last > 0:
                rotate_ckpts(out_dir, prefix, keep_last)

        step += 1

    save_ckpt(out_dir / f"{prefix}_final.pt", step, model, ema, opt, cfg)
    log_f.close()
    print("done.")


def rotate_ckpts(out_dir: Path, prefix: str, keep: int) -> None:
    candidates = sorted(
        glob.glob(str(out_dir / f"{prefix}_[0-9]*.pt")),
        key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)),
    )
    for old in candidates[:-keep]:
        try:
            Path(old).unlink()
        except OSError:
            pass


def save_ckpt(
    path: Path,
    step: int,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    opt: AdamW,
    cfg: dict,
) -> None:
    # Stamp an explicit arch marker into the saved model config so
    # Denoiser.load can distinguish NagiV2 checkpoints from NagiNR ones.
    cfg = dict(cfg)
    cfg["model"] = dict(cfg.get("model") or {})
    cfg["model"]["arch"] = "nagi_v2"
    payload = {
        "step": step,
        "state_dict": ema.state_dict(),
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer": opt.state_dict(),
        "config": cfg,
        "model_kind": "nagi_v2",
    }
    torch.save(payload, str(path))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
