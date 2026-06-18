"""Train NagiPerfect HDR-first pilot models."""
from __future__ import annotations

import argparse
import glob
import math
import re
import time
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import ChunkedShuffleSampler, SIDDPatchDataset, find_polyu_pairs, find_sidd_pairs
from .devices import resolve_device
from .losses import NagiPerfectLoss
from .nagiperfect import NagiPerfect, build_nagiperfect_preset


def lr_at(step: int, total: int, warmup: int, lr: float, lr_min: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


def build_model(cfg: dict) -> NagiPerfect:
    model_cfg = dict(cfg["model"])
    kind = str(model_cfg.pop("kind", "nagiperfect"))
    if kind != "nagiperfect":
        raise ValueError(f"train_perfect only supports kind='nagiperfect', got {kind!r}")
    preset = model_cfg.pop("preset", None)
    if preset:
        model = build_nagiperfect_preset(str(preset))
        if model_cfg:
            merged = {
                "img_channels": model.img_channels,
                "width": model.width,
                "enc_blk_nums": model.enc_blk_nums,
                "middle_blk_num": model.middle_blk_num,
                "dec_blk_nums": model.dec_blk_nums,
                "asinh_k": model.asinh_k,
                "detail_scale": model.detail_scale,
                "compressed_output_clamp": model.compressed_output_clamp,
                "confidence_bias": model._confidence_bias,
                "highlight_protect_threshold": model.highlight_protect_threshold,
                "highlight_protect_transition": model.highlight_protect_transition,
                "highlight_protect_strength": model.highlight_protect_strength,
                "chroma_branch": model.chroma_branch,
                "chroma_branch_scale": model.chroma_branch_scale,
                "chroma_gate_threshold": model.chroma_gate_threshold,
                "chroma_gate_transition": model.chroma_gate_transition,
                "chroma_gate_kernel_size": model.chroma_gate_kernel_size,
                "chroma_gate_highlight_threshold": model.chroma_gate_highlight_threshold,
                "chroma_gate_highlight_transition": model.chroma_gate_highlight_transition,
                "chroma_gate_strength": model.chroma_gate_strength,
                "chroma_branch_width": model.chroma_branch_width,
                "chroma_branch_blocks": model.chroma_branch_blocks,
                "chroma_branch_use_input": model.chroma_branch_use_input,
                "chroma_smooth_branch": model.chroma_smooth_branch,
                "chroma_smooth_strength": model.chroma_smooth_strength,
                "chroma_smooth_kernel_size": model.chroma_smooth_kernel_size,
                "chroma_smooth_gate_bias": model.chroma_smooth_gate_bias,
                "chroma_smooth_gate_scale": model.chroma_smooth_gate_scale,
            }
            merged.update(model_cfg)
            model = NagiPerfect(**merged)
        return model
    return NagiPerfect(**model_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NagiPerfect pilot.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sidd-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--max-iters", type=int, default=0, help="Debug override for train.total_iters.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt-prefix", default="nagiperfect")
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
    prefix = args.ckpt_prefix

    opt = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    loss_cfg = cfg["loss"]
    criterion = NagiPerfectLoss(
        compress_fn=model.compress,
        final_weight=float(loss_cfg.get("final_weight", 1.0)),
        base_weight=float(loss_cfg.get("base_weight", 0.35)),
        srgb_weight=float(loss_cfg.get("srgb_weight", 0.0)),
        srgb_base_weight=float(loss_cfg.get("srgb_base_weight", 0.0)),
        body_srgb_weight=float(loss_cfg.get("body_srgb_weight", 0.0)),
        body_srgb_base_weight=float(loss_cfg.get("body_srgb_base_weight", 0.0)),
        flat_srgb_hf_weight=float(loss_cfg.get("flat_srgb_hf_weight", 0.0)),
        flat_luma_hf_weight=float(loss_cfg.get("flat_luma_hf_weight", 0.0)),
        flat_chroma_hf_weight=float(loss_cfg.get("flat_chroma_hf_weight", 0.0)),
        flat_chroma_damp_weight=float(loss_cfg.get("flat_chroma_damp_weight", 0.0)),
        flat_chroma_distill_weight=float(loss_cfg.get("flat_chroma_distill_weight", 0.0)),
        flat_chroma_distill_strength=float(loss_cfg.get("flat_chroma_distill_strength", 0.5)),
        flat_chroma_distill_kernel_size=int(loss_cfg.get("flat_chroma_distill_kernel_size", 9)),
        flat_chroma_delta_distill_weight=float(loss_cfg.get("flat_chroma_delta_distill_weight", 0.0)),
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
        if getattr(model, "chroma_head", None) is None and getattr(model, "chroma_smooth_head", None) is None:
            raise ValueError("freeze_for_chroma_branch requires a chroma branch")
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("chroma_head.") or name.startswith("chroma_smooth_head."))
        for name, param in ema.named_parameters():
            param.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"freeze_for_chroma_branch: trainable params={trainable}")

    log_path = out_dir / "train.log"
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== train-perfect {prefix} ===\n")

    data_iter = iter(dl)
    step = start_step
    t0 = time.time()
    model.train()
    while step < total_iters:
        cur_lr = lr_at(step, total_iters, warmup, lr, lr_min)
        criterion.set_step(step, total_iters)
        for group in opt.param_groups:
            group["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        accum_log: dict[str, float] = {}

        for _ in range(accum_steps):
            try:
                noisy, clean = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                noisy, clean = next(data_iter)

            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(noisy, return_aux=True)
            losses = criterion(pred, clean)
            if not torch.isfinite(losses["total"]).item():
                msg = f"[fatal {step:7d}] non-finite loss detected; stopping before optimizer step/checkpoint"
                print(msg)
                log_f.write(msg + "\n")
                log_f.close()
                raise FloatingPointError(msg)
            (losses["total"] / accum_steps).backward()
            for key, value in losses.items():
                accum_log[key] = accum_log.get(key, 0.0) + float(value.detach().item()) / accum_steps

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
                "srgb",
                "srgb_base",
                "body_srgb",
                "body_srgb_base",
                "flat_srgb_hf",
                "flat_luma_hf",
                "flat_chroma_hf",
                "flat_chroma_damp",
                "flat_chroma_distill",
                "flat_chroma_delta_distill",
                "flat_chroma_distill_mask_mean",
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
            ):
                if key in accum_log:
                    msg += f"{key}={accum_log[key]:.4f} "
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
    payload = {
        "step": step,
        "state_dict": ema.state_dict(),
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer": opt.state_dict(),
        "config": cfg,
        "model_kind": "nagiperfect",
    }
    torch.save(payload, str(path))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
