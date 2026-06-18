"""Train Nagi NR on SIDD Medium sRGB.

CLI entry point: ``nagi-train``.
"""
from __future__ import annotations
import argparse
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import ChunkedShuffleSampler, SIDDPatchDataset
from .devices import resolve_device
from .losses import NagiDistillLoss, NagiLoss
from .model import NagiNR


def lr_at(step: int, total: int, warmup: int, lr: float, lr_min: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nagi-train", description="Train Nagi NR on SIDD Medium sRGB.")
    p.add_argument("--config", required=True,
                   help="Path to a Nagi NR YAML config (e.g. packages/nagi_nr/configs/nagi_nr_s.yaml)")
    p.add_argument("--sidd-root", required=True, help="Path to SIDD_Medium_Srgb (or its Data/ dir)")
    p.add_argument("--output", required=True, help="Output directory (e.g. runs/nagi_nr_s)")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--resume-latest", action="store_true",
                   help="Auto-resume from the latest checkpoint in --output (ignores --resume)")
    p.add_argument("--init-from", default=None,
                   help="Path to checkpoint whose weights initialize the model (transfer "
                        "learning). Unlike --resume, the optimizer state and step counter "
                        "are NOT restored, so a fresh LR schedule applies.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-prefix", default="nagi_nr",
                   help="Prefix for saved checkpoint filenames (default: nagi_nr)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    data_cfg = cfg["data"]
    ds = SIDDPatchDataset(
        root=args.sidd_root,
        patch_size=data_cfg["patch_size"],
        patches_per_image=data_cfg["patches_per_image"],
        exposure_jitter=tuple(data_cfg["exposure_jitter"]) if data_cfg.get("exposure_jitter") else None,
        flip_rot=data_cfg.get("flip_rot", True),
        seed=args.seed,
        synth_prob=float(data_cfg.get("synth_prob", 0.0)),
        gauss_sigma_max=float(data_cfg.get("gauss_sigma_max", 30.0)),
        poisson_lambda_range=tuple(data_cfg.get("poisson_lambda_range", (10.0, 100.0))),
        jpeg_q_range=tuple(data_cfg.get("jpeg_q_range", (60, 100))),
        return_teacher=bool(data_cfg.get("return_teacher", False)),
    )
    num_workers = data_cfg.get("num_workers", 4)
    sampler = ChunkedShuffleSampler(
        num_pairs=len(ds.pairs),
        patches_per_image=ds.patches_per_image,
        chunk_size=data_cfg.get("chunk_size", ds.patches_per_image),
        seed=args.seed,
    )
    dl_kwargs = dict(
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0 and "prefetch_factor" in data_cfg:
        dl_kwargs["prefetch_factor"] = int(data_cfg["prefetch_factor"])
    dl = DataLoader(ds, **dl_kwargs)
    print(f"device: {device}")
    print(f"SIDD pairs: {len(ds.pairs)}, dataset items: {len(ds)}")

    # ---- Model ----
    model = NagiNR(**cfg["model"]).to(device=device, dtype=torch.float32)
    print(f"params: {model.param_count() / 1e3:.1f}K")

    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema_decay = float(cfg["train"]["ema_decay"])

    # ---- Loss ----
    loss_cfg = cfg["loss"]
    use_distill = bool(loss_cfg.get("distill", False))
    if use_distill:
        if not ds.return_teacher:
            raise ValueError("loss.distill=true requires data.return_teacher=true")
        criterion = NagiDistillLoss(
            compress_fn=model.compress,
            gt_weight=float(loss_cfg.get("gt_weight", 0.5)),
            distill_weight=float(loss_cfg.get("distill_weight", 0.5)),
            charbonnier_eps=float(loss_cfg.get("charbonnier_eps", 1e-3)),
        ).to(device)
    else:
        criterion = NagiLoss(
            compress_fn=model.compress,
            fft_weight=loss_cfg["fft_weight"],
            linear_weight=loss_cfg["linear_weight"],
            linear_clamp=loss_cfg["linear_clamp"],
            charbonnier_eps=loss_cfg["charbonnier_eps"],
            recon_space=loss_cfg.get("recon_space", "compressed"),
            luma_ms_weight=float(loss_cfg.get("luma_ms_weight", 0.0)),
            luma_ms_scales=tuple(loss_cfg.get("luma_ms_scales", (4, 8))),
        ).to(device)

    # ---- Optimizer ----
    train_cfg = cfg["train"]
    opt = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=train_cfg["weight_decay"],
    )
    total_iters = int(train_cfg["total_iters"])
    warmup = int(train_cfg["warmup_iters"])
    lr = float(train_cfg["lr"])
    lr_min = float(train_cfg["lr_min"])
    grad_clip = float(train_cfg["grad_clip"])
    log_every = int(train_cfg["log_every"])
    save_every = int(train_cfg["save_every"])
    keep_last = int(train_cfg.get("keep_last_ckpts", 0))  # 0 = keep all

    # ---- Resume / Init-from ----
    start_step = 0
    # --resume-latest: scan output dir for the highest-step checkpoint
    if args.resume_latest:
        import glob, re
        final_ckpt = out_dir / f"{args.ckpt_prefix}_final.pt"
        if final_ckpt.exists():
            ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=False)
            if int(ckpt.get("step", -1)) >= total_iters:
                args.resume = str(final_ckpt)
                print(f"--resume-latest: found completed {args.resume}")
        if not args.resume:
            pattern = str(out_dir / f"{args.ckpt_prefix}_[0-9]*.pt")
            candidates = sorted(
                glob.glob(pattern),
                key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1))
            )
            if candidates:
                args.resume = candidates[-1]
                print(f"--resume-latest: found {args.resume}")
            else:
                print("--resume-latest: no checkpoint found, starting from scratch")

    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {args.resume} at step {start_step}")
    elif args.init_from:
        # Transfer-learning init: weights only, fresh optimizer + LR schedule.
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        # Prefer EMA weights (state_dict) for clean fine-tune signal; fall back to live weights.
        init_state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        model.load_state_dict(init_state)
        # Mirror into EMA so the EMA tracker starts from the same point.
        ema.load_state_dict(init_state)
        print(f"initialized weights from {args.init_from} (fresh optimizer, step 0)")

    # ---- Train loop ----
    model.train()
    step = start_step
    t0 = time.time()
    log_path = out_dir / "train.log"
    log_f = open(log_path, "a", buffering=1)

    prefix = args.ckpt_prefix

    while step < total_iters:
        for batch in dl:
            if step >= total_iters:
                break

            if use_distill:
                noisy, clean, teacher, has_teacher = batch
                teacher = teacher.to(device, non_blocking=True)
                # Synchronous copy: this is a tiny (B,) tensor and using
                # non_blocking on the orphaned .float() temp races the free.
                has_teacher = has_teacher.float().to(device)
            else:
                noisy, clean = batch
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            cur_lr = lr_at(step, total_iters, warmup, lr, lr_min)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            pred = model(noisy)
            if use_distill:
                losses = criterion(pred, clean, teacher, has_teacher)
            else:
                losses = criterion(pred, clean)
            loss = losses["total"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            with torch.no_grad():
                for p, ep in zip(model.parameters(), ema.parameters()):
                    ep.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

            if step % log_every == 0:
                dt = time.time() - t0
                ips = (step - start_step + 1) / max(dt, 1e-6)
                msg = f"[{step:7d}/{total_iters}] loss={loss.item():.4f} "
                if use_distill:
                    msg += (
                        f"gt={losses['gt'].item():.4f} "
                        f"dist={losses['distill'].item():.4f} "
                        f"tfrac={losses['teacher_frac'].item():.2f} "
                    )
                else:
                    msg += f"c={losses['recon'].item():.4f} "
                    if "fft" in losses:
                        msg += f"fft={losses['fft'].item():.4f} "
                    if "linear" in losses:
                        msg += f"lin={losses['linear'].item():.4f} "
                    if "luma_ms" in losses:
                        msg += f"luma_ms={losses['luma_ms'].item():.4f} "
                msg += f"lr={cur_lr:.2e} ips={ips:.2f}"
                print(msg)
                log_f.write(msg + "\n")

            if save_every > 0 and step > 0 and step % save_every == 0:
                save_ckpt(out_dir / f"{prefix}_{step:07d}.pt", step, model, ema, opt, cfg)
                if keep_last > 0:
                    _rotate_ckpts(out_dir, prefix, keep_last)

            step += 1

    save_ckpt(out_dir / f"{prefix}_final.pt", step, model, ema, opt, cfg)
    log_f.close()
    print("done.")


def _rotate_ckpts(out_dir: Path, prefix: str, keep: int) -> None:
    """Delete oldest step checkpoints, keeping only the `keep` most recent.
    Never touches *_final.pt or any non-step checkpoint.
    """
    import glob, re
    pattern = str(out_dir / f"{prefix}_[0-9]*.pt")
    candidates = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1))
    )
    for old in candidates[:-keep]:
        try:
            Path(old).unlink()
        except OSError:
            pass


def save_ckpt(path: Path, step: int, model, ema, opt, cfg) -> None:
    torch.save(
        {
            "step": step,
            "state_dict": ema.state_dict(),       # EMA weights are recommended for inference
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer": opt.state_dict(),
            "config": cfg,
        },
        str(path),
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
