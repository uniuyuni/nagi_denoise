"""Train ChromaGuard to predict a chroma-NR strength map."""
from __future__ import annotations

import argparse
import glob
import math
import re
import time
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .chromaguard import (
    ChromaGuard,
    adaptive_chroma_gate,
    adaptive_luma_gate,
    heuristic_chroma_gate,
    paired_luma_noise_gate,
)
from .data import ChunkedShuffleSampler, SIDDPatchDataset, find_polyu_pairs, find_sidd_pairs
from .devices import resolve_device


def lr_at(step: int, total: int, warmup: int, lr: float, lr_min: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    t = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


def latest_checkpoint(out_dir: Path, prefix: str) -> Path | None:
    best_step = -1
    best_path = None
    for name in glob.glob(str(out_dir / f"{prefix}_*.pt")):
        m = re.search(r"_(\d+)\.pt$", name)
        if m and int(m.group(1)) > best_step:
            best_step = int(m.group(1))
            best_path = Path(name)
    final = out_dir / f"{prefix}_final.pt"
    if final.exists():
        try:
            final_step = int(torch.load(final, map_location="cpu", weights_only=False).get("step", -1))
        except Exception:
            final_step = -1
        if final_step >= best_step:
            return final
    return best_path


def save_checkpoint(path: Path, model: ChromaGuard, opt: AdamW, step: int, cfg: dict) -> None:
    torch.save(
        {
            "step": int(step),
            "config": cfg,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ChromaGuard.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sidd-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--max-iters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt-prefix", default="chromaguard")
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
        polyu_pairs = find_polyu_pairs(str(polyu_root), split=str(data_cfg.get("polyu_split", "CroppedImages")))
        max_polyu_pairs = int(data_cfg.get("polyu_max_pairs", 0))
        if max_polyu_pairs > 0:
            polyu_pairs = polyu_pairs[:max_polyu_pairs]
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
        output_space="linear",
        randomize_each_access=bool(data_cfg.get("randomize_each_access", True)),
    )
    sampler = ChunkedShuffleSampler(
        num_pairs=len(ds.pairs),
        patches_per_image=ds.patches_per_image,
        chunk_size=int(data_cfg.get("chunk_size", ds.patches_per_image)),
        seed=args.seed,
    )
    num_workers = int(data_cfg.get("num_workers", 0))
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = ChromaGuard(**dict(cfg["model"])).to(device=device, dtype=torch.float32)
    train_cfg = cfg["train"]
    total_iters = int(args.max_iters or train_cfg["total_iters"])
    warmup = int(train_cfg["warmup_iters"])
    lr = float(train_cfg["lr"])
    lr_min = float(train_cfg["lr_min"])
    grad_clip = float(train_cfg["grad_clip"])
    log_every = int(train_cfg["log_every"])
    save_every = int(train_cfg["save_every"])
    prefix = args.ckpt_prefix
    opt = AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.0)))

    teacher_cfg = dict(cfg["teacher"])
    teacher_kind = str(teacher_cfg.pop("kind", "fixed"))
    start_step = 0
    if args.resume_latest:
        ckpt_path = latest_checkpoint(out_dir, prefix)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            start_step = int(ckpt.get("step", 0))
            print(f"resumed from {ckpt_path} at step {start_step}")

    print(f"device: {device}")
    print(f"pairs: {len(ds.pairs)}, dataset items: {len(ds)}")
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
    print(f"output: {out_dir}")

    log_path = out_dir / "train.log"
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== train-chromaguard {prefix} ===\n")

    data_iter = iter(dl)
    step = start_step
    t0 = time.time()
    model.train()
    while step < total_iters:
        cur_lr = lr_at(step, total_iters, warmup, lr, lr_min)
        for group in opt.param_groups:
            group["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        try:
            noisy, _clean = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            noisy, _clean = next(data_iter)
        noisy = noisy.to(device, non_blocking=True)
        if teacher_kind == "paired_luma":
            _clean = _clean.to(device, non_blocking=True)
        with torch.no_grad():
            if teacher_kind == "adaptive":
                target = adaptive_chroma_gate(noisy, **teacher_cfg)
            elif teacher_kind == "luma":
                target = adaptive_luma_gate(noisy, **teacher_cfg)
            elif teacher_kind == "paired_luma":
                target = paired_luma_noise_gate(noisy, _clean, **teacher_cfg)
            elif teacher_kind == "fixed":
                target = heuristic_chroma_gate(noisy, **teacher_cfg)
            else:
                raise ValueError(f"unknown teacher kind: {teacher_kind!r}")
        pred = model(noisy)
        loss_mse = torch.mean((pred - target).pow(2))
        loss_smooth = torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])) + torch.mean(
            torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        )
        loss = loss_mse + float(train_cfg.get("smooth_weight", 0.0)) * loss_smooth
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step % log_every == 0:
            dt = time.time() - t0
            msg = (
                f"[{step:7d}/{total_iters}] loss={float(loss.detach()):.5f} "
                f"mse={float(loss_mse.detach()):.5f} smooth={float(loss_smooth.detach()):.5f} "
                f"pred_mean={float(pred.detach().mean()):.4f} target_mean={float(target.mean()):.4f} "
                f"lr={cur_lr:.2e} ips={(step - start_step + 1) / max(dt, 1.0e-6):.2f}"
            )
            print(msg, flush=True)
            log_f.write(msg + "\n")

        if save_every > 0 and step > start_step and step % save_every == 0:
            path = out_dir / f"{prefix}_{step:07d}.pt"
            save_checkpoint(path, model, opt, step, cfg)
            print(f"saved {path}")

        step += 1

    final_path = out_dir / f"{prefix}_final.pt"
    save_checkpoint(final_path, model, opt, step, cfg)
    print(f"saved {final_path}")
    log_f.write("done.\n")
    log_f.close()


if __name__ == "__main__":
    main()
