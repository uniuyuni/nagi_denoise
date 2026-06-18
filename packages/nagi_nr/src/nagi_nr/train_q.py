"""Train NagiQ sRGB student models.

CLI entry point used through pixi tasks / launch scripts.
"""
from __future__ import annotations

import argparse
import glob
import math
import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import ChunkedShuffleSampler, SIDDPatchDataset, find_polyu_pairs, find_sidd_pairs
from .devices import resolve_device
from .losses import Charbonnier
from .gamair import GAMAIR, build_gamair_preset
from .nagiq import NagiQ, build_nagiq_preset
from .realfast import NagiRealFast, build_realfast_preset
from nagi_nr_bench.eval_sidd_val import psnr_srgb


def lr_at(step: int, total: int, warmup: int, lr: float, lr_min: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


class NagiQSrgbDistillLoss(nn.Module):
    """sRGB GT + teacher distillation loss for NagiQ."""

    def __init__(
        self,
        kind: str = "charbonnier_distill",
        charbonnier_eps: float = 1e-3,
        grad_weight: float = 0.0,
        range_weight: float = 0.0,
        chroma_weight: float = 0.0,
        lowfreq_weight: float = 0.0,
        lowfreq_kernel: int = 8,
        clamp_pred: bool = False,
    ):
        super().__init__()
        if kind not in ("charbonnier_distill", "mse_distill"):
            raise ValueError(f"unknown NagiQ loss kind {kind!r}")
        self.kind = kind
        self.charb = Charbonnier(eps=charbonnier_eps)
        self.eps2 = float(charbonnier_eps) ** 2
        self.grad_weight = float(grad_weight)
        self.range_weight = float(range_weight)
        self.chroma_weight = float(chroma_weight)
        self.lowfreq_weight = float(lowfreq_weight)
        self.lowfreq_kernel = int(lowfreq_kernel)
        self.clamp_pred = bool(clamp_pred)

    def _charb_per_sample(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a - b
        return torch.sqrt(diff * diff + self.eps2).flatten(1).mean(dim=1)

    def _mse_per_sample(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a - b
        return (diff * diff).flatten(1).mean(dim=1)

    def _per_sample(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.kind == "mse_distill":
            return self._mse_per_sample(a, b)
        return self._charb_per_sample(a, b)

    def _grad_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dx_p = pred[..., :, 1:] - pred[..., :, :-1]
        dx_t = target[..., :, 1:] - target[..., :, :-1]
        dy_p = pred[..., 1:, :] - pred[..., :-1, :]
        dy_t = target[..., 1:, :] - target[..., :-1, :]
        return 0.5 * (self.charb(dx_p, dx_t) + self.charb(dy_p, dy_t))

    def _range_loss(self, pred: torch.Tensor) -> torch.Tensor:
        below = torch.relu(-pred)
        above = torch.relu(pred - 1.0)
        return (below * below + above * above).mean()

    def _chroma(self, x: torch.Tensor) -> torch.Tensor:
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        return torch.cat((cb, cr), dim=1)

    def _lowfreq(self, x: torch.Tensor) -> torch.Tensor:
        k = max(1, self.lowfreq_kernel)
        if k <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=k, stride=k)

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        teacher: torch.Tensor,
        has_teacher: torch.Tensor,
        gt_weight: float,
        teacher_weight: float,
    ) -> dict[str, torch.Tensor]:
        if self.clamp_pred:
            pred = pred.clamp(0.0, 1.0)
        gt_loss_ps = self._per_sample(pred, gt)
        teacher_loss_ps = self._per_sample(pred, teacher)

        ht = has_teacher.to(pred.dtype).view(-1)
        mixed = torch.where(
            ht > 0,
            float(gt_weight) * gt_loss_ps + float(teacher_weight) * teacher_loss_ps,
            gt_loss_ps,
        )
        total = mixed.mean()
        out = {
            "total": total,
            "gt": gt_loss_ps.mean().detach(),
            "teacher": ((teacher_loss_ps * ht).sum() / ht.sum().clamp_min(1.0)).detach(),
            "teacher_frac": ht.mean().detach(),
        }
        if self.grad_weight > 0:
            grad = self._grad_loss(pred, gt)
            total = total + self.grad_weight * grad
            out["total"] = total
            out["grad"] = grad.detach()
        if self.range_weight > 0:
            range_loss = self._range_loss(pred)
            total = total + self.range_weight * range_loss
            out["total"] = total
            out["range"] = range_loss.detach()
        if self.chroma_weight > 0:
            chroma = self.charb(self._chroma(pred), self._chroma(gt))
            total = total + self.chroma_weight * chroma
            out["total"] = total
            out["chroma"] = chroma.detach()
        if self.lowfreq_weight > 0:
            lowfreq = self.charb(self._lowfreq(pred), self._lowfreq(gt))
            total = total + self.lowfreq_weight * lowfreq
            out["total"] = total
            out["lowfreq"] = lowfreq.detach()
        return out


def teacher_weight_at(step: int, total: int, loss_cfg: dict) -> tuple[float, float]:
    start = float(loss_cfg.get("teacher_weight_start", 0.7))
    end = float(loss_cfg.get("teacher_weight_end", 0.3))
    t = min(max(step / max(1, total - 1), 0.0), 1.0)
    teacher = start + (end - start) * t
    gt = 1.0 - teacher
    return gt, teacher


def build_model(cfg: dict) -> nn.Module:
    model_cfg = dict(cfg["model"])
    kind = str(model_cfg.pop("kind", "nagiq"))
    if kind == "realfast":
        preset = model_cfg.pop("preset", None)
        if preset:
            model = build_realfast_preset(str(preset))
            if model_cfg:
                merged = {
                    "img_channels": model.img_channels,
                    "width": model.width,
                    "enc_blk_nums": model.enc_blk_nums,
                    "middle_blk_num": model.middle_blk_num,
                    "dec_blk_nums": model.dec_blk_nums,
                    "high_expand": model.high_expand,
                    "high_ffn_expand": model.high_ffn_expand,
                    "high_ffn_enc_stages": model.high_ffn_enc_stages,
                    "high_ffn_dec_stages": model.high_ffn_dec_stages,
                    "low_expand": model.low_expand,
                    "ffn_expand": model.ffn_expand,
                    "residual_init": model.residual_init,
                    "ending_init_std": model.ending_init_std,
                }
                merged.update(model_cfg)
                model = NagiRealFast(**merged)
            return model
        return NagiRealFast(**model_cfg)
    if kind == "gamair":
        preset = model_cfg.pop("preset", None)
        if preset:
            model = build_gamair_preset(str(preset))
            if model_cfg:
                merged = {
                    "img_channels": model.img_channels,
                    "width": model.width,
                    "depth": model.depth,
                    "levels": model.levels,
                    "enc_blk_nums": model.enc_blk_nums,
                    "middle_blk_num": model.middle_blk_num,
                    "dec_blk_nums": model.dec_blk_nums,
                    "dw_expand": model.dw_expand,
                    "ffn_expand": model.ffn_expand,
                    "gama_kernel": model.gama_kernel,
                    "residual_init": model.residual_init,
                    "ending_init_std": model.ending_init_std,
                }
                merged.update(model_cfg)
                model = GAMAIR(**merged)
            return model
        return GAMAIR(**model_cfg)
    if kind != "nagiq":
        raise ValueError(f"unknown model kind {kind!r}")
    preset = model_cfg.pop("preset", None)
    if preset:
        model = build_nagiq_preset(str(preset))
        if model_cfg:
            # Allow explicit overrides after the preset.
            merged = {
                "img_channels": model.img_channels,
                "width": model.width,
                "enc_blk_nums": model.enc_blk_nums,
                "middle_blk_num": model.middle_blk_num,
                "dec_blk_nums": model.dec_blk_nums,
                "dw_expand": model.dw_expand,
                "ffn_expand": model.ffn_expand,
                "drop_out_rate": model.drop_out_rate,
            }
            merged.update(model_cfg)
            model = NagiQ(**merged)
        return model
    return NagiQ(**model_cfg)


def _trainable_for_mode(name: str, mode: str) -> bool:
    if mode == "full":
        return True
    if mode in ("ending", "head"):
        return name.startswith("ending.")
    if mode == "decoder":
        return name.startswith(("ups.", "decoders.", "ending."))
    if mode == "tail":
        return name.startswith(("middle_blks.", "ups.", "decoders.", "ending."))
    raise ValueError(f"unknown train phase mode {mode!r}")


def apply_train_mode(model: nn.Module, mode: str) -> tuple[int, int]:
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        enabled = _trainable_for_mode(name, mode)
        param.requires_grad_(enabled)
        if enabled:
            trainable += param.numel()
    return trainable, total


def train_mode_at(step: int, train_cfg: dict) -> str:
    phases = train_cfg.get("phases")
    if not phases:
        return "full"
    for phase in phases:
        until = int(phase["until"])
        if step < until:
            return str(phase["train"])
    return str(phases[-1]["train"])


def load_training_state(path: str, state_key: str = "state_dict") -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if state_key in ckpt:
            return ckpt[state_key]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _iter_validation_patches(noisy: np.ndarray, gt: np.ndarray, max_patches: int):
    total = noisy.shape[0] * noisy.shape[1]
    limit = total if max_patches <= 0 else min(max_patches, total)
    done = 0
    for i in range(noisy.shape[0]):
        for j in range(noisy.shape[1]):
            if done >= limit:
                return
            yield noisy[i, j], gt[i, j]
            done += 1


@torch.inference_mode()
def evaluate_sidd_val128(
    model: nn.Module,
    device: torch.device,
    noisy_mat: str,
    gt_mat: str,
    max_patches: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    noisy_data = sio.loadmat(noisy_mat)
    nkey = next(k for k in noisy_data if not k.startswith("__"))
    noisy = noisy_data[nkey]
    gt_data = sio.loadmat(gt_mat)
    gkey = next(k for k in gt_data if not k.startswith("__"))
    gt = gt_data[gkey]
    if noisy.shape != gt.shape:
        raise RuntimeError(f"shape mismatch: noisy={noisy.shape}, gt={gt.shape}")

    psnr_in: list[float] = []
    psnr_out: list[float] = []
    start = time.time()
    done = 0
    for n_patch, g_patch in _iter_validation_patches(noisy, gt, max_patches):
        t = torch.from_numpy(n_patch).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        t = t.to(device=device, dtype=torch.float32)
        out = model(t)
        out_np = out.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out_u8 = (out_np * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        psnr_in.append(psnr_srgb(n_patch, g_patch))
        psnr_out.append(psnr_srgb(out_u8, g_patch))
        done += 1
    if was_training:
        model.train()
    elapsed = time.time() - start
    return {
        "patches": float(done),
        "psnr_in": float(np.mean(psnr_in)),
        "psnr_out": float(np.mean(psnr_out)),
        "sec": elapsed,
        "ms_patch": elapsed / max(1, done) * 1000.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nagiq-train", description="Train NagiQ sRGB student.")
    p.add_argument("--config", required=True)
    p.add_argument("--sidd-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--resume", default=None)
    p.add_argument("--resume-latest", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-prefix", default="nagiq")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
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
        patch_size=data_cfg["patch_size"],
        patches_per_image=data_cfg["patches_per_image"],
        exposure_jitter=tuple(data_cfg["exposure_jitter"]) if data_cfg.get("exposure_jitter") else None,
        flip_rot=data_cfg.get("flip_rot", True),
        seed=args.seed,
        return_teacher=bool(data_cfg.get("return_teacher", True)),
        output_space=data_cfg.get("output_space", "srgb"),
        randomize_each_access=bool(data_cfg.get("randomize_each_access", False)),
        pairs=pairs,
    )
    if not ds.return_teacher:
        raise ValueError("NagiQ training expects data.return_teacher=true for the current plan")

    num_workers = int(data_cfg.get("num_workers", 1))
    sampler = ChunkedShuffleSampler(
        num_pairs=len(ds.pairs),
        patches_per_image=ds.patches_per_image,
        chunk_size=data_cfg.get("chunk_size", ds.patches_per_image),
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

    print(f"device: {device}")
    print(f"SIDD pairs: {len(ds.pairs)}, dataset items: {len(ds)}")
    print(f"params: {model.param_count() / 1e6:.2f}M")
    print(f"effective batch: {cfg['train']['batch_size']} x {cfg['train'].get('grad_accum_steps', 1)}")
    print(f"randomize_each_access: {ds.randomize_each_access}")

    train_cfg = cfg["train"]
    opt = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=(0.9, 0.999),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = NagiQSrgbDistillLoss(
        kind=str(cfg["loss"].get("kind", "charbonnier_distill")),
        charbonnier_eps=float(cfg["loss"].get("charbonnier_eps", 1e-3)),
        grad_weight=float(cfg["loss"].get("grad_weight", 0.0)),
        range_weight=float(cfg["loss"].get("range_weight", 0.0)),
        chroma_weight=float(cfg["loss"].get("chroma_weight", 0.0)),
        lowfreq_weight=float(cfg["loss"].get("lowfreq_weight", 0.0)),
        lowfreq_kernel=int(cfg["loss"].get("lowfreq_kernel", 8)),
        clamp_pred=bool(cfg["loss"].get("clamp_pred", False)),
    ).to(device)

    total_iters = int(train_cfg["total_iters"])
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
    val_cfg = cfg.get("validation", {})
    val_enabled = bool(val_cfg.get("enabled", False))
    val_every = int(val_cfg.get("every", save_every))
    val_max_patches = int(val_cfg.get("max_patches", 128))
    val_noisy_mat = str(val_cfg.get("noisy_mat", "data/ValidationNoisyBlocksSrgb.mat"))
    val_gt_mat = str(val_cfg.get("gt_mat", "data/ValidationGtBlocksSrgb.mat"))
    best_metric = float("-inf")
    last_val_step = -1

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
        live_state = ckpt.get("model_state_dict", ckpt["state_dict"])
        ema_state = ckpt.get("ema_state_dict", ckpt["state_dict"])
        model.load_state_dict(live_state)
        ema.load_state_dict(ema_state)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_metric = float(ckpt.get("best_val_psnr", best_metric))
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        init_ckpt = train_cfg.get("init_checkpoint")
        if init_ckpt:
            state_key = str(train_cfg.get("init_state", "state_dict"))
            init_state = load_training_state(str(init_ckpt), state_key=state_key)
            model.load_state_dict(init_state, strict=True)
            ema.load_state_dict(init_state, strict=True)
            print(f"initialized from {init_ckpt} ({state_key})")

    log_path = out_dir / "train.log"
    log_f = open(log_path, "a", buffering=1)
    data_iter = iter(dl)
    step = start_step
    t0 = time.time()
    model.train()
    active_train_mode = ""

    while step < total_iters:
        train_mode = train_mode_at(step, train_cfg)
        if train_mode != active_train_mode:
            trainable, total_params = apply_train_mode(model, train_mode)
            active_train_mode = train_mode
            msg = (
                f"[phase {step:7d}] train={train_mode} "
                f"trainable={trainable / 1e6:.2f}M/{total_params / 1e6:.2f}M"
            )
            print(msg)
            log_f.write(msg + "\n")

        cur_lr = lr_at(step, total_iters, warmup, lr, lr_min)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        accum_log: dict[str, float] = {}
        for _ in range(accum_steps):
            try:
                noisy, clean, teacher, has_teacher = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                noisy, clean, teacher, has_teacher = next(data_iter)

            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            has_teacher = has_teacher.float().to(device)

            pred = model(noisy)
            gt_w, teacher_w = teacher_weight_at(step, total_iters, cfg["loss"])
            losses = criterion(pred, clean, teacher, has_teacher, gt_w, teacher_w)
            (losses["total"] / accum_steps).backward()
            for key, value in losses.items():
                accum_log[key] = accum_log.get(key, 0.0) + float(value.detach().item()) / accum_steps
            accum_log["gt_weight"] = gt_w
            accum_log["teacher_weight"] = teacher_w

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        with torch.no_grad():
            for p, ep in zip(model.parameters(), ema.parameters()):
                ep.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

        if step % log_every == 0:
            dt = time.time() - t0
            ips = (step - start_step + 1) / max(dt, 1e-6)
            msg = (
                f"[{step:7d}/{total_iters}] "
                f"loss={accum_log['total']:.4f} gt={accum_log['gt']:.4f} "
                f"teach={accum_log['teacher']:.4f} tfrac={accum_log['teacher_frac']:.2f} "
            )
            if "grad" in accum_log:
                msg += f"grad={accum_log['grad']:.4f} "
            if "range" in accum_log:
                msg += f"range={accum_log['range']:.4f} "
            if "chroma" in accum_log:
                msg += f"chroma={accum_log['chroma']:.4f} "
            if "lowfreq" in accum_log:
                msg += f"lowfreq={accum_log['lowfreq']:.4f} "
            msg += (
                f"phase={active_train_mode} "
                f"wgt={accum_log['gt_weight']:.2f}/{accum_log['teacher_weight']:.2f} "
                f"lr={cur_lr:.2e} ips={ips:.2f}"
            )
            print(msg)
            log_f.write(msg + "\n")

        if save_every > 0 and step > 0 and step % save_every == 0:
            save_ckpt(out_dir / f"{prefix}_{step:07d}.pt", step, model, ema, opt, cfg, best_metric)
            if keep_last > 0:
                rotate_ckpts(out_dir, prefix, keep_last)

        if val_enabled and val_every > 0 and step > 0 and step % val_every == 0:
            val = evaluate_sidd_val128(
                ema,
                device=device,
                noisy_mat=val_noisy_mat,
                gt_mat=val_gt_mat,
                max_patches=val_max_patches,
            )
            msg = (
                f"[val {step:7d}] patches={val['patches']:.0f} "
                f"psnr={val['psnr_out']:.3f} noisy={val['psnr_in']:.3f} "
                f"ms={val['ms_patch']:.1f}"
            )
            print(msg)
            log_f.write(msg + "\n")
            last_val_step = step
            if val["psnr_out"] > best_metric:
                best_metric = float(val["psnr_out"])
                save_ckpt(
                    out_dir / f"{prefix}_best.pt",
                    step,
                    model,
                    ema,
                    opt,
                    cfg,
                    best_metric,
                    metrics={"val128_psnr": best_metric, "val128": val},
                )

        step += 1

    final_metrics = None
    if val_enabled and step != last_val_step:
        val = evaluate_sidd_val128(
            ema,
            device=device,
            noisy_mat=val_noisy_mat,
            gt_mat=val_gt_mat,
            max_patches=val_max_patches,
        )
        final_metrics = {"val128_psnr": float(val["psnr_out"]), "val128": val}
        msg = (
            f"[val {step:7d}] patches={val['patches']:.0f} "
            f"psnr={val['psnr_out']:.3f} noisy={val['psnr_in']:.3f} "
            f"ms={val['ms_patch']:.1f}"
        )
        print(msg)
        log_f.write(msg + "\n")
        if val["psnr_out"] > best_metric:
            best_metric = float(val["psnr_out"])
            save_ckpt(
                out_dir / f"{prefix}_best.pt",
                step,
                model,
                ema,
                opt,
                cfg,
                best_metric,
                metrics=final_metrics,
            )

    save_ckpt(
        out_dir / f"{prefix}_final.pt",
        step,
        model,
        ema,
        opt,
        cfg,
        best_metric,
        metrics=final_metrics,
    )
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
    model: nn.Module,
    ema: nn.Module,
    opt: AdamW,
    cfg: dict,
    best_val_psnr: float = float("-inf"),
    metrics: dict | None = None,
) -> None:
    payload = {
        "step": step,
        "state_dict": ema.state_dict(),
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer": opt.state_dict(),
        "config": cfg,
        "model_kind": str((cfg.get("model") or {}).get("kind", "nagiq")),
        "best_val_psnr": best_val_psnr,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, str(path))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
