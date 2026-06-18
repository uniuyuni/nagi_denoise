"""Profile exact NAFNet inference for the NAFNet-Fast track."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from nagi_nr.devices import resolve_device
from nagi_nr_bench.eval_sidd_val import psnr_srgb
from nagi_nr_bench.third_party.nafnet import NAFNet


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _load_nafnet(weights: str, device: torch.device, channels_last: bool) -> NAFNet:
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
    model.to(device=device, dtype=torch.float32)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def _to_input_tensor(np_noisy_uint8: np.ndarray, device: torch.device, channels_last: bool) -> torch.Tensor:
    t = torch.from_numpy(np_noisy_uint8).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    if channels_last:
        t = t.contiguous(memory_format=torch.channels_last)
    return t.to(device=device, dtype=torch.float32)


@torch.inference_mode()
def benchmark_forward(
    model: NAFNet,
    device: torch.device,
    height: int,
    width: int,
    warmup: int,
    iters: int,
    channels_last: bool,
) -> float:
    x = torch.rand(1, 3, height, width, device=device, dtype=torch.float32)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    for _ in range(warmup):
        y = model(x)
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        y = model(x)
    _sync(device)
    if tuple(y.shape) != (1, 3, height, width):
        raise RuntimeError(f"unexpected output shape: {tuple(y.shape)}")
    return (time.perf_counter() - start) / max(1, iters) * 1000.0


def _record_phase(totals: dict[str, float], key: str, device: torch.device, fn):
    start = time.perf_counter()
    value = fn()
    _sync(device)
    totals[key] = totals.get(key, 0.0) + time.perf_counter() - start
    return value


@torch.inference_mode()
def profile_stages(
    model: NAFNet,
    device: torch.device,
    height: int,
    width: int,
    warmup: int,
    iters: int,
    channels_last: bool,
) -> list[tuple[str, float]]:
    x = torch.rand(1, 3, height, width, device=device, dtype=torch.float32)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    for _ in range(warmup):
        y = model(x)
    _sync(device)

    totals: dict[str, float] = {}
    for _ in range(iters):
        h, w = x.shape[-2:]
        z = _record_phase(totals, "pad", device, lambda: model.check_image_size(x))
        z = _record_phase(totals, "intro", device, lambda: model.intro(z))
        skips = []
        for idx, (encoder, down) in enumerate(zip(model.encoders, model.downs)):
            z = _record_phase(totals, f"enc{idx}", device, lambda encoder=encoder, z=z: encoder(z))
            skips.append(z)
            z = _record_phase(totals, f"down{idx}", device, lambda down=down, z=z: down(z))
        z = _record_phase(totals, "middle", device, lambda: model.middle_blks(z))
        for idx, (decoder, up, skip) in enumerate(zip(model.decoders, model.ups, skips[::-1])):
            z = _record_phase(totals, f"up{idx}", device, lambda up=up, z=z: up(z))
            z = _record_phase(totals, f"skip_add{idx}", device, lambda z=z, skip=skip: z + skip)
            z = _record_phase(totals, f"dec{idx}", device, lambda decoder=decoder, z=z: decoder(z))
        z = _record_phase(totals, "ending", device, lambda: model.ending(z))
        z = _record_phase(totals, "residual_add_crop", device, lambda z=z: (z + x)[:, :, :h, :w])
        y = z

    if tuple(y.shape) != (1, 3, height, width):
        raise RuntimeError(f"unexpected stage-profile output shape: {tuple(y.shape)}")
    return [(key, value / max(1, iters) * 1000.0) for key, value in totals.items()]


@torch.inference_mode()
def profile_validation_loop(
    model: NAFNet,
    device: torch.device,
    noisy_mat: str,
    gt_mat: str,
    max_patches: int,
    channels_last: bool,
) -> dict[str, float]:
    noisy = sio.loadmat(noisy_mat)
    nkey = next(k for k in noisy if not k.startswith("__"))
    noisy = noisy[nkey]
    gt = sio.loadmat(gt_mat)
    gkey = next(k for k in gt if not k.startswith("__"))
    gt = gt[gkey]
    if noisy.shape != gt.shape:
        raise RuntimeError(f"shape mismatch: noisy={noisy.shape}, gt={gt.shape}")

    total = noisy.shape[0] * noisy.shape[1]
    limit = total if max_patches <= 0 else min(max_patches, total)

    phase = {
        "preprocess_cpu": 0.0,
        "to_device": 0.0,
        "forward": 0.0,
        "postprocess_cpu": 0.0,
        "psnr": 0.0,
    }
    psnr_in: list[float] = []
    psnr_out: list[float] = []
    done = 0
    loop_start = time.perf_counter()

    for i in range(noisy.shape[0]):
        for j in range(noisy.shape[1]):
            if done >= limit:
                break
            n_patch = noisy[i, j]
            g_patch = gt[i, j]

            t0 = time.perf_counter()
            t_cpu = torch.from_numpy(n_patch).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            if channels_last:
                t_cpu = t_cpu.contiguous(memory_format=torch.channels_last)
            phase["preprocess_cpu"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            t = t_cpu.to(device=device, dtype=torch.float32)
            _sync(device)
            phase["to_device"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            out = model(t)
            _sync(device)
            phase["forward"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            out_np = out.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_u8 = (out_np * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
            phase["postprocess_cpu"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            psnr_in.append(psnr_srgb(n_patch, g_patch))
            psnr_out.append(psnr_srgb(out_u8, g_patch))
            phase["psnr"] += time.perf_counter() - t0

            done += 1
        if done >= limit:
            break

    loop_total = time.perf_counter() - loop_start
    out = {
        "patches": float(done),
        "psnr_in": float(np.mean(psnr_in)),
        "psnr_out": float(np.mean(psnr_out)),
        "loop_ms_patch": loop_total / max(1, done) * 1000.0,
    }
    for key, value in phase.items():
        out[f"{key}_ms_patch"] = value / max(1, done) * 1000.0
    out["accounted_ms_patch"] = sum(phase.values()) / max(1, done) * 1000.0
    return out


def _markdown(
    args: argparse.Namespace,
    device: torch.device,
    forward_ms: float,
    val: dict[str, float],
    stage_rows: list[tuple[str, float]],
) -> str:
    mode = "channels_last" if args.channels_last else "baseline"
    lines = [
        "# NAFNet-Fast Profile",
        "",
        f"Mode: `{mode}`",
        f"Device: `{device}`",
        f"Weights: `{args.weights}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| random forward ms/patch | {forward_ms:.1f} |",
    ]
    if val:
        lines.extend(
            [
                f"| validation patches | {int(val['patches'])} |",
                f"| noisy PSNR | {val['psnr_in']:.3f} dB |",
                f"| NAFNet PSNR | {val['psnr_out']:.3f} dB |",
                f"| validation loop ms/patch | {val['loop_ms_patch']:.1f} |",
                f"| accounted timed ms/patch | {val['accounted_ms_patch']:.1f} |",
                f"| preprocess CPU ms/patch | {val['preprocess_cpu_ms_patch']:.1f} |",
                f"| to device ms/patch | {val['to_device_ms_patch']:.1f} |",
                f"| forward ms/patch | {val['forward_ms_patch']:.1f} |",
                f"| postprocess CPU ms/patch | {val['postprocess_cpu_ms_patch']:.1f} |",
                f"| PSNR compute ms/patch | {val['psnr_ms_patch']:.1f} |",
            ]
        )
    if stage_rows:
        total = sum(ms for _, ms in stage_rows)
        lines.extend(
            [
                "",
                "| stage | ms/patch | share |",
                "| --- | ---: | ---: |",
            ]
        )
        for name, ms in stage_rows:
            share = 0.0 if total <= 0 else ms / total * 100.0
            lines.append(f"| {name} | {ms:.1f} | {share:.1f}% |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile exact NAFNet inference.")
    ap.add_argument("--weights", default="benchmarks/nafnet/NAFNet-SIDD-width64.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--stage-profile", action="store_true")
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    ap.add_argument("--max-patches", type=int, default=32)
    ap.add_argument("--output-md", default="runs/nafnet_fast_profile.md")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model = _load_nafnet(args.weights, device, args.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"mode: {'channels_last' if args.channels_last else 'baseline'}")
    print(f"params: {n_params / 1e6:.2f}M")

    forward_ms = benchmark_forward(
        model,
        device=device,
        height=args.height,
        width=args.width,
        warmup=args.warmup,
        iters=args.iters,
        channels_last=args.channels_last,
    )
    print(f"random forward: {forward_ms:.1f} ms/patch")

    stage_rows: list[tuple[str, float]] = []
    if args.stage_profile:
        stage_rows = profile_stages(
            model,
            device=device,
            height=args.height,
            width=args.width,
            warmup=max(1, min(args.warmup, 3)),
            iters=max(1, min(args.iters, 5)),
            channels_last=args.channels_last,
        )
        stage_total = sum(ms for _, ms in stage_rows)
        print(f"stage-profile accounted: {stage_total:.1f} ms/patch")
        for name, ms in sorted(stage_rows, key=lambda row: row[1], reverse=True)[:8]:
            print(f"  {name}: {ms:.1f} ms")

    val = {}
    if args.max_patches != 0:
        val = profile_validation_loop(
            model,
            device=device,
            noisy_mat=args.noisy_mat,
            gt_mat=args.gt_mat,
            max_patches=args.max_patches,
            channels_last=args.channels_last,
        )
        print(
            "validation: patches={patches:.0f} psnr={psnr_out:.3f} "
            "loop={loop_ms_patch:.1f}ms forward={forward_ms_patch:.1f}ms "
            "to_device={to_device_ms_patch:.1f}ms post={postprocess_cpu_ms_patch:.1f}ms".format(
                **val
            )
        )

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_markdown(args, device, forward_ms, val, stage_rows), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
