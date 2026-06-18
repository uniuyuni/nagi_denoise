"""Benchmark untrained NagiQ presets.

This is the first gate before training: if a candidate is already too slow on
random 256x256 input, do not spend hours or days training it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nagi_nr.devices import resolve_device
from nagi_nr.nagiq import NAGIQ_PRESETS, NagiQ, build_nagiq_preset


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _conv_macs(in_ch: int, out_ch: int, kernel: int, h: int, w: int, groups: int = 1) -> int:
    return out_ch * (in_ch // groups) * kernel * kernel * h * w


def _block_macs(channels: int, h: int, w: int, dw_expand: int, ffn_expand: int) -> int:
    dw_channels = channels * dw_expand
    ffn_channels = channels * ffn_expand
    macs = 0
    macs += _conv_macs(channels, dw_channels, 1, h, w)
    macs += _conv_macs(dw_channels, dw_channels, 3, h, w, groups=dw_channels)
    macs += _conv_macs(dw_channels // 2, channels, 1, h, w)
    # SCA 1x1 at 1x1 spatial.
    macs += _conv_macs(dw_channels // 2, dw_channels // 2, 1, 1, 1)
    macs += _conv_macs(channels, ffn_channels, 1, h, w)
    macs += _conv_macs(ffn_channels // 2, channels, 1, h, w)
    return macs


def estimate_gmac(model: NagiQ, h: int, w: int) -> float:
    m = model.size_multiple
    h = h + (m - h % m) % m
    w = w + (m - w % m) % m
    channels = model.width
    total = _conv_macs(model.img_channels, channels, 3, h, w)

    for n_blocks in model.enc_blk_nums:
        total += n_blocks * _block_macs(channels, h, w, model.dw_expand, model.ffn_expand)
        total += _conv_macs(channels, channels * 2, 2, h // 2, w // 2)
        channels *= 2
        h //= 2
        w //= 2

    total += model.middle_blk_num * _block_macs(channels, h, w, model.dw_expand, model.ffn_expand)

    for n_blocks in model.dec_blk_nums:
        total += _conv_macs(channels, channels * 2, 1, h, w)
        channels //= 2
        h *= 2
        w *= 2
        total += n_blocks * _block_macs(channels, h, w, model.dw_expand, model.ffn_expand)

    total += _conv_macs(channels, model.img_channels, 3, h, w)
    return total / 1e9


@torch.inference_mode()
def benchmark_one(
    name: str,
    device: torch.device,
    height: int,
    width: int,
    warmup: int,
    iters: int,
) -> dict[str, float | str]:
    model = build_nagiq_preset(name).to(device=device, dtype=torch.float32).eval()
    x = torch.rand(1, 3, height, width, device=device, dtype=torch.float32)

    for _ in range(warmup):
        y = model(x)
    _sync(device)

    start = time.perf_counter()
    for _ in range(iters):
        y = model(x)
    _sync(device)
    elapsed = time.perf_counter() - start

    # Keep y alive until after sync, and sanity-check shape.
    if tuple(y.shape) != (1, 3, height, width):
        raise RuntimeError(f"{name} returned shape {tuple(y.shape)}")

    params = model.param_count() / 1e6
    gmac = estimate_gmac(model, height, width)
    ms = elapsed / iters * 1000.0
    return {
        "preset": name,
        "params_m": params,
        "gmac": gmac,
        "ms_patch": ms,
        "gmac_per_ms": gmac / ms,
    }


def _markdown(rows: list[dict[str, float | str]], device: torch.device, h: int, w: int) -> str:
    lines = [
        "# NagiQ Speed Benchmark",
        "",
        f"Device: `{device}`",
        f"Input: `1x3x{h}x{w}`",
        "",
        "| preset | params | GMAC | ms/patch | GMAC/ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {preset} | {params_m:.2f}M | {gmac:.2f} | {ms_patch:.1f} | {gmac_per_ms:.4f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark NagiQ presets on random input.")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--presets",
        nargs="+",
        default=["q48-trim", "q48-fast", "q56-trim", "q40-trim"],
        choices=sorted(NAGIQ_PRESETS),
    )
    ap.add_argument("--output-md", default="runs/nagiq_speed.md")
    args = ap.parse_args()

    device = resolve_device(args.device)
    rows = []
    for preset in args.presets:
        row = benchmark_one(preset, device, args.height, args.width, args.warmup, args.iters)
        rows.append(row)
        print(
            "{preset:9s} params={params_m:6.2f}M gmac={gmac:6.2f} "
            "ms={ms_patch:7.1f} gmac/ms={gmac_per_ms:.4f}".format(**row),
            flush=True,
        )

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_markdown(rows, device, args.height, args.width), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
