"""Benchmark Nagi-RealFast presets before spending training time."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nagi_nr.devices import resolve_device
from nagi_nr.realfast import REALFAST_PRESETS, build_realfast_preset


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def estimate_conv_gmac(model: torch.nn.Module, x: torch.Tensor) -> float:
    total = 0
    handles = []

    def hook(module, inp, out):
        nonlocal total
        if not isinstance(module, torch.nn.Conv2d):
            return
        y = out[0] if isinstance(out, tuple) else out
        batch, out_ch, out_h, out_w = y.shape
        kh, kw = module.kernel_size
        total += batch * out_ch * out_h * out_w * (module.in_channels // module.groups) * kh * kw

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(x)
    for handle in handles:
        handle.remove()
    return total / 1e9


@torch.inference_mode()
def benchmark_one(
    preset: str,
    device: torch.device,
    height: int,
    width: int,
    warmup: int,
    iters: int,
) -> dict[str, float | str]:
    model = build_realfast_preset(preset).to(device=device, dtype=torch.float32).eval()
    x = torch.rand(1, 3, height, width, device=device, dtype=torch.float32)
    gmac = estimate_conv_gmac(model, x)
    _sync(device)

    for _ in range(warmup):
        y = model(x)
    _sync(device)

    start = time.perf_counter()
    for _ in range(iters):
        y = model(x)
    _sync(device)
    elapsed = time.perf_counter() - start

    if tuple(y.shape) != (1, 3, height, width):
        raise RuntimeError(f"{preset} returned shape {tuple(y.shape)}")

    params = model.param_count() / 1e6
    ms = elapsed / max(1, iters) * 1000.0
    return {
        "preset": preset,
        "params_m": params,
        "gmac": gmac,
        "ms_patch": ms,
        "gmac_per_ms": gmac / ms,
    }


def markdown(rows: list[dict[str, float | str]], device: torch.device, h: int, w: int) -> str:
    lines = [
        "# Nagi-RealFast Speed Benchmark",
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
    ap = argparse.ArgumentParser(description="Benchmark Nagi-RealFast presets on random input.")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--presets", nargs="+", default=sorted(REALFAST_PRESETS), choices=sorted(REALFAST_PRESETS))
    ap.add_argument("--output-md", default="runs/realfast_speed.md")
    args = ap.parse_args()

    device = resolve_device(args.device)
    rows = []
    for preset in args.presets:
        row = benchmark_one(preset, device, args.height, args.width, args.warmup, args.iters)
        rows.append(row)
        print(
            "{preset:16s} params={params_m:6.2f}M gmac={gmac:6.2f} "
            "ms={ms_patch:7.1f} gmac/ms={gmac_per_ms:.4f}".format(**row),
            flush=True,
        )

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(rows, device, args.height, args.width), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
