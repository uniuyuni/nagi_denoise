"""Benchmark GAMA-IR presets on random input."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nagi_nr.devices import resolve_device
from nagi_nr.gamair import GAMAIR_PRESETS, build_gamair_preset


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.inference_mode()
def benchmark_one(
    preset: str,
    device: torch.device,
    height: int,
    width: int,
    warmup: int,
    iters: int,
) -> dict[str, float | str]:
    model = build_gamair_preset(preset).to(device=device, dtype=torch.float32).eval()
    x = torch.rand(1, 3, height, width, device=device, dtype=torch.float32)
    for _ in range(warmup):
        y = model(x)
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        y = model(x)
    _sync(device)
    elapsed = time.perf_counter() - start
    if tuple(y.shape) != (1, 3, height, width):
        raise RuntimeError(f"unexpected output shape: {tuple(y.shape)}")
    return {
        "preset": preset,
        "params_m": model.param_count() / 1e6,
        "ms_patch": elapsed / max(1, iters) * 1000.0,
    }


def markdown(rows: list[dict[str, float | str]], device: torch.device, h: int, w: int) -> str:
    lines = [
        "# GAMA-IR Speed Benchmark",
        "",
        f"Device: `{device}`",
        f"Input: `1x3x{h}x{w}`",
        "",
        "| preset | params | ms/patch |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {preset} | {params_m:.2f}M | {ms_patch:.1f} |".format(**row))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark GAMA-IR presets.")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--presets", nargs="+", default=sorted(GAMAIR_PRESETS), choices=sorted(GAMAIR_PRESETS))
    ap.add_argument("--output-md", default="runs/gamair_speed.md")
    args = ap.parse_args()

    device = resolve_device(args.device)
    rows = []
    for preset in args.presets:
        row = benchmark_one(preset, device, args.height, args.width, args.warmup, args.iters)
        rows.append(row)
        print("{preset:8s} params={params_m:6.2f}M ms={ms_patch:7.1f}".format(**row), flush=True)
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(rows, device, args.height, args.width), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
