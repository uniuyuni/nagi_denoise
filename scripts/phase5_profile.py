"""Phase 5 profiling harness: times the production denoise() pipeline stages.

Usage:
    pixi run python scripts/phase5_profile.py --scene ice --tile 768 --overlap 64
    pixi run python scripts/phase5_profile.py --scene occi --tile 768 --overlap 64 --batch 4 --amp fp16

This does NOT modify the production pipeline; it calls the same building
blocks (`Denoiser.denoise_array`, `smooth_chroma`) with wall-clock timers
around each stage, plus `torch.mps.synchronize()` so GPU work is actually
accounted for before a timer stops.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from nagi_denoise.devices import resolve_device
from nagi_denoise.infer import Denoiser
from nagi_denoise.pipeline.flat_chroma_smoother import smooth_chroma
from nagi_denoise.pipeline.probe import read_image
from nagi_denoise.pipeline.denoise import CH, PRODUCTION_WEIGHTS

SCENES = {
    "ice": "K-5 Ice noisy.EXR",
    "occi": "X-T5 Occi noisy.EXR",
    "room": "X-T5 Room.EXR",
    "fix": "Z7 fix noisy.EXR",
    "dance": "K-5 Dance noisy.EXR",
    "cat": "X-T5 Cat noisy.EXR",
}

TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="ice", choices=list(SCENES))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--amp", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--chroma", action="store_true", default=True)
    ap.add_argument("--no-chroma", dest="chroma", action="store_false")
    args = ap.parse_args()

    weights = args.weights or str(PRODUCTION_WEIGHTS)
    device = resolve_device(args.device)

    path = TEST_PHOTOS / SCENES[args.scene]
    t0 = time.perf_counter()
    img = read_image(path)
    t_read = time.perf_counter() - t0
    mp = img.shape[0] * img.shape[1] / 1e6
    print(f"scene={args.scene} shape={img.shape} ({mp:.1f} MP) read={t_read:.2f}s")

    t0 = time.perf_counter()
    dn = Denoiser.load(weights, device=device)
    sync(device)
    t_load = time.perf_counter() - t0
    print(f"model load: {t_load:.2f}s")

    kwargs = {}
    if args.batch != 1:
        kwargs["batch_size"] = args.batch
    if args.amp != "fp32":
        kwargs["amp_dtype"] = torch.float16

    times = []
    for i in range(args.repeat):
        t0 = time.perf_counter()
        out = dn.denoise_array(img, input_space="linear", tile=args.tile, overlap=args.overlap, **kwargs)
        sync(device)
        t_model = time.perf_counter() - t0
        times.append(t_model)
        print(f"  run {i}: model inference = {t_model:.2f}s")

    t_model_med = float(np.median(times))

    t0 = time.perf_counter()
    if args.chroma:
        out_c, stats, _gate = smooth_chroma(out, **CH)
    else:
        out_c = out
    t_chroma = time.perf_counter() - t0
    print(f"chroma pass: {t_chroma:.2f}s")

    total = t_model_med + t_chroma
    print(
        f"SUMMARY scene={args.scene} mp={mp:.1f} tile={args.tile} overlap={args.overlap} "
        f"batch={args.batch} amp={args.amp} chroma={args.chroma} "
        f"model={t_model_med:.2f}s chroma={t_chroma:.2f}s total={total:.2f}s "
        f"mp_per_s={mp/total:.3f}"
    )


if __name__ == "__main__":
    main()
