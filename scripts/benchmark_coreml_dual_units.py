"""Benchmark manual CPU+NE and CPU+GPU parallel tiling."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from denoise_exr_coreml import compute_unit, load_exr_rgb, linear_to_srgb, pad_reflect_to_tile


def load_model(path: str, compute_units: str):
    import coremltools as ct

    return ct.models.MLModel(path, compute_units=compute_unit(compute_units))


def starts(length: int, tile: int) -> list[int]:
    if length <= tile:
        return [0]
    out = list(range(0, length - tile + 1, tile))
    if out[-1] != length - tile:
        out.append(length - tile)
    return out


def run_group(model, padded: np.ndarray, coords: list[tuple[int, int]], tile: int, batch: int) -> float:
    start = time.perf_counter()
    for offset in range(0, len(coords), batch):
        group = coords[offset : offset + batch]
        patches = [padded[y : y + tile, x : x + tile].transpose(2, 0, 1) for y, x in group]
        inp = np.stack(patches, axis=0).astype(np.float32, copy=False)
        if len(group) < batch:
            inp = np.concatenate([inp, np.repeat(inp[-1:], batch - len(group), axis=0)], axis=0)
        _ = model.predict({"input": inp})
    return time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="samples/coreml_exr_input/sample_cat_noisy.EXR")
    ap.add_argument("--crop", default="3096,1808,1536,1536")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--ne-model", default="runs/ane_coreml_experiment/nafnet_width64_neuralnetwork_b1_256.mlmodel")
    ap.add_argument("--gpu-model", default="runs/nafnet_fast_coreml/nafnet_width64_fp16_b4_256.mlpackage")
    ap.add_argument("--gpu-fraction", type=float, default=0.25)
    ap.add_argument("--ne-batch", type=int, default=1)
    ap.add_argument("--gpu-batch", type=int, default=4)
    args = ap.parse_args()

    exr = load_exr_rgb(Path(args.input))
    if args.crop:
        x, y, w, h = [int(v) for v in args.crop.split(",")]
        exr = np.ascontiguousarray(exr[y : y + h, x : x + w])
    srgb = np.clip(linear_to_srgb(exr), 0.0, 1.0).astype(np.float32)
    padded, _, _ = pad_reflect_to_tile(srgb, args.tile)
    coords = [(y, x) for y in starts(padded.shape[0], args.tile) for x in starts(padded.shape[1], args.tile)]
    n_gpu = int(round(len(coords) * args.gpu_fraction))
    gpu_coords = coords[:n_gpu]
    ne_coords = coords[n_gpu:]

    ne = load_model(args.ne_model, "cpu_and_ne")
    gpu = load_model(args.gpu_model, "cpu_and_gpu")

    # Warm both paths once.
    if ne_coords:
        run_group(ne, padded, ne_coords[:1], args.tile, args.ne_batch)
    if gpu_coords:
        run_group(gpu, padded, gpu_coords[: min(args.gpu_batch, len(gpu_coords))], args.tile, args.gpu_batch)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_ne = pool.submit(run_group, ne, padded, ne_coords, args.tile, args.ne_batch)
        fut_gpu = pool.submit(run_group, gpu, padded, gpu_coords, args.tile, args.gpu_batch)
        ne_sec = fut_ne.result()
        gpu_sec = fut_gpu.result()
    elapsed = time.perf_counter() - start
    print(
        {
            "tiles": len(coords),
            "ne_tiles": len(ne_coords),
            "gpu_tiles": len(gpu_coords),
            "gpu_fraction": args.gpu_fraction,
            "elapsed_sec": elapsed,
            "ms_per_tile": elapsed / max(1, len(coords)) * 1000.0,
            "ne_worker_sec": ne_sec,
            "gpu_worker_sec": gpu_sec,
        }
    )


if __name__ == "__main__":
    main()
