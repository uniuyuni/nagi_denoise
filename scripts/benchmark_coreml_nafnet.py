"""Benchmark and validate a Core ML NAFNet-Fast export."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

from nagi_nr_bench.eval_sidd_val import psnr_srgb


def compute_unit(name: str):
    import coremltools as ct

    return {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[name]


def load_model(path: str, compute_units: str):
    import coremltools as ct

    return ct.models.MLModel(path, compute_units=compute_unit(compute_units))


def predict_patch(model, patch_u8: np.ndarray, batch: int = 1) -> np.ndarray:
    x = patch_u8.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    if batch > 1:
        x = np.repeat(x, batch, axis=0)
    pred = model.predict({"input": x})
    y = pred["output"]
    y = np.asarray(y)[0].transpose(1, 2, 0)
    return (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def benchmark_random(model, height: int, width: int, warmup: int, iters: int, batch: int) -> float:
    x = np.random.rand(batch, 3, height, width).astype(np.float32)
    for _ in range(warmup):
        _ = model.predict({"input": x})
    start = time.perf_counter()
    for _ in range(iters):
        out = model.predict({"input": x})
    elapsed = time.perf_counter() - start
    y = out["output"]
    if tuple(y.shape) != (batch, 3, height, width):
        raise RuntimeError(f"unexpected output shape: {tuple(y.shape)}")
    return elapsed / max(1, iters) * 1000.0


def load_validation(noisy_mat: str, gt_mat: str) -> tuple[np.ndarray, np.ndarray]:
    noisy = sio.loadmat(noisy_mat)
    nkey = next(k for k in noisy if not k.startswith("__"))
    noisy_arr = noisy[nkey]
    gt = sio.loadmat(gt_mat)
    gkey = next(k for k in gt if not k.startswith("__"))
    gt_arr = gt[gkey]
    if noisy_arr.shape != gt_arr.shape:
        raise RuntimeError(f"shape mismatch: noisy={noisy_arr.shape}, gt={gt_arr.shape}")
    return noisy_arr, gt_arr


def evaluate_validation(
    model,
    noisy_mat: str,
    gt_mat: str,
    max_patches: int,
    batch: int,
    profile_loop: bool = False,
    progress_every: int = 0,
) -> dict[str, float]:
    noisy, gt = load_validation(noisy_mat, gt_mat)
    total = noisy.shape[0] * noisy.shape[1]
    limit = total if max_patches <= 0 else min(max_patches, total)
    psnr_in: list[float] = []
    psnr_out: list[float] = []
    phase = {
        "preprocess": 0.0,
        "predict": 0.0,
        "postprocess": 0.0,
        "psnr": 0.0,
    }
    done = 0
    start = time.perf_counter()
    for i in range(noisy.shape[0]):
        for j in range(noisy.shape[1]):
            if done >= limit:
                break
            n_patch = noisy[i, j]
            g_patch = gt[i, j]

            t0 = time.perf_counter()
            x = n_patch.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            if batch > 1:
                x = np.repeat(x, batch, axis=0)
            phase["preprocess"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            pred = model.predict({"input": x})
            phase["predict"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            y = np.asarray(pred["output"])[0].transpose(1, 2, 0)
            out = (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
            phase["postprocess"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            psnr_in.append(psnr_srgb(n_patch, g_patch))
            psnr_out.append(psnr_srgb(out, g_patch))
            phase["psnr"] += time.perf_counter() - t0
            done += 1
            if progress_every > 0 and (done % progress_every == 0 or done == limit):
                elapsed = time.perf_counter() - start
                eta = elapsed / max(1, done) * (limit - done)
                print(
                    f"[{done:4d}/{limit}] psnr={np.mean(psnr_out):.3f} "
                    f"ms={elapsed / done * 1000.0:.1f} eta={eta:.0f}s",
                    flush=True,
                )
        if done >= limit:
            break
    elapsed = time.perf_counter() - start
    result = {
        "patches": float(done),
        "psnr_in": float(np.mean(psnr_in)),
        "psnr_out": float(np.mean(psnr_out)),
        "ms_patch": elapsed / max(1, done) * 1000.0,
    }
    if profile_loop:
        for key, value in phase.items():
            result[f"{key}_ms_patch"] = value / max(1, done) * 1000.0
        result["profile_accounted_ms_patch"] = sum(phase.values()) / max(1, done) * 1000.0
    return result


def _markdown(args: argparse.Namespace, random_ms: float, val: dict[str, float]) -> str:
    lines = [
        "# Core ML NAFNet-Fast Benchmark",
        "",
        f"Model: `{args.model}`",
        f"Compute units: `{args.compute_units}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| random forward ms/patch | {random_ms:.1f} |",
    ]
    if val:
        lines.extend(
            [
                f"| validation patches | {int(val['patches'])} |",
                f"| noisy PSNR | {val['psnr_in']:.3f} dB |",
                f"| denoised PSNR | {val['psnr_out']:.3f} dB |",
                f"| validation ms/patch | {val['ms_patch']:.1f} |",
            ]
        )
        if "predict_ms_patch" in val:
            lines.extend(
                [
                    f"| preprocess ms/patch | {val['preprocess_ms_patch']:.1f} |",
                    f"| Core ML predict ms/patch | {val['predict_ms_patch']:.1f} |",
                    f"| postprocess ms/patch | {val['postprocess_ms_patch']:.1f} |",
                    f"| PSNR ms/patch | {val['psnr_ms_patch']:.1f} |",
                    f"| accounted ms/patch | {val['profile_accounted_ms_patch']:.1f} |",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark a Core ML NAFNet-Fast model.")
    ap.add_argument("--model", default="runs/nafnet_fast_coreml/nafnet_width64_fp32.mlpackage")
    ap.add_argument("--compute-units", choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"], default="all")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    ap.add_argument("--max-patches", type=int, default=32)
    ap.add_argument("--profile-loop", action="store_true")
    ap.add_argument("--progress-every", type=int, default=0)
    ap.add_argument("--output-md", default="runs/nafnet_fast_coreml/benchmark.md")
    args = ap.parse_args()

    model = load_model(args.model, args.compute_units)
    random_ms = benchmark_random(model, args.height, args.width, args.warmup, args.iters, args.batch)
    print(f"random forward: {random_ms:.1f} ms/batch ({random_ms / args.batch:.1f} ms/item)")

    val = {}
    if args.max_patches != 0:
        val = evaluate_validation(
            model,
            args.noisy_mat,
            args.gt_mat,
            args.max_patches,
            args.batch,
            args.profile_loop,
            args.progress_every,
        )
        print(
            "validation: patches={patches:.0f} psnr={psnr_out:.3f} "
            "ms={ms_patch:.1f}".format(**val)
        )
        if args.profile_loop:
            print(
                "loop profile: preprocess={preprocess_ms_patch:.1f} "
                "predict={predict_ms_patch:.1f} post={postprocess_ms_patch:.1f} "
                "psnr={psnr_ms_patch:.1f}".format(**val)
            )

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_markdown(args, random_ms, val), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
