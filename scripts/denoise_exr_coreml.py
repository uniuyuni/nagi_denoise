"""Run Core ML NAFNet-Fast on an EXR image and write 16-bit TIFF previews."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F


def compute_unit(name: str):
    import coremltools as ct

    return {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[name]


def load_coreml_model(path: str, compute_units: str):
    import coremltools as ct

    return ct.models.MLModel(path, compute_units=compute_unit(compute_units))


def load_exr_rgb(path: Path) -> np.ndarray:
    import OpenEXR

    part = OpenEXR.File(str(path)).parts[0]
    if "RGB" in part.channels:
        arr = part.channels["RGB"].pixels
    else:
        names = part.channels.keys()
        if not all(ch in names for ch in ("R", "G", "B")):
            raise RuntimeError(f"EXR has no RGB or R/G/B channels: {sorted(names)}")
        arr = np.stack([part.channels[ch].pixels for ch in ("R", "G", "B")], axis=-1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise RuntimeError(f"unexpected EXR RGB shape: {arr.shape}")
    arr = np.asarray(arr[..., :3], dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def to_u16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def resize_rgb(x: np.ndarray, scale: float | None = None, size: tuple[int, int] | None = None) -> np.ndarray:
    if scale is None and size is None:
        raise ValueError("scale or size is required")
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0).float()
    if size is None:
        y = F.interpolate(t, scale_factor=float(scale), mode="bilinear", align_corners=False)
    else:
        y = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return y.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32, copy=False)


def pad_reflect_to_tile(x: np.ndarray, tile: int) -> tuple[np.ndarray, int, int]:
    h, w = x.shape[:2]
    pad_h = (tile - h % tile) % tile
    pad_w = (tile - w % tile) % tile
    if pad_h or pad_w:
        x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return x, pad_h, pad_w


def _make_weight(tile: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((tile, tile, 1), dtype=np.float32)
    ramp = np.ones(tile, dtype=np.float32)
    edge = min(overlap, tile // 2)
    vals = np.linspace(0.0, 1.0, edge + 2, dtype=np.float32)[1:-1]
    ramp[:edge] = vals
    ramp[-edge:] = vals[::-1]
    w = ramp[:, None] * ramp[None, :]
    return w[..., None].astype(np.float32)


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(1, length - tile + 1), stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def run_tiled(
    model,
    image_srgb: np.ndarray,
    tile: int,
    progress_every: int,
    batch: int,
    overlap: int,
) -> tuple[np.ndarray, dict[str, float]]:
    padded, pad_h, pad_w = pad_reflect_to_tile(image_srgb, tile)
    h, w = image_srgb.shape[:2]
    hp, wp = padded.shape[:2]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile")
    ys = _tile_starts(hp, tile, stride)
    xs = _tile_starts(wp, tile, stride)
    coords = [(y, x) for y in ys for x in xs]
    total = len(coords)
    out_acc = np.zeros_like(padded, dtype=np.float32)
    weight_acc = np.zeros((*padded.shape[:2], 1), dtype=np.float32)
    weight = _make_weight(tile, overlap)

    predict_time = 0.0
    start = time.perf_counter()
    done = 0
    batch = max(1, int(batch))
    for offset in range(0, total, batch):
        group = coords[offset : offset + batch]
        patches = []
        for y0, x0 in group:
            patch = padded[y0 : y0 + tile, x0 : x0 + tile]
            patches.append(patch.transpose(2, 0, 1))
        inp = np.stack(patches, axis=0).astype(np.float32, copy=False)
        if len(group) < batch:
            pad_n = batch - len(group)
            inp = np.concatenate([inp, np.repeat(inp[-1:], pad_n, axis=0)], axis=0)
        t0 = time.perf_counter()
        pred = model.predict({"input": inp})
        predict_time += time.perf_counter() - t0
        ys_pred = np.asarray(pred["output"], dtype=np.float32)
        for i, (y0, x0) in enumerate(group):
            y = ys_pred[i].transpose(1, 2, 0)
            out_acc[y0 : y0 + tile, x0 : x0 + tile] += y * weight
            weight_acc[y0 : y0 + tile, x0 : x0 + tile] += weight
        done += len(group)
        if progress_every > 0 and (done % progress_every == 0 or done == total):
            elapsed = time.perf_counter() - start
            eta = elapsed / max(1, done) * (total - done)
            print(
                f"[{done:4d}/{total}] elapsed={elapsed:.1f}s "
                f"ms/tile={elapsed / done * 1000.0:.1f} eta={eta:.0f}s",
                flush=True,
            )

    out = out_acc / np.maximum(weight_acc, 1e-8)
    out = out[:h, :w]
    elapsed = time.perf_counter() - start
    stats = {
        "tiles": float(total),
        "batch": float(batch),
        "overlap": float(overlap),
        "elapsed_sec": elapsed,
        "ms_per_tile": elapsed / max(1, total) * 1000.0,
        "predict_ms_per_tile": predict_time / max(1, total) * 1000.0,
        "pad_h": float(pad_h),
        "pad_w": float(pad_w),
    }
    return out, stats


def array_stats(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "p01": float(np.percentile(x, 1)),
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
        "p999": float(np.percentile(x, 99.9)),
    }


def parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,w,h")
    x, y, w, h = parts
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("crop values must be non-negative and size must be positive")
    return x, y, w, h


def output_warnings(input_stats: dict[str, float], denoised_stats: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    if denoised_stats["p50"] > 0.98 and input_stats["p50"] < 0.95:
        warnings.append(
            "denoised median is near white while input median is not; output may be clipped or invalid"
        )
    if denoised_stats["mean"] > input_stats["mean"] + 0.2:
        warnings.append("denoised mean is much brighter than input; verify Core ML compute unit correctness")
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser(description="Denoise an EXR via Core ML NAFNet-Fast.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="runs/nafnet_fast_coreml/nafnet_width64_fp32.mlpackage")
    ap.add_argument("--output-dir", default="runs/coreml_exr_outputs")
    ap.add_argument("--compute-units", choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"], default="cpu_and_gpu")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--input-space", choices=["linear", "srgb"], default="linear")
    ap.add_argument("--exposure", type=float, default=1.0, help="Multiply EXR values before sRGB conversion.")
    ap.add_argument("--crop", type=parse_crop, default=None, help="Optional EXR crop as x,y,w,h before processing.")
    ap.add_argument("--process-scale", type=float, default=1.0, help="Resize before Core ML inference.")
    ap.add_argument(
        "--residual-upsample",
        action="store_true",
        help="At process-scale < 1, upsample denoise residual and add it to the full-res input.",
    )
    ap.add_argument("--progress-every", type=int, default=16)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem

    print(f"loading {in_path}")
    exr = load_exr_rgb(in_path)
    print(f"EXR shape={exr.shape} dtype={exr.dtype}")
    if args.crop is not None:
        x, y, w, h = args.crop
        if x + w > exr.shape[1] or y + h > exr.shape[0]:
            raise ValueError(f"crop {args.crop} exceeds image shape {exr.shape}")
        exr = np.ascontiguousarray(exr[y : y + h, x : x + w])
        stem = f"{stem}_crop_x{x}_y{y}_w{w}_h{h}"
        print(f"cropped EXR to {exr.shape} from x={x} y={y} w={w} h={h}")

    scaled = exr * float(args.exposure)
    if args.input_space == "linear":
        input_srgb = linear_to_srgb(scaled)
    else:
        input_srgb = scaled
    input_srgb = np.clip(input_srgb, 0.0, 1.0).astype(np.float32, copy=False)
    process_srgb = input_srgb
    if args.process_scale != 1.0:
        resize_start = time.perf_counter()
        process_srgb = resize_rgb(input_srgb, scale=args.process_scale)
        print(
            f"resized for processing: {input_srgb.shape[:2]} -> {process_srgb.shape[:2]} "
            f"({time.perf_counter() - resize_start:.1f}s)"
        )

    model = load_coreml_model(args.model, args.compute_units)
    print(f"running Core ML tiles, compute_units={args.compute_units}")
    denoised_srgb, timing = run_tiled(
        model,
        process_srgb,
        args.tile,
        args.progress_every,
        args.batch,
        args.overlap,
    )
    denoised_srgb = np.clip(denoised_srgb, 0.0, 1.0)
    if args.process_scale != 1.0:
        up_start = time.perf_counter()
        if args.residual_upsample:
            residual = denoised_srgb - process_srgb
            residual_full = resize_rgb(residual, size=input_srgb.shape[:2])
            denoised_srgb = np.clip(input_srgb + residual_full, 0.0, 1.0)
        else:
            denoised_srgb = np.clip(resize_rgb(denoised_srgb, size=input_srgb.shape[:2]), 0.0, 1.0)
        timing["upscale_sec"] = time.perf_counter() - up_start

    input_tif = out_dir / f"{stem}_input_srgb16.tiff"
    denoised_tif = out_dir / f"{stem}_coreml_nafnet_srgb16.tiff"
    denoised_linear_tif = out_dir / f"{stem}_coreml_nafnet_linear16.tiff"
    meta_path = out_dir / f"{stem}_coreml_nafnet_meta.json"

    tifffile.imwrite(input_tif, to_u16(input_srgb), photometric="rgb")
    tifffile.imwrite(denoised_tif, to_u16(denoised_srgb), photometric="rgb")
    tifffile.imwrite(denoised_linear_tif, to_u16(srgb_to_linear(denoised_srgb)), photometric="rgb")

    exr_stats = array_stats(exr)
    input_stats = array_stats(input_srgb)
    denoised_stats = array_stats(denoised_srgb)
    warnings = output_warnings(input_stats, denoised_stats)
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)

    meta = {
        "input": str(in_path),
        "model": args.model,
        "compute_units": args.compute_units,
        "tile": args.tile,
        "batch": args.batch,
        "overlap": args.overlap,
        "input_space": args.input_space,
        "exposure": args.exposure,
        "crop": args.crop,
        "process_scale": args.process_scale,
        "residual_upsample": args.residual_upsample,
        "shape": list(exr.shape),
        "timing": timing,
        "warnings": warnings,
        "stats": {
            "exr": exr_stats,
            "input_srgb": input_stats,
            "denoised_srgb": denoised_stats,
        },
        "outputs": {
            "input_srgb16": str(input_tif),
            "denoised_srgb16": str(denoised_tif),
            "denoised_linear16": str(denoised_linear_tif),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {input_tif}")
    print(f"wrote {denoised_tif}")
    print(f"wrote {denoised_linear_tif}")
    print(f"wrote {meta_path}")
    print(f"elapsed={timing['elapsed_sec']:.1f}s ms/tile={timing['ms_per_tile']:.1f}")


if __name__ == "__main__":
    main()
