"""Perfect NR diagnostic image probe.

This script is deliberately model-agnostic. It reads an image, reports linear/HDR
statistics, and writes a stable preview PNG plus JSON metadata. Use it before
judging any denoiser output by eye.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _read_exr(path: Path) -> np.ndarray:
    import OpenEXR

    file = OpenEXR.InputFile(str(path))
    header = file.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    channel_names = header["channels"].keys()
    channels = []
    for name in ("R", "G", "B"):
        if name not in channel_names:
            raise ValueError(f"EXR is missing channel {name!r}: {path}")
        raw = file.channel(name)
        channels.append(np.frombuffer(raw, dtype=np.float32).reshape(height, width))
    return np.stack(channels, axis=-1).astype(np.float32, copy=False)


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".exr":
        return _read_exr(path)
    if suffix in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        arr = arr[..., :3]
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
        return arr.astype(np.float32, copy=False)
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def srgb_oetf(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def make_preview(img: np.ndarray, exposure: float = 1.0, tone: str = "reinhard") -> np.ndarray:
    x = np.nan_to_num(img.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x * float(exposure), 0.0, None)
    if tone == "reinhard":
        x = x / (1.0 + x)
    elif tone == "clip":
        x = np.clip(x, 0.0, 1.0)
    else:
        raise ValueError(f"unknown tone curve: {tone}")
    return (srgb_oetf(x) * 255.0 + 0.5).astype(np.uint8)


def image_stats(img: np.ndarray) -> dict:
    x = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(x)
    safe = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = safe[..., :3]
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    peak = np.max(rgb, axis=2)

    def q(a: np.ndarray, value: float) -> float:
        return float(np.quantile(a, value))

    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite_fraction": float(np.mean(finite)),
        "rgb_min": float(np.nanmin(x)),
        "rgb_max": float(np.nanmax(x)),
        "rgb_p50": q(rgb, 0.50),
        "rgb_p95": q(rgb, 0.95),
        "rgb_p99": q(rgb, 0.99),
        "rgb_p999": q(rgb, 0.999),
        "luma_min": float(np.nanmin(luma)),
        "luma_max": float(np.nanmax(luma)),
        "luma_p50": q(luma, 0.50),
        "luma_p95": q(luma, 0.95),
        "luma_p99": q(luma, 0.99),
        "luma_p999": q(luma, 0.999),
        "peak_gt_1_fraction": float(np.mean(peak > 1.0)),
        "luma_gt_1_fraction": float(np.mean(luma > 1.0)),
        "peak_gt_2_fraction": float(np.mean(peak > 2.0)),
        "peak_gt_4_fraction": float(np.mean(peak > 4.0)),
        "channel_p99": [q(rgb[..., i], 0.99) for i in range(3)],
        "channel_max": [float(np.nanmax(rgb[..., i])) for i in range(3)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe HDR/detail statistics for Perfect NR crops.")
    parser.add_argument("input")
    parser.add_argument("--output-dir", default="runs/nagi_v2/probes")
    parser.add_argument("--name", default=None)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--tone", choices=["reinhard", "clip"], default="reinhard")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or input_path.stem.replace(" ", "_")

    img = read_image(input_path)
    stats = image_stats(img)
    stats.update(
        {
            "input": str(input_path),
            "preview_exposure": args.exposure,
            "preview_tone": args.tone,
        }
    )

    preview = make_preview(img, exposure=args.exposure, tone=args.tone)
    preview_path = out_dir / f"{name}_preview.png"
    json_path = out_dir / f"{name}_stats.json"
    Image.fromarray(preview).save(preview_path)
    json_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"wrote {preview_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()

