"""Restore v8 in blue/cyan structure where neutral chroma cleanup can overreach."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma
from apply_luma_hf_shrink_filter import sigmoid01
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import image_stats, make_preview, read_image


def display(image: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(np.asarray(image, dtype=np.float32)[..., :3], nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(linear_to_srgb_np(np.clip(rgb, 0.0, None)), 0.0, 1.0).astype(np.float32, copy=False)


def blue_structure_mask(
    base_linear: np.ndarray,
    *,
    blue_threshold: float,
    blue_transition: float,
    chroma_threshold: float,
    chroma_transition: float,
    luma_min: float,
    luma_max: float,
) -> np.ndarray:
    base = display(base_linear)
    y = luma(base, LUMA_SRGB)
    chroma = base - y[..., None]
    low = gaussian_filter(chroma, sigma=(2.0, 2.0, 0.0), mode="reflect")
    low_blue = low[..., 2] - 0.5 * (low[..., 0] + low[..., 1])
    low_mag = np.sqrt(np.sum(low * low, axis=2))
    blue = sigmoid01((low_blue - float(blue_threshold)) / float(blue_transition))
    chroma_open = sigmoid01((low_mag - float(chroma_threshold)) / float(chroma_transition))
    luma_open = sigmoid01((y - float(luma_min)) / 0.06) * sigmoid01((float(luma_max) - y) / 0.12)
    return np.clip(blue * chroma_open * luma_open, 0.0, 1.0).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Protect blue/cyan structure after neutral chroma cleanup.")
    parser.add_argument("--base", required=True, help="Pre-neutral v8 EXR.")
    parser.add_argument("--candidate", required=True, help="Neutral-cleaned v9 EXR.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strength", type=float, default=0.72)
    parser.add_argument("--blue-threshold", type=float, default=0.052)
    parser.add_argument("--blue-transition", type=float, default=0.024)
    parser.add_argument("--chroma-threshold", type=float, default=0.064)
    parser.add_argument("--chroma-transition", type=float, default=0.030)
    parser.add_argument("--luma-min", type=float, default=0.045)
    parser.add_argument("--luma-max", type=float, default=0.82)
    parser.add_argument("--preview", default=None)
    parser.add_argument("--mask-preview", default=None)
    parser.add_argument("--meta", default=None)
    args = parser.parse_args()

    base = read_image(Path(args.base))
    candidate = read_image(Path(args.candidate))
    if base.shape != candidate.shape:
        raise ValueError(f"shape mismatch base={base.shape} candidate={candidate.shape}")
    mask = blue_structure_mask(
        base,
        blue_threshold=args.blue_threshold,
        blue_transition=args.blue_transition,
        chroma_threshold=args.chroma_threshold,
        chroma_transition=args.chroma_transition,
        luma_min=args.luma_min,
        luma_max=args.luma_max,
    )
    restore = np.clip(mask * float(args.strength), 0.0, 1.0)
    out = candidate * (1.0 - restore[..., None]) + base * restore[..., None]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_exr(out_path, out)
    preview_path = Path(args.preview) if args.preview else out_path.with_name(out_path.stem + "_preview.png")
    mask_path = Path(args.mask_preview) if args.mask_preview else out_path.with_name(out_path.stem + "_mask.png")
    meta_path = Path(args.meta) if args.meta else out_path.with_suffix(".json")
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(restore * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(mask_path)
    meta = {
        "base": str(Path(args.base)),
        "candidate": str(Path(args.candidate)),
        "output": str(out_path),
        "preview": str(preview_path),
        "mask_preview": str(mask_path),
        "restore_mean": float(np.mean(restore)),
        "restore_p95": float(np.quantile(restore, 0.95)),
        "restore_p99": float(np.quantile(restore, 0.99)),
        "output_stats": image_stats(out),
        "args": vars(args),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
