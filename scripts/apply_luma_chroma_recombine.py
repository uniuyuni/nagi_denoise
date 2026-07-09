"""Recombine luma from one candidate with chroma/HDR from a safe base."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def display(image: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb_np(np.clip(_safe_rgb(image), 0.0, None)), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def display_chroma(display_rgb: np.ndarray) -> np.ndarray:
    y = luma(display_rgb, LUMA_SRGB)
    return (display_rgb / np.maximum(y[..., None], 1.0e-6)).astype(np.float32, copy=False)


def recombine_luma_chroma(
    base: np.ndarray,
    luma_source: np.ndarray,
    *,
    luma_strength: float,
    chroma_source_mix: float,
    chroma_limit: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
) -> tuple[np.ndarray, dict[str, float]]:
    base_rgb = np.clip(_safe_rgb(base), 0.0, None)
    luma_rgb = np.clip(_safe_rgb(luma_source), 0.0, None)
    if base_rgb.shape[:2] != luma_rgb.shape[:2]:
        raise ValueError(f"shape mismatch base={base_rgb.shape} luma_source={luma_rgb.shape}")

    base_d = display(base_rgb)
    luma_d = display(luma_rgb)
    base_y = luma(base_d, LUMA_SRGB)
    source_y = luma(luma_d, LUMA_SRGB)
    y = np.clip(base_y * (1.0 - float(luma_strength)) + source_y * float(luma_strength), 0.0, 1.0)

    base_chroma = display_chroma(base_d)
    source_chroma = display_chroma(luma_d)
    cm = np.clip(float(chroma_source_mix), 0.0, 1.0)
    chroma = base_chroma * (1.0 - cm) + source_chroma * cm
    chroma = np.clip(chroma, 0.0, float(chroma_limit))
    out_display = np.clip(chroma * y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)

    base_peak = np.max(base_rgb, axis=2)
    hdr = smoothstep01((base_peak - float(hdr_peak_threshold)) / max(float(hdr_transition), 1.0e-6))
    out = out * (1.0 - hdr[..., None]) + base_rgb * hdr[..., None]

    stats = {
        "luma_strength": float(luma_strength),
        "chroma_source_mix": float(chroma_source_mix),
        "chroma_limit": float(chroma_limit),
        "hdr_restore_mean": float(np.mean(hdr)),
        "display_luma_delta_abs_mean": float(np.mean(np.abs(y - base_y))),
        "display_luma_delta_abs_p99": float(np.quantile(np.abs(y - base_y), 0.99)),
        "source_luma_delta_abs_mean": float(np.mean(np.abs(source_y - base_y))),
        "source_luma_delta_abs_p99": float(np.quantile(np.abs(source_y - base_y), 0.99)),
    }
    return out.astype(np.float32, copy=False), stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Recombine luma from one image with chroma/HDR from another.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--luma-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--luma-strength", type=float, default=1.0)
    parser.add_argument("--chroma-source-mix", type=float, default=0.0)
    parser.add_argument("--chroma-limit", type=float, default=8.0)
    parser.add_argument("--hdr-peak-threshold", type=float, default=0.82)
    parser.add_argument("--hdr-transition", type=float, default=0.25)
    args = parser.parse_args()

    base_path = Path(args.base).expanduser()
    luma_path = Path(args.luma_source).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_luma_chroma_recombine"

    base = read_image(base_path)
    luma_source = read_image(luma_path)
    out, stats = recombine_luma_chroma(
        base,
        luma_source,
        luma_strength=args.luma_strength,
        chroma_source_mix=args.chroma_source_mix,
        chroma_limit=args.chroma_limit,
        hdr_peak_threshold=args.hdr_peak_threshold,
        hdr_transition=args.hdr_transition,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    meta = {
        "base": str(base_path),
        "luma_source": str(luma_path),
        "outputs": {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path)},
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


if __name__ == "__main__":
    main()
