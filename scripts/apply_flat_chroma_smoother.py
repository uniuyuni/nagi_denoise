"""Prototype flat-region chroma smoothing for real-photo NR diagnostics.

This is not meant as the final pipeline. It is a falsification tool: if direct
flat chroma smoothing cannot reduce the measured/visible color grain without
hurting luma edges, then asking the model to learn the same behavior is the
wrong next move.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "nagi_nr" / "src"))

from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


LUMA_SRGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)
LUMA_LINEAR = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    x = x[..., :3]
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def linear_to_srgb_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None).astype(np.float32, copy=False)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_to_linear_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def luma(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(_safe_rgb(rgb) * weights.reshape(1, 1, 3), axis=2)


def sigmoid01(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False)


def flat_gate(
    base_linear: np.ndarray,
    *,
    detail_sigma: float,
    threshold: float,
    transition: float,
    highlight_threshold: float,
    highlight_transition: float,
) -> np.ndarray:
    base_srgb = linear_to_srgb_np(_safe_rgb(base_linear))
    y_display = luma(base_srgb, LUMA_SRGB)
    detail = np.abs(y_display - gaussian_filter(y_display, sigma=float(detail_sigma), mode="reflect"))
    flat = sigmoid01((float(threshold) - detail) / max(float(transition), 1.0e-6))

    y_linear = luma(base_linear, LUMA_LINEAR)
    highlight = sigmoid01(
        (y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6)
    )
    return (flat * (1.0 - highlight)).astype(np.float32, copy=False)


def smooth_chroma(
    base_linear: np.ndarray,
    *,
    strength: float,
    chroma_sigma: float,
    detail_sigma: float,
    threshold: float,
    transition: float,
    highlight_threshold: float,
    highlight_transition: float,
) -> tuple[np.ndarray, dict, np.ndarray]:
    base = _safe_rgb(base_linear)
    base_srgb = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    y = luma(base_srgb, LUMA_SRGB)[..., None]
    chroma = base_srgb - y
    low_chroma = gaussian_filter(chroma, sigma=(float(chroma_sigma), float(chroma_sigma), 0.0), mode="reflect")
    gate = flat_gate(
        base,
        detail_sigma=detail_sigma,
        threshold=threshold,
        transition=transition,
        highlight_threshold=highlight_threshold,
        highlight_transition=highlight_transition,
    )
    blend = np.clip(gate * float(strength), 0.0, 1.0)[..., None]
    out_srgb = y + chroma * (1.0 - blend) + low_chroma * blend
    out = srgb_to_linear_np(out_srgb)
    stats = {
        "strength": float(strength),
        "chroma_sigma": float(chroma_sigma),
        "detail_sigma": float(detail_sigma),
        "threshold": float(threshold),
        "transition": float(transition),
        "highlight_threshold": float(highlight_threshold),
        "highlight_transition": float(highlight_transition),
        "gate_mean": float(np.mean(gate)),
        "gate_p50": float(np.quantile(gate, 0.50)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
    }
    return out.astype(np.float32, copy=False), stats, gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply diagnostic flat chroma smoothing.")
    parser.add_argument("--input", required=True, help="Denoised linear EXR/TIFF input.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--chroma-sigma", type=float, default=1.4)
    parser.add_argument("--detail-sigma", type=float, default=1.2)
    parser.add_argument("--threshold", type=float, default=0.010)
    parser.add_argument("--transition", type=float, default=0.006)
    parser.add_argument("--highlight-threshold", type=float, default=1.0)
    parser.add_argument("--highlight-transition", type=float, default=0.2)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_flat_chroma_smooth"

    base = read_image(input_path)
    out, stats, gate = smooth_chroma(
        base,
        strength=args.strength,
        chroma_sigma=args.chroma_sigma,
        detail_sigma=args.detail_sigma,
        threshold=args.threshold,
        transition=args.transition,
        highlight_threshold=args.highlight_threshold,
        highlight_transition=args.highlight_transition,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}_meta.json"

    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)

    meta = {
        "input": str(input_path),
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
        },
        "smoother": stats,
        "input_stats": image_stats(base),
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
