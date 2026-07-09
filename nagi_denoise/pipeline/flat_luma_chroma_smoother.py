"""Diagnostic flat-region luma+chroma smoothing for very noisy real photos.

This is a falsification tool, not the final model path. It answers one practical
question: if we add a bounded flat-region luma smoother on top of the current
pipeline, does the noisy real photo become visually comparable? If yes, the
model needs a luma-noise mechanism; if no, the issue is deeper than a missing
flat luma suppressor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

from .detail_guard import write_exr, write_tiff
from .probe import image_stats, make_preview, read_image


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


def sigmoid01(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False)


def smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def luma(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(_safe_rgb(rgb) * weights.reshape(1, 1, 3), axis=2)


def make_flat_gate(
    y_display: np.ndarray,
    y_linear: np.ndarray,
    *,
    structure_sigma: float,
    edge_sigma: float,
    flat_threshold: float,
    flat_transition: float,
    edge_threshold: float,
    edge_transition: float,
    highlight_threshold: float,
    highlight_transition: float,
) -> np.ndarray:
    structure = gaussian_filter(y_display, sigma=float(structure_sigma), mode="reflect")
    detail = np.abs(structure - gaussian_filter(structure, sigma=float(structure_sigma) * 2.0, mode="reflect"))
    edge = gaussian_gradient_magnitude(structure, sigma=float(edge_sigma), mode="reflect")
    flat = sigmoid01((float(flat_threshold) - detail) / max(float(flat_transition), 1.0e-6))
    non_edge = sigmoid01((float(edge_threshold) - edge) / max(float(edge_transition), 1.0e-6))
    highlight = sigmoid01(
        (y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6)
    )
    return (flat * non_edge * (1.0 - highlight)).astype(np.float32, copy=False)


def smooth_flat_luma_chroma(
    image: np.ndarray,
    *,
    luma_strength: float,
    chroma_strength: float,
    luma_sigma: float,
    chroma_sigma: float,
    structure_sigma: float,
    edge_sigma: float,
    flat_threshold: float,
    flat_transition: float,
    edge_threshold: float,
    edge_transition: float,
    highlight_threshold: float,
    highlight_transition: float,
    hdr_restore_threshold: float,
    hdr_restore_transition: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    base = _safe_rgb(image)
    display = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)
    y_linear = luma(base, LUMA_LINEAR)
    chroma = display - y[..., None]

    gate = make_flat_gate(
        y,
        y_linear,
        structure_sigma=structure_sigma,
        edge_sigma=edge_sigma,
        flat_threshold=flat_threshold,
        flat_transition=flat_transition,
        edge_threshold=edge_threshold,
        edge_transition=edge_transition,
        highlight_threshold=highlight_threshold,
        highlight_transition=highlight_transition,
    )
    luma_blend = np.clip(gate * float(luma_strength), 0.0, 1.0)
    chroma_blend = np.clip(gate * float(chroma_strength), 0.0, 1.0)[..., None]

    y_low = gaussian_filter(y, sigma=float(luma_sigma), mode="reflect")
    chroma_low = gaussian_filter(chroma, sigma=(float(chroma_sigma), float(chroma_sigma), 0.0), mode="reflect")
    out_y = y * (1.0 - luma_blend) + y_low * luma_blend
    out_chroma = chroma * (1.0 - chroma_blend) + chroma_low * chroma_blend
    out_display = np.clip(out_y[..., None] + out_chroma, 0.0, 1.0)
    out = srgb_to_linear_np(out_display)

    # Do not let the diagnostic smoother destroy true HDR peaks.
    hdr_restore = smoothstep((y_linear - float(hdr_restore_threshold)) / max(float(hdr_restore_transition), 1.0e-6))
    out = out * (1.0 - hdr_restore[..., None]) + base * hdr_restore[..., None]

    stats = {
        "luma_strength": float(luma_strength),
        "chroma_strength": float(chroma_strength),
        "luma_sigma": float(luma_sigma),
        "chroma_sigma": float(chroma_sigma),
        "gate_mean": float(np.mean(gate)),
        "gate_p50": float(np.quantile(gate, 0.50)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
    }
    return out.astype(np.float32, copy=False), stats, gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply diagnostic flat luma+chroma smoothing.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--luma-strength", type=float, default=0.55)
    parser.add_argument("--chroma-strength", type=float, default=0.80)
    parser.add_argument("--luma-sigma", type=float, default=1.1)
    parser.add_argument("--chroma-sigma", type=float, default=1.4)
    parser.add_argument("--structure-sigma", type=float, default=1.2)
    parser.add_argument("--edge-sigma", type=float, default=1.0)
    parser.add_argument("--flat-threshold", type=float, default=0.018)
    parser.add_argument("--flat-transition", type=float, default=0.010)
    parser.add_argument("--edge-threshold", type=float, default=0.030)
    parser.add_argument("--edge-transition", type=float, default=0.015)
    parser.add_argument("--highlight-threshold", type=float, default=1.0)
    parser.add_argument("--highlight-transition", type=float, default=0.25)
    parser.add_argument("--hdr-restore-threshold", type=float, default=0.85)
    parser.add_argument("--hdr-restore-transition", type=float, default=0.25)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_flat_luma_chroma_smooth"

    image = read_image(input_path)
    out, stats, gate = smooth_flat_luma_chroma(
        image,
        luma_strength=args.luma_strength,
        chroma_strength=args.chroma_strength,
        luma_sigma=args.luma_sigma,
        chroma_sigma=args.chroma_sigma,
        structure_sigma=args.structure_sigma,
        edge_sigma=args.edge_sigma,
        flat_threshold=args.flat_threshold,
        flat_transition=args.flat_transition,
        edge_threshold=args.edge_threshold,
        edge_transition=args.edge_transition,
        highlight_threshold=args.highlight_threshold,
        highlight_transition=args.highlight_transition,
        hdr_restore_threshold=args.hdr_restore_threshold,
        hdr_restore_transition=args.hdr_restore_transition,
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
        "input_stats": image_stats(image),
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
