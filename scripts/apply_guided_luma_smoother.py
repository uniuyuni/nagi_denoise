"""Apply HDR-safe guided luma smoothing on top of a chroma-denoised image.

This is a practical diagnostic for the remaining grain in very noisy real
photos: preserve the candidate chroma/detail, but suppress display-luma grain
with an edge-aware guided filter and a flat-region gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, uniform_filter

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


def sigmoid01(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False)


def guided_filter_gray(guide: np.ndarray, src: np.ndarray, *, radius: int, eps: float) -> np.ndarray:
    size = int(radius) * 2 + 1
    mean_i = uniform_filter(guide, size=size, mode="reflect")
    mean_p = uniform_filter(src, size=size, mode="reflect")
    corr_i = uniform_filter(guide * guide, size=size, mode="reflect")
    corr_ip = uniform_filter(guide * src, size=size, mode="reflect")
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    mean_a = uniform_filter(a, size=size, mode="reflect")
    mean_b = uniform_filter(b, size=size, mode="reflect")
    return (mean_a * guide + mean_b).astype(np.float32, copy=False)


def make_luma_gate(
    guide_y: np.ndarray,
    guide_y_linear: np.ndarray,
    *,
    structure_sigma: float,
    detail_sigma: float,
    detail_threshold: float,
    detail_transition: float,
    edge_sigma: float,
    edge_threshold: float,
    edge_transition: float,
    highlight_threshold: float,
    highlight_transition: float,
) -> np.ndarray:
    structure = gaussian_filter(guide_y, sigma=float(structure_sigma), mode="reflect")
    detail = np.abs(structure - gaussian_filter(structure, sigma=float(detail_sigma), mode="reflect"))
    edge = gaussian_gradient_magnitude(structure, sigma=float(edge_sigma), mode="reflect")
    flat = sigmoid01((float(detail_threshold) - detail) / max(float(detail_transition), 1.0e-6))
    non_edge = sigmoid01((float(edge_threshold) - edge) / max(float(edge_transition), 1.0e-6))
    highlight = smoothstep((guide_y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6))
    return (flat * non_edge * (1.0 - highlight)).astype(np.float32, copy=False)


def apply_guided_luma_smoothing(
    image: np.ndarray,
    guide_image: np.ndarray,
    *,
    strength: float,
    radius: int,
    eps: float,
    guide_sigma: float,
    structure_sigma: float,
    detail_sigma: float,
    detail_threshold: float,
    detail_transition: float,
    edge_sigma: float,
    edge_threshold: float,
    edge_transition: float,
    highlight_threshold: float,
    highlight_transition: float,
    hdr_restore_peak_threshold: float,
    hdr_restore_threshold: float,
    hdr_restore_transition: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    base = np.nan_to_num(image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    guide_base = np.nan_to_num(guide_image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    display = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    guide_display = np.clip(linear_to_srgb_np(guide_base), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)
    guide_y = luma(guide_display, LUMA_SRGB)
    guide_y_linear = luma(np.clip(guide_base, 0.0, None), LUMA_LINEAR)

    structure = gaussian_filter(guide_y, sigma=float(guide_sigma), mode="reflect")
    y_smooth = guided_filter_gray(structure, y, radius=int(radius), eps=float(eps))
    gate = make_luma_gate(
        guide_y,
        guide_y_linear,
        structure_sigma=structure_sigma,
        detail_sigma=detail_sigma,
        detail_threshold=detail_threshold,
        detail_transition=detail_transition,
        edge_sigma=edge_sigma,
        edge_threshold=edge_threshold,
        edge_transition=edge_transition,
        highlight_threshold=highlight_threshold,
        highlight_transition=highlight_transition,
    )
    blend = np.clip(gate * float(strength), 0.0, 1.0)
    out_y = y * (1.0 - blend) + y_smooth * blend

    chroma = display / np.maximum(y[..., None], 1.0e-6)
    out_display = np.clip(chroma * np.maximum(out_y[..., None], 0.0), 0.0, 1.0)
    out = srgb_to_linear_np(out_display)

    y_linear = luma(base, LUMA_LINEAR)
    peak_linear = np.max(base, axis=2)
    hdr_signal = np.maximum(y_linear - float(hdr_restore_threshold), peak_linear - float(hdr_restore_peak_threshold))
    hdr_restore = smoothstep(hdr_signal / max(float(hdr_restore_transition), 1.0e-6))
    out = out * (1.0 - hdr_restore[..., None]) + base * hdr_restore[..., None]

    stats = {
        "strength": float(strength),
        "radius": int(radius),
        "eps": float(eps),
        "guide_sigma": float(guide_sigma),
        "gate_mean": float(np.mean(gate)),
        "gate_p50": float(np.quantile(gate, 0.50)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
    }
    return out.astype(np.float32, copy=False), stats, gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply guided luma smoothing to a denoised image.")
    parser.add_argument("--input", required=True, help="Chroma-denoised image to smooth.")
    parser.add_argument("--guide-input", default=None, help="Original noisy image used for edge/highlight gating.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.0025)
    parser.add_argument("--guide-sigma", type=float, default=1.0)
    parser.add_argument("--structure-sigma", type=float, default=1.2)
    parser.add_argument("--detail-sigma", type=float, default=2.6)
    parser.add_argument("--detail-threshold", type=float, default=0.018)
    parser.add_argument("--detail-transition", type=float, default=0.010)
    parser.add_argument("--edge-sigma", type=float, default=1.0)
    parser.add_argument("--edge-threshold", type=float, default=0.030)
    parser.add_argument("--edge-transition", type=float, default=0.015)
    parser.add_argument("--highlight-threshold", type=float, default=1.0)
    parser.add_argument("--highlight-transition", type=float, default=0.25)
    parser.add_argument("--hdr-restore-peak-threshold", type=float, default=0.95)
    parser.add_argument("--hdr-restore-threshold", type=float, default=0.85)
    parser.add_argument("--hdr-restore-transition", type=float, default=0.25)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide_input).expanduser() if args.guide_input else input_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_guided_luma"

    image = read_image(input_path)
    guide = read_image(guide_path)
    if image.shape[:2] != guide.shape[:2]:
        raise ValueError(f"shape mismatch: input={image.shape}, guide={guide.shape}")
    out, stats, gate = apply_guided_luma_smoothing(
        image,
        guide,
        strength=args.strength,
        radius=args.radius,
        eps=args.eps,
        guide_sigma=args.guide_sigma,
        structure_sigma=args.structure_sigma,
        detail_sigma=args.detail_sigma,
        detail_threshold=args.detail_threshold,
        detail_transition=args.detail_transition,
        edge_sigma=args.edge_sigma,
        edge_threshold=args.edge_threshold,
        edge_transition=args.edge_transition,
        highlight_threshold=args.highlight_threshold,
        highlight_transition=args.highlight_transition,
        hdr_restore_peak_threshold=args.hdr_restore_peak_threshold,
        hdr_restore_threshold=args.hdr_restore_threshold,
        hdr_restore_transition=args.hdr_restore_transition,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}.json"

    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    meta = {
        "input": str(input_path),
        "guide_input": str(guide_path),
        "outputs": {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path), "gate": str(gate_path)},
        "smoother": stats,
        "input_stats": image_stats(image),
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
