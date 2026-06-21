"""Suppress tiny magenta chroma dots without changing display luma."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, maximum_filter

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from apply_luma_hf_shrink_filter import make_flat_gate, sigmoid01
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


PRESETS = {
    "mild": {
        "strength": 0.70,
        "dot_threshold": 0.010,
        "dot_transition": 0.005,
        "absolute_threshold": 0.018,
        "absolute_transition": 0.010,
        "target_blend": 0.80,
        "median_size": 5,
        "local_sigma": 1.2,
    },
    "strong": {
        "strength": 0.90,
        "dot_threshold": 0.007,
        "dot_transition": 0.004,
        "absolute_threshold": 0.014,
        "absolute_transition": 0.008,
        "target_blend": 0.90,
        "median_size": 5,
        "local_sigma": 1.4,
    },
    "xstrong": {
        "strength": 1.0,
        "dot_threshold": 0.0045,
        "dot_transition": 0.003,
        "absolute_threshold": 0.010,
        "absolute_transition": 0.006,
        "target_blend": 1.0,
        "median_size": 7,
        "local_sigma": 1.6,
    },
}


def apply_magenta_dot_filter(
    image: np.ndarray,
    guide_image: np.ndarray,
    *,
    strength: float,
    dot_threshold: float,
    dot_transition: float,
    absolute_threshold: float,
    absolute_transition: float,
    target_blend: float,
    median_size: int,
    local_sigma: float,
    dot_size: int,
    magenta_only: bool,
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
    display = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)
    chroma = display - y[..., None]

    flat_gate, flat_stats = make_flat_gate(
        guide_image,
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

    magenta = 0.5 * (display[..., 0] + display[..., 2]) - display[..., 1]
    local = gaussian_filter(magenta, sigma=float(local_sigma), mode="reflect")
    local_median = median_filter(magenta, size=int(median_size), mode="reflect")
    dot_signal = magenta - np.minimum(local, local_median)
    if int(dot_size) > 1:
        dot_signal = maximum_filter(dot_signal, size=int(dot_size), mode="reflect")

    dot_gate = sigmoid01((dot_signal - float(dot_threshold)) / max(float(dot_transition), 1.0e-6))
    abs_gate = sigmoid01((magenta - float(absolute_threshold)) / max(float(absolute_transition), 1.0e-6))
    gate = np.clip(flat_gate * dot_gate * abs_gate * float(strength), 0.0, 1.0).astype(np.float32, copy=False)

    target_chroma = median_filter(chroma, size=(int(median_size), int(median_size), 1), mode="reflect")
    if magenta_only:
        # Remove the magenta axis component while preserving other chroma axes.
        current_axis = np.stack([np.full_like(magenta, 0.5), np.full_like(magenta, -1.0), np.full_like(magenta, 0.5)], axis=2)
        target_magenta = 0.5 * (target_chroma[..., 0] + target_chroma[..., 2]) - target_chroma[..., 1]
        delta = np.maximum(magenta - target_magenta, 0.0) * float(target_blend)
        target_chroma = chroma - delta[..., None] * current_axis
    else:
        target_chroma = chroma * (1.0 - float(target_blend)) + target_chroma * float(target_blend)

    out_chroma = chroma * (1.0 - gate[..., None]) + target_chroma * gate[..., None]
    out_display = np.clip(y[..., None] + out_chroma, 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)

    y_linear = luma(np.clip(base, 0.0, None), LUMA_LINEAR)
    peak_linear = np.max(base, axis=2)
    hdr_signal = np.maximum(
        y_linear - float(hdr_restore_threshold),
        peak_linear - float(hdr_restore_peak_threshold),
    )
    hdr_restore = smoothstep(hdr_signal / max(float(hdr_restore_transition), 1.0e-6))
    out = out * (1.0 - hdr_restore[..., None]) + base * hdr_restore[..., None]

    stats = {
        "strength": float(strength),
        "dot_threshold": float(dot_threshold),
        "dot_transition": float(dot_transition),
        "absolute_threshold": float(absolute_threshold),
        "absolute_transition": float(absolute_transition),
        "target_blend": float(target_blend),
        "median_size": int(median_size),
        "local_sigma": float(local_sigma),
        "dot_size": int(dot_size),
        "magenta_only": bool(magenta_only),
        "magenta_mean": float(np.mean(magenta)),
        "magenta_p95": float(np.quantile(magenta, 0.95)),
        "magenta_p99": float(np.quantile(magenta, 0.99)),
        "dot_signal_p95": float(np.quantile(dot_signal, 0.95)),
        "dot_signal_p99": float(np.quantile(dot_signal, 0.99)),
        "gate_mean": float(np.mean(gate)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
        **flat_stats,
    }
    return out.astype(np.float32, copy=False), stats, gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply tiny magenta-dot chroma cleanup.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--guide-input", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="strong")
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--dot-threshold", type=float, default=None)
    parser.add_argument("--dot-transition", type=float, default=None)
    parser.add_argument("--absolute-threshold", type=float, default=None)
    parser.add_argument("--absolute-transition", type=float, default=None)
    parser.add_argument("--target-blend", type=float, default=None)
    parser.add_argument("--median-size", type=int, default=None)
    parser.add_argument("--local-sigma", type=float, default=None)
    parser.add_argument("--dot-size", type=int, default=1)
    parser.add_argument("--full-chroma-target", action="store_true")
    parser.add_argument("--structure-sigma", type=float, default=1.2)
    parser.add_argument("--detail-sigma", type=float, default=2.8)
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

    params = dict(PRESETS[args.preset])
    for attr, key in (
        ("strength", "strength"),
        ("dot_threshold", "dot_threshold"),
        ("dot_transition", "dot_transition"),
        ("absolute_threshold", "absolute_threshold"),
        ("absolute_transition", "absolute_transition"),
        ("target_blend", "target_blend"),
        ("median_size", "median_size"),
        ("local_sigma", "local_sigma"),
    ):
        value = getattr(args, attr)
        if value is not None:
            params[key] = value

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide_input).expanduser() if args.guide_input else input_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_magenta_dot_{args.preset}"

    image = read_image(input_path)
    guide = read_image(guide_path)
    if image.shape[:2] != guide.shape[:2]:
        raise ValueError(f"shape mismatch: input={image.shape}, guide={guide.shape}")

    out, stats, gate = apply_magenta_dot_filter(
        image,
        guide,
        strength=float(params["strength"]),
        dot_threshold=float(params["dot_threshold"]),
        dot_transition=float(params["dot_transition"]),
        absolute_threshold=float(params["absolute_threshold"]),
        absolute_transition=float(params["absolute_transition"]),
        target_blend=float(params["target_blend"]),
        median_size=int(params["median_size"]),
        local_sigma=float(params["local_sigma"]),
        dot_size=int(args.dot_size),
        magenta_only=not bool(args.full_chroma_target),
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
        "preset": args.preset,
        "params": {**params, "dot_size": args.dot_size, "magenta_only": not bool(args.full_chroma_target)},
        "outputs": {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path), "gate": str(gate_path)},
        "filter": stats,
        "input_stats": image_stats(image),
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
