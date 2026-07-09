"""Blend a detail candidate's display RGB under a region mask, then restore HDR.

This is useful for probes such as SCUNet: the candidate may have attractive
RGB-space detail reconstruction, but its highlights/chroma are not globally
safe. We borrow it only under the existing hair/detail mask and keep HDR peaks
from the base image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from apply_flat_chroma_smoother import linear_to_srgb_np, srgb_to_linear_np
from apply_hair_region_luma_blend import hair_mask
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


def blend_region_rgb(
    reference: np.ndarray,
    base: np.ndarray,
    detail: np.ndarray,
    *,
    mask_mode: str,
    blend_strength: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
    **mask_kwargs: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    base_rgb = np.clip(_safe_rgb(base), 0.0, None)
    detail_rgb = np.clip(_safe_rgb(detail), 0.0, None)
    if base_rgb.shape[:2] != detail_rgb.shape[:2]:
        raise ValueError(f"shape mismatch base={base_rgb.shape} detail={detail_rgb.shape}")
    if mask_mode == "all":
        mask = np.ones(base_rgb.shape[:2], dtype=np.float32)
        mask_stats = {
            "mask_mode": mask_mode,
            "mask_mean": 1.0,
            "mask_p90": 1.0,
            "mask_p99": 1.0,
        }
    elif mask_mode == "hair":
        mask, mask_stats = hair_mask(reference, base, None, **mask_kwargs)
        mask_stats["mask_mode"] = mask_mode
    else:
        raise ValueError(f"unknown mask mode: {mask_mode!r}")
    blend = np.clip(mask * float(blend_strength), 0.0, 1.0).astype(np.float32, copy=False)

    base_d = display(base_rgb)
    detail_d = display(detail_rgb)
    out_display = np.clip(base_d * (1.0 - blend[..., None]) + detail_d * blend[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)

    base_peak = np.max(base_rgb, axis=2)
    hdr = smoothstep01((base_peak - float(hdr_peak_threshold)) / max(float(hdr_transition), 1.0e-6))
    out = out * (1.0 - hdr[..., None]) + base_rgb * hdr[..., None]
    stats = {
        "blend_strength": float(blend_strength),
        "blend_mean": float(np.mean(blend)),
        "blend_p90": float(np.quantile(blend, 0.90)),
        "blend_p99": float(np.quantile(blend, 0.99)),
        "hdr_restore_mean": float(np.mean(hdr)),
        "display_delta_abs_mean": float(np.mean(np.abs(out_display - base_d) * blend[..., None])),
        "display_delta_abs_p99": float(np.quantile(np.abs(out_display - base_d) * blend[..., None], 0.99)),
        **mask_stats,
    }
    return out.astype(np.float32, copy=False), blend, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend detail display RGB under a hair/detail region mask.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--mask-mode", choices=["hair", "all"], default="hair")
    parser.add_argument("--blend-strength", type=float, default=0.45)
    parser.add_argument("--dark-low", type=float, default=0.020)
    parser.add_argument("--dark-high", type=float, default=0.62)
    parser.add_argument("--dark-transition", type=float, default=0.10)
    parser.add_argument("--skin-inhibit", type=float, default=0.95)
    parser.add_argument("--skin-proximity-sigma", type=float, default=70.0)
    parser.add_argument("--skin-proximity-threshold", type=float, default=0.014)
    parser.add_argument("--skin-proximity-transition", type=float, default=0.026)
    parser.add_argument("--texture-threshold", type=float, default=0.004)
    parser.add_argument("--texture-transition", type=float, default=0.012)
    parser.add_argument("--coherence-threshold", type=float, default=0.28)
    parser.add_argument("--coherence-transition", type=float, default=0.18)
    parser.add_argument("--coherence-energy-threshold", type=float, default=0.006)
    parser.add_argument("--coherence-energy-transition", type=float, default=0.006)
    parser.add_argument("--mask-guide-dark-high", type=float, default=0.46)
    parser.add_argument("--mask-guide-transition", type=float, default=0.10)
    parser.add_argument("--mask-guide-weight", type=float, default=0.0)
    parser.add_argument("--mask-blur-sigma", type=float, default=2.0)
    parser.add_argument("--hdr-peak-threshold", type=float, default=0.82)
    parser.add_argument("--hdr-transition", type=float, default=0.25)
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    detail_path = Path(args.detail).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_region_rgb_blend"

    reference = read_image(reference_path)
    base = read_image(base_path)
    detail = read_image(detail_path)
    out, blend, stats = blend_region_rgb(
        reference,
        base,
        detail,
        mask_mode=args.mask_mode,
        blend_strength=args.blend_strength,
        dark_low=args.dark_low,
        dark_high=args.dark_high,
        dark_transition=args.dark_transition,
        skin_inhibit=args.skin_inhibit,
        skin_proximity_sigma=args.skin_proximity_sigma,
        skin_proximity_threshold=args.skin_proximity_threshold,
        skin_proximity_transition=args.skin_proximity_transition,
        texture_threshold=args.texture_threshold,
        texture_transition=args.texture_transition,
        coherence_threshold=args.coherence_threshold,
        coherence_transition=args.coherence_transition,
        coherence_energy_threshold=args.coherence_energy_threshold,
        coherence_energy_transition=args.coherence_energy_transition,
        mask_guide_dark_high=args.mask_guide_dark_high,
        mask_guide_transition=args.mask_guide_transition,
        mask_guide_weight=args.mask_guide_weight,
        mask_blur_sigma=args.mask_blur_sigma,
        hdr_peak_threshold=args.hdr_peak_threshold,
        hdr_transition=args.hdr_transition,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    blend_path = out_dir / f"{name}_blend.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(blend * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(blend_path)
    meta = {
        "reference": str(reference_path),
        "base": str(base_path),
        "detail": str(detail_path),
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "blend": str(blend_path),
        },
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


if __name__ == "__main__":
    main()
