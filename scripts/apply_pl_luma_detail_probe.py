"""Probe PL-derived luma detail as an upper-bound hair-detail reference.

This does not treat PhotoLab as the final teacher. It transfers only local luma
mid/high-frequency detail from a same-resolution PL TIFF into the HDR-safe base
under the existing hair-like mask. Color, global exposure, and HDR peaks stay
from the base.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_hair_region_luma_blend import hair_mask
from apply_luma_tail_speckle_filter import sigmoid01
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def base_display(image: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb_np(np.clip(_safe_rgb(image), 0.0, None)), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def transfer_pl_luma_detail(
    reference: np.ndarray,
    base: np.ndarray,
    pl_display: np.ndarray,
    *,
    mid_weight: float,
    fine_weight: float,
    low_sigma: float,
    mid_sigma: float,
    fine_sigma: float,
    detail_limit: float,
    dark_lift_strength: float,
    dark_lift_limit: float,
    blend_strength: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
    **mask_kwargs: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    base_rgb = np.clip(_safe_rgb(base), 0.0, None)
    base_d = base_display(base_rgb)
    pl_d = np.clip(_safe_rgb(pl_display), 0.0, 1.0)
    if base_d.shape != pl_d.shape:
        raise ValueError(f"shape mismatch base={base_d.shape} pl={pl_d.shape}")

    mask, mask_stats = hair_mask(reference, base, **mask_kwargs)
    base_y = luma(base_d, LUMA_SRGB)
    pl_y = luma(pl_d, LUMA_SRGB)

    pl_mid = gaussian_filter(pl_y, sigma=float(mid_sigma), mode="reflect") - gaussian_filter(
        pl_y, sigma=float(low_sigma), mode="reflect"
    )
    pl_fine = pl_y - gaussian_filter(pl_y, sigma=float(fine_sigma), mode="reflect")
    detail = np.clip(
        pl_mid * float(mid_weight) + pl_fine * float(fine_weight),
        -float(detail_limit),
        float(detail_limit),
    )
    if dark_lift_strength > 0:
        dark = sigmoid01((0.52 - base_y) / 0.10)
        lift = np.clip(np.sqrt(np.maximum(base_y, 0.0)) - base_y, 0.0, float(dark_lift_limit))
        detail += lift * dark * float(dark_lift_strength)

    blend = np.clip(mask * float(blend_strength), 0.0, 1.0)
    out_y = np.clip(base_y + detail * blend, 0.0, 1.0)
    chroma = base_d / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)

    base_peak = np.max(base_rgb, axis=2)
    hdr = smoothstep01((base_peak - float(hdr_peak_threshold)) / max(float(hdr_transition), 1.0e-6))
    out = out * (1.0 - hdr[..., None]) + base_rgb * hdr[..., None]
    stats = {
        "mid_weight": float(mid_weight),
        "fine_weight": float(fine_weight),
        "low_sigma": float(low_sigma),
        "mid_sigma": float(mid_sigma),
        "fine_sigma": float(fine_sigma),
        "detail_limit": float(detail_limit),
        "dark_lift_strength": float(dark_lift_strength),
        "dark_lift_limit": float(dark_lift_limit),
        "blend_strength": float(blend_strength),
        "blend_mean": float(np.mean(blend)),
        "detail_abs_p95": float(np.quantile(np.abs(detail * blend), 0.95)),
        "detail_abs_p99": float(np.quantile(np.abs(detail * blend), 0.99)),
        "hdr_restore_mean": float(np.mean(hdr)),
        **mask_stats,
    }
    return out.astype(np.float32, copy=False), mask.astype(np.float32, copy=False), stats


def crop(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, min(w - size, int(x) - size // 2))
    y0 = max(0, min(h - size, int(y) - size // 2))
    return image[y0 : y0 + size, x0 : x0 + size]


def render_compare(path: Path, panels: list[tuple[str, np.ndarray]], *, scale: int) -> None:
    previews = []
    for _, image in panels:
        preview = Image.fromarray(make_preview(image, exposure=1.0, tone="reinhard"))
        if scale != 1:
            preview = preview.resize((preview.width * scale, preview.height * scale), Image.Resampling.NEAREST)
        previews.append(preview)
    width, height = previews[0].size
    label_h = 26
    canvas = Image.new("RGB", (width * len(previews), height + label_h), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    for i, ((label, _), preview) in enumerate(zip(panels, previews, strict=True)):
        canvas.paste(preview, (i * width, label_h))
        draw.text((i * width + 6, 6), label, fill=(235, 235, 235))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer PL luma detail into hair regions as an upper-bound probe.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--pl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--mid-weight", type=float, default=0.75)
    parser.add_argument("--fine-weight", type=float, default=0.35)
    parser.add_argument("--low-sigma", type=float, default=8.0)
    parser.add_argument("--mid-sigma", type=float, default=1.1)
    parser.add_argument("--fine-sigma", type=float, default=0.75)
    parser.add_argument("--detail-limit", type=float, default=0.070)
    parser.add_argument("--dark-lift-strength", type=float, default=0.25)
    parser.add_argument("--dark-lift-limit", type=float, default=0.055)
    parser.add_argument("--blend-strength", type=float, default=0.95)
    parser.add_argument("--dark-low", type=float, default=0.020)
    parser.add_argument("--dark-high", type=float, default=0.62)
    parser.add_argument("--dark-transition", type=float, default=0.10)
    parser.add_argument("--skin-inhibit", type=float, default=0.95)
    parser.add_argument("--skin-proximity-sigma", type=float, default=130.0)
    parser.add_argument("--skin-proximity-threshold", type=float, default=0.004)
    parser.add_argument("--skin-proximity-transition", type=float, default=0.020)
    parser.add_argument("--texture-threshold", type=float, default=0.004)
    parser.add_argument("--texture-transition", type=float, default=0.012)
    parser.add_argument("--coherence-threshold", type=float, default=0.30)
    parser.add_argument("--coherence-transition", type=float, default=0.18)
    parser.add_argument("--coherence-energy-threshold", type=float, default=0.006)
    parser.add_argument("--coherence-energy-transition", type=float, default=0.006)
    parser.add_argument("--mask-blur-sigma", type=float, default=2.8)
    parser.add_argument("--hdr-peak-threshold", type=float, default=0.82)
    parser.add_argument("--hdr-transition", type=float, default=0.25)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--crop-scale", type=int, default=2)
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    pl_path = Path(args.pl).expanduser()
    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_pl_luma_detail"

    reference = read_image(reference_path)
    base = read_image(base_path)
    pl = read_image(pl_path)
    out, mask, stats = transfer_pl_luma_detail(
        reference,
        base,
        pl,
        mid_weight=args.mid_weight,
        fine_weight=args.fine_weight,
        low_sigma=args.low_sigma,
        mid_sigma=args.mid_sigma,
        fine_sigma=args.fine_sigma,
        detail_limit=args.detail_limit,
        dark_lift_strength=args.dark_lift_strength,
        dark_lift_limit=args.dark_lift_limit,
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
        mask_blur_sigma=args.mask_blur_sigma,
        hdr_peak_threshold=args.hdr_peak_threshold,
        hdr_transition=args.hdr_transition,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    mask_path = out_dir / f"{name}_mask.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(mask_path)

    panels = [("reference", reference), ("base", base), ("pl_probe", out), ("PL_XD", pl)]
    for roi_name, (x, y) in {
        "bangs": (2030, 1510),
        "top_hair": (2420, 1040),
        "face_hair": (2120, 1260),
    }.items():
        render_compare(
            crop_dir / f"{name}_{roi_name}_compare.png",
            [(label, crop(image, x, y, args.crop_size)) for label, image in panels],
            scale=args.crop_scale,
        )

    meta = {
        "reference": str(reference_path),
        "base": str(base_path),
        "pl": str(pl_path),
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "mask": str(mask_path),
        },
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


if __name__ == "__main__":
    main()
