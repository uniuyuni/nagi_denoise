"""Blend a strong luma rebuild only into hair-like regions.

The strong structure-graft candidates recover hair shape, but they also restore
too much texture in trees/background. This compositor keeps the HDR-safe base
everywhere, and borrows only display-luma from the detail candidate where a
simple hair mask agrees: dark, non-skin, near skin, and structured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from apply_region_aware_luma_cleanup import make_coherent_structure_mask, make_skin_mask, make_texture_mask
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


def hair_mask(
    reference: np.ndarray,
    base: np.ndarray,
    mask_guide: np.ndarray | None = None,
    *,
    dark_low: float,
    dark_high: float,
    dark_transition: float,
    skin_inhibit: float,
    skin_proximity_sigma: float,
    skin_proximity_threshold: float,
    skin_proximity_transition: float,
    texture_threshold: float,
    texture_transition: float,
    coherence_threshold: float,
    coherence_transition: float,
    coherence_energy_threshold: float,
    coherence_energy_transition: float,
    mask_guide_dark_high: float,
    mask_guide_transition: float,
    mask_guide_weight: float,
    mask_blur_sigma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    ref_d = display(reference)
    base_d = display(base)
    base_y = luma(base_d, LUMA_SRGB)
    dark = sigmoid01((base_y - float(dark_low)) / 0.030) * sigmoid01(
        (float(dark_high) - base_y) / max(float(dark_transition), 1.0e-6)
    )
    skin = make_skin_mask(base_d, blur_sigma=1.4)
    not_skin = 1.0 - np.clip(skin * float(skin_inhibit), 0.0, 1.0)
    skin_prox = gaussian_filter(skin, sigma=float(skin_proximity_sigma), mode="reflect")
    skin_prox_gate = smoothstep01(
        (skin_prox - float(skin_proximity_threshold)) / max(float(skin_proximity_transition), 1.0e-6)
    )
    texture = make_texture_mask(
        ref_d,
        texture_threshold=float(texture_threshold),
        texture_transition=float(texture_transition),
    )
    coherent = make_coherent_structure_mask(
        ref_d,
        coherence_threshold=float(coherence_threshold),
        coherence_transition=float(coherence_transition),
        energy_threshold=float(coherence_energy_threshold),
        energy_transition=float(coherence_energy_transition),
    )
    structure = np.clip(np.maximum(texture, coherent), 0.0, 1.0)
    mask = np.clip(dark * not_skin * skin_prox_gate * structure, 0.0, 1.0).astype(np.float32, copy=False)
    guide_gate_mean = 0.0
    if mask_guide is not None and mask_guide_weight > 0:
        guide_d = np.clip(_safe_rgb(mask_guide), 0.0, 1.0)
        guide_y = luma(guide_d, LUMA_SRGB)
        guide_gate = sigmoid01(
            (float(mask_guide_dark_high) - guide_y) / max(float(mask_guide_transition), 1.0e-6)
        )
        guide_gate_mean = float(np.mean(guide_gate))
        guide_mix = np.clip(float(mask_guide_weight), 0.0, 1.0)
        mask *= 1.0 - guide_mix + guide_gate * guide_mix
    if mask_blur_sigma > 0:
        mask = gaussian_filter(mask, sigma=float(mask_blur_sigma), mode="reflect")
        mask = np.clip(mask, 0.0, 1.0)
    stats = {
        "dark_mean": float(np.mean(dark)),
        "skin_mean": float(np.mean(skin)),
        "skin_proximity_mean": float(np.mean(skin_prox_gate)),
        "texture_mean": float(np.mean(texture)),
        "coherent_mean": float(np.mean(coherent)),
        "mask_guide_gate_mean": guide_gate_mean,
        "mask_mean": float(np.mean(mask)),
        "mask_p90": float(np.quantile(mask, 0.90)),
        "mask_p99": float(np.quantile(mask, 0.99)),
    }
    return mask, stats


def blend_hair_luma(
    reference: np.ndarray,
    base: np.ndarray,
    detail: np.ndarray,
    mask_guide: np.ndarray | None = None,
    *,
    blend_strength: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
    **mask_kwargs: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    base_rgb = np.clip(_safe_rgb(base), 0.0, None)
    detail_rgb = np.clip(_safe_rgb(detail), 0.0, None)
    if base_rgb.shape[:2] != detail_rgb.shape[:2]:
        raise ValueError(f"shape mismatch base={base_rgb.shape} detail={detail_rgb.shape}")
    mask, mask_stats = hair_mask(reference, base, mask_guide, **mask_kwargs)
    base_d = display(base_rgb)
    detail_d = display(detail_rgb)
    base_y = luma(base_d, LUMA_SRGB)
    detail_y = luma(detail_d, LUMA_SRGB)
    blend = np.clip(mask * float(blend_strength), 0.0, 1.0)
    out_y = base_y * (1.0 - blend) + detail_y * blend
    chroma = base_d / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(chroma * np.maximum(out_y[..., None], 0.0), 0.0, 1.0)
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
        "delta_abs_mean": float(np.mean(np.abs(out_y - base_y) * blend)),
        "delta_abs_p99": float(np.quantile(np.abs(out_y - base_y) * blend, 0.99)),
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
    parser = argparse.ArgumentParser(description="Blend detail candidate only into hair-like luma regions.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--mask-guide", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--blend-strength", type=float, default=0.85)
    parser.add_argument("--dark-low", type=float, default=0.020)
    parser.add_argument("--dark-high", type=float, default=0.58)
    parser.add_argument("--dark-transition", type=float, default=0.10)
    parser.add_argument("--skin-inhibit", type=float, default=0.95)
    parser.add_argument("--skin-proximity-sigma", type=float, default=90.0)
    parser.add_argument("--skin-proximity-threshold", type=float, default=0.006)
    parser.add_argument("--skin-proximity-transition", type=float, default=0.018)
    parser.add_argument("--texture-threshold", type=float, default=0.005)
    parser.add_argument("--texture-transition", type=float, default=0.012)
    parser.add_argument("--coherence-threshold", type=float, default=0.34)
    parser.add_argument("--coherence-transition", type=float, default=0.18)
    parser.add_argument("--coherence-energy-threshold", type=float, default=0.006)
    parser.add_argument("--coherence-energy-transition", type=float, default=0.006)
    parser.add_argument("--mask-guide-dark-high", type=float, default=0.46)
    parser.add_argument("--mask-guide-transition", type=float, default=0.10)
    parser.add_argument("--mask-guide-weight", type=float, default=0.0)
    parser.add_argument("--mask-blur-sigma", type=float, default=2.2)
    parser.add_argument("--hdr-peak-threshold", type=float, default=0.82)
    parser.add_argument("--hdr-transition", type=float, default=0.25)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--crop-scale", type=int, default=2)
    parser.add_argument("--compare", action="append", default=[], help="LABEL=PATH panels for crop comparison.")
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    detail_path = Path(args.detail).expanduser()
    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_hair_region_blend"

    reference = read_image(reference_path)
    base = read_image(base_path)
    detail = read_image(detail_path)
    mask_guide = read_image(Path(args.mask_guide).expanduser()) if args.mask_guide else None
    out, mask, stats = blend_hair_luma(
        reference,
        base,
        detail,
        mask_guide,
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
    mask_path = out_dir / f"{name}_mask.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(mask_path)

    compare_panels: list[tuple[str, np.ndarray]] = [("reference", reference), ("base", base), ("detail", detail), ("blend", out)]
    for item in args.compare:
        label, path = item.split("=", 1)
        p = Path(path).expanduser()
        if p.exists():
            compare_panels.append((label, read_image(p)))
    rois = {
        "bangs": (2030, 1510),
        "top_hair": (2420, 1040),
        "face_hair": (2120, 1260),
    }
    for roi_name, (x, y) in rois.items():
        render_compare(
            crop_dir / f"{name}_{roi_name}_compare.png",
            [(label, crop(image, x, y, args.crop_size)) for label, image in compare_panels],
            scale=args.crop_scale,
        )

    meta = {
        "reference": str(reference_path),
        "base": str(base_path),
        "detail": str(detail_path),
        "mask_guide": str(Path(args.mask_guide).expanduser()) if args.mask_guide else None,
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
