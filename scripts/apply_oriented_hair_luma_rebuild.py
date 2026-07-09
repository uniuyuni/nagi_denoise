"""Rebuild dark hair luma with orientation-aware reference structure.

This is a targeted probe for the Occi hair failure. It estimates local strand
orientation from a shadow-lifted noisy reference, denoises luma along that
orientation, then blends only luma into the denoised base under a hair-like
mask. Chroma and HDR highlights are left to the base/HDR-restore steps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve, gaussian_filter

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from apply_region_aware_luma_cleanup import make_skin_mask
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


def shadow_lift_luma(y: np.ndarray, *, gamma: float, floor: float) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=np.float32), 0.0, 1.0)
    f = max(float(floor), 0.0)
    g = max(float(gamma), 1.0e-3)
    lo = f**g
    hi = (1.0 + f) ** g
    return np.clip(((y + f) ** g - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def line_kernel(length: int, theta: float) -> np.ndarray:
    size = int(length) | 1
    c = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    dx = float(np.cos(theta))
    dy = float(np.sin(theta))
    for t in range(-c, c + 1):
        x = int(round(c + t * dx))
        y = int(round(c + t * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    kernel /= max(float(np.sum(kernel)), 1.0)
    return kernel


def oriented_line_average(
    source: np.ndarray,
    orientation_source: np.ndarray,
    *,
    length: int,
    directions: int,
    orientation_sigma: float,
    sharpness: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    structure = gaussian_filter(orientation_source, sigma=float(orientation_sigma), mode="reflect")
    gy, gx = np.gradient(structure)
    line_angle = np.arctan2(gy, gx) + np.pi * 0.5

    jxx = gaussian_filter(gx * gx, sigma=1.4, mode="reflect")
    jyy = gaussian_filter(gy * gy, sigma=1.4, mode="reflect")
    jxy = gaussian_filter(gx * gy, sigma=1.4, mode="reflect")
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / np.maximum(jxx + jyy, 1.0e-7)

    thetas = np.linspace(0.0, np.pi, int(directions), endpoint=False, dtype=np.float32)
    filtered = []
    weights = []
    for theta in thetas:
        filtered.append(convolve(source, line_kernel(length, float(theta)), mode="reflect"))
        w = np.maximum(0.0, np.cos(2.0 * (line_angle - float(theta))))
        weights.append(np.power(w, float(sharpness)).astype(np.float32, copy=False))
    stack = np.stack(filtered, axis=2)
    weight_stack = np.stack(weights, axis=2)
    denom = np.maximum(np.sum(weight_stack, axis=2), 1.0e-6)
    out = np.sum(stack * weight_stack, axis=2) / denom
    stats = {
        "coherence_mean": float(np.mean(coherence)),
        "coherence_p90": float(np.quantile(coherence, 0.90)),
        "coherence_p99": float(np.quantile(coherence, 0.99)),
        "orientation_weight_mean": float(np.mean(denom)),
    }
    return out.astype(np.float32, copy=False), coherence.astype(np.float32, copy=False), stats


def rebuild_hair_luma(
    reference: np.ndarray,
    base: np.ndarray,
    *,
    strength: float,
    line_length: int,
    directions: int,
    orientation_sigma: float,
    orientation_sharpness: float,
    gamma: float,
    lift_floor: float,
    dark_low: float,
    dark_high: float,
    coherence_threshold: float,
    coherence_transition: float,
    detail_threshold: float,
    detail_transition: float,
    skin_inhibit: float,
    gate_blur_sigma: float,
    correction_limit: float,
    max_detail_frac: float,
    dodge_strength: float,
    dodge_gamma: float,
    dodge_limit: float,
    lifted_contrast_boost: float,
    lifted_contrast_sigma: float,
    lifted_contrast_limit: float,
    mid_boost: float,
    mid_limit: float,
    fine_boost: float,
    fine_limit: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    ref = display(reference)
    base_d = display(base)
    ref_y = luma(ref, LUMA_SRGB)
    base_y = luma(base_d, LUMA_SRGB)
    lifted = shadow_lift_luma(ref_y, gamma=gamma, floor=lift_floor)
    oriented_y, coherence, orient_stats = oriented_line_average(
        ref_y,
        lifted,
        length=line_length,
        directions=directions,
        orientation_sigma=orientation_sigma,
        sharpness=orientation_sharpness,
    )
    oriented_lifted_y, _, _ = oriented_line_average(
        lifted,
        lifted,
        length=line_length,
        directions=directions,
        orientation_sigma=orientation_sigma,
        sharpness=orientation_sharpness,
    )

    lifted_band = np.abs(lifted - gaussian_filter(lifted, sigma=2.0, mode="reflect"))
    detail_gate = sigmoid01((lifted_band - float(detail_threshold)) / max(float(detail_transition), 1.0e-6))
    coherent_gate = sigmoid01(
        (coherence - float(coherence_threshold)) / max(float(coherence_transition), 1.0e-6)
    )
    dark_gate = sigmoid01((base_y - float(dark_low)) / 0.035) * sigmoid01((float(dark_high) - base_y) / 0.080)
    skin = make_skin_mask(base_d, blur_sigma=1.4)
    skin_gate = 1.0 - np.clip(skin * float(skin_inhibit), 0.0, 1.0)
    gate = np.clip(detail_gate * coherent_gate * dark_gate * skin_gate, 0.0, 1.0).astype(np.float32, copy=False)
    if gate_blur_sigma > 0:
        gate = gaussian_filter(gate, sigma=float(gate_blur_sigma), mode="reflect")
        gate = np.clip(gate, 0.0, 1.0)

    correction = oriented_y - base_y
    if dodge_strength > 0:
        dodge_y = shadow_lift_luma(base_y, gamma=dodge_gamma, floor=0.010)
        correction += np.clip(dodge_y - base_y, 0.0, float(dodge_limit)) * float(dodge_strength)
    if lifted_contrast_boost > 0:
        lifted_contrast = oriented_lifted_y - gaussian_filter(
            oriented_lifted_y, sigma=float(lifted_contrast_sigma), mode="reflect"
        )
        correction += (
            np.clip(lifted_contrast, -float(lifted_contrast_limit), float(lifted_contrast_limit))
            * float(lifted_contrast_boost)
        )
    if mid_boost > 0:
        ref_mid = gaussian_filter(oriented_y, sigma=0.8, mode="reflect") - gaussian_filter(
            oriented_y, sigma=3.0, mode="reflect"
        )
        base_mid = gaussian_filter(base_y, sigma=0.8, mode="reflect") - gaussian_filter(
            base_y, sigma=3.0, mode="reflect"
        )
        correction += np.clip(ref_mid - base_mid, -float(mid_limit), float(mid_limit)) * float(mid_boost)
    if fine_boost > 0:
        ref_fine = oriented_y - gaussian_filter(oriented_y, sigma=0.75, mode="reflect")
        base_fine = base_y - gaussian_filter(base_y, sigma=0.75, mode="reflect")
        correction += np.clip(ref_fine - base_fine, -float(fine_limit), float(fine_limit)) * float(fine_boost)

    local_limit = np.minimum(float(correction_limit), np.maximum(base_y, 0.0) * float(max_detail_frac))
    correction = np.clip(correction, -local_limit, local_limit) * gate * float(strength)
    out_y = np.clip(base_y + correction, 0.0, 1.0)
    chroma = base_d / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    stats = {
        "strength": float(strength),
        "line_length": int(line_length),
        "directions": int(directions),
        "orientation_sigma": float(orientation_sigma),
        "orientation_sharpness": float(orientation_sharpness),
        "gamma": float(gamma),
        "lift_floor": float(lift_floor),
        "dark_low": float(dark_low),
        "dark_high": float(dark_high),
        "coherence_threshold": float(coherence_threshold),
        "detail_threshold": float(detail_threshold),
        "skin_inhibit": float(skin_inhibit),
        "dodge_strength": float(dodge_strength),
        "dodge_gamma": float(dodge_gamma),
        "dodge_limit": float(dodge_limit),
        "lifted_contrast_boost": float(lifted_contrast_boost),
        "lifted_contrast_sigma": float(lifted_contrast_sigma),
        "lifted_contrast_limit": float(lifted_contrast_limit),
        "gate_mean": float(np.mean(gate)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "correction_abs_mean": float(np.mean(np.abs(correction))),
        "correction_abs_p95": float(np.quantile(np.abs(correction), 0.95)),
        "correction_abs_p99": float(np.quantile(np.abs(correction), 0.99)),
        **orient_stats,
    }
    return out, stats, gate, oriented_y


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
    parser = argparse.ArgumentParser(description="Apply orientation-aware dark hair luma rebuild.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--strength", type=float, default=0.72)
    parser.add_argument("--line-length", type=int, default=21)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--orientation-sigma", type=float, default=0.9)
    parser.add_argument("--orientation-sharpness", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=0.48)
    parser.add_argument("--lift-floor", type=float, default=0.020)
    parser.add_argument("--dark-low", type=float, default=0.020)
    parser.add_argument("--dark-high", type=float, default=0.58)
    parser.add_argument("--coherence-threshold", type=float, default=0.34)
    parser.add_argument("--coherence-transition", type=float, default=0.18)
    parser.add_argument("--detail-threshold", type=float, default=0.010)
    parser.add_argument("--detail-transition", type=float, default=0.012)
    parser.add_argument("--skin-inhibit", type=float, default=0.85)
    parser.add_argument("--gate-blur-sigma", type=float, default=0.45)
    parser.add_argument("--correction-limit", type=float, default=0.095)
    parser.add_argument("--max-detail-frac", type=float, default=0.42)
    parser.add_argument("--dodge-strength", type=float, default=0.0)
    parser.add_argument("--dodge-gamma", type=float, default=0.72)
    parser.add_argument("--dodge-limit", type=float, default=0.055)
    parser.add_argument("--lifted-contrast-boost", type=float, default=0.0)
    parser.add_argument("--lifted-contrast-sigma", type=float, default=4.0)
    parser.add_argument("--lifted-contrast-limit", type=float, default=0.045)
    parser.add_argument("--mid-boost", type=float, default=0.35)
    parser.add_argument("--mid-limit", type=float, default=0.070)
    parser.add_argument("--fine-boost", type=float, default=0.18)
    parser.add_argument("--fine-limit", type=float, default=0.035)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--crop-scale", type=int, default=2)
    parser.add_argument("--compare", action="append", default=[], help="LABEL=PATH panels for crop comparison.")
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_oriented_hair_luma"

    reference = read_image(reference_path)
    base = read_image(base_path)
    out, stats, gate, oriented_y = rebuild_hair_luma(
        reference,
        base,
        strength=args.strength,
        line_length=args.line_length,
        directions=args.directions,
        orientation_sigma=args.orientation_sigma,
        orientation_sharpness=args.orientation_sharpness,
        gamma=args.gamma,
        lift_floor=args.lift_floor,
        dark_low=args.dark_low,
        dark_high=args.dark_high,
        coherence_threshold=args.coherence_threshold,
        coherence_transition=args.coherence_transition,
        detail_threshold=args.detail_threshold,
        detail_transition=args.detail_transition,
        skin_inhibit=args.skin_inhibit,
        gate_blur_sigma=args.gate_blur_sigma,
        correction_limit=args.correction_limit,
        max_detail_frac=args.max_detail_frac,
        dodge_strength=args.dodge_strength,
        dodge_gamma=args.dodge_gamma,
        dodge_limit=args.dodge_limit,
        lifted_contrast_boost=args.lifted_contrast_boost,
        lifted_contrast_sigma=args.lifted_contrast_sigma,
        lifted_contrast_limit=args.lifted_contrast_limit,
        mid_boost=args.mid_boost,
        mid_limit=args.mid_limit,
        fine_boost=args.fine_boost,
        fine_limit=args.fine_limit,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    structure_path = out_dir / f"{name}_oriented_luma.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    Image.fromarray(np.clip(oriented_y * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(structure_path)

    compare_panels: list[tuple[str, np.ndarray]] = [("reference", reference), ("base", base), ("hair_rebuild", out)]
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
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
            "oriented_luma": str(structure_path),
        },
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


if __name__ == "__main__":
    main()
