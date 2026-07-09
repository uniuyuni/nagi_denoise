"""Synthesize subtle luma strand texture in flat hair regions.

When the source image is already flat, luma grafting cannot restore real
detail. This probe generates plausible, orientation-aligned luma texture inside
a restricted hair region. It is intentionally luma-only: chroma and HDR peaks
stay from the base image.
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
from apply_oriented_hair_luma_rebuild import oriented_line_average, shadow_lift_luma
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


def smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def ellipse_mask(shape: tuple[int, int], specs: list[str], *, feather: float) -> np.ndarray:
    h, w = shape
    if not specs:
        return np.ones((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    f = max(float(feather), 1.0e-6)
    for spec in specs:
        parts = [float(p) for p in spec.split(",")]
        if len(parts) != 4:
            raise ValueError(f"ellipse must be x,y,rx,ry: {spec!r}")
        x, y, rx, ry = parts
        d = np.sqrt(((xx - x) / max(rx, 1.0e-6)) ** 2 + ((yy - y) / max(ry, 1.0e-6)) ** 2)
        out = np.maximum(out, smoothstep01((1.0 + f - d) / f))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def local_normalize(x: np.ndarray, *, sigma: float) -> np.ndarray:
    mean = gaussian_filter(x, sigma=float(sigma), mode="reflect")
    var = gaussian_filter((x - mean) ** 2, sigma=float(sigma), mode="reflect")
    return ((x - mean) / np.sqrt(np.maximum(var, 1.0e-6))).astype(np.float32, copy=False)


def synthesize_hair_texture(
    reference: np.ndarray,
    base: np.ndarray,
    orientation_guide: np.ndarray,
    *,
    seed: int,
    ellipses: list[str],
    ellipse_feather: float,
    gamma: float,
    lift_floor: float,
    dark_low: float,
    dark_high: float,
    dark_transition: float,
    low_detail_sigma: float,
    low_detail_threshold: float,
    low_detail_transition: float,
    coherence_threshold: float,
    coherence_transition: float,
    skin_inhibit: float,
    skin_blur_sigma: float,
    line_length: int,
    directions: int,
    orientation_sigma: float,
    orientation_sharpness: float,
    noise_smooth_sigma: float,
    sparse_density: float,
    sparse_positive_fraction: float,
    normalize_sigma: float,
    strand_strength: float,
    positive_bias: float,
    texture_limit: float,
    gate_blur_sigma: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    ref_d = display(reference)
    base_rgb = np.clip(_safe_rgb(base), 0.0, None)
    base_d = display(base_rgb)
    guide_d = display(orientation_guide)
    ref_y = luma(ref_d, LUMA_SRGB)
    base_y = luma(base_d, LUMA_SRGB)
    guide_y = luma(guide_d, LUMA_SRGB)
    lifted = shadow_lift_luma(np.maximum(ref_y, guide_y), gamma=gamma, floor=lift_floor)

    _, coherence, orient_stats = oriented_line_average(
        lifted,
        lifted,
        length=line_length,
        directions=directions,
        orientation_sigma=orientation_sigma,
        sharpness=orientation_sharpness,
    )
    rng = np.random.default_rng(int(seed))
    if sparse_density > 0:
        active = rng.random(size=base_y.shape) < float(sparse_density)
        signs = np.where(rng.random(size=base_y.shape) < float(sparse_positive_fraction), 1.0, -1.0)
        amplitudes = rng.uniform(0.35, 1.0, size=base_y.shape)
        noise = (active * signs * amplitudes).astype(np.float32)
    else:
        noise = rng.normal(0.0, 1.0, size=base_y.shape).astype(np.float32)
    if noise_smooth_sigma > 0:
        noise = gaussian_filter(noise, sigma=float(noise_smooth_sigma), mode="reflect")
    line_noise, _, _ = oriented_line_average(
        noise,
        lifted,
        length=line_length,
        directions=directions,
        orientation_sigma=orientation_sigma,
        sharpness=orientation_sharpness,
    )
    texture = local_normalize(line_noise, sigma=normalize_sigma)
    texture = texture * float(strand_strength) + np.maximum(texture, 0.0) * float(positive_bias)
    texture = np.clip(texture, -float(texture_limit), float(texture_limit))

    base_hp = np.abs(base_y - gaussian_filter(base_y, sigma=float(low_detail_sigma), mode="reflect"))
    low_detail = sigmoid01(
        (float(low_detail_threshold) - base_hp) / max(float(low_detail_transition), 1.0e-6)
    )
    dark = sigmoid01((base_y - float(dark_low)) / 0.030) * sigmoid01(
        (float(dark_high) - base_y) / max(float(dark_transition), 1.0e-6)
    )
    coherent = sigmoid01(
        (coherence - float(coherence_threshold)) / max(float(coherence_transition), 1.0e-6)
    )
    skin = make_skin_mask(base_d, blur_sigma=float(skin_blur_sigma))
    not_skin = 1.0 - np.clip(skin * float(skin_inhibit), 0.0, 1.0)
    roi = ellipse_mask(base_y.shape, ellipses, feather=ellipse_feather)
    gate = np.clip(dark * low_detail * coherent * not_skin * roi, 0.0, 1.0).astype(np.float32, copy=False)
    if gate_blur_sigma > 0:
        gate = gaussian_filter(gate, sigma=float(gate_blur_sigma), mode="reflect")
        gate = np.clip(gate, 0.0, 1.0)

    out_y = np.clip(base_y + texture * gate, 0.0, 1.0)
    chroma = base_d / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    peak = np.max(base_rgb, axis=2)
    hdr = smoothstep01((peak - float(hdr_peak_threshold)) / max(float(hdr_transition), 1.0e-6))
    out = out * (1.0 - hdr[..., None]) + base_rgb * hdr[..., None]

    stats = {
        "seed": int(seed),
        "strand_strength": float(strand_strength),
        "positive_bias": float(positive_bias),
        "texture_limit": float(texture_limit),
        "sparse_density": float(sparse_density),
        "sparse_positive_fraction": float(sparse_positive_fraction),
        "gate_mean": float(np.mean(gate)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "texture_abs_p95": float(np.quantile(np.abs(texture * gate), 0.95)),
        "texture_abs_p99": float(np.quantile(np.abs(texture * gate), 0.99)),
        "low_detail_mean": float(np.mean(low_detail)),
        "dark_mean": float(np.mean(dark)),
        "coherent_gate_mean": float(np.mean(coherent)),
        "skin_mean": float(np.mean(skin)),
        "not_skin_mean": float(np.mean(not_skin)),
        "roi_mean": float(np.mean(roi)),
        "hdr_restore_mean": float(np.mean(hdr)),
        **orient_stats,
    }
    return out.astype(np.float32, copy=False), gate, texture * gate, stats


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
    parser = argparse.ArgumentParser(description="Synthesize plausible luma hair strand texture.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--orientation-guide", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=7519)
    parser.add_argument("--ellipse", action="append", default=[], help="Soft ROI: x,y,rx,ry. Repeatable.")
    parser.add_argument("--ellipse-feather", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.50)
    parser.add_argument("--lift-floor", type=float, default=0.020)
    parser.add_argument("--dark-low", type=float, default=0.020)
    parser.add_argument("--dark-high", type=float, default=0.58)
    parser.add_argument("--dark-transition", type=float, default=0.10)
    parser.add_argument("--low-detail-sigma", type=float, default=1.6)
    parser.add_argument("--low-detail-threshold", type=float, default=0.020)
    parser.add_argument("--low-detail-transition", type=float, default=0.012)
    parser.add_argument("--coherence-threshold", type=float, default=0.34)
    parser.add_argument("--coherence-transition", type=float, default=0.18)
    parser.add_argument("--skin-inhibit", type=float, default=0.95)
    parser.add_argument("--skin-blur-sigma", type=float, default=1.4)
    parser.add_argument("--line-length", type=int, default=33)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--orientation-sigma", type=float, default=1.1)
    parser.add_argument("--orientation-sharpness", type=float, default=3.0)
    parser.add_argument("--noise-smooth-sigma", type=float, default=0.55)
    parser.add_argument("--sparse-density", type=float, default=0.0)
    parser.add_argument("--sparse-positive-fraction", type=float, default=0.70)
    parser.add_argument("--normalize-sigma", type=float, default=9.0)
    parser.add_argument("--strand-strength", type=float, default=0.010)
    parser.add_argument("--positive-bias", type=float, default=0.004)
    parser.add_argument("--texture-limit", type=float, default=0.030)
    parser.add_argument("--gate-blur-sigma", type=float, default=1.0)
    parser.add_argument("--hdr-peak-threshold", type=float, default=0.82)
    parser.add_argument("--hdr-transition", type=float, default=0.25)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--crop-scale", type=int, default=2)
    parser.add_argument("--compare", action="append", default=[], help="LABEL=PATH panels for crop comparison.")
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    guide_path = Path(args.orientation_guide).expanduser() if args.orientation_guide else base_path
    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_synthetic_hair_texture"

    reference = read_image(reference_path)
    base = read_image(base_path)
    guide = read_image(guide_path)
    out, gate, texture, stats = synthesize_hair_texture(
        reference,
        base,
        guide,
        seed=args.seed,
        ellipses=args.ellipse,
        ellipse_feather=args.ellipse_feather,
        gamma=args.gamma,
        lift_floor=args.lift_floor,
        dark_low=args.dark_low,
        dark_high=args.dark_high,
        dark_transition=args.dark_transition,
        low_detail_sigma=args.low_detail_sigma,
        low_detail_threshold=args.low_detail_threshold,
        low_detail_transition=args.low_detail_transition,
        coherence_threshold=args.coherence_threshold,
        coherence_transition=args.coherence_transition,
        skin_inhibit=args.skin_inhibit,
        skin_blur_sigma=args.skin_blur_sigma,
        line_length=args.line_length,
        directions=args.directions,
        orientation_sigma=args.orientation_sigma,
        orientation_sharpness=args.orientation_sharpness,
        noise_smooth_sigma=args.noise_smooth_sigma,
        sparse_density=args.sparse_density,
        sparse_positive_fraction=args.sparse_positive_fraction,
        normalize_sigma=args.normalize_sigma,
        strand_strength=args.strand_strength,
        positive_bias=args.positive_bias,
        texture_limit=args.texture_limit,
        gate_blur_sigma=args.gate_blur_sigma,
        hdr_peak_threshold=args.hdr_peak_threshold,
        hdr_transition=args.hdr_transition,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    texture_path = out_dir / f"{name}_texture.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    Image.fromarray(np.clip(texture / max(args.texture_limit, 1.0e-6) * 127.0 + 128.0, 0, 255).astype(np.uint8)).save(
        texture_path
    )

    compare_panels: list[tuple[str, np.ndarray]] = [("reference", reference), ("base", base), ("synth", out)]
    for item in args.compare:
        label, path = item.split("=", 1)
        p = Path(path).expanduser()
        if p.exists():
            compare_panels.append((label, read_image(p)))
    for roi_name, (x, y) in {
        "bangs": (2030, 1510),
        "top_hair": (2420, 1040),
        "face_hair": (2120, 1260),
    }.items():
        render_compare(
            crop_dir / f"{name}_{roi_name}_compare.png",
            [(label, crop(image, x, y, args.crop_size)) for label, image in compare_panels],
            scale=args.crop_scale,
        )

    meta = {
        "reference": str(reference_path),
        "base": str(base_path),
        "orientation_guide": str(guide_path),
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
            "texture": str(texture_path),
        },
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


if __name__ == "__main__":
    main()
