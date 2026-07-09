"""Add a guarded luma texture floor to dark line-like regions.

This is intended for the "plastic / over-denoised flat patch" failure after
v18. It keeps chroma from the denoised base and borrows only clipped, local
zero-mean luma texture from the original reference where the base has become
too texture-poor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve, gaussian_filter

from apply_dark_coherent_hair_detail_rescue import coherent_line_gate
from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from build_frequency_split_pseudo_teacher import RUN_ROOT, SCENES
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


BASE_DIR = RUN_ROOT / "signed_chroma_outlier_v18_dark_line_microcontrast"


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def display(image: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb_np(np.clip(_safe_rgb(image), 0.0, None)), 0.0, 1.0).astype(np.float32, copy=False)


def saturation(rgb: np.ndarray) -> np.ndarray:
    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    return ((mx - mn) / np.maximum(mx, 1.0e-6)).astype(np.float32, copy=False)


def shadow_lift_luma(luma_plane: np.ndarray, *, gamma: float, floor: float) -> np.ndarray:
    """Brighten shadows for feature detection without changing final exposure."""
    y = np.clip(np.asarray(luma_plane, dtype=np.float32), 0.0, 1.0)
    f = max(float(floor), 0.0)
    g = max(float(gamma), 1.0e-3)
    lo = f**g
    hi = (1.0 + f) ** g
    return np.clip(((y + f) ** g - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def oklab_from_display_srgb(rgb: np.ndarray) -> np.ndarray:
    linear = np.clip(srgb_to_linear_np(np.clip(rgb, 0.0, 1.0)), 0.0, 1.0)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    lms_l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    lms_m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    lms_s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = np.cbrt(np.maximum(lms_l, 0.0))
    m_ = np.cbrt(np.maximum(lms_m, 0.0))
    s_ = np.cbrt(np.maximum(lms_s, 0.0))
    lab_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    lab_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    lab_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([lab_l, lab_a, lab_b], axis=2).astype(np.float32, copy=False)


def oklab_edge_gate(
    rgb: np.ndarray,
    *,
    sigma: float,
    edge_threshold: float,
    edge_transition: float,
    chroma_weight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    lab = oklab_from_display_srgb(rgb)
    edge2 = np.zeros(lab.shape[:2], dtype=np.float32)
    weights = (1.0, float(chroma_weight), float(chroma_weight))
    for channel, weight in enumerate(weights):
        plane = gaussian_filter(lab[..., channel], sigma=float(sigma), mode="reflect")
        gy, gx = np.gradient(plane)
        edge2 += float(weight) * (gx * gx + gy * gy)
    edge = np.sqrt(np.maximum(edge2, 0.0)).astype(np.float32, copy=False)
    gate = sigmoid01((edge - float(edge_threshold)) / max(float(edge_transition), 1.0e-6))
    stats = {
        "oklab_edge_mean": float(np.mean(edge)),
        "oklab_edge_p95": float(np.quantile(edge, 0.95)),
        "oklab_edge_p99": float(np.quantile(edge, 0.99)),
        "oklab_edge_gate_mean": float(np.mean(gate)),
    }
    return gate.astype(np.float32, copy=False), stats


def directional_line_filter(source: np.ndarray, base_y: np.ndarray, *, length: int, sharpness: float) -> tuple[np.ndarray, dict[str, float]]:
    size = int(length) | 1
    kernels = []
    for direction in ("h", "v", "d1", "d2"):
        kernel = np.zeros((size, size), dtype=np.float32)
        c = size // 2
        if direction == "h":
            kernel[c, :] = 1.0
        elif direction == "v":
            kernel[:, c] = 1.0
        elif direction == "d1":
            np.fill_diagonal(kernel, 1.0)
        else:
            np.fill_diagonal(np.fliplr(kernel), 1.0)
        kernel /= np.sum(kernel)
        kernels.append(kernel)

    structure = gaussian_filter(base_y, sigma=0.8, mode="reflect")
    gy, gx = np.gradient(structure)
    line_angle = np.arctan2(gy, gx) + np.pi * 0.5
    thetas = np.array([0.0, np.pi * 0.5, np.pi * 0.25, np.pi * 0.75], dtype=np.float32)
    weights = []
    for theta in thetas:
        # Lines are pi-periodic, so use cos(2*dtheta).
        w = np.maximum(0.0, np.cos(2.0 * (line_angle - float(theta))))
        weights.append(np.power(w, float(sharpness)).astype(np.float32, copy=False))
    weight_stack = np.stack(weights, axis=2)
    filtered_stack = np.stack([convolve(source, kernel, mode="reflect") for kernel in kernels], axis=2)
    denom = np.maximum(np.sum(weight_stack, axis=2), 1.0e-6)
    out = np.sum(filtered_stack * weight_stack, axis=2) / denom
    stats = {
        "oriented_weight_mean": float(np.mean(denom)),
        "oriented_weight_p95": float(np.quantile(denom, 0.95)),
        "oriented_source_abs_p95": float(np.quantile(np.abs(out), 0.95)),
    }
    return out.astype(np.float32, copy=False), stats


def texture_floor(
    reference: np.ndarray,
    base: np.ndarray,
    *,
    strength: float,
    fine_sigma: float,
    coarse_sigma: float,
    amp_sigma: float,
    amp_ratio: float,
    min_missing_amp: float,
    correction_limit: float,
    max_detail_frac: float,
    dark_low: float,
    dark_high: float,
    saturation_high: float,
    coherence_threshold: float,
    energy_threshold: float,
    gate_blur_sigma: float,
    density_sigma: float,
    density_threshold: float,
    density_transition: float,
    always_floor: bool,
    floor_amp_limit: float,
    use_oklab_edge: bool,
    oklab_edge_sigma: float,
    oklab_edge_threshold: float,
    oklab_edge_transition: float,
    oklab_chroma_weight: float,
    oklab_mix: float,
    texture_soft_threshold: float,
    texture_soft_knee: float,
    oriented_texture: bool,
    oriented_length: int,
    oriented_sharpness: float,
    shadow_lift_gamma: float,
    shadow_lift_floor: float,
    shadow_lift_mix: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    ref = display(reference)
    base_d = display(base)
    ref_y = luma(ref, LUMA_SRGB)
    base_y = luma(base_d, LUMA_SRGB)
    sat = saturation(base_d)
    lift_mix = np.clip(float(shadow_lift_mix), 0.0, 1.0)
    if lift_mix > 0.0 and abs(float(shadow_lift_gamma) - 1.0) > 1.0e-4:
        ref_lift_y = shadow_lift_luma(ref_y, gamma=shadow_lift_gamma, floor=shadow_lift_floor)
        base_lift_y = shadow_lift_luma(base_y, gamma=shadow_lift_gamma, floor=shadow_lift_floor)
        ref_detail_y = ref_y * (1.0 - lift_mix) + ref_lift_y * lift_mix
        base_detail_y = base_y * (1.0 - lift_mix) + base_lift_y * lift_mix
    else:
        ref_detail_y = ref_y
        base_detail_y = base_y

    base_line_gate, line_stats = coherent_line_gate(
        base_detail_y,
        coherence_threshold=coherence_threshold,
        energy_threshold=energy_threshold,
    )
    if use_oklab_edge:
        oklab_gate, oklab_stats = oklab_edge_gate(
            base_d,
            sigma=oklab_edge_sigma,
            edge_threshold=oklab_edge_threshold,
            edge_transition=oklab_edge_transition,
            chroma_weight=oklab_chroma_weight,
        )
        base_line_gate = np.maximum(base_line_gate, np.clip(oklab_gate * float(oklab_mix), 0.0, 1.0))
    else:
        oklab_stats = {
            "oklab_edge_mean": 0.0,
            "oklab_edge_p95": 0.0,
            "oklab_edge_p99": 0.0,
            "oklab_edge_gate_mean": 0.0,
        }
    ref_band = gaussian_filter(ref_detail_y, sigma=float(fine_sigma), mode="reflect") - gaussian_filter(
        ref_detail_y, sigma=float(coarse_sigma), mode="reflect"
    )
    base_band = gaussian_filter(base_detail_y, sigma=float(fine_sigma), mode="reflect") - gaussian_filter(
        base_detail_y, sigma=float(coarse_sigma), mode="reflect"
    )
    ref_amp = gaussian_filter(np.abs(ref_band), sigma=float(amp_sigma), mode="reflect")
    base_amp = gaussian_filter(np.abs(base_band), sigma=float(amp_sigma), mode="reflect")
    missing_amp = np.maximum(0.0, ref_amp * float(amp_ratio) - base_amp)
    if always_floor:
        missing_gate = np.ones_like(missing_amp, dtype=np.float32)
    else:
        missing_gate = sigmoid01((missing_amp - float(min_missing_amp)) / max(float(min_missing_amp), 0.002))

    dark_gate = sigmoid01((base_y - float(dark_low)) / 0.030) * sigmoid01((float(dark_high) - base_y) / 0.070)
    sat_gate = sigmoid01((float(saturation_high) - sat) / 0.10)
    line_density = gaussian_filter(base_line_gate, sigma=float(density_sigma), mode="reflect")
    density_gate = sigmoid01(
        (line_density - float(density_threshold)) / max(float(density_transition), 1.0e-6)
    )
    gate = np.clip(base_line_gate * density_gate * missing_gate * dark_gate * sat_gate, 0.0, 1.0).astype(
        np.float32, copy=False
    )
    if gate_blur_sigma > 0:
        gate = gaussian_filter(gate, sigma=float(gate_blur_sigma), mode="reflect")

    source = ref_band - gaussian_filter(ref_band, sigma=10.0, mode="reflect") * 0.35
    if texture_soft_threshold > 0:
        abs_source = np.abs(source)
        if texture_soft_knee > 0:
            shrink = sigmoid01((abs_source - float(texture_soft_threshold)) / float(texture_soft_knee))
            source = np.sign(source) * np.maximum(0.0, abs_source - float(texture_soft_threshold) * (1.0 - shrink))
        else:
            source = np.sign(source) * np.maximum(0.0, abs_source - float(texture_soft_threshold))
    if oriented_texture:
        source, oriented_stats = directional_line_filter(
            source,
            base_detail_y,
            length=oriented_length,
            sharpness=oriented_sharpness,
        )
    else:
        oriented_stats = {
            "oriented_weight_mean": 0.0,
            "oriented_weight_p95": 0.0,
            "oriented_source_abs_p95": 0.0,
        }
    local_limit = np.minimum(float(correction_limit), np.maximum(base_y, 0.0) * float(max_detail_frac))
    if always_floor:
        amp_limit = np.full_like(missing_amp, float(floor_amp_limit), dtype=np.float32)
    else:
        amp_limit = np.maximum(missing_amp * 0.85, 0.0015)
    limit = np.minimum(local_limit, amp_limit)
    correction = np.clip(source, -limit, limit) * gate * float(strength)
    out_y = np.clip(base_y + correction, 0.0, 1.0)

    base_chroma = base_d / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(base_chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    stats = {
        "strength": float(strength),
        "fine_sigma": float(fine_sigma),
        "coarse_sigma": float(coarse_sigma),
        "amp_sigma": float(amp_sigma),
        "amp_ratio": float(amp_ratio),
        "min_missing_amp": float(min_missing_amp),
        "correction_limit": float(correction_limit),
        "max_detail_frac": float(max_detail_frac),
        "dark_low": float(dark_low),
        "dark_high": float(dark_high),
        "saturation_high": float(saturation_high),
        "coherence_threshold": float(coherence_threshold),
        "energy_threshold": float(energy_threshold),
        "density_sigma": float(density_sigma),
        "density_threshold": float(density_threshold),
        "density_transition": float(density_transition),
        "always_floor": bool(always_floor),
        "floor_amp_limit": float(floor_amp_limit),
        "use_oklab_edge": bool(use_oklab_edge),
        "oklab_edge_sigma": float(oklab_edge_sigma),
        "oklab_edge_threshold": float(oklab_edge_threshold),
        "oklab_edge_transition": float(oklab_edge_transition),
        "oklab_chroma_weight": float(oklab_chroma_weight),
        "oklab_mix": float(oklab_mix),
        "texture_soft_threshold": float(texture_soft_threshold),
        "texture_soft_knee": float(texture_soft_knee),
        "oriented_texture": bool(oriented_texture),
        "oriented_length": int(oriented_length),
        "oriented_sharpness": float(oriented_sharpness),
        "shadow_lift_gamma": float(shadow_lift_gamma),
        "shadow_lift_floor": float(shadow_lift_floor),
        "shadow_lift_mix": float(shadow_lift_mix),
        **oriented_stats,
        **oklab_stats,
        "line_energy_p99": line_stats["line_energy_p99"],
        "line_gate_mean": line_stats["line_gate_mean"],
        "density_gate_mean": float(np.mean(density_gate)),
        "missing_gate_mean": float(np.mean(missing_gate)),
        "dark_gate_mean": float(np.mean(dark_gate)),
        "sat_gate_mean": float(np.mean(sat_gate)),
        "gate_mean": float(np.mean(gate)),
        "gate_p95": float(np.quantile(gate, 0.95)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "missing_amp_p95": float(np.quantile(missing_amp, 0.95)),
        "correction_abs_mean": float(np.mean(np.abs(correction))),
        "correction_abs_p95": float(np.quantile(np.abs(correction), 0.95)),
        "correction_abs_p99": float(np.quantile(np.abs(correction), 0.99)),
        "correction_abs_max": float(np.max(np.abs(correction))),
    }
    return out, stats, gate, correction.astype(np.float32, copy=False)


def base_path(base_dir: Path, scene_name: str, tag: str) -> Path:
    return base_dir / f"{scene_name}_{tag}.exr"


def crop(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, min(w - size, int(x) - size // 2))
    y0 = max(0, min(h - size, int(y) - size // 2))
    return image[y0 : y0 + size, x0 : x0 + size]


def render_compare(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    previews = [Image.fromarray(make_preview(image)) for _, image in panels]
    width, height = previews[0].size
    label_h = 24
    canvas = Image.new("RGB", (width * len(previews), height + label_h), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    for i, ((label, _), preview) in enumerate(zip(panels, previews)):
        canvas.paste(preview, (i * width, label_h))
        draw.text((i * width + 8, 5), label, fill=(235, 235, 235))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply dark-line luma texture floor on top of v18 outputs.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "signed_chroma_outlier_v19_dark_line_texture_floor"))
    parser.add_argument("--base-tag", default="v18_dark_line_microcontrast")
    parser.add_argument("--tag", default="v19_dark_line_texture_floor")
    parser.add_argument("--strength", type=float, default=0.34)
    parser.add_argument("--fine-sigma", type=float, default=0.75)
    parser.add_argument("--coarse-sigma", type=float, default=2.6)
    parser.add_argument("--amp-sigma", type=float, default=2.0)
    parser.add_argument("--amp-ratio", type=float, default=0.52)
    parser.add_argument("--min-missing-amp", type=float, default=0.0035)
    parser.add_argument("--correction-limit", type=float, default=0.018)
    parser.add_argument("--max-detail-frac", type=float, default=0.055)
    parser.add_argument("--dark-low", type=float, default=0.030)
    parser.add_argument("--dark-high", type=float, default=0.55)
    parser.add_argument("--saturation-high", type=float, default=0.60)
    parser.add_argument("--coherence-threshold", type=float, default=0.30)
    parser.add_argument("--energy-threshold", type=float, default=0.0015)
    parser.add_argument("--gate-blur-sigma", type=float, default=0.35)
    parser.add_argument("--density-sigma", type=float, default=8.0)
    parser.add_argument("--density-threshold", type=float, default=0.18)
    parser.add_argument("--density-transition", type=float, default=0.10)
    parser.add_argument("--always-floor", action="store_true")
    parser.add_argument("--floor-amp-limit", type=float, default=0.0035)
    parser.add_argument("--use-oklab-edge", action="store_true")
    parser.add_argument("--oklab-edge-sigma", type=float, default=0.85)
    parser.add_argument("--oklab-edge-threshold", type=float, default=0.006)
    parser.add_argument("--oklab-edge-transition", type=float, default=0.010)
    parser.add_argument("--oklab-chroma-weight", type=float, default=0.75)
    parser.add_argument("--oklab-mix", type=float, default=0.75)
    parser.add_argument("--texture-soft-threshold", type=float, default=0.0)
    parser.add_argument("--texture-soft-knee", type=float, default=0.002)
    parser.add_argument("--oriented-texture", action="store_true")
    parser.add_argument("--oriented-length", type=int, default=9)
    parser.add_argument("--oriented-sharpness", type=float, default=2.0)
    parser.add_argument("--shadow-lift-gamma", type=float, default=1.0)
    parser.add_argument("--shadow-lift-floor", type=float, default=0.010)
    parser.add_argument("--shadow-lift-mix", type=float, default=0.0)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--scene", action="append", default=[], help="Scene key to process. Repeatable.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    scenes = {name: SCENES[name] for name in args.scene} if args.scene else SCENES
    for scene_name, scene in scenes.items():
        reference = read_image(scene.noisy)
        base = read_image(base_path(base_dir, scene_name, args.base_tag))
        out, stats, gate, correction = texture_floor(
            reference,
            base,
            strength=args.strength,
            fine_sigma=args.fine_sigma,
            coarse_sigma=args.coarse_sigma,
            amp_sigma=args.amp_sigma,
            amp_ratio=args.amp_ratio,
            min_missing_amp=args.min_missing_amp,
            correction_limit=args.correction_limit,
            max_detail_frac=args.max_detail_frac,
            dark_low=args.dark_low,
            dark_high=args.dark_high,
            saturation_high=args.saturation_high,
            coherence_threshold=args.coherence_threshold,
            energy_threshold=args.energy_threshold,
            gate_blur_sigma=args.gate_blur_sigma,
            density_sigma=args.density_sigma,
            density_threshold=args.density_threshold,
            density_transition=args.density_transition,
            always_floor=args.always_floor,
            floor_amp_limit=args.floor_amp_limit,
            use_oklab_edge=args.use_oklab_edge,
            oklab_edge_sigma=args.oklab_edge_sigma,
            oklab_edge_threshold=args.oklab_edge_threshold,
            oklab_edge_transition=args.oklab_edge_transition,
            oklab_chroma_weight=args.oklab_chroma_weight,
            oklab_mix=args.oklab_mix,
            texture_soft_threshold=args.texture_soft_threshold,
            texture_soft_knee=args.texture_soft_knee,
            oriented_texture=args.oriented_texture,
            oriented_length=args.oriented_length,
            oriented_sharpness=args.oriented_sharpness,
            shadow_lift_gamma=args.shadow_lift_gamma,
            shadow_lift_floor=args.shadow_lift_floor,
            shadow_lift_mix=args.shadow_lift_mix,
        )
        stem = f"{scene_name}_{args.tag}"
        exr_path = out_dir / f"{stem}.exr"
        tiff_path = out_dir / f"{stem}.tiff"
        preview_path = out_dir / f"{stem}_preview.png"
        gate_path = out_dir / f"{stem}_gate.png"
        correction_path = out_dir / f"{stem}_correction.png"
        write_exr(exr_path, out)
        write_tiff(tiff_path, out)
        Image.fromarray(make_preview(out)).save(preview_path)
        Image.fromarray((np.clip(gate, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(gate_path)
        corr_preview = np.clip(correction / max(args.correction_limit, 1.0e-6) * 0.5 + 0.5, 0.0, 1.0)
        Image.fromarray((corr_preview * 255.0 + 0.5).astype(np.uint8)).save(correction_path)
        for roi_name, x, y in scene.rois:
            render_compare(
                crop_dir / f"{scene_name}_{roi_name}_{args.tag}_compare.png",
                [
                    ("noisy", crop(reference, x, y, args.crop_size)),
                    ("v18", crop(base, x, y, args.crop_size)),
                    (args.tag, crop(out, x, y, args.crop_size)),
                ],
            )
        report[scene_name] = {
            "reference": str(scene.noisy),
            "base": str(base_path(base_dir, scene_name, args.base_tag)),
            "output": str(exr_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
            "correction": str(correction_path),
            "filter": stats,
            "output_stats": image_stats(out),
        }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
