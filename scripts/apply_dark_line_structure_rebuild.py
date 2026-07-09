"""Rebuild dark coherent luma structure on top of v10 outputs.

This targets the Occi hair failure more directly than generic luma grafting:
borrow smoothed luma structure from the noisy reference only in dark, line-like
regions, while keeping the v10 chroma cleanup intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, uniform_filter

from apply_dark_coherent_hair_detail_rescue import coherent_line_gate
from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_guided_luma_smoother import guided_filter_gray
from apply_luma_tail_speckle_filter import sigmoid01
from build_frequency_split_pseudo_teacher import RUN_ROOT, SCENES
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


V10_DIR = RUN_ROOT / "signed_chroma_outlier_v10_adaptive_detail_blend"


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


def rebuild_dark_line_structure(
    reference: np.ndarray,
    base: np.ndarray,
    *,
    strength: float,
    guide_sigma: float,
    guided_radius: int,
    guided_eps: float,
    correction_limit: float,
    max_detail_frac: float,
    dark_low: float,
    dark_high: float,
    saturation_high: float,
    coherence_threshold: float,
    energy_threshold: float,
    contrast_threshold: float,
    contrast_transition: float,
    gate_blur_sigma: float,
    band_only: bool,
    band_sigma: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    ref = display(reference)
    base_d = display(base)
    ref_y = luma(ref, LUMA_SRGB)
    base_y = luma(base_d, LUMA_SRGB)

    guide = gaussian_filter(ref_y, sigma=float(guide_sigma), mode="reflect")
    ref_structure = guided_filter_gray(guide, ref_y, radius=int(guided_radius), eps=float(guided_eps))
    ref_structure = gaussian_filter(ref_structure, sigma=0.15, mode="reflect")

    base_line_gate, base_line_stats = coherent_line_gate(
        base_y,
        coherence_threshold=coherence_threshold,
        energy_threshold=energy_threshold,
    )
    ref_line_gate, ref_line_stats = coherent_line_gate(
        ref_structure,
        coherence_threshold=max(0.30, coherence_threshold - 0.05),
        energy_threshold=max(0.0015, energy_threshold * 0.75),
    )
    local_mean = uniform_filter(ref_structure, size=11, mode="reflect")
    contrast = np.abs(ref_structure - local_mean)
    contrast_gate = sigmoid01((contrast - float(contrast_threshold)) / max(float(contrast_transition), 1.0e-6))

    sat = saturation(base_d)
    dark_gate = sigmoid01((base_y - float(dark_low)) / 0.030) * sigmoid01((float(dark_high) - base_y) / 0.070)
    sat_gate = sigmoid01((float(saturation_high) - sat) / 0.10)
    line_gate = np.maximum(base_line_gate, 0.72 * ref_line_gate)
    gate = np.clip(line_gate * contrast_gate * dark_gate * sat_gate, 0.0, 1.0).astype(np.float32, copy=False)
    if gate_blur_sigma > 0:
        gate = gaussian_filter(gate, sigma=float(gate_blur_sigma), mode="reflect")

    if band_only:
        ref_low = gaussian_filter(ref_structure, sigma=float(band_sigma), mode="reflect")
        base_low = gaussian_filter(base_y, sigma=float(band_sigma), mode="reflect")
        ref_signal = ref_structure - ref_low
        base_signal = base_y - base_low
        restore_gate = sigmoid01((np.abs(ref_signal) - np.abs(base_signal) * 1.02) / 0.004)
        correction_source = ref_signal - base_signal
        gate = np.clip(gate * restore_gate, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        restore_gate = np.ones_like(gate, dtype=np.float32)
        correction_source = ref_structure - base_y
    limit = np.minimum(float(correction_limit), np.maximum(base_y, 0.0) * float(max_detail_frac))
    correction = np.clip(correction_source, -limit, limit) * gate * float(strength)
    correction -= gaussian_filter(correction, sigma=18.0, mode="reflect") * 0.25
    out_y = np.clip(base_y + correction, 0.0, 1.0)

    base_chroma = base_d / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(base_chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    stats = {
        "strength": float(strength),
        "guide_sigma": float(guide_sigma),
        "guided_radius": int(guided_radius),
        "guided_eps": float(guided_eps),
        "correction_limit": float(correction_limit),
        "max_detail_frac": float(max_detail_frac),
        "dark_low": float(dark_low),
        "dark_high": float(dark_high),
        "saturation_high": float(saturation_high),
        "coherence_threshold": float(coherence_threshold),
        "energy_threshold": float(energy_threshold),
        "contrast_threshold": float(contrast_threshold),
        "band_only": bool(band_only),
        "band_sigma": float(band_sigma),
        "base_line_gate_mean": float(np.mean(base_line_gate)),
        "ref_line_gate_mean": float(np.mean(ref_line_gate)),
        "base_line_energy_p99": base_line_stats["line_energy_p99"],
        "ref_line_energy_p99": ref_line_stats["line_energy_p99"],
        "contrast_gate_mean": float(np.mean(contrast_gate)),
        "restore_gate_mean": float(np.mean(restore_gate)),
        "dark_gate_mean": float(np.mean(dark_gate)),
        "sat_gate_mean": float(np.mean(sat_gate)),
        "gate_mean": float(np.mean(gate)),
        "gate_p95": float(np.quantile(gate, 0.95)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "correction_abs_mean": float(np.mean(np.abs(correction))),
        "correction_abs_p95": float(np.quantile(np.abs(correction), 0.95)),
        "correction_abs_p99": float(np.quantile(np.abs(correction), 0.99)),
        "correction_abs_max": float(np.max(np.abs(correction))),
    }
    return out, stats, gate, ref_structure.astype(np.float32, copy=False)


def v10_path(scene_name: str) -> Path:
    return V10_DIR / f"{scene_name}_signed_chroma_outlier_v10_adaptive_detail_blend.exr"


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
    parser = argparse.ArgumentParser(description="Apply dark-line luma structure rebuild to v10 outputs.")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "signed_chroma_outlier_v16_dark_line_structure_rebuild"))
    parser.add_argument("--tag", default="v16_dark_line_structure_rebuild")
    parser.add_argument("--strength", type=float, default=0.52)
    parser.add_argument("--guide-sigma", type=float, default=0.65)
    parser.add_argument("--guided-radius", type=int, default=2)
    parser.add_argument("--guided-eps", type=float, default=0.00045)
    parser.add_argument("--correction-limit", type=float, default=0.075)
    parser.add_argument("--max-detail-frac", type=float, default=0.18)
    parser.add_argument("--dark-low", type=float, default=0.035)
    parser.add_argument("--dark-high", type=float, default=0.50)
    parser.add_argument("--saturation-high", type=float, default=0.52)
    parser.add_argument("--coherence-threshold", type=float, default=0.36)
    parser.add_argument("--energy-threshold", type=float, default=0.0022)
    parser.add_argument("--contrast-threshold", type=float, default=0.0040)
    parser.add_argument("--contrast-transition", type=float, default=0.0060)
    parser.add_argument("--gate-blur-sigma", type=float, default=0.35)
    parser.add_argument("--band-only", action="store_true")
    parser.add_argument("--band-sigma", type=float, default=3.2)
    parser.add_argument("--crop-size", type=int, default=768)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for scene_name, scene in SCENES.items():
        reference = read_image(scene.noisy)
        base = read_image(v10_path(scene_name))
        out, stats, gate, ref_structure = rebuild_dark_line_structure(
            reference,
            base,
            strength=args.strength,
            guide_sigma=args.guide_sigma,
            guided_radius=args.guided_radius,
            guided_eps=args.guided_eps,
            correction_limit=args.correction_limit,
            max_detail_frac=args.max_detail_frac,
            dark_low=args.dark_low,
            dark_high=args.dark_high,
            saturation_high=args.saturation_high,
            coherence_threshold=args.coherence_threshold,
            energy_threshold=args.energy_threshold,
            contrast_threshold=args.contrast_threshold,
            contrast_transition=args.contrast_transition,
            gate_blur_sigma=args.gate_blur_sigma,
            band_only=args.band_only,
            band_sigma=args.band_sigma,
        )
        stem = f"{scene_name}_{args.tag}"
        exr_path = out_dir / f"{stem}.exr"
        tiff_path = out_dir / f"{stem}.tiff"
        preview_path = out_dir / f"{stem}_preview.png"
        gate_path = out_dir / f"{stem}_gate.png"
        structure_path = out_dir / f"{stem}_reference_structure.png"
        write_exr(exr_path, out)
        write_tiff(tiff_path, out)
        Image.fromarray(make_preview(out)).save(preview_path)
        Image.fromarray((np.clip(gate, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(gate_path)
        Image.fromarray((np.clip(ref_structure, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(structure_path)
        for roi_name, x, y in scene.rois:
            render_compare(
                crop_dir / f"{scene_name}_{roi_name}_{args.tag}_compare.png",
                [
                    ("noisy", crop(reference, x, y, args.crop_size)),
                    ("v10", crop(base, x, y, args.crop_size)),
                    (args.tag, crop(out, x, y, args.crop_size)),
                ],
            )
        report[scene_name] = {
            "reference": str(scene.noisy),
            "base": str(v10_path(scene_name)),
            "output": str(exr_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
            "reference_structure": str(structure_path),
            "filter": stats,
            "output_stats": image_stats(out),
        }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
