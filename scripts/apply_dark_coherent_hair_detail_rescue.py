"""Rescue dark coherent hair/detail texture on top of v10/v13 outputs.

This is a targeted experiment for the Occi hair failure: restore a little more
signed luma detail only where the denoised base already contains dark coherent
line texture. It avoids broad flat regions and keeps chroma from the base.
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


def coherent_line_gate(y: np.ndarray, *, coherence_threshold: float, energy_threshold: float) -> tuple[np.ndarray, dict[str, float]]:
    structure = gaussian_filter(y, sigma=0.75, mode="reflect")
    gy, gx = np.gradient(structure)
    jxx = gaussian_filter(gx * gx, sigma=1.8, mode="reflect")
    jyy = gaussian_filter(gy * gy, sigma=1.8, mode="reflect")
    jxy = gaussian_filter(gx * gy, sigma=1.8, mode="reflect")
    energy = jxx + jyy
    coherence = np.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy) / np.maximum(energy, 1.0e-8)
    gate = sigmoid01((coherence - float(coherence_threshold)) / 0.14) * sigmoid01(
        (energy - float(energy_threshold)) / max(float(energy_threshold) * 0.75, 0.0018)
    )
    gate = gaussian_filter(np.clip(gate, 0.0, 1.0).astype(np.float32, copy=False), sigma=0.7, mode="reflect")
    stats = {
        "coherence_mean": float(np.mean(coherence)),
        "coherence_p95": float(np.quantile(coherence, 0.95)),
        "line_energy_p99": float(np.quantile(energy, 0.99)),
        "line_gate_mean": float(np.mean(gate)),
        "line_gate_p99": float(np.quantile(gate, 0.99)),
    }
    return gate.astype(np.float32, copy=False), stats


def apply_dark_coherent_hair_detail_rescue(
    reference: np.ndarray,
    base: np.ndarray,
    *,
    strength: float,
    detail_sigma: float,
    max_correction: float,
    max_detail_frac: float,
    dark_low: float,
    dark_high: float,
    saturation_high: float,
    coherence_threshold: float,
    energy_threshold: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    ref = display(reference)
    base_d = display(base)
    ref_y = luma(ref, LUMA_SRGB)
    base_y = luma(base_d, LUMA_SRGB)
    sat = saturation(base_d)

    ref_detail = ref_y - gaussian_filter(ref_y, sigma=float(detail_sigma), mode="reflect")
    base_detail = base_y - gaussian_filter(base_y, sigma=float(detail_sigma), mode="reflect")
    missing = ref_detail - base_detail

    line_gate, line_stats = coherent_line_gate(
        base_y,
        coherence_threshold=coherence_threshold,
        energy_threshold=energy_threshold,
    )
    dark_gate = sigmoid01((base_y - float(dark_low)) / 0.035) * sigmoid01((float(dark_high) - base_y) / 0.075)
    sat_gate = sigmoid01((float(saturation_high) - sat) / 0.10)
    ref_energy = gaussian_filter(np.abs(ref_detail), sigma=1.2, mode="reflect")
    base_energy = gaussian_filter(np.abs(base_detail), sigma=1.2, mode="reflect")
    missing_gate = sigmoid01((ref_energy - base_energy * 0.90) / 0.004)
    gate = np.clip(line_gate * dark_gate * sat_gate * missing_gate, 0.0, 1.0).astype(np.float32, copy=False)
    gate = gaussian_filter(gate, sigma=0.8, mode="reflect")

    limit = np.minimum(float(max_correction), np.maximum(base_y, 0.0) * float(max_detail_frac))
    correction = np.clip(missing, -limit, limit) * gate * float(strength)
    out_y = np.clip(base_y + correction, 0.0, 1.0)
    base_chroma = base_d / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(base_chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    stats = {
        **line_stats,
        "strength": float(strength),
        "detail_sigma": float(detail_sigma),
        "max_correction": float(max_correction),
        "max_detail_frac": float(max_detail_frac),
        "dark_low": float(dark_low),
        "dark_high": float(dark_high),
        "saturation_high": float(saturation_high),
        "dark_gate_mean": float(np.mean(dark_gate)),
        "sat_gate_mean": float(np.mean(sat_gate)),
        "missing_gate_mean": float(np.mean(missing_gate)),
        "gate_mean": float(np.mean(gate)),
        "gate_p95": float(np.quantile(gate, 0.95)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "correction_abs_mean": float(np.mean(np.abs(correction))),
        "correction_abs_p95": float(np.quantile(np.abs(correction), 0.95)),
        "correction_abs_p99": float(np.quantile(np.abs(correction), 0.99)),
    }
    return out, stats, gate


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
    parser = argparse.ArgumentParser(description="Apply dark coherent hair/detail rescue to v10 outputs.")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "signed_chroma_outlier_v14_dark_hair_detail_rescue"))
    parser.add_argument("--strength", type=float, default=0.34)
    parser.add_argument("--detail-sigma", type=float, default=0.85)
    parser.add_argument("--max-correction", type=float, default=0.014)
    parser.add_argument("--max-detail-frac", type=float, default=0.045)
    parser.add_argument("--dark-low", type=float, default=0.045)
    parser.add_argument("--dark-high", type=float, default=0.48)
    parser.add_argument("--saturation-high", type=float, default=0.50)
    parser.add_argument("--coherence-threshold", type=float, default=0.38)
    parser.add_argument("--energy-threshold", type=float, default=0.0028)
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
        out, stats, gate = apply_dark_coherent_hair_detail_rescue(
            reference,
            base,
            strength=args.strength,
            detail_sigma=args.detail_sigma,
            max_correction=args.max_correction,
            max_detail_frac=args.max_detail_frac,
            dark_low=args.dark_low,
            dark_high=args.dark_high,
            saturation_high=args.saturation_high,
            coherence_threshold=args.coherence_threshold,
            energy_threshold=args.energy_threshold,
        )
        stem = f"{scene_name}_v14_dark_hair_detail_rescue"
        exr_path = out_dir / f"{stem}.exr"
        tiff_path = out_dir / f"{stem}.tiff"
        preview_path = out_dir / f"{stem}_preview.png"
        gate_path = out_dir / f"{stem}_gate.png"
        write_exr(exr_path, out)
        write_tiff(tiff_path, out)
        Image.fromarray(make_preview(out)).save(preview_path)
        Image.fromarray((np.clip(gate, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(gate_path)
        for roi_name, x, y in scene.rois:
            render_compare(
                crop_dir / f"{scene_name}_{roi_name}_v14_dark_hair_detail_rescue_compare.png",
                [
                    ("noisy", crop(reference, x, y, args.crop_size)),
                    ("v10", crop(base, x, y, args.crop_size)),
                    ("v14 hair", crop(out, x, y, args.crop_size)),
                ],
            )
        report[scene_name] = {
            "reference": str(scene.noisy),
            "base": str(v10_path(scene_name)),
            "output": str(exr_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
            "filter": stats,
            "output_stats": image_stats(out),
        }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
