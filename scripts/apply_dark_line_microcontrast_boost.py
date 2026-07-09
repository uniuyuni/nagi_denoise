"""Boost existing dark coherent hair/detail microcontrast on v10 outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from apply_dark_coherent_hair_detail_rescue import coherent_line_gate
from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from apply_dark_line_texture_floor import oklab_edge_gate
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


def boost_dark_line_microcontrast(
    base: np.ndarray,
    *,
    strength: float,
    fine_strength: float,
    band_sigma: float,
    fine_sigma: float,
    correction_limit: float,
    max_detail_frac: float,
    dark_low: float,
    dark_high: float,
    saturation_high: float,
    coherence_threshold: float,
    energy_threshold: float,
    gate_blur_sigma: float,
    use_oklab_edge: bool,
    oklab_edge_sigma: float,
    oklab_edge_threshold: float,
    oklab_edge_transition: float,
    oklab_chroma_weight: float,
    oklab_mix: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    base_d = display(base)
    base_y = luma(base_d, LUMA_SRGB)
    sat = saturation(base_d)

    line_gate, line_stats = coherent_line_gate(
        base_y,
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
        line_gate = np.maximum(line_gate, np.clip(oklab_gate * float(oklab_mix), 0.0, 1.0))
    else:
        oklab_stats = {
            "oklab_edge_mean": 0.0,
            "oklab_edge_p95": 0.0,
            "oklab_edge_p99": 0.0,
            "oklab_edge_gate_mean": 0.0,
        }
    dark_gate = sigmoid01((base_y - float(dark_low)) / 0.030) * sigmoid01((float(dark_high) - base_y) / 0.070)
    sat_gate = sigmoid01((float(saturation_high) - sat) / 0.10)
    gate = np.clip(line_gate * dark_gate * sat_gate, 0.0, 1.0).astype(np.float32, copy=False)
    if gate_blur_sigma > 0:
        gate = gaussian_filter(gate, sigma=float(gate_blur_sigma), mode="reflect")

    low = gaussian_filter(base_y, sigma=float(band_sigma), mode="reflect")
    medium = base_y - low
    fine = base_y - gaussian_filter(base_y, sigma=float(fine_sigma), mode="reflect")
    source = medium * float(strength) + fine * float(fine_strength)
    limit = np.minimum(float(correction_limit), np.maximum(base_y, 0.0) * float(max_detail_frac))
    correction = np.clip(source, -limit, limit) * gate
    out_y = np.clip(base_y + correction, 0.0, 1.0)

    base_chroma = base_d / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(base_chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    stats = {
        "strength": float(strength),
        "fine_strength": float(fine_strength),
        "band_sigma": float(band_sigma),
        "fine_sigma": float(fine_sigma),
        "correction_limit": float(correction_limit),
        "max_detail_frac": float(max_detail_frac),
        "dark_low": float(dark_low),
        "dark_high": float(dark_high),
        "saturation_high": float(saturation_high),
        "coherence_threshold": float(coherence_threshold),
        "energy_threshold": float(energy_threshold),
        "use_oklab_edge": bool(use_oklab_edge),
        "oklab_edge_sigma": float(oklab_edge_sigma),
        "oklab_edge_threshold": float(oklab_edge_threshold),
        "oklab_edge_transition": float(oklab_edge_transition),
        "oklab_chroma_weight": float(oklab_chroma_weight),
        "oklab_mix": float(oklab_mix),
        **oklab_stats,
        "line_energy_p99": line_stats["line_energy_p99"],
        "line_gate_mean": line_stats["line_gate_mean"],
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
    parser = argparse.ArgumentParser(description="Apply dark-line microcontrast boost to v10 outputs.")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "signed_chroma_outlier_v18_dark_line_microcontrast"))
    parser.add_argument("--tag", default="v18_dark_line_microcontrast")
    parser.add_argument("--strength", type=float, default=0.34)
    parser.add_argument("--fine-strength", type=float, default=0.12)
    parser.add_argument("--band-sigma", type=float, default=3.2)
    parser.add_argument("--fine-sigma", type=float, default=0.85)
    parser.add_argument("--correction-limit", type=float, default=0.035)
    parser.add_argument("--max-detail-frac", type=float, default=0.12)
    parser.add_argument("--dark-low", type=float, default=0.030)
    parser.add_argument("--dark-high", type=float, default=0.56)
    parser.add_argument("--saturation-high", type=float, default=0.60)
    parser.add_argument("--coherence-threshold", type=float, default=0.30)
    parser.add_argument("--energy-threshold", type=float, default=0.0015)
    parser.add_argument("--gate-blur-sigma", type=float, default=0.45)
    parser.add_argument("--use-oklab-edge", action="store_true")
    parser.add_argument("--oklab-edge-sigma", type=float, default=0.85)
    parser.add_argument("--oklab-edge-threshold", type=float, default=0.006)
    parser.add_argument("--oklab-edge-transition", type=float, default=0.010)
    parser.add_argument("--oklab-chroma-weight", type=float, default=0.75)
    parser.add_argument("--oklab-mix", type=float, default=0.75)
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
        out, stats, gate = boost_dark_line_microcontrast(
            base,
            strength=args.strength,
            fine_strength=args.fine_strength,
            band_sigma=args.band_sigma,
            fine_sigma=args.fine_sigma,
            correction_limit=args.correction_limit,
            max_detail_frac=args.max_detail_frac,
            dark_low=args.dark_low,
            dark_high=args.dark_high,
            saturation_high=args.saturation_high,
            coherence_threshold=args.coherence_threshold,
            energy_threshold=args.energy_threshold,
            gate_blur_sigma=args.gate_blur_sigma,
            use_oklab_edge=args.use_oklab_edge,
            oklab_edge_sigma=args.oklab_edge_sigma,
            oklab_edge_threshold=args.oklab_edge_threshold,
            oklab_edge_transition=args.oklab_edge_transition,
            oklab_chroma_weight=args.oklab_chroma_weight,
            oklab_mix=args.oklab_mix,
        )
        stem = f"{scene_name}_{args.tag}"
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
            "filter": stats,
            "output_stats": image_stats(out),
        }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
