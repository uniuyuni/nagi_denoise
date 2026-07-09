"""Apply a conservative luma detail restore on top of v10 chroma cleanup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from apply_perceptual_luma_detail_restore import apply_perceptual_luma_detail_restore
from build_frequency_split_pseudo_teacher import RUN_ROOT, SCENES
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


V10_DIR = RUN_ROOT / "signed_chroma_outlier_v10_adaptive_detail_blend"


def v10_path(scene_name: str) -> Path:
    return V10_DIR / f"{scene_name}_signed_chroma_outlier_v10_adaptive_detail_blend.exr"


def crop(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, min(width - size, int(x) - size // 2))
    y0 = max(0, min(height - size, int(y) - size // 2))
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
    parser = argparse.ArgumentParser(description="Apply conservative luma detail restore to v10 outputs.")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "signed_chroma_outlier_v12_luma_detail_restore"))
    parser.add_argument("--tag", default="v12_luma_detail_restore")
    parser.add_argument("--strength", type=float, default=0.18)
    parser.add_argument("--energy-threshold", type=float, default=0.013)
    parser.add_argument("--coherence-threshold", type=float, default=0.42)
    parser.add_argument("--correction-limit", type=float, default=0.012)
    parser.add_argument("--max-detail-frac", type=float, default=0.035)
    parser.add_argument("--base-energy-threshold", type=float, default=0.0)
    parser.add_argument("--base-energy-transition", type=float, default=0.006)
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
        out, stats, gate = apply_perceptual_luma_detail_restore(
            reference,
            base,
            strength=args.strength,
            detail_sigma=1.0,
            coherence_sigma=1.2,
            coherence_threshold=args.coherence_threshold,
            coherence_transition=0.16,
            energy_sigma=1.6,
            energy_threshold=args.energy_threshold,
            energy_transition=0.006,
            base_detail_saturation=0.72,
            max_detail_frac=args.max_detail_frac,
            min_detail_limit=0.003,
            correction_limit=args.correction_limit,
            zero_mean_sigma=8.0,
            base_energy_threshold=args.base_energy_threshold,
            base_energy_transition=args.base_energy_transition,
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
                crop_dir / f"{scene_name}_{roi_name}_v12_luma_detail_restore_compare.png",
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
