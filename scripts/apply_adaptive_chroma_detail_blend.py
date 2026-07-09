"""Blend strong and detail-safe chroma dot cleanup outputs by texture.

The strong signed chroma outlier filter removes more chroma dots, while the
density-inhibited variant preserves a little more edge energy. This script
keeps the strong output in flat regions and blends toward the detail-safe output
only on coherent/texture regions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from apply_flat_chroma_smoother import linear_to_srgb_np
from apply_region_aware_luma_cleanup import make_coherent_structure_mask, make_texture_mask
from build_frequency_split_pseudo_teacher import RUN_ROOT, SCENES
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import make_preview, read_image


STRONG_DIR = RUN_ROOT / "signed_chroma_outlier_scan_v5_plsafe/full_outputs"
DETAIL_SAFE_DIR = RUN_ROOT / "signed_chroma_outlier_scan_v9_density035_full"


def display_linear(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.clip(linear_to_srgb_np(np.clip(x[..., :3], 0.0, None)), 0.0, 1.0).astype(np.float32, copy=False)


def make_detail_protect_mask(
    base_linear: np.ndarray,
    *,
    texture_weight: float,
    coherent_weight: float,
    blur_sigma: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    base_display = display_linear(base_linear)
    texture = make_texture_mask(base_display, texture_threshold=0.010, texture_transition=0.014)
    coherent = make_coherent_structure_mask(
        base_display,
        coherence_threshold=0.40,
        coherence_transition=0.16,
        energy_threshold=0.0045,
        energy_transition=0.0038,
    )
    protect = np.maximum(texture * float(texture_weight), coherent * float(coherent_weight))
    if blur_sigma > 0:
        protect = gaussian_filter(protect.astype(np.float32, copy=False), sigma=float(blur_sigma), mode="reflect")
    protect = np.clip(protect, 0.0, 1.0).astype(np.float32, copy=False)
    stats = {
        "texture_weight": float(texture_weight),
        "coherent_weight": float(coherent_weight),
        "protect_mean": float(np.mean(protect)),
        "protect_p90": float(np.quantile(protect, 0.90)),
        "protect_p99": float(np.quantile(protect, 0.99)),
        "texture_mean": float(np.mean(texture)),
        "coherent_mean": float(np.mean(coherent)),
    }
    return protect, stats, {"texture": texture, "coherent": coherent}


def strong_path(scene_name: str) -> Path:
    return STRONG_DIR / f"{scene_name}_signed_chroma_outlier_v5_plsafe.exr"


def detail_safe_path(scene_name: str) -> Path:
    return DETAIL_SAFE_DIR / f"{scene_name}_signed_chroma_outlier_v9_density035.exr"


def crop(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, min(w - size, int(x) - size // 2))
    y0 = max(0, min(h - size, int(y) - size // 2))
    return image[y0 : y0 + size, x0 : x0 + size]


def render_compare(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    previews = [Image.fromarray(make_preview(image)) for _, image in panels]
    w, h = previews[0].size
    label_h = 24
    canvas = Image.new("RGB", (w * len(previews), h + label_h), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    for i, ((label, _), preview) in enumerate(zip(panels, previews)):
        canvas.paste(preview, (i * w, label_h))
        draw.text((i * w + 8, 5), label, fill=(235, 235, 235))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend strong/detail-safe chroma cleanup by texture.")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "signed_chroma_outlier_v10_adaptive_detail_blend"))
    parser.add_argument("--texture-weight", type=float, default=0.75)
    parser.add_argument("--coherent-weight", type=float, default=0.65)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--crop-size", type=int, default=768)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for scene_name, scene in SCENES.items():
        base = read_image(scene.base)
        strong = read_image(strong_path(scene_name))
        detail_safe = read_image(detail_safe_path(scene_name))
        protect, stats, masks = make_detail_protect_mask(
            base,
            texture_weight=args.texture_weight,
            coherent_weight=args.coherent_weight,
            blur_sigma=args.blur_sigma,
        )
        out = strong * (1.0 - protect[..., None]) + detail_safe * protect[..., None]
        stem = f"{scene_name}_signed_chroma_outlier_v10_adaptive_detail_blend"
        exr_path = out_dir / f"{stem}.exr"
        tiff_path = out_dir / f"{stem}.tiff"
        preview_path = out_dir / f"{stem}_preview.png"
        protect_path = out_dir / f"{stem}_protect.png"
        texture_path = out_dir / f"{stem}_texture.png"
        coherent_path = out_dir / f"{stem}_coherent.png"
        write_exr(exr_path, out)
        write_tiff(tiff_path, out)
        Image.fromarray(make_preview(out)).save(preview_path)
        Image.fromarray((protect * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(protect_path)
        Image.fromarray((masks["texture"] * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(texture_path)
        Image.fromarray((masks["coherent"] * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(coherent_path)
        for roi_name, x, y in scene.rois:
            render_compare(
                crop_dir / f"{scene_name}_{roi_name}_adaptive_detail_blend_compare.png",
                [
                    ("base", crop(base, x, y, args.crop_size)),
                    ("v5 strong", crop(strong, x, y, args.crop_size)),
                    ("density035", crop(detail_safe, x, y, args.crop_size)),
                    ("adaptive", crop(out, x, y, args.crop_size)),
                ],
            )
        report[scene_name] = {
            "base": str(scene.base),
            "strong": str(strong_path(scene_name)),
            "detail_safe": str(detail_safe_path(scene_name)),
            "output": str(exr_path),
            "preview": str(preview_path),
            "protect": str(protect_path),
            "stats": stats,
        }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
