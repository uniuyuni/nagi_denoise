"""Build an oracle blend between strong and detail-safe chroma cleanup outputs.

This is not a deployable final filter because it uses the PL-safe pseudo teacher
at selection time. Its purpose is to test whether a learned selector is worth
training: if the teacher-oracle blend cannot beat the hand adaptive blend, a
small learned selector will not have enough useful signal either.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma
from build_frequency_split_pseudo_teacher import RUN_ROOT, SCENES
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import make_preview, read_image


STRONG_DIR = RUN_ROOT / "signed_chroma_outlier_scan_v5_plsafe/full_outputs"
DETAIL_SAFE_DIR = RUN_ROOT / "signed_chroma_outlier_scan_v9_density035_full"
TEACHER_DIR = RUN_ROOT / "frequency_split_pseudo_teacher_v3_chroma_safe"


def display(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.clip(linear_to_srgb_np(np.clip(x[..., :3], 0.0, None)), 0.0, 1.0).astype(np.float32, copy=False)


def chroma(display_rgb: np.ndarray) -> np.ndarray:
    y = luma(display_rgb, LUMA_SRGB)
    return display_rgb - y[..., None]


def local_loss(candidate: np.ndarray, teacher: np.ndarray, *, chroma_weight: float, luma_weight: float, sigma: float) -> np.ndarray:
    cand_display = display(candidate)
    teacher_display = display(teacher)
    cand_y = luma(cand_display, LUMA_SRGB)
    teacher_y = luma(teacher_display, LUMA_SRGB)
    chroma_loss = np.mean(np.abs(chroma(cand_display) - chroma(teacher_display)), axis=2)
    luma_loss = np.abs(cand_y - teacher_y)
    loss = chroma_loss * float(chroma_weight) + luma_loss * float(luma_weight)
    return gaussian_filter(loss.astype(np.float32, copy=False), sigma=float(sigma), mode="reflect")


def sigmoid01(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32, copy=False)


def strong_path(scene_name: str) -> Path:
    return STRONG_DIR / f"{scene_name}_signed_chroma_outlier_v5_plsafe.exr"


def detail_safe_path(scene_name: str) -> Path:
    return DETAIL_SAFE_DIR / f"{scene_name}_signed_chroma_outlier_v9_density035.exr"


def teacher_path(scene_name: str) -> Path:
    return TEACHER_DIR / f"{scene_name}_freqsplit_teacher_v3_chroma_safe.exr"


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
    parser = argparse.ArgumentParser(description="Build PL-safe teacher oracle chroma blend.")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "signed_chroma_outlier_v11_teacher_oracle_blend"))
    parser.add_argument("--chroma-weight", type=float, default=1.0)
    parser.add_argument("--luma-weight", type=float, default=0.18)
    parser.add_argument("--loss-sigma", type=float, default=2.0)
    parser.add_argument("--transition", type=float, default=0.0012)
    parser.add_argument("--mask-blur", type=float, default=1.4)
    parser.add_argument("--detail-safe-floor", type=float, default=0.0)
    parser.add_argument("--detail-safe-ceiling", type=float, default=0.90)
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
        teacher = read_image(teacher_path(scene_name))
        strong_loss = local_loss(strong, teacher, chroma_weight=args.chroma_weight, luma_weight=args.luma_weight, sigma=args.loss_sigma)
        detail_loss = local_loss(detail_safe, teacher, chroma_weight=args.chroma_weight, luma_weight=args.luma_weight, sigma=args.loss_sigma)
        detail_weight = sigmoid01((strong_loss - detail_loss) / max(float(args.transition), 1.0e-6))
        if args.mask_blur > 0:
            detail_weight = gaussian_filter(detail_weight, sigma=float(args.mask_blur), mode="reflect")
        detail_weight = np.clip(
            float(args.detail_safe_floor) + detail_weight * (float(args.detail_safe_ceiling) - float(args.detail_safe_floor)),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        out = strong * (1.0 - detail_weight[..., None]) + detail_safe * detail_weight[..., None]

        stem = f"{scene_name}_signed_chroma_outlier_v11_teacher_oracle_blend"
        exr_path = out_dir / f"{stem}.exr"
        tiff_path = out_dir / f"{stem}.tiff"
        preview_path = out_dir / f"{stem}_preview.png"
        mask_path = out_dir / f"{stem}_detail_weight.png"
        write_exr(exr_path, out)
        write_tiff(tiff_path, out)
        Image.fromarray(make_preview(out)).save(preview_path)
        Image.fromarray((detail_weight * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(mask_path)
        for roi_name, x, y in scene.rois:
            render_compare(
                crop_dir / f"{scene_name}_{roi_name}_teacher_oracle_blend_compare.png",
                [
                    ("base", crop(base, x, y, args.crop_size)),
                    ("v5 strong", crop(strong, x, y, args.crop_size)),
                    ("density035", crop(detail_safe, x, y, args.crop_size)),
                    ("oracle", crop(out, x, y, args.crop_size)),
                    ("teacher", crop(teacher, x, y, args.crop_size)),
                ],
            )
        report[scene_name] = {
            "base": str(scene.base),
            "strong": str(strong_path(scene_name)),
            "detail_safe": str(detail_safe_path(scene_name)),
            "teacher": str(teacher_path(scene_name)),
            "output": str(exr_path),
            "preview": str(preview_path),
            "detail_weight": str(mask_path),
            "detail_weight_mean": float(np.mean(detail_weight)),
            "detail_weight_p90": float(np.quantile(detail_weight, 0.90)),
            "detail_weight_p99": float(np.quantile(detail_weight, 0.99)),
            "strong_loss_mean": float(np.mean(strong_loss)),
            "detail_loss_mean": float(np.mean(detail_loss)),
        }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
