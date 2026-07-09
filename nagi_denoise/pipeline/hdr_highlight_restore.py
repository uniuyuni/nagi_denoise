"""Restore HDR highlights after display-space Perfect NR experiments.

Several hand-crafted detail stages operate in display space and clip output to
0..1 before converting back to EXR. This restores highlight values from the
original HDR reference with a smooth mask, keeping the denoised candidate in
non-highlight regions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from .flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma
from .luma_tail_speckle import sigmoid01
from .build_pseudo_teacher import RUN_ROOT, SCENES
from .detail_guard import write_exr, write_tiff
from .probe import image_stats, make_preview, read_image


DEFAULT_BASE_DIR = RUN_ROOT / "signed_chroma_outlier_v24b_oklab_oriented_texture_floor_mild"


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def restore_hdr_highlights(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    start: float,
    full: float,
    blur_sigma: float,
    chroma_start: float,
    chroma_full: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    ref = _safe_rgb(reference)
    cand = np.clip(_safe_rgb(candidate), 0.0, None)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch reference={ref.shape} candidate={cand.shape}")

    ref_pos = np.clip(ref, 0.0, None)
    ref_peak = np.max(ref_pos, axis=2)
    highlight = sigmoid01((ref_peak - float(start)) / max(float(full) - float(start), 1.0e-6))
    if blur_sigma > 0:
        highlight = gaussian_filter(highlight, sigma=float(blur_sigma), mode="reflect")
    highlight = np.clip(highlight, 0.0, 1.0).astype(np.float32, copy=False)

    # Preserve full HDR intensity in highlights. Near the transition, keep more
    # candidate chroma to avoid reintroducing color noise around non-HDR edges.
    ref_display = np.clip(linear_to_srgb_np(ref_pos), 0.0, 1.0)
    ref_y = luma(ref_display, LUMA_SRGB)
    cand_display = np.clip(linear_to_srgb_np(cand), 0.0, 1.0)
    cand_y = luma(cand_display, LUMA_SRGB)
    ref_ratio = ref_pos / np.maximum(np.sum(ref_pos * LUMA_SRGB, axis=2, keepdims=True), 1.0e-6)
    cand_ratio = cand / np.maximum(np.sum(cand * LUMA_SRGB, axis=2, keepdims=True), 1.0e-6)
    chroma_gate = sigmoid01((ref_peak - float(chroma_start)) / max(float(chroma_full) - float(chroma_start), 1.0e-6))
    chroma_gate = np.clip(chroma_gate[..., None], 0.0, 1.0)
    ratio = cand_ratio * (1.0 - chroma_gate) + ref_ratio * chroma_gate
    restored_luma = np.sum(ref_pos * LUMA_SRGB, axis=2, keepdims=True)
    hdr_restored = np.clip(ratio * restored_luma, 0.0, None)
    out = cand * (1.0 - highlight[..., None]) + hdr_restored * highlight[..., None]

    stats = {
        "start": float(start),
        "full": float(full),
        "blur_sigma": float(blur_sigma),
        "chroma_start": float(chroma_start),
        "chroma_full": float(chroma_full),
        "mask_mean": float(np.mean(highlight)),
        "mask_p95": float(np.quantile(highlight, 0.95)),
        "mask_p99": float(np.quantile(highlight, 0.99)),
        "reference_max": float(np.nanmax(ref)),
        "candidate_max": float(np.nanmax(cand)),
        "output_max": float(np.nanmax(out)),
        "reference_gt1_frac": float(np.mean(ref_peak > 1.0)),
        "output_gt1_frac": float(np.mean(np.max(out, axis=2) > 1.0)),
        "display_luma_delta_mean": float(np.mean(luma(np.clip(linear_to_srgb_np(np.clip(out, 0.0, None)), 0.0, 1.0), LUMA_SRGB) - cand_y)),
        "reference_display_luma_mean": float(np.mean(ref_y)),
    }
    return out.astype(np.float32, copy=False), highlight, stats


def base_path(base_dir: Path, scene_name: str, tag: str) -> Path:
    return base_dir / f"{scene_name}_{tag}.exr"


def crop(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, min(w - size, int(x) - size // 2))
    y0 = max(0, min(h - size, int(y) - size // 2))
    return image[y0 : y0 + size, x0 : x0 + size]


def render_compare(path: Path, panels: list[tuple[str, np.ndarray]], *, exposure: float = 1.0) -> None:
    previews = [Image.fromarray(make_preview(image, exposure=exposure, tone="reinhard")) for _, image in panels]
    width, height = previews[0].size
    label_h = 24
    canvas = Image.new("RGB", (width * len(previews), height + label_h), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    for i, ((label, _), preview) in enumerate(zip(panels, previews)):
        canvas.paste(preview, (i * width, label_h))
        draw.text((i * width + 8, 5), label, fill=(235, 235, 235))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore HDR highlights from the original EXR.")
    parser.add_argument("--reference-file", default=None, help="Original HDR EXR for a single generic restore.")
    parser.add_argument("--candidate-file", default=None, help="Candidate EXR/TIFF for a single generic restore.")
    parser.add_argument("--name", default=None, help="Output stem for generic restore mode.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--base-tag", default="v24b_oklab_oriented_texture_floor_mild")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "signed_chroma_outlier_v25_hdr_restore"))
    parser.add_argument("--tag", default="v25_hdr_restore")
    parser.add_argument("--start", type=float, default=0.82)
    parser.add_argument("--full", type=float, default=1.08)
    parser.add_argument("--blur-sigma", type=float, default=1.8)
    parser.add_argument("--chroma-start", type=float, default=0.92)
    parser.add_argument("--chroma-full", type=float, default=1.35)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--scene", action="append", default=[], help="Scene key to process. Repeatable.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_file or args.candidate_file:
        if not (args.reference_file and args.candidate_file):
            raise SystemExit("--reference-file and --candidate-file must be passed together")
        reference_path = Path(args.reference_file).expanduser()
        candidate_path = Path(args.candidate_file).expanduser()
        reference = read_image(reference_path)
        candidate = read_image(candidate_path)
        out, mask, stats = restore_hdr_highlights(
            reference,
            candidate,
            start=args.start,
            full=args.full,
            blur_sigma=args.blur_sigma,
            chroma_start=args.chroma_start,
            chroma_full=args.chroma_full,
        )
        stem = args.name or f"{candidate_path.stem}_{args.tag}"
        exr_path = out_dir / f"{stem}.exr"
        tiff_path = out_dir / f"{stem}.tiff"
        preview_path = out_dir / f"{stem}_preview.png"
        mask_path = out_dir / f"{stem}_mask.png"
        report_path = out_dir / f"{stem}.json"
        write_exr(exr_path, out)
        write_tiff(tiff_path, out)
        Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
        Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(mask_path)
        report = {
            "reference": str(reference_path),
            "candidate": str(candidate_path),
            "output": str(exr_path),
            "preview": str(preview_path),
            "mask": str(mask_path),
            "filter": stats,
            "output_stats": image_stats(out),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {report_path}")
        print(f"wrote {exr_path}")
        return

    report = {}
    scenes = {name: SCENES[name] for name in args.scene} if args.scene else SCENES
    for scene_name, scene in scenes.items():
        reference = read_image(scene.noisy)
        candidate = read_image(base_path(base_dir, scene_name, args.base_tag))
        out, mask, stats = restore_hdr_highlights(
            reference,
            candidate,
            start=args.start,
            full=args.full,
            blur_sigma=args.blur_sigma,
            chroma_start=args.chroma_start,
            chroma_full=args.chroma_full,
        )
        stem = f"{scene_name}_{args.tag}"
        exr_path = out_dir / f"{stem}.exr"
        tiff_path = out_dir / f"{stem}.tiff"
        preview_path = out_dir / f"{stem}_preview.png"
        mask_path = out_dir / f"{stem}_mask.png"
        write_exr(exr_path, out)
        write_tiff(tiff_path, out)
        Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
        Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(mask_path)
        for roi_name, x, y in scene.rois:
            render_compare(
                crop_dir / f"{scene_name}_{roi_name}_{args.tag}_compare.png",
                [
                    ("noisy", crop(reference, x, y, args.crop_size)),
                    (args.base_tag, crop(candidate, x, y, args.crop_size)),
                    (args.tag, crop(out, x, y, args.crop_size)),
                ],
            )
        report[scene_name] = {
            "reference": str(scene.noisy),
            "base": str(base_path(base_dir, scene_name, args.base_tag)),
            "output": str(exr_path),
            "preview": str(preview_path),
            "mask": str(mask_path),
            "filter": stats,
            "output_stats": image_stats(out),
        }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
