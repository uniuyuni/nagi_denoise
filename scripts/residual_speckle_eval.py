"""Evaluate isolated residual speckles in denoised real photos.

The existing real-photo evaluator measures broad high-frequency energy. This
script focuses on small isolated luma/chroma impulses, which often remain
visually annoying even when average HF metrics look good.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

from perfect_nr_probe import read_image
from real_photo_noise_eval import (
    LUMA_SRGB,
    display_rgb,
    luma,
    make_masks,
    parse_candidate,
    quantile_masked,
    safe_ratio,
)


def impulse(x: np.ndarray, size: int) -> np.ndarray:
    base = median_filter(x, size=int(size), mode="reflect")
    return (x - base).astype(np.float32, copy=False)


def add_neutral_flat_masks(masks: dict[str, np.ndarray], reference_display: np.ndarray) -> dict[str, np.ndarray]:
    rgb = np.asarray(reference_display, dtype=np.float32)[..., :3]
    y = luma(rgb, LUMA_SRGB)
    chroma = rgb - y[..., None]
    low_chroma = gaussian_filter(chroma, sigma=(2.0, 2.0, 0.0), mode="reflect")
    low_chroma_mag = np.sqrt(np.sum(low_chroma * low_chroma, axis=2))
    low_blue = low_chroma[..., 2] - 0.5 * (low_chroma[..., 0] + low_chroma[..., 1])
    neutral_flat = masks["flat"] & (low_chroma_mag < 0.075)
    blue_struct_flat = masks["flat"] & (low_blue > 0.055)
    return {**masks, "neutral_flat": neutral_flat, "blue_struct_flat": blue_struct_flat}


def candidate_speckle_metrics(
    candidate_display: np.ndarray,
    reference_metrics: dict[str, float] | None,
    masks: dict[str, np.ndarray],
    *,
    median_size: int,
) -> dict[str, float]:
    rgb = np.asarray(candidate_display, dtype=np.float32)[..., :3]
    y = luma(rgb, LUMA_SRGB)
    rg = rgb[..., 0] - rgb[..., 1]
    by = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])
    magenta = 0.5 * (rgb[..., 0] + rgb[..., 2]) - rgb[..., 1]
    blue = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])

    luma_imp = np.abs(impulse(y, median_size))
    rg_imp = impulse(rg, median_size)
    by_imp = impulse(by, median_size)
    chroma_imp = np.sqrt(0.5 * (rg_imp * rg_imp + by_imp * by_imp))
    magenta_pos = np.maximum(impulse(magenta, median_size), 0.0)
    blue_pos = np.maximum(impulse(blue, median_size), 0.0)
    visibility = 1.0 / np.sqrt(np.maximum(y, 0.02))

    flat = masks["flat"]
    shadow = masks["shadow_flat"]
    neutral = masks.get("neutral_flat", np.zeros_like(flat, dtype=bool))
    blue_struct = masks.get("blue_struct_flat", np.zeros_like(flat, dtype=bool))
    raw = {
        "flat_luma_imp_p99": quantile_masked(luma_imp, flat, 0.99),
        "flat_luma_imp_p999": quantile_masked(luma_imp, flat, 0.999),
        "flat_luma_imp_visible": float(np.mean((luma_imp * visibility)[flat])) if np.any(flat) else 0.0,
        "flat_chroma_imp_p99": quantile_masked(chroma_imp, flat, 0.99),
        "flat_chroma_imp_p999": quantile_masked(chroma_imp, flat, 0.999),
        "flat_chroma_imp_visible": float(np.mean((chroma_imp * visibility)[flat])) if np.any(flat) else 0.0,
        "flat_magenta_pos_p999": quantile_masked(magenta_pos, flat, 0.999),
        "shadow_magenta_pos_p999": quantile_masked(magenta_pos, shadow, 0.999),
        "flat_blue_pos_p999": quantile_masked(blue_pos, flat, 0.999),
        "shadow_blue_pos_p999": quantile_masked(blue_pos, shadow, 0.999),
        "neutral_chroma_imp_visible": float(np.mean((chroma_imp * visibility)[neutral])) if np.any(neutral) else 0.0,
        "neutral_magenta_pos_p999": quantile_masked(magenta_pos, neutral, 0.999),
        "neutral_blue_pos_p999": quantile_masked(blue_pos, neutral, 0.999),
        "blue_struct_blue_pos_p999": quantile_masked(blue_pos, blue_struct, 0.999),
    }
    if reference_metrics is None:
        return {**raw, **{f"{k}_ratio": 1.0 for k in raw}}
    ratios = {f"{k}_ratio": safe_ratio(v, reference_metrics[k]) for k, v in raw.items()}
    return {**raw, **ratios}


def save_speckle_preview(path: Path, candidate_display: np.ndarray, masks: dict[str, np.ndarray], median_size: int) -> None:
    rgb = np.asarray(candidate_display, dtype=np.float32)[..., :3]
    y = luma(rgb, LUMA_SRGB)
    rg = rgb[..., 0] - rgb[..., 1]
    by = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])
    luma_imp = np.abs(impulse(y, median_size))
    chroma_imp = np.sqrt(0.5 * (impulse(rg, median_size) ** 2 + impulse(by, median_size) ** 2))
    heat = np.zeros((*y.shape, 3), dtype=np.float32)
    heat[..., 0] = chroma_imp * 10.0
    heat[..., 1] = luma_imp * 10.0
    heat[..., 2] = chroma_imp * 6.0
    heat[~masks["flat"]] *= 0.18
    Image.fromarray(np.clip(heat * 255.0, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate isolated residual speckles.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--tone", choices=["reinhard", "clip"], default="reinhard")
    parser.add_argument("--median-size", type=int, default=3)
    args = parser.parse_args()

    if not args.candidate:
        raise SystemExit("at least one --candidate is required")

    out_dir = Path(args.output_dir)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference).expanduser()
    reference = read_image(reference_path)
    reference_display = display_rgb(reference, exposure=args.exposure, tone=args.tone)
    masks = make_masks(
        reference,
        reference_display,
        hf_sigma=1.2,
        edge_sigma=1.0,
        flat_hf_threshold=0.018,
        flat_edge_threshold=0.030,
        min_display_luma=0.035,
        max_display_luma=0.92,
        shadow_display_luma=0.22,
        highlight_linear_luma=1.0,
        edge_threshold=0.050,
        top_luma_percent=1.0,
    )
    masks = add_neutral_flat_masks(masks, reference_display)
    baseline = candidate_speckle_metrics(reference_display, None, masks, median_size=args.median_size)
    report = {
        "reference": str(reference_path),
        "settings": {"exposure": args.exposure, "tone": args.tone, "median_size": args.median_size},
        "masks": {name: {"fraction": float(np.mean(mask)), "pixels": int(np.sum(mask))} for name, mask in masks.items()},
        "candidates": [{"name": "input", "path": str(reference_path), "metrics": baseline}],
    }
    save_speckle_preview(preview_dir / "input_speckle.png", reference_display, masks, args.median_size)

    for spec in args.candidate:
        name, path = parse_candidate(spec)
        candidate = read_image(path)
        if candidate.shape[:2] != reference.shape[:2]:
            raise ValueError(f"shape mismatch for {name}: reference={reference.shape}, candidate={candidate.shape}")
        candidate_display = display_rgb(candidate, exposure=args.exposure, tone=args.tone)
        metrics = candidate_speckle_metrics(candidate_display, baseline, masks, median_size=args.median_size)
        save_speckle_preview(preview_dir / f"{name}_speckle.png", candidate_display, masks, args.median_size)
        report["candidates"].append({"name": name, "path": str(path), "metrics": metrics})

    json_path = out_dir / "residual_speckle_eval.json"
    md_path = out_dir / "residual_speckle_eval.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Residual Speckle Evaluation",
        "",
        f"Reference: `{reference_path}`",
        "",
        "| candidate | luma p99 | luma p999 | luma visible | chroma p99 | chroma p999 | chroma visible | magenta p999 | shadow magenta p999 | blue p999 | shadow blue p999 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["candidates"]:
        m = item["metrics"]
        lines.append(
            "| {name} | {lp99:.3f} | {lp999:.3f} | {lv:.3f} | {cp99:.3f} | {cp999:.3f} | {cv:.3f} | {mp999:.3f} | {smp999:.3f} | {bp999:.3f} | {sbp999:.3f} |".format(
                name=item["name"],
                lp99=m["flat_luma_imp_p99_ratio"],
                lp999=m["flat_luma_imp_p999_ratio"],
                lv=m["flat_luma_imp_visible_ratio"],
                cp99=m["flat_chroma_imp_p99_ratio"],
                cp999=m["flat_chroma_imp_p999_ratio"],
                cv=m["flat_chroma_imp_visible_ratio"],
                mp999=m["flat_magenta_pos_p999_ratio"],
                smp999=m["shadow_magenta_pos_p999_ratio"],
                bp999=m["flat_blue_pos_p999_ratio"],
                sbp999=m["shadow_blue_pos_p999_ratio"],
            )
        )
    lines.extend(
        [
            "",
            "Neutral / blue-structure split:",
            "",
            "| candidate | neutral chroma visible | neutral magenta p999 | neutral blue p999 | blue-structure blue p999 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["candidates"]:
        m = item["metrics"]
        lines.append(
            "| {name} | {ncv:.3f} | {nmp:.3f} | {nbp:.3f} | {bsp:.3f} |".format(
                name=item["name"],
                ncv=m["neutral_chroma_imp_visible_ratio"],
                nmp=m["neutral_magenta_pos_p999_ratio"],
                nbp=m["neutral_blue_pos_p999_ratio"],
                bsp=m["blue_struct_blue_pos_p999_ratio"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
