"""Evaluate real-photo residual noise and preservation tradeoffs.

This is intentionally reference-free: the original real image is treated as the
baseline, not as a clean GT. The metrics answer practical questions:

* Did flat-region luma/chroma high-frequency energy go down?
* Did edges/thin lines keep their high-frequency energy?
* Did highlights drift in chroma/luma?

Use this alongside visual inspection. It is a microscope for tradeoffs, not a
final perceptual score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

from perfect_nr_probe import read_image, srgb_oetf


LUMA_LINEAR = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
LUMA_SRGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    x = x[..., :3]
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def display_rgb(img: np.ndarray, exposure: float, tone: str) -> np.ndarray:
    x = np.clip(_safe_rgb(img) * float(exposure), 0.0, None)
    if tone == "reinhard":
        x = x / (1.0 + x)
    elif tone == "clip":
        x = np.clip(x, 0.0, 1.0)
    else:
        raise ValueError(f"unknown tone curve: {tone!r}")
    return srgb_oetf(x).astype(np.float32, copy=False)


def luma(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(_safe_rgb(rgb) * weights.reshape(1, 1, 3), axis=2)


def chroma_ratio_linear(img: np.ndarray) -> np.ndarray:
    rgb = np.clip(_safe_rgb(img), 0.0, None)
    y = np.sum(rgb * LUMA_LINEAR.reshape(1, 1, 3), axis=2, keepdims=True)
    return rgb / np.maximum(y, 1.0e-6)


def highpass(x: np.ndarray, sigma: float) -> np.ndarray:
    return x - gaussian_filter(x, sigma=float(sigma), mode="reflect")


def chroma_hf_energy(rgb_display: np.ndarray, sigma: float) -> np.ndarray:
    rgb = _safe_rgb(rgb_display)
    rg = rgb[..., 0] - rgb[..., 1]
    by = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])
    rg_hf = highpass(rg, sigma)
    by_hf = highpass(by, sigma)
    return np.sqrt(0.5 * (rg_hf * rg_hf + by_hf * by_hf))


def magenta_dot_hf_energy(rgb_display: np.ndarray, sigma: float) -> np.ndarray:
    rgb = _safe_rgb(rgb_display)
    magenta_axis = 0.5 * (rgb[..., 0] + rgb[..., 2]) - rgb[..., 1]
    return np.maximum(highpass(magenta_axis, sigma), 0.0)


def luma_hf_energy(rgb_display: np.ndarray, sigma: float) -> np.ndarray:
    y = luma(rgb_display, LUMA_SRGB)
    return np.abs(highpass(y, sigma))


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser()


def make_masks(
    reference_linear: np.ndarray,
    reference_display: np.ndarray,
    *,
    hf_sigma: float,
    edge_sigma: float,
    flat_hf_threshold: float,
    flat_edge_threshold: float,
    min_display_luma: float,
    max_display_luma: float,
    highlight_linear_luma: float,
    edge_threshold: float,
    top_luma_percent: float,
) -> dict[str, np.ndarray]:
    y_display = luma(reference_display, LUMA_SRGB)
    y_linear = luma(reference_linear, LUMA_LINEAR)
    flat_hf = luma_hf_energy(reference_display, hf_sigma)
    edge_mag = gaussian_gradient_magnitude(y_display, sigma=float(edge_sigma), mode="reflect")
    midtone = (y_display >= min_display_luma) & (y_display <= max_display_luma)
    non_highlight = y_linear < highlight_linear_luma
    flat = (flat_hf < flat_hf_threshold) & (edge_mag < flat_edge_threshold) & midtone & non_highlight
    edge = (edge_mag >= edge_threshold) & midtone & non_highlight
    q = 1.0 - float(top_luma_percent) / 100.0
    top_luma = y_linear > float(np.quantile(y_linear, q))
    return {
        "flat": flat,
        "edge": edge,
        "highlight": top_luma,
        "midtone": midtone,
        "non_highlight": non_highlight,
    }


def mean_masked(a: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(a[mask]))


def quantile_masked(a: np.ndarray, mask: np.ndarray, q: float) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.quantile(a[mask], float(q)))


def safe_ratio(value: float, base: float) -> float:
    return float(value / max(base, 1.0e-8))


def candidate_metrics(
    reference_linear: np.ndarray,
    candidate_linear: np.ndarray,
    reference_display: np.ndarray,
    candidate_display: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    hf_sigma: float,
) -> dict:
    flat = masks["flat"]
    edge = masks["edge"]
    highlight = masks["highlight"]

    ref_luma_hf = luma_hf_energy(reference_display, hf_sigma)
    out_luma_hf = luma_hf_energy(candidate_display, hf_sigma)
    ref_chroma_hf = chroma_hf_energy(reference_display, hf_sigma)
    out_chroma_hf = chroma_hf_energy(candidate_display, hf_sigma)
    ref_magenta_dot_hf = magenta_dot_hf_energy(reference_display, hf_sigma)
    out_magenta_dot_hf = magenta_dot_hf_energy(candidate_display, hf_sigma)
    y_display = luma(reference_display, LUMA_SRGB)
    visibility_weight = 1.0 / np.sqrt(np.maximum(y_display, 0.02))

    ref_edge_hf = ref_luma_hf
    out_edge_hf = out_luma_hf

    ref_y_lin = luma(reference_linear, LUMA_LINEAR)
    out_y_lin = luma(candidate_linear, LUMA_LINEAR)
    ref_chroma_lin = chroma_ratio_linear(reference_linear)
    out_chroma_lin = chroma_ratio_linear(candidate_linear)

    flat_luma = mean_masked(out_luma_hf, flat)
    flat_chroma = mean_masked(out_chroma_hf, flat)
    ref_flat_luma = mean_masked(ref_luma_hf, flat)
    ref_flat_chroma = mean_masked(ref_chroma_hf, flat)
    flat_luma_p95 = quantile_masked(out_luma_hf, flat, 0.95)
    flat_luma_p99 = quantile_masked(out_luma_hf, flat, 0.99)
    flat_chroma_p95 = quantile_masked(out_chroma_hf, flat, 0.95)
    flat_chroma_p99 = quantile_masked(out_chroma_hf, flat, 0.99)
    ref_flat_luma_p95 = quantile_masked(ref_luma_hf, flat, 0.95)
    ref_flat_luma_p99 = quantile_masked(ref_luma_hf, flat, 0.99)
    ref_flat_chroma_p95 = quantile_masked(ref_chroma_hf, flat, 0.95)
    ref_flat_chroma_p99 = quantile_masked(ref_chroma_hf, flat, 0.99)
    flat_magenta_dot_p95 = quantile_masked(out_magenta_dot_hf, flat, 0.95)
    flat_magenta_dot_p99 = quantile_masked(out_magenta_dot_hf, flat, 0.99)
    ref_flat_magenta_dot_p95 = quantile_masked(ref_magenta_dot_hf, flat, 0.95)
    ref_flat_magenta_dot_p99 = quantile_masked(ref_magenta_dot_hf, flat, 0.99)
    flat_luma_visible = mean_masked(out_luma_hf * visibility_weight, flat)
    flat_chroma_visible = mean_masked(out_chroma_hf * visibility_weight, flat)
    flat_magenta_dot_visible = mean_masked(out_magenta_dot_hf * visibility_weight, flat)
    ref_flat_luma_visible = mean_masked(ref_luma_hf * visibility_weight, flat)
    ref_flat_chroma_visible = mean_masked(ref_chroma_hf * visibility_weight, flat)
    ref_flat_magenta_dot_visible = mean_masked(ref_magenta_dot_hf * visibility_weight, flat)
    edge_hf = mean_masked(out_edge_hf, edge)
    ref_edge = mean_masked(ref_edge_hf, edge)

    if np.any(highlight):
        chroma_delta = np.mean(np.abs((out_chroma_lin - ref_chroma_lin)[highlight]))
    else:
        chroma_delta = 0.0

    return {
        "flat_luma_hf": flat_luma,
        "reference_flat_luma_hf": ref_flat_luma,
        "flat_luma_hf_ratio": safe_ratio(flat_luma, ref_flat_luma),
        "flat_luma_hf_reduction": 1.0 - safe_ratio(flat_luma, ref_flat_luma),
        "flat_luma_hf_p95": flat_luma_p95,
        "reference_flat_luma_hf_p95": ref_flat_luma_p95,
        "flat_luma_hf_p95_ratio": safe_ratio(flat_luma_p95, ref_flat_luma_p95),
        "flat_luma_hf_p99": flat_luma_p99,
        "reference_flat_luma_hf_p99": ref_flat_luma_p99,
        "flat_luma_hf_p99_ratio": safe_ratio(flat_luma_p99, ref_flat_luma_p99),
        "flat_luma_visible_ratio": safe_ratio(flat_luma_visible, ref_flat_luma_visible),
        "flat_chroma_hf": flat_chroma,
        "reference_flat_chroma_hf": ref_flat_chroma,
        "flat_chroma_hf_ratio": safe_ratio(flat_chroma, ref_flat_chroma),
        "flat_chroma_hf_reduction": 1.0 - safe_ratio(flat_chroma, ref_flat_chroma),
        "flat_chroma_hf_p95": flat_chroma_p95,
        "reference_flat_chroma_hf_p95": ref_flat_chroma_p95,
        "flat_chroma_hf_p95_ratio": safe_ratio(flat_chroma_p95, ref_flat_chroma_p95),
        "flat_chroma_hf_p99": flat_chroma_p99,
        "reference_flat_chroma_hf_p99": ref_flat_chroma_p99,
        "flat_chroma_hf_p99_ratio": safe_ratio(flat_chroma_p99, ref_flat_chroma_p99),
        "flat_chroma_visible_ratio": safe_ratio(flat_chroma_visible, ref_flat_chroma_visible),
        "flat_magenta_dot_hf_p95": flat_magenta_dot_p95,
        "reference_flat_magenta_dot_hf_p95": ref_flat_magenta_dot_p95,
        "flat_magenta_dot_hf_p95_ratio": safe_ratio(flat_magenta_dot_p95, ref_flat_magenta_dot_p95),
        "flat_magenta_dot_hf_p99": flat_magenta_dot_p99,
        "reference_flat_magenta_dot_hf_p99": ref_flat_magenta_dot_p99,
        "flat_magenta_dot_hf_p99_ratio": safe_ratio(flat_magenta_dot_p99, ref_flat_magenta_dot_p99),
        "flat_magenta_dot_visible_ratio": safe_ratio(flat_magenta_dot_visible, ref_flat_magenta_dot_visible),
        "edge_luma_hf": edge_hf,
        "reference_edge_luma_hf": ref_edge,
        "edge_luma_hf_retention": safe_ratio(edge_hf, ref_edge),
        "highlight_luma_delta": mean_masked(out_y_lin - ref_y_lin, highlight),
        "highlight_chroma_drift": float(chroma_delta),
        "global_display_luma_mae": float(
            np.mean(np.abs(luma(candidate_display, LUMA_SRGB) - luma(reference_display, LUMA_SRGB)))
        ),
        "global_display_chroma_mae": float(np.mean(np.abs(candidate_display - reference_display))),
        "rgb_max": float(np.nanmax(candidate_linear)),
    }


def save_mask_preview(
    preview_dir: Path,
    reference_display: np.ndarray,
    masks: dict[str, np.ndarray],
) -> None:
    base = np.clip(reference_display * 255.0 + 0.5, 0, 255).astype(np.uint8)
    overlay = base.astype(np.float32)
    colors = {
        "flat": np.array([40, 220, 80], dtype=np.float32),
        "edge": np.array([255, 190, 30], dtype=np.float32),
        "highlight": np.array([80, 160, 255], dtype=np.float32),
    }
    for name, color in colors.items():
        mask = masks[name]
        overlay[mask] = overlay[mask] * 0.45 + color * 0.55
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(preview_dir / "mask_overlay.png")
    for name in ("flat", "edge", "highlight"):
        mask_img = (masks[name].astype(np.uint8) * 255)
        Image.fromarray(mask_img).save(preview_dir / f"mask_{name}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real-photo residual noise tradeoffs.")
    parser.add_argument("--reference", required=True, help="Original noisy real image.")
    parser.add_argument("--candidate", action="append", default=[], help="NAME=PATH or PATH. Repeatable.")
    parser.add_argument("--output-dir", default="runs/perfect_nr/real_noise_eval")
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--tone", choices=["reinhard", "clip"], default="reinhard")
    parser.add_argument("--hf-sigma", type=float, default=1.2)
    parser.add_argument("--edge-sigma", type=float, default=1.0)
    parser.add_argument("--flat-hf-threshold", type=float, default=0.018)
    parser.add_argument("--flat-edge-threshold", type=float, default=0.030)
    parser.add_argument("--edge-threshold", type=float, default=0.050)
    parser.add_argument("--min-display-luma", type=float, default=0.035)
    parser.add_argument("--max-display-luma", type=float, default=0.92)
    parser.add_argument("--highlight-linear-luma", type=float, default=1.0)
    parser.add_argument("--top-luma-percent", type=float, default=1.0)
    args = parser.parse_args()

    if not args.candidate:
        raise SystemExit("at least one --candidate is required")

    out_dir = Path(args.output_dir)
    preview_dir = out_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference).expanduser()
    reference = read_image(reference_path)
    reference_display = display_rgb(reference, exposure=args.exposure, tone=args.tone)
    masks = make_masks(
        reference,
        reference_display,
        hf_sigma=args.hf_sigma,
        edge_sigma=args.edge_sigma,
        flat_hf_threshold=args.flat_hf_threshold,
        flat_edge_threshold=args.flat_edge_threshold,
        min_display_luma=args.min_display_luma,
        max_display_luma=args.max_display_luma,
        highlight_linear_luma=args.highlight_linear_luma,
        edge_threshold=args.edge_threshold,
        top_luma_percent=args.top_luma_percent,
    )
    save_mask_preview(preview_dir, reference_display, masks)
    Image.fromarray((reference_display * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(
        preview_dir / "reference_preview.png"
    )

    report = {
        "reference": str(reference_path),
        "settings": {
            "exposure": args.exposure,
            "tone": args.tone,
            "hf_sigma": args.hf_sigma,
            "edge_sigma": args.edge_sigma,
            "flat_hf_threshold": args.flat_hf_threshold,
            "flat_edge_threshold": args.flat_edge_threshold,
            "edge_threshold": args.edge_threshold,
            "min_display_luma": args.min_display_luma,
            "max_display_luma": args.max_display_luma,
            "highlight_linear_luma": args.highlight_linear_luma,
            "top_luma_percent": args.top_luma_percent,
        },
        "masks": {
            name: {"fraction": float(np.mean(mask)), "pixels": int(np.sum(mask))}
            for name, mask in masks.items()
        },
        "candidates": [],
    }

    # Include the input itself as a baseline row.
    baseline_metrics = candidate_metrics(
        reference,
        reference,
        reference_display,
        reference_display,
        masks,
        hf_sigma=args.hf_sigma,
    )
    report["candidates"].append(
        {"name": "input", "path": str(reference_path), "preview": str(preview_dir / "reference_preview.png"), "metrics": baseline_metrics}
    )

    for spec in args.candidate:
        name, path = parse_candidate(spec)
        candidate = read_image(path)
        if candidate.shape[:2] != reference.shape[:2]:
            raise ValueError(f"shape mismatch for {name}: reference={reference.shape}, candidate={candidate.shape}")
        candidate_display = display_rgb(candidate, exposure=args.exposure, tone=args.tone)
        preview_path = preview_dir / f"{name}_preview.png"
        Image.fromarray((candidate_display * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(preview_path)
        metrics = candidate_metrics(
            reference,
            candidate,
            reference_display,
            candidate_display,
            masks,
            hf_sigma=args.hf_sigma,
        )
        report["candidates"].append(
            {"name": name, "path": str(path), "preview": str(preview_path), "metrics": metrics}
        )

    json_path = out_dir / "real_noise_eval.json"
    md_path = out_dir / "real_noise_eval.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Real Photo Noise Evaluation",
        "",
        f"Reference: `{reference_path}`",
        "",
        "Masks:",
        "",
        "| mask | fraction | pixels |",
        "| --- | ---: | ---: |",
    ]
    for name in ("flat", "edge", "highlight"):
        m = report["masks"][name]
        lines.append(f"| {name} | {m['fraction']:.6f} | {m['pixels']} |")

    lines += [
        "",
        "| candidate | flat luma ratio | flat chroma ratio | flat chroma reduction | edge HF retention | highlight chroma drift | highlight luma delta | rgb max | preview |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["candidates"]:
        m = item["metrics"]
        lines.append(
            "| {name} | {fl:.3f} | {fc:.3f} | {fcr:.1%} | {edge:.3f} | {hc:.6f} | {hl:.6f} | {rgb:.3f} | `{preview}` |".format(
                name=item["name"],
                fl=m["flat_luma_hf_ratio"],
                fc=m["flat_chroma_hf_ratio"],
                fcr=m["flat_chroma_hf_reduction"],
                edge=m["edge_luma_hf_retention"],
                hc=m["highlight_chroma_drift"],
                hl=m["highlight_luma_delta"],
                rgb=m["rgb_max"],
                preview=item["preview"],
            )
        )
    lines += [
        "",
        "Tail / visibility metrics:",
        "",
        "| candidate | luma p95 ratio | luma p99 ratio | luma visible ratio | chroma p95 ratio | chroma p99 ratio | chroma visible ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["candidates"]:
        m = item["metrics"]
        lines.append(
            "| {name} | {lp95:.3f} | {lp99:.3f} | {lv:.3f} | {cp95:.3f} | {cp99:.3f} | {cv:.3f} |".format(
                name=item["name"],
                lp95=m["flat_luma_hf_p95_ratio"],
                lp99=m["flat_luma_hf_p99_ratio"],
                lv=m["flat_luma_visible_ratio"],
                cp95=m["flat_chroma_hf_p95_ratio"],
                cp99=m["flat_chroma_hf_p99_ratio"],
                cv=m["flat_chroma_visible_ratio"],
            )
        )
    lines += [
        "",
        "Magenta dot metrics:",
        "",
        "| candidate | magenta dot p95 ratio | magenta dot p99 ratio | magenta dot visible ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in report["candidates"]:
        m = item["metrics"]
        lines.append(
            "| {name} | {mp95:.3f} | {mp99:.3f} | {mv:.3f} |".format(
                name=item["name"],
                mp95=m["flat_magenta_dot_hf_p95_ratio"],
                mp99=m["flat_magenta_dot_hf_p99_ratio"],
                mv=m["flat_magenta_dot_visible_ratio"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {preview_dir / 'mask_overlay.png'}")


if __name__ == "__main__":
    main()
