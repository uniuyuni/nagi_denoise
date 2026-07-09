"""Apply a PL-hair detail blend with a flat/spill close gate.

This is the reusable form of the v32 Occi pilot. It keeps the strong v30
hair/detail opening prior, then closes that prior where the detail candidate
appears to add luma/chroma impulses in flat or weak-structure regions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, median_filter

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_hair_region_luma_blend import hair_mask, smoothstep01
from apply_luma_tail_speckle_filter import sigmoid01
from apply_region_aware_luma_cleanup import make_coherent_structure_mask, make_texture_mask
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import image_stats, make_preview, read_image


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def display(image: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb_np(np.clip(_safe_rgb(image), 0.0, None)), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def chroma(rgb: np.ndarray) -> np.ndarray:
    y = luma(rgb, LUMA_SRGB)
    return (rgb - y[..., None]).astype(np.float32, copy=False)


def impulse(x: np.ndarray, size: int) -> np.ndarray:
    return (x - median_filter(x, size=int(size), mode="reflect")).astype(np.float32, copy=False)


def chroma_impulse(rgb: np.ndarray, size: int) -> np.ndarray:
    c = chroma(rgb)
    rg = c[..., 0] - c[..., 1]
    by = c[..., 2] - 0.5 * (c[..., 0] + c[..., 1])
    rg_imp = impulse(rg, size)
    by_imp = impulse(by, size)
    return np.sqrt(0.5 * (rg_imp * rg_imp + by_imp * by_imp)).astype(np.float32, copy=False)


def positive_impulses(rgb: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    magenta = 0.5 * (rgb[..., 0] + rgb[..., 2]) - rgb[..., 1]
    blue = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])
    magenta_imp = np.maximum(magenta - median_filter(magenta, size=int(size), mode="reflect"), 0.0)
    blue_imp = np.maximum(blue - median_filter(blue, size=int(size), mode="reflect"), 0.0)
    return magenta_imp.astype(np.float32, copy=False), blue_imp.astype(np.float32, copy=False)


def build_close_gate(
    reference: np.ndarray,
    base: np.ndarray,
    detail: np.ndarray,
    *,
    impulse_size: int,
    texture_threshold: float,
    texture_transition: float,
    coherence_threshold: float,
    coherence_transition: float,
    coherence_energy_threshold: float,
    coherence_energy_transition: float,
    flat_detail_threshold: float,
    flat_detail_transition: float,
    flat_edge_threshold: float,
    flat_edge_transition: float,
    luma_risk_threshold: float,
    luma_risk_transition: float,
    chroma_risk_threshold: float,
    chroma_risk_transition: float,
    impulse_risk_threshold: float,
    impulse_risk_transition: float,
    benefit_threshold: float,
    benefit_transition: float,
    close_risk_weight: float,
    close_flat_weight: float,
    close_benefit_weight: float,
    close_blur_sigma: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    ref = display(reference)
    base_d = display(base)
    detail_d = display(detail)
    base_y = luma(base_d, LUMA_SRGB)
    detail_y = luma(detail_d, LUMA_SRGB)

    texture = make_texture_mask(ref, texture_threshold=float(texture_threshold), texture_transition=float(texture_transition))
    coherent = make_coherent_structure_mask(
        ref,
        coherence_threshold=float(coherence_threshold),
        coherence_transition=float(coherence_transition),
        energy_threshold=float(coherence_energy_threshold),
        energy_transition=float(coherence_energy_transition),
    )
    structure = np.clip(np.maximum(texture, coherent), 0.0, 1.0).astype(np.float32, copy=False)
    edge = gaussian_gradient_magnitude(gaussian_filter(base_y, sigma=0.9, mode="reflect"), sigma=0.9, mode="reflect")
    local_detail = np.abs(base_y - gaussian_filter(base_y, sigma=1.0, mode="reflect"))
    flat = sigmoid01((float(flat_detail_threshold) - local_detail) / max(float(flat_detail_transition), 1.0e-6))
    flat *= sigmoid01((float(flat_edge_threshold) - edge) / max(float(flat_edge_transition), 1.0e-6))

    base_ci = chroma_impulse(base_d, impulse_size)
    detail_ci = chroma_impulse(detail_d, impulse_size)
    base_li = np.abs(impulse(base_y, impulse_size))
    detail_li = np.abs(impulse(detail_y, impulse_size))
    base_contrast = np.abs(base_y - gaussian_filter(base_y, sigma=2.0, mode="reflect"))
    detail_contrast = np.abs(detail_y - gaussian_filter(detail_y, sigma=2.0, mode="reflect"))
    base_mag, base_blue = positive_impulses(base_d, impulse_size)
    detail_mag, detail_blue = positive_impulses(detail_d, impulse_size)

    luma_risk = sigmoid01((detail_li - base_li - float(luma_risk_threshold)) / max(float(luma_risk_transition), 1.0e-6))
    chroma_risk = sigmoid01((detail_ci - base_ci - float(chroma_risk_threshold)) / max(float(chroma_risk_transition), 1.0e-6))
    magenta_risk = sigmoid01((detail_mag - base_mag - float(impulse_risk_threshold)) / max(float(impulse_risk_transition), 1.0e-6))
    blue_risk = sigmoid01((detail_blue - base_blue - float(impulse_risk_threshold)) / max(float(impulse_risk_transition), 1.0e-6))
    risk = np.maximum.reduce([luma_risk, chroma_risk, magenta_risk, blue_risk]).astype(np.float32, copy=False)

    color_benefit = np.maximum.reduce(
        [
            sigmoid01((base_ci - detail_ci + float(benefit_threshold)) / max(float(benefit_transition), 1.0e-6)),
            sigmoid01((base_mag - detail_mag + float(benefit_threshold)) / max(float(benefit_transition), 1.0e-6)),
            sigmoid01((base_blue - detail_blue + float(benefit_threshold)) / max(float(benefit_transition), 1.0e-6)),
        ]
    )
    contrast_benefit = sigmoid01((detail_contrast - base_contrast + float(benefit_threshold)) / max(float(benefit_transition), 1.0e-6))
    benefit = np.clip(0.60 * color_benefit + 0.25 * contrast_benefit + 0.15 * structure, 0.0, 1.0)
    dark = sigmoid01((base_y - 0.02) / 0.030) * sigmoid01((0.60 - base_y) / 0.10)
    flat_spill = np.clip(flat * (1.0 - structure) * (0.35 + 0.65 * dark), 0.0, 1.0)
    close = np.clip(
        float(close_risk_weight) * risk + float(close_flat_weight) * flat_spill - float(close_benefit_weight) * benefit,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    if close_blur_sigma > 0:
        close = gaussian_filter(close, sigma=float(close_blur_sigma), mode="reflect").astype(np.float32, copy=False)
        close = np.clip(close, 0.0, 1.0)
    stats = {
        "close_mean": float(np.mean(close)),
        "close_p90": float(np.quantile(close, 0.90)),
        "close_p99": float(np.quantile(close, 0.99)),
        "risk_mean": float(np.mean(risk)),
        "flat_spill_mean": float(np.mean(flat_spill)),
        "benefit_mean": float(np.mean(benefit)),
        "structure_mean": float(np.mean(structure)),
    }
    masks = {"close": close, "risk": risk, "flat_spill": flat_spill, "benefit": benefit, "structure": structure}
    return close, stats, masks


def apply_plhair_close_gate(
    reference: np.ndarray,
    base: np.ndarray,
    detail: np.ndarray,
    *,
    close_strength: float,
    close_min_keep: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
    **kwargs: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    hair_kwargs = {
        key: kwargs[key]
        for key in kwargs
        if key
        in {
            "dark_low",
            "dark_high",
            "dark_transition",
            "skin_inhibit",
            "skin_proximity_sigma",
            "skin_proximity_threshold",
            "skin_proximity_transition",
            "texture_threshold",
            "texture_transition",
            "coherence_threshold",
            "coherence_transition",
            "coherence_energy_threshold",
            "coherence_energy_transition",
            "mask_guide_dark_high",
            "mask_guide_transition",
            "mask_guide_weight",
            "mask_blur_sigma",
        }
    }
    close_keys = {
        "impulse_size",
        "texture_threshold",
        "texture_transition",
        "coherence_threshold",
        "coherence_transition",
        "coherence_energy_threshold",
        "coherence_energy_transition",
        "flat_detail_threshold",
        "flat_detail_transition",
        "flat_edge_threshold",
        "flat_edge_transition",
        "luma_risk_threshold",
        "luma_risk_transition",
        "chroma_risk_threshold",
        "chroma_risk_transition",
        "impulse_risk_threshold",
        "impulse_risk_transition",
        "benefit_threshold",
        "benefit_transition",
        "close_risk_weight",
        "close_flat_weight",
        "close_benefit_weight",
        "close_blur_sigma",
    }
    close_kwargs = {key: kwargs[key] for key in close_keys}
    base_gate, hair_stats = hair_mask(reference, base, None, **hair_kwargs)
    close, close_stats, close_masks = build_close_gate(reference, base, detail, **close_kwargs)
    limiter = np.clip(1.0 - float(close_strength) * close, float(close_min_keep), 1.0).astype(np.float32, copy=False)
    gate = np.clip(base_gate * limiter, 0.0, 1.0).astype(np.float32, copy=False)

    base_rgb = np.clip(_safe_rgb(base), 0.0, None)
    detail_rgb = np.clip(_safe_rgb(detail), 0.0, None)
    base_d = display(base_rgb)
    detail_d = display(detail_rgb)
    out_display = np.clip(base_d * (1.0 - gate[..., None]) + detail_d * gate[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    base_peak = np.max(base_rgb, axis=2)
    hdr = smoothstep01((base_peak - float(hdr_peak_threshold)) / max(float(hdr_transition), 1.0e-6))
    out = out * (1.0 - hdr[..., None]) + base_rgb * hdr[..., None]

    stats = {
        **{f"hair_{key}": value for key, value in hair_stats.items()},
        **close_stats,
        "close_strength": float(close_strength),
        "close_min_keep": float(close_min_keep),
        "limiter_mean": float(np.mean(limiter)),
        "limiter_p05": float(np.quantile(limiter, 0.05)),
        "gate_mean": float(np.mean(gate)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "hdr_restore_mean": float(np.mean(hdr)),
    }
    masks = {"base_gate": base_gate, "limiter": limiter, "gate": gate, **close_masks}
    return out.astype(np.float32, copy=False), stats, masks


def write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply PL-hair RGB detail blend with flat/spill close gate.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--close-strength", type=float, default=0.65)
    parser.add_argument("--close-min-keep", type=float, default=0.16)
    parser.add_argument("--dark-low", type=float, default=0.020)
    parser.add_argument("--dark-high", type=float, default=0.60)
    parser.add_argument("--dark-transition", type=float, default=0.10)
    parser.add_argument("--skin-inhibit", type=float, default=0.92)
    parser.add_argument("--skin-proximity-sigma", type=float, default=24.0)
    parser.add_argument("--skin-proximity-threshold", type=float, default=0.080)
    parser.add_argument("--skin-proximity-transition", type=float, default=0.035)
    parser.add_argument("--texture-threshold", type=float, default=0.018)
    parser.add_argument("--texture-transition", type=float, default=0.010)
    parser.add_argument("--coherence-threshold", type=float, default=0.55)
    parser.add_argument("--coherence-transition", type=float, default=0.12)
    parser.add_argument("--coherence-energy-threshold", type=float, default=0.010)
    parser.add_argument("--coherence-energy-transition", type=float, default=0.005)
    parser.add_argument("--mask-guide-dark-high", type=float, default=0.46)
    parser.add_argument("--mask-guide-transition", type=float, default=0.10)
    parser.add_argument("--mask-guide-weight", type=float, default=0.0)
    parser.add_argument("--mask-blur-sigma", type=float, default=0.45)
    parser.add_argument("--impulse-size", type=int, default=3)
    parser.add_argument("--flat-detail-threshold", type=float, default=0.024)
    parser.add_argument("--flat-detail-transition", type=float, default=0.010)
    parser.add_argument("--flat-edge-threshold", type=float, default=0.030)
    parser.add_argument("--flat-edge-transition", type=float, default=0.015)
    parser.add_argument("--luma-risk-threshold", type=float, default=0.0003)
    parser.add_argument("--luma-risk-transition", type=float, default=0.0045)
    parser.add_argument("--chroma-risk-threshold", type=float, default=0.0003)
    parser.add_argument("--chroma-risk-transition", type=float, default=0.0045)
    parser.add_argument("--impulse-risk-threshold", type=float, default=0.0002)
    parser.add_argument("--impulse-risk-transition", type=float, default=0.006)
    parser.add_argument("--benefit-threshold", type=float, default=0.0002)
    parser.add_argument("--benefit-transition", type=float, default=0.006)
    parser.add_argument("--close-risk-weight", type=float, default=0.72)
    parser.add_argument("--close-flat-weight", type=float, default=0.55)
    parser.add_argument("--close-benefit-weight", type=float, default=0.55)
    parser.add_argument("--close-blur-sigma", type=float, default=0.75)
    parser.add_argument("--hdr-peak-threshold", type=float, default=0.82)
    parser.add_argument("--hdr-transition", type=float, default=0.25)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-masks", action="store_true")
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    detail_path = Path(args.detail).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_plhair_close_gate"

    reference = read_image(reference_path)
    base = read_image(base_path)
    detail = read_image(detail_path)
    params = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}
    out, stats, masks = apply_plhair_close_gate(
        reference,
        base,
        detail,
        close_strength=args.close_strength,
        close_min_keep=args.close_min_keep,
        hdr_peak_threshold=args.hdr_peak_threshold,
        hdr_transition=args.hdr_transition,
        dark_low=args.dark_low,
        dark_high=args.dark_high,
        dark_transition=args.dark_transition,
        skin_inhibit=args.skin_inhibit,
        skin_proximity_sigma=args.skin_proximity_sigma,
        skin_proximity_threshold=args.skin_proximity_threshold,
        skin_proximity_transition=args.skin_proximity_transition,
        texture_threshold=args.texture_threshold,
        texture_transition=args.texture_transition,
        coherence_threshold=args.coherence_threshold,
        coherence_transition=args.coherence_transition,
        coherence_energy_threshold=args.coherence_energy_threshold,
        coherence_energy_transition=args.coherence_energy_transition,
        mask_guide_dark_high=args.mask_guide_dark_high,
        mask_guide_transition=args.mask_guide_transition,
        mask_guide_weight=args.mask_guide_weight,
        mask_blur_sigma=args.mask_blur_sigma,
        impulse_size=args.impulse_size,
        flat_detail_threshold=args.flat_detail_threshold,
        flat_detail_transition=args.flat_detail_transition,
        flat_edge_threshold=args.flat_edge_threshold,
        flat_edge_transition=args.flat_edge_transition,
        luma_risk_threshold=args.luma_risk_threshold,
        luma_risk_transition=args.luma_risk_transition,
        chroma_risk_threshold=args.chroma_risk_threshold,
        chroma_risk_transition=args.chroma_risk_transition,
        impulse_risk_threshold=args.impulse_risk_threshold,
        impulse_risk_transition=args.impulse_risk_transition,
        benefit_threshold=args.benefit_threshold,
        benefit_transition=args.benefit_transition,
        close_risk_weight=args.close_risk_weight,
        close_flat_weight=args.close_flat_weight,
        close_benefit_weight=args.close_benefit_weight,
        close_blur_sigma=args.close_blur_sigma,
    )

    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    outputs = {"exr": str(exr_path)}
    if not args.no_preview:
        Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
        outputs["preview"] = str(preview_path)
    if not args.no_masks:
        for key in ("base_gate", "close", "limiter", "gate", "risk", "flat_spill", "benefit", "structure"):
            mask_path = out_dir / f"{name}_{key}.png"
            write_mask(mask_path, masks[key])
            outputs[key] = str(mask_path)
    meta = {
        "inputs": {"reference": str(reference_path), "base": str(base_path), "detail": str(detail_path)},
        "outputs": outputs,
        "params": params,
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


if __name__ == "__main__":
    main()
