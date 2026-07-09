"""Selectively restore PL luma hair detail without adopting PL color/tone.

This is a conservative hair-detail probe. It uses PhotoLab only as a local
detail reference, matches its low-frequency luma to the HDR-safe base, and
transfers signed mid/fine luma detail only where a hair-like mask and a PL
structure gate agree. Chroma, global tone, and HDR peaks stay from the base.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_hair_region_luma_blend import hair_mask
from apply_luma_tail_speckle_filter import sigmoid01
from apply_region_aware_luma_cleanup import make_coherent_structure_mask, make_texture_mask
from perfect_nr_detail_guard import write_exr, write_tiff
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


def smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def luma_detail_bands(
    y: np.ndarray,
    *,
    low_sigma: float,
    mid_sigma: float,
    fine_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = gaussian_filter(y, sigma=float(low_sigma), mode="reflect")
    mid_low = gaussian_filter(y, sigma=float(mid_sigma), mode="reflect")
    fine_low = gaussian_filter(y, sigma=float(fine_sigma), mode="reflect")
    mid = mid_low - low
    fine = y - fine_low
    return low.astype(np.float32, copy=False), mid.astype(np.float32, copy=False), fine.astype(np.float32, copy=False)


def transfer_selective_pl_hair_detail(
    reference: np.ndarray,
    base: np.ndarray,
    pl: np.ndarray,
    *,
    low_sigma: float,
    mid_sigma: float,
    fine_sigma: float,
    mid_weight: float,
    fine_weight: float,
    detail_limit: float,
    local_luma_frac: float,
    pl_excess_threshold: float,
    pl_excess_transition: float,
    base_low_detail_threshold: float,
    base_low_detail_transition: float,
    pl_texture_threshold: float,
    pl_texture_transition: float,
    pl_coherence_threshold: float,
    pl_coherence_transition: float,
    pl_coherence_energy_threshold: float,
    pl_coherence_energy_transition: float,
    blend_strength: float,
    gate_blur_sigma: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
    **mask_kwargs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    base_rgb = np.clip(_safe_rgb(base), 0.0, None)
    base_d = display(base_rgb)
    pl_d = np.clip(_safe_rgb(pl), 0.0, 1.0)
    if base_d.shape != pl_d.shape:
        raise ValueError(f"shape mismatch base={base_d.shape} pl={pl_d.shape}")

    base_y = luma(base_d, LUMA_SRGB)
    pl_y = luma(pl_d, LUMA_SRGB)

    base_low, base_mid, base_fine = luma_detail_bands(
        base_y,
        low_sigma=low_sigma,
        mid_sigma=mid_sigma,
        fine_sigma=fine_sigma,
    )
    pl_low, pl_mid_raw, pl_fine_raw = luma_detail_bands(
        pl_y,
        low_sigma=low_sigma,
        mid_sigma=mid_sigma,
        fine_sigma=fine_sigma,
    )

    # Match PhotoLab's slow luma component to the base before reading bands.
    pl_matched = np.clip(pl_y - pl_low + base_low, 0.0, 1.0)
    _, pl_mid, pl_fine = luma_detail_bands(
        pl_matched,
        low_sigma=low_sigma,
        mid_sigma=mid_sigma,
        fine_sigma=fine_sigma,
    )

    signed_mid = pl_mid - base_mid
    signed_fine = pl_fine - base_fine
    detail = np.clip(
        signed_mid * float(mid_weight) + signed_fine * float(fine_weight),
        -float(detail_limit),
        float(detail_limit),
    )
    local_limit = np.maximum(base_y, 0.030) * float(local_luma_frac)
    detail = np.clip(detail, -local_limit, local_limit)

    hair, hair_stats = hair_mask(reference, base, None, **mask_kwargs)
    base_energy = np.abs(base_mid) + np.abs(base_fine)
    pl_energy = np.abs(pl_mid_raw) + np.abs(pl_fine_raw)
    excess = np.maximum(pl_energy - base_energy, 0.0)
    excess_gate = sigmoid01(
        (excess - float(pl_excess_threshold)) / max(float(pl_excess_transition), 1.0e-6)
    )
    low_detail_gate = sigmoid01(
        (float(base_low_detail_threshold) - base_energy) / max(float(base_low_detail_transition), 1.0e-6)
    )
    pl_texture = make_texture_mask(
        pl_d,
        texture_threshold=float(pl_texture_threshold),
        texture_transition=float(pl_texture_transition),
    )
    pl_coherent = make_coherent_structure_mask(
        pl_d,
        coherence_threshold=float(pl_coherence_threshold),
        coherence_transition=float(pl_coherence_transition),
        energy_threshold=float(pl_coherence_energy_threshold),
        energy_transition=float(pl_coherence_energy_transition),
    )
    pl_structure = np.clip(np.maximum(pl_texture, pl_coherent), 0.0, 1.0)

    gate = np.clip(hair * pl_structure * excess_gate * low_detail_gate * float(blend_strength), 0.0, 1.0)
    if gate_blur_sigma > 0:
        gate = gaussian_filter(gate, sigma=float(gate_blur_sigma), mode="reflect")
        gate = np.clip(gate, 0.0, 1.0)

    out_y = np.clip(base_y + detail * gate, 0.0, 1.0)
    chroma = base_d / np.maximum(base_y[..., None], 1.0e-6)
    out_display = np.clip(chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)

    base_peak = np.max(base_rgb, axis=2)
    hdr = smoothstep01((base_peak - float(hdr_peak_threshold)) / max(float(hdr_transition), 1.0e-6))
    out = out * (1.0 - hdr[..., None]) + base_rgb * hdr[..., None]

    applied = detail * gate
    stats = {
        "low_sigma": float(low_sigma),
        "mid_sigma": float(mid_sigma),
        "fine_sigma": float(fine_sigma),
        "mid_weight": float(mid_weight),
        "fine_weight": float(fine_weight),
        "detail_limit": float(detail_limit),
        "local_luma_frac": float(local_luma_frac),
        "blend_strength": float(blend_strength),
        "gate_mean": float(np.mean(gate)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "applied_abs_mean": float(np.mean(np.abs(applied))),
        "applied_abs_p95": float(np.quantile(np.abs(applied), 0.95)),
        "applied_abs_p99": float(np.quantile(np.abs(applied), 0.99)),
        "pl_excess_mean": float(np.mean(excess_gate)),
        "base_low_detail_mean": float(np.mean(low_detail_gate)),
        "pl_texture_mean": float(np.mean(pl_texture)),
        "pl_coherent_mean": float(np.mean(pl_coherent)),
        "hdr_restore_mean": float(np.mean(hdr)),
        **hair_stats,
    }
    return out.astype(np.float32, copy=False), gate.astype(np.float32, copy=False), applied, stats


def crop(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, min(w - size, int(x) - size // 2))
    y0 = max(0, min(h - size, int(y) - size // 2))
    return image[y0 : y0 + size, x0 : x0 + size]


def render_compare(path: Path, panels: list[tuple[str, np.ndarray]], *, scale: int) -> None:
    previews = []
    for _, image in panels:
        preview = Image.fromarray(make_preview(image, exposure=1.0, tone="reinhard"))
        if scale != 1:
            preview = preview.resize((preview.width * scale, preview.height * scale), Image.Resampling.NEAREST)
        previews.append(preview)
    width, height = previews[0].size
    label_h = 26
    canvas = Image.new("RGB", (width * len(previews), height + label_h), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    for i, ((label, _), preview) in enumerate(zip(panels, previews, strict=True)):
        canvas.paste(preview, (i * width, label_h))
        draw.text((i * width + 6, 6), label, fill=(235, 235, 235))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore only selective PL luma detail into hair-like regions.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--pl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--low-sigma", type=float, default=9.0)
    parser.add_argument("--mid-sigma", type=float, default=1.3)
    parser.add_argument("--fine-sigma", type=float, default=0.70)
    parser.add_argument("--mid-weight", type=float, default=0.65)
    parser.add_argument("--fine-weight", type=float, default=0.22)
    parser.add_argument("--detail-limit", type=float, default=0.055)
    parser.add_argument("--local-luma-frac", type=float, default=0.33)
    parser.add_argument("--pl-excess-threshold", type=float, default=0.006)
    parser.add_argument("--pl-excess-transition", type=float, default=0.010)
    parser.add_argument("--base-low-detail-threshold", type=float, default=0.032)
    parser.add_argument("--base-low-detail-transition", type=float, default=0.020)
    parser.add_argument("--pl-texture-threshold", type=float, default=0.004)
    parser.add_argument("--pl-texture-transition", type=float, default=0.012)
    parser.add_argument("--pl-coherence-threshold", type=float, default=0.26)
    parser.add_argument("--pl-coherence-transition", type=float, default=0.18)
    parser.add_argument("--pl-coherence-energy-threshold", type=float, default=0.004)
    parser.add_argument("--pl-coherence-energy-transition", type=float, default=0.006)
    parser.add_argument("--blend-strength", type=float, default=0.90)
    parser.add_argument("--gate-blur-sigma", type=float, default=1.4)
    parser.add_argument("--dark-low", type=float, default=0.020)
    parser.add_argument("--dark-high", type=float, default=0.62)
    parser.add_argument("--dark-transition", type=float, default=0.10)
    parser.add_argument("--skin-inhibit", type=float, default=0.95)
    parser.add_argument("--skin-proximity-sigma", type=float, default=130.0)
    parser.add_argument("--skin-proximity-threshold", type=float, default=0.004)
    parser.add_argument("--skin-proximity-transition", type=float, default=0.020)
    parser.add_argument("--texture-threshold", type=float, default=0.004)
    parser.add_argument("--texture-transition", type=float, default=0.012)
    parser.add_argument("--coherence-threshold", type=float, default=0.30)
    parser.add_argument("--coherence-transition", type=float, default=0.18)
    parser.add_argument("--coherence-energy-threshold", type=float, default=0.006)
    parser.add_argument("--coherence-energy-transition", type=float, default=0.006)
    parser.add_argument("--mask-guide-dark-high", type=float, default=0.46)
    parser.add_argument("--mask-guide-transition", type=float, default=0.10)
    parser.add_argument("--mask-guide-weight", type=float, default=0.0)
    parser.add_argument("--mask-blur-sigma", type=float, default=2.6)
    parser.add_argument("--hdr-peak-threshold", type=float, default=0.82)
    parser.add_argument("--hdr-transition", type=float, default=0.25)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--crop-scale", type=int, default=2)
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    pl_path = Path(args.pl).expanduser()
    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_selective_pl_hair_detail"

    reference = read_image(reference_path)
    base = read_image(base_path)
    pl = read_image(pl_path)
    out, gate, applied, stats = transfer_selective_pl_hair_detail(
        reference,
        base,
        pl,
        low_sigma=args.low_sigma,
        mid_sigma=args.mid_sigma,
        fine_sigma=args.fine_sigma,
        mid_weight=args.mid_weight,
        fine_weight=args.fine_weight,
        detail_limit=args.detail_limit,
        local_luma_frac=args.local_luma_frac,
        pl_excess_threshold=args.pl_excess_threshold,
        pl_excess_transition=args.pl_excess_transition,
        base_low_detail_threshold=args.base_low_detail_threshold,
        base_low_detail_transition=args.base_low_detail_transition,
        pl_texture_threshold=args.pl_texture_threshold,
        pl_texture_transition=args.pl_texture_transition,
        pl_coherence_threshold=args.pl_coherence_threshold,
        pl_coherence_transition=args.pl_coherence_transition,
        pl_coherence_energy_threshold=args.pl_coherence_energy_threshold,
        pl_coherence_energy_transition=args.pl_coherence_energy_transition,
        blend_strength=args.blend_strength,
        gate_blur_sigma=args.gate_blur_sigma,
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
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    detail_path = out_dir / f"{name}_applied_detail.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    detail_vis = np.clip(applied / max(args.detail_limit, 1.0e-6) * 127.0 + 128.0, 0, 255).astype(np.uint8)
    Image.fromarray(detail_vis).save(detail_path)

    panels = [("reference", reference), ("base", base), ("selective", out), ("PL_XD", pl)]
    for roi_name, (x, y) in {
        "bangs": (2030, 1510),
        "top_hair": (2420, 1040),
        "face_hair": (2120, 1260),
    }.items():
        render_compare(
            crop_dir / f"{name}_{roi_name}_compare.png",
            [(label, crop(image, x, y, args.crop_size)) for label, image in panels],
            scale=args.crop_scale,
        )

    meta = {
        "reference": str(reference_path),
        "base": str(base_path),
        "pl": str(pl_path),
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
            "applied_detail": str(detail_path),
        },
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


if __name__ == "__main__":
    main()
