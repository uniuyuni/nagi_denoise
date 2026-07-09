"""Region-aware luma cleanup for aggressive rebuild outputs.

This keeps the luma structure recovered by ``apply_structure_luma_graft.py`` in
textured areas, while blending skin-like flat regions toward a cleaner luma
target. Chroma is kept from the rebuild image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, uniform_filter

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from apply_guided_luma_smoother import guided_filter_gray
from apply_luma_tail_speckle_filter import sigmoid01
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import image_stats, make_preview, read_image


PRESETS = {
    "skin_base": {
        "texture_source": "reference",
        "texture_threshold": 0.005,
        "texture_transition": 0.012,
        "base_texture_threshold": 0.018,
        "base_texture_transition": 0.014,
        "base_texture_weight": 0.85,
        "agreement_texture_weight": 0.85,
        "skin_weight": 1.25,
        "skin_mask_blur": 1.4,
        "flat_mask_blur": 1.2,
        "smooth_target_weight": 0.0,
        "base_target_weight": 1.0,
        "smooth_radius": 3,
        "smooth_eps": 0.003,
        "smooth_guide_sigma": 0.8,
        "non_skin_flat_weight": 0.0,
        "non_skin_dark_threshold": 1.0,
        "non_skin_dark_transition": 0.08,
        "non_skin_low_sat_threshold": 1.0,
        "non_skin_low_sat_transition": 0.15,
        "coherent_protect_weight": 0.0,
        "coherence_threshold": 0.42,
        "coherence_transition": 0.18,
        "coherence_energy_threshold": 0.006,
        "coherence_energy_transition": 0.006,
    },
    "hybrid_clean": {
        "texture_source": "reference",
        "texture_threshold": 0.005,
        "texture_transition": 0.012,
        "base_texture_threshold": 0.018,
        "base_texture_transition": 0.014,
        "base_texture_weight": 0.85,
        "agreement_texture_weight": 0.85,
        "skin_weight": 1.55,
        "skin_mask_blur": 1.4,
        "flat_mask_blur": 1.2,
        "smooth_target_weight": 0.65,
        "base_target_weight": 0.35,
        "smooth_radius": 3,
        "smooth_eps": 0.003,
        "smooth_guide_sigma": 0.8,
        "non_skin_flat_weight": 0.0,
        "non_skin_dark_threshold": 1.0,
        "non_skin_dark_transition": 0.08,
        "non_skin_low_sat_threshold": 1.0,
        "non_skin_low_sat_transition": 0.15,
        "coherent_protect_weight": 0.0,
        "coherence_threshold": 0.42,
        "coherence_transition": 0.18,
        "coherence_energy_threshold": 0.006,
        "coherence_energy_transition": 0.006,
    },
    "flat_protect": {
        "texture_source": "base",
        "texture_threshold": 0.030,
        "texture_transition": 0.020,
        "base_texture_threshold": 0.030,
        "base_texture_transition": 0.020,
        "base_texture_weight": 1.0,
        "agreement_texture_weight": 0.0,
        "skin_weight": 1.20,
        "skin_mask_blur": 1.4,
        "flat_mask_blur": 1.2,
        "smooth_target_weight": 0.65,
        "base_target_weight": 0.35,
        "smooth_radius": 3,
        "smooth_eps": 0.003,
        "smooth_guide_sigma": 0.8,
        "non_skin_flat_weight": 1.0,
        "non_skin_dark_threshold": 1.0,
        "non_skin_dark_transition": 0.08,
        "non_skin_low_sat_threshold": 1.0,
        "non_skin_low_sat_transition": 0.15,
        "coherent_protect_weight": 0.0,
        "coherence_threshold": 0.42,
        "coherence_transition": 0.18,
        "coherence_energy_threshold": 0.006,
        "coherence_energy_transition": 0.006,
    },
    "adaptive_flat": {
        "texture_source": "agreement",
        "texture_threshold": 0.006,
        "texture_transition": 0.012,
        "base_texture_threshold": 0.018,
        "base_texture_transition": 0.014,
        "base_texture_weight": 0.80,
        "agreement_texture_weight": 1.00,
        "skin_weight": 1.20,
        "skin_mask_blur": 1.4,
        "flat_mask_blur": 1.2,
        "smooth_target_weight": 0.65,
        "base_target_weight": 0.35,
        "smooth_radius": 3,
        "smooth_eps": 0.003,
        "smooth_guide_sigma": 0.8,
        "non_skin_flat_weight": 0.85,
        "non_skin_dark_threshold": 1.0,
        "non_skin_dark_transition": 0.08,
        "non_skin_low_sat_threshold": 1.0,
        "non_skin_low_sat_transition": 0.15,
        "coherent_protect_weight": 0.0,
        "coherence_threshold": 0.42,
        "coherence_transition": 0.18,
        "coherence_energy_threshold": 0.006,
        "coherence_energy_transition": 0.006,
    },
    "adaptive_sky": {
        "texture_source": "agreement",
        "texture_threshold": 0.006,
        "texture_transition": 0.012,
        "base_texture_threshold": 0.018,
        "base_texture_transition": 0.014,
        "base_texture_weight": 0.80,
        "agreement_texture_weight": 1.00,
        "skin_weight": 1.20,
        "skin_mask_blur": 1.4,
        "flat_mask_blur": 1.2,
        "smooth_target_weight": 0.65,
        "base_target_weight": 0.35,
        "smooth_radius": 3,
        "smooth_eps": 0.003,
        "smooth_guide_sigma": 0.8,
        "non_skin_flat_weight": 0.95,
        "non_skin_dark_threshold": 0.34,
        "non_skin_dark_transition": 0.08,
        "non_skin_low_sat_threshold": 0.42,
        "non_skin_low_sat_transition": 0.12,
        "coherent_protect_weight": 0.0,
        "coherence_threshold": 0.42,
        "coherence_transition": 0.18,
        "coherence_energy_threshold": 0.006,
        "coherence_energy_transition": 0.006,
    },
    "adaptive_coherent": {
        "texture_source": "agreement",
        "texture_threshold": 0.006,
        "texture_transition": 0.012,
        "base_texture_threshold": 0.018,
        "base_texture_transition": 0.014,
        "base_texture_weight": 0.80,
        "agreement_texture_weight": 1.00,
        "skin_weight": 1.20,
        "skin_mask_blur": 1.4,
        "flat_mask_blur": 1.2,
        "smooth_target_weight": 0.65,
        "base_target_weight": 0.35,
        "smooth_radius": 3,
        "smooth_eps": 0.003,
        "smooth_guide_sigma": 0.8,
        "non_skin_flat_weight": 0.95,
        "non_skin_dark_threshold": 0.34,
        "non_skin_dark_transition": 0.08,
        "non_skin_low_sat_threshold": 0.42,
        "non_skin_low_sat_transition": 0.12,
        "coherent_protect_weight": 0.85,
        "coherence_threshold": 0.40,
        "coherence_transition": 0.18,
        "coherence_energy_threshold": 0.006,
        "coherence_energy_transition": 0.006,
    },
}


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def _display(image: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb_np(np.clip(_safe_rgb(image), 0.0, None)), 0.0, 1.0)


def make_texture_mask(
    reference_display: np.ndarray,
    *,
    texture_threshold: float,
    texture_transition: float,
) -> np.ndarray:
    ref_y = luma(reference_display, LUMA_SRGB)
    structure = gaussian_filter(ref_y, sigma=0.7, mode="reflect")
    edge = gaussian_gradient_magnitude(structure, sigma=0.8, mode="reflect")
    contrast = np.abs(structure - uniform_filter(structure, size=11, mode="reflect"))
    texture = np.maximum(
        smoothstep((edge - float(texture_threshold)) / max(float(texture_transition), 1.0e-6)),
        smoothstep((contrast - float(texture_threshold)) / max(float(texture_transition), 1.0e-6)),
    )
    return gaussian_filter(np.clip(texture, 0.0, 1.0).astype(np.float32, copy=False), sigma=0.7, mode="reflect")


def make_skin_mask(base_display: np.ndarray, *, blur_sigma: float) -> np.ndarray:
    base_y = luma(base_display, LUMA_SRGB)
    r = base_display[..., 0]
    g = base_display[..., 1]
    b = base_display[..., 2]
    mx = np.max(base_display, axis=2)
    mn = np.min(base_display, axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0e-6)
    skin = (
        sigmoid01((r - g - 0.004) / 0.020)
        * sigmoid01((g - b + 0.018) / 0.035)
        * sigmoid01((base_y - 0.16) / 0.045)
        * sigmoid01((0.86 - base_y) / 0.080)
        * sigmoid01((sat - 0.035) / 0.035)
        * sigmoid01((0.60 - sat) / 0.140)
    )
    if blur_sigma > 0:
        skin = gaussian_filter(np.clip(skin, 0.0, 1.0).astype(np.float32, copy=False), sigma=float(blur_sigma), mode="reflect")
    return np.clip(skin, 0.0, 1.0).astype(np.float32, copy=False)


def make_coherent_structure_mask(
    reference_display: np.ndarray,
    *,
    coherence_threshold: float,
    coherence_transition: float,
    energy_threshold: float,
    energy_transition: float,
) -> np.ndarray:
    ref_y = luma(reference_display, LUMA_SRGB)
    structure = gaussian_filter(ref_y, sigma=0.8, mode="reflect")
    gy, gx = np.gradient(structure)
    jxx = gaussian_filter(gx * gx, sigma=2.0, mode="reflect")
    jyy = gaussian_filter(gy * gy, sigma=2.0, mode="reflect")
    jxy = gaussian_filter(gx * gy, sigma=2.0, mode="reflect")
    energy = jxx + jyy
    coherence = np.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy) / np.maximum(energy, 1.0e-8)
    coherence_gate = sigmoid01(
        (coherence - float(coherence_threshold)) / max(float(coherence_transition), 1.0e-6)
    )
    energy_gate = sigmoid01((energy - float(energy_threshold)) / max(float(energy_transition), 1.0e-6))
    mask = coherence_gate * energy_gate
    return gaussian_filter(np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False), sigma=0.8, mode="reflect")


def apply_region_aware_luma_cleanup(
    reference: np.ndarray,
    base: np.ndarray,
    rebuild: np.ndarray,
    *,
    texture_source: str,
    texture_threshold: float,
    texture_transition: float,
    base_texture_threshold: float,
    base_texture_transition: float,
    base_texture_weight: float,
    agreement_texture_weight: float,
    skin_weight: float,
    skin_mask_blur: float,
    flat_mask_blur: float,
    smooth_target_weight: float,
    base_target_weight: float,
    smooth_radius: int,
    smooth_eps: float,
    smooth_guide_sigma: float,
    non_skin_flat_weight: float,
    non_skin_dark_threshold: float,
    non_skin_dark_transition: float,
    non_skin_low_sat_threshold: float,
    non_skin_low_sat_transition: float,
    coherent_protect_weight: float,
    coherence_threshold: float,
    coherence_transition: float,
    coherence_energy_threshold: float,
    coherence_energy_transition: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    ref_display = _display(reference)
    base_display = _display(base)
    rebuild_display = _display(rebuild)
    if ref_display.shape[:2] != rebuild_display.shape[:2] or base_display.shape[:2] != rebuild_display.shape[:2]:
        raise ValueError(
            f"shape mismatch: reference={ref_display.shape}, base={base_display.shape}, rebuild={rebuild_display.shape}"
        )

    texture_inputs = {
        "reference": ref_display,
        "base": base_display,
        "rebuild": rebuild_display,
    }
    if texture_source not in {*texture_inputs, "agreement"}:
        raise ValueError(f"unknown texture_source: {texture_source!r}")
    if texture_source == "agreement":
        reference_texture = make_texture_mask(
            ref_display,
            texture_threshold=texture_threshold,
            texture_transition=texture_transition,
        )
        base_texture = make_texture_mask(
            base_display,
            texture_threshold=base_texture_threshold,
            texture_transition=base_texture_transition,
        )
        texture = np.maximum(
            base_texture * float(base_texture_weight),
            reference_texture * base_texture * float(agreement_texture_weight),
        )
        texture = np.clip(texture, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        texture = make_texture_mask(
            texture_inputs[texture_source],
            texture_threshold=texture_threshold,
            texture_transition=texture_transition,
        )
        reference_texture = texture if texture_source == "reference" else None
        base_texture = texture if texture_source == "base" else None
    skin = make_skin_mask(base_display, blur_sigma=skin_mask_blur)
    flat = np.clip(1.0 - texture, 0.0, 1.0).astype(np.float32, copy=False)
    skin_flat = np.clip(skin * flat, 0.0, 1.0)
    if flat_mask_blur > 0:
        skin_flat = gaussian_filter(skin_flat, sigma=float(flat_mask_blur), mode="reflect")

    base_y = luma(base_display, LUMA_SRGB)
    rebuild_y = luma(rebuild_display, LUMA_SRGB)
    smooth_base_y = guided_filter_gray(
        gaussian_filter(base_y, sigma=float(smooth_guide_sigma), mode="reflect"),
        base_y,
        radius=int(smooth_radius),
        eps=float(smooth_eps),
    )
    target_sum = max(float(smooth_target_weight) + float(base_target_weight), 1.0e-6)
    clean_target_y = (
        smooth_base_y * float(smooth_target_weight) + base_y * float(base_target_weight)
    ) / target_sum

    skin_blend = np.clip(skin_flat * float(skin_weight), 0.0, 1.0)
    out_y = rebuild_y * (1.0 - skin_blend) + clean_target_y * skin_blend
    if non_skin_flat_weight > 0:
        coherent_structure = make_coherent_structure_mask(
            ref_display,
            coherence_threshold=coherence_threshold,
            coherence_transition=coherence_transition,
            energy_threshold=coherence_energy_threshold,
            energy_transition=coherence_energy_transition,
        )
        base_y_for_gate = base_y
        mx = np.max(base_display, axis=2)
        mn = np.min(base_display, axis=2)
        sat = (mx - mn) / np.maximum(mx, 1.0e-6)
        dark_gate = sigmoid01(
            (float(non_skin_dark_threshold) - base_y_for_gate) / max(float(non_skin_dark_transition), 1.0e-6)
        )
        low_sat_gate = sigmoid01(
            (float(non_skin_low_sat_threshold) - sat) / max(float(non_skin_low_sat_transition), 1.0e-6)
        )
        flat_region_gate = dark_gate * low_sat_gate
        coherent_protect = 1.0 - np.clip(coherent_structure * float(coherent_protect_weight), 0.0, 1.0)
        non_skin_flat = np.clip(
            (1.0 - skin) * flat * flat_region_gate * coherent_protect * float(non_skin_flat_weight), 0.0, 1.0
        )
        out_y = out_y * (1.0 - non_skin_flat) + clean_target_y * non_skin_flat
    else:
        non_skin_flat = np.zeros_like(out_y, dtype=np.float32)
        coherent_structure = np.zeros_like(out_y, dtype=np.float32)

    rebuild_chroma = rebuild_display / np.maximum(rebuild_y[..., None], 1.0e-6)
    out_display = np.clip(rebuild_chroma * np.maximum(out_y[..., None], 0.0), 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    masks = {
        "texture": texture,
        "skin": skin,
        "skin_flat": skin_flat,
        "skin_blend": skin_blend,
        "non_skin_flat": non_skin_flat,
        "coherent_structure": coherent_structure,
    }
    if texture_source == "agreement":
        masks["reference_texture"] = reference_texture
        masks["base_texture"] = base_texture
    stats = {
        "texture_mean": float(np.mean(texture)),
        "texture_source": str(texture_source),
        "texture_p90": float(np.quantile(texture, 0.90)),
        "base_texture_threshold": float(base_texture_threshold),
        "base_texture_weight": float(base_texture_weight),
        "agreement_texture_weight": float(agreement_texture_weight),
        "skin_mean": float(np.mean(skin)),
        "skin_p90": float(np.quantile(skin, 0.90)),
        "skin_flat_mean": float(np.mean(skin_flat)),
        "skin_blend_mean": float(np.mean(skin_blend)),
        "skin_blend_p95": float(np.quantile(skin_blend, 0.95)),
        "skin_blend_p99": float(np.quantile(skin_blend, 0.99)),
        "non_skin_flat_mean": float(np.mean(non_skin_flat)),
        "non_skin_dark_threshold": float(non_skin_dark_threshold),
        "non_skin_low_sat_threshold": float(non_skin_low_sat_threshold),
        "coherent_structure_mean": float(np.mean(coherent_structure)),
        "coherent_structure_p90": float(np.quantile(coherent_structure, 0.90)),
        "coherent_protect_weight": float(coherent_protect_weight),
        "delta_vs_rebuild_abs_mean": float(np.mean(np.abs(out_y - rebuild_y))),
        "delta_vs_rebuild_abs_p95": float(np.quantile(np.abs(out_y - rebuild_y), 0.95)),
        "delta_vs_rebuild_abs_p99": float(np.quantile(np.abs(out_y - rebuild_y), 0.99)),
    }
    return out, stats, masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply region-aware luma cleanup to a rebuild output.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--rebuild", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="hybrid_clean")
    for key, default_value in next(iter(PRESETS.values())).items():
        if isinstance(default_value, str):
            parser.add_argument(f"--{key.replace('_', '-')}", default=None)
        else:
            parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=None)
    args = parser.parse_args()

    params = dict(PRESETS[args.preset])
    for key in params:
        value = getattr(args, key)
        if value is not None:
            if isinstance(params[key], str):
                params[key] = str(value)
            else:
                params[key] = int(value) if key == "smooth_radius" else float(value)

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    rebuild_path = Path(args.rebuild).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{rebuild_path.stem}_region_luma_cleanup_{args.preset}"

    out, stats, masks = apply_region_aware_luma_cleanup(
        read_image(reference_path),
        read_image(base_path),
        read_image(rebuild_path),
        **params,
    )
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    mask_outputs = {}
    for mask_name, mask in masks.items():
        mask_path = out_dir / f"{name}_{mask_name}.png"
        Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(mask_path)
        mask_outputs[mask_name] = str(mask_path)

    meta = {
        "reference": str(reference_path),
        "base": str(base_path),
        "rebuild": str(rebuild_path),
        "outputs": {"exr": str(exr_path), "preview": str(preview_path), "masks": mask_outputs},
        "params": {"preset": args.preset, **params},
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


if __name__ == "__main__":
    main()
