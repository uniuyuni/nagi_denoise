"""Apply a learned flat-cleanup gate with region-aware strength.

The learned gate pilot is good at finding residual grain, but full-strength
application softens hair/branches/fabric. This stage keeps the gate, then
rescales it spatially:

* stronger in low-detail flat dark/low-saturation regions,
* weaker on coherent structures, texture, skin, and highlights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, uniform_filter

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from apply_region_aware_luma_cleanup import make_coherent_structure_mask, make_skin_mask, make_texture_mask
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image
from train_flat_cleanup_gate import SMOOTH_PARAMS


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def _display(image: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb_np(np.clip(_safe_rgb(image), 0.0, None)), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def _read_gate(path: Path, shape: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != (shape[1], shape[0]):
        raise ValueError(f"gate size mismatch: gate={img.size}, image={(shape[1], shape[0])}")
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def saturation(display: np.ndarray) -> np.ndarray:
    mx = np.max(display, axis=2)
    mn = np.min(display, axis=2)
    return ((mx - mn) / np.maximum(mx, 1.0e-6)).astype(np.float32, copy=False)


def build_strength_map(
    reference_linear: np.ndarray,
    current_linear: np.ndarray,
    *,
    base_strength: float,
    flat_boost: float,
    skin_strength: float,
    shadow_flat_boost: float,
    structure_suppress: float,
    highlight_suppress: float,
    flat_threshold: float,
    flat_transition: float,
    edge_threshold: float,
    edge_transition: float,
    min_strength: float,
    max_strength: float,
    blur_sigma: float,
    shadow_luma_threshold: float = 0.50,
    shadow_luma_transition: float = 0.18,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    reference = _display(reference_linear)
    current = _display(current_linear)
    current_y = luma(current, LUMA_SRGB)
    current_y_linear = luma(_safe_rgb(current_linear), LUMA_LINEAR)

    structure = gaussian_filter(current_y, sigma=1.0, mode="reflect")
    local_detail = np.abs(structure - gaussian_filter(structure, sigma=3.0, mode="reflect"))
    local_contrast = np.abs(structure - uniform_filter(structure, size=13, mode="reflect"))
    edge = gaussian_gradient_magnitude(structure, sigma=0.9, mode="reflect")
    flat = (
        sigmoid01((float(flat_threshold) - np.maximum(local_detail, local_contrast * 0.72)) / max(float(flat_transition), 1.0e-6))
        * sigmoid01((float(edge_threshold) - edge) / max(float(edge_transition), 1.0e-6))
    )

    sat = saturation(current)
    low_sat = sigmoid01((0.56 - sat) / 0.15)
    shadow = sigmoid01((float(shadow_luma_threshold) - current_y) / max(float(shadow_luma_transition), 1.0e-6))
    flat_target = np.clip(flat * (0.72 * low_sat + 0.28 * shadow), 0.0, 1.0).astype(np.float32, copy=False)

    ref_texture = make_texture_mask(reference, texture_threshold=0.007, texture_transition=0.013)
    cur_texture = make_texture_mask(current, texture_threshold=0.014, texture_transition=0.014)
    coherent = make_coherent_structure_mask(
        reference,
        coherence_threshold=0.38,
        coherence_transition=0.17,
        energy_threshold=0.0048,
        energy_transition=0.0055,
    )
    edge_structure = sigmoid01((edge - 0.018) / 0.010)
    texture = np.maximum(ref_texture * 0.85, cur_texture)
    structure_protect = np.clip(np.maximum.reduce([texture, coherent, edge_structure]), 0.0, 1.0)
    skin = make_skin_mask(current, blur_sigma=1.4)
    highlight = sigmoid01((current_y_linear - 0.92) / 0.24)
    shadow_flat = np.clip(flat * low_sat * shadow, 0.0, 1.0).astype(np.float32, copy=False)

    strength = (
        float(base_strength) * (1.0 - structure_protect * float(structure_suppress))
        + float(flat_boost) * flat_target * (1.0 - structure_protect * 0.78)
        + float(shadow_flat_boost) * shadow_flat * (1.0 - structure_protect * 0.86)
        + float(skin_strength) * skin * flat * (1.0 - structure_protect * 0.55)
    )
    strength *= 1.0 - highlight * float(highlight_suppress)
    strength = np.clip(strength, float(min_strength), float(max_strength)).astype(np.float32, copy=False)
    if blur_sigma > 0:
        strength = gaussian_filter(strength, sigma=float(blur_sigma), mode="reflect")
        strength = np.clip(strength, float(min_strength), float(max_strength)).astype(np.float32, copy=False)

    masks = {
        "strength": strength,
        "flat": flat.astype(np.float32, copy=False),
        "flat_target": flat_target,
        "shadow_flat": shadow_flat,
        "texture": texture.astype(np.float32, copy=False),
        "coherent": coherent.astype(np.float32, copy=False),
        "edge_structure": edge_structure.astype(np.float32, copy=False),
        "structure_protect": structure_protect.astype(np.float32, copy=False),
        "skin": skin.astype(np.float32, copy=False),
        "highlight": highlight.astype(np.float32, copy=False),
    }
    stats = {f"{name}_mean": float(np.mean(mask)) for name, mask in masks.items()}
    stats.update({f"{name}_p95": float(np.quantile(mask, 0.95)) for name, mask in masks.items()})
    return strength, stats, masks


def build_reopen_map(
    masks: dict[str, np.ndarray],
    *,
    reopen_strength: float = 0.0,
    reopen_shadow_weight: float = 0.85,
    reopen_structure_suppress: float = 1.0,
    reopen_min: float = 1.0,
    reopen_max: float = 1.45,
    reopen_shadow_threshold: float = 0.0,
    reopen_shadow_transition: float = 0.12,
) -> np.ndarray:
    if reopen_strength <= 0:
        return np.ones_like(masks["flat"], dtype=np.float32)
    shadow_gate = np.clip(
        (masks["shadow_flat"] - float(reopen_shadow_threshold)) / max(float(reopen_shadow_transition), 1.0e-6),
        0.0,
        1.0,
    )
    sky_flat = np.clip(
        masks["flat"]
        * shadow_gate
        * (masks["shadow_flat"] * float(reopen_shadow_weight) + masks["flat_target"] * (1.0 - float(reopen_shadow_weight))),
        0.0,
        1.0,
    )
    safe = np.clip(1.0 - masks["structure_protect"] * float(reopen_structure_suppress), 0.0, 1.0)
    reopen = 1.0 + float(reopen_strength) * sky_flat * safe
    return np.clip(reopen, float(reopen_min), float(reopen_max)).astype(np.float32, copy=False)


def build_effective_gate_limiter(
    masks: dict[str, np.ndarray],
    *,
    limiter_strength: float = 0.0,
    limiter_min: float = 0.35,
    limiter_flat_threshold: float = 0.62,
    limiter_flat_transition: float = 0.18,
    limiter_shadow_threshold: float = 0.32,
    limiter_shadow_transition: float = 0.18,
    limiter_structure_suppress: float = 1.0,
) -> np.ndarray:
    if limiter_strength <= 0:
        return np.ones_like(masks["flat"], dtype=np.float32)
    flat_conf = np.clip(
        (masks["flat"] - float(limiter_flat_threshold)) / max(float(limiter_flat_transition), 1.0e-6),
        0.0,
        1.0,
    )
    shadow_conf = np.clip(
        (masks["shadow_flat"] - float(limiter_shadow_threshold)) / max(float(limiter_shadow_transition), 1.0e-6),
        0.0,
        1.0,
    )
    safe = np.clip(1.0 - masks["structure_protect"] * float(limiter_structure_suppress), 0.0, 1.0)
    confidence = np.clip(flat_conf * shadow_conf * safe, 0.0, 1.0)
    limiter = 1.0 - float(limiter_strength) * (1.0 - confidence)
    return np.clip(limiter, float(limiter_min), 1.0).astype(np.float32, copy=False)


def apply_region_aware_gate(
    reference_linear: np.ndarray,
    current_linear: np.ndarray,
    gate: np.ndarray,
    smooth_params: dict[str, float] | None = None,
    **kwargs: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    reopen_params = {
        "reopen_strength": float(kwargs.pop("reopen_strength", 0.0)),
        "reopen_shadow_weight": float(kwargs.pop("reopen_shadow_weight", 0.85)),
        "reopen_structure_suppress": float(kwargs.pop("reopen_structure_suppress", 1.0)),
        "reopen_min": float(kwargs.pop("reopen_min", 1.0)),
        "reopen_max": float(kwargs.pop("reopen_max", 1.45)),
        "reopen_shadow_threshold": float(kwargs.pop("reopen_shadow_threshold", 0.0)),
        "reopen_shadow_transition": float(kwargs.pop("reopen_shadow_transition", 0.12)),
    }
    limiter_params = {
        "limiter_strength": float(kwargs.pop("limiter_strength", 0.0)),
        "limiter_min": float(kwargs.pop("limiter_min", 0.35)),
        "limiter_flat_threshold": float(kwargs.pop("limiter_flat_threshold", 0.62)),
        "limiter_flat_transition": float(kwargs.pop("limiter_flat_transition", 0.18)),
        "limiter_shadow_threshold": float(kwargs.pop("limiter_shadow_threshold", 0.32)),
        "limiter_shadow_transition": float(kwargs.pop("limiter_shadow_transition", 0.18)),
        "limiter_structure_suppress": float(kwargs.pop("limiter_structure_suppress", 1.0)),
    }
    strength, stats, masks = build_strength_map(reference_linear, current_linear, **kwargs)
    reopen = build_reopen_map(masks, **reopen_params)
    limiter = build_effective_gate_limiter(masks, **limiter_params)
    smooth = SMOOTH_PARAMS if smooth_params is None else smooth_params
    current = _display(current_linear)
    y = luma(current, LUMA_SRGB)
    chroma = current - y[..., None]
    y_low = gaussian_filter(y, sigma=float(smooth["luma_sigma"]), mode="reflect")
    chroma_low = gaussian_filter(chroma, sigma=(float(smooth["chroma_sigma"]), float(smooth["chroma_sigma"]), 0.0), mode="reflect")
    effective_gate = np.clip(gate * strength * reopen * limiter, 0.0, 1.0).astype(np.float32, copy=False)
    luma_blend = np.clip(effective_gate * float(smooth["luma_strength"]), 0.0, 1.0)
    chroma_blend = np.clip(effective_gate * float(smooth["chroma_strength"]), 0.0, 1.0)[..., None]
    out_y = y * (1.0 - luma_blend) + y_low * luma_blend
    out_chroma = chroma * (1.0 - chroma_blend) + chroma_low * chroma_blend
    out_display = np.clip(out_y[..., None] + out_chroma, 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)

    current_rgb = _safe_rgb(current_linear)
    peak = np.max(current_rgb, axis=2)
    hdr_restore = sigmoid01((peak - 0.92) / 0.24)
    out = (out * (1.0 - hdr_restore[..., None]) + current_rgb * hdr_restore[..., None]).astype(
        np.float32, copy=False
    )
    masks["reopen"] = reopen
    masks["limiter"] = limiter
    masks["effective_gate"] = effective_gate
    masks["hdr_restore"] = hdr_restore.astype(np.float32, copy=False)
    stats.update(
        {
            "gate_mean": float(np.mean(gate)),
            "gate_p95": float(np.quantile(gate, 0.95)),
            "reopen_mean": float(np.mean(reopen)),
            "reopen_p95": float(np.quantile(reopen, 0.95)),
            "limiter_mean": float(np.mean(limiter)),
            "limiter_p05": float(np.quantile(limiter, 0.05)),
            "limiter_p95": float(np.quantile(limiter, 0.95)),
            "effective_gate_mean": float(np.mean(effective_gate)),
            "effective_gate_p95": float(np.quantile(effective_gate, 0.95)),
            "hdr_restore_mean": float(np.mean(hdr_restore)),
            "hdr_restore_p95": float(np.quantile(hdr_restore, 0.95)),
        }
    )
    return out, stats, masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply region-aware learned flat gate.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--base-strength", type=float, default=0.30)
    parser.add_argument("--flat-boost", type=float, default=0.42)
    parser.add_argument("--skin-strength", type=float, default=0.14)
    parser.add_argument("--shadow-flat-boost", type=float, default=0.0)
    parser.add_argument("--shadow-luma-threshold", type=float, default=0.50)
    parser.add_argument("--shadow-luma-transition", type=float, default=0.18)
    parser.add_argument("--structure-suppress", type=float, default=0.86)
    parser.add_argument("--highlight-suppress", type=float, default=0.82)
    parser.add_argument("--flat-threshold", type=float, default=0.032)
    parser.add_argument("--flat-transition", type=float, default=0.014)
    parser.add_argument("--edge-threshold", type=float, default=0.028)
    parser.add_argument("--edge-transition", type=float, default=0.013)
    parser.add_argument("--min-strength", type=float, default=0.04)
    parser.add_argument("--max-strength", type=float, default=0.78)
    parser.add_argument("--blur-sigma", type=float, default=1.15)
    parser.add_argument("--luma-strength", type=float, default=SMOOTH_PARAMS["luma_strength"])
    parser.add_argument("--chroma-strength", type=float, default=SMOOTH_PARAMS["chroma_strength"])
    parser.add_argument("--luma-sigma", type=float, default=SMOOTH_PARAMS["luma_sigma"])
    parser.add_argument("--chroma-sigma", type=float, default=SMOOTH_PARAMS["chroma_sigma"])
    parser.add_argument("--reopen-strength", type=float, default=0.0)
    parser.add_argument("--reopen-shadow-weight", type=float, default=0.85)
    parser.add_argument("--reopen-structure-suppress", type=float, default=1.0)
    parser.add_argument("--reopen-min", type=float, default=1.0)
    parser.add_argument("--reopen-max", type=float, default=1.45)
    parser.add_argument("--reopen-shadow-threshold", type=float, default=0.0)
    parser.add_argument("--reopen-shadow-transition", type=float, default=0.12)
    parser.add_argument("--limiter-strength", type=float, default=0.0)
    parser.add_argument("--limiter-min", type=float, default=0.35)
    parser.add_argument("--limiter-flat-threshold", type=float, default=0.62)
    parser.add_argument("--limiter-flat-transition", type=float, default=0.18)
    parser.add_argument("--limiter-shadow-threshold", type=float, default=0.32)
    parser.add_argument("--limiter-shadow-transition", type=float, default=0.18)
    parser.add_argument("--limiter-structure-suppress", type=float, default=1.0)
    parser.add_argument("--no-tiff", action="store_true")
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    input_path = Path(args.input).expanduser()
    gate_path = Path(args.gate).expanduser()
    out_dir = Path(args.output_dir)
    mask_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    reference = read_image(reference_path)
    current = read_image(input_path)
    if reference.shape[:2] != current.shape[:2]:
        raise ValueError(f"shape mismatch reference={reference.shape} input={current.shape}")
    gate = _read_gate(gate_path, current.shape[:2])
    params = {
        "base_strength": args.base_strength,
        "flat_boost": args.flat_boost,
        "skin_strength": args.skin_strength,
        "shadow_flat_boost": args.shadow_flat_boost,
        "shadow_luma_threshold": args.shadow_luma_threshold,
        "shadow_luma_transition": args.shadow_luma_transition,
        "structure_suppress": args.structure_suppress,
        "highlight_suppress": args.highlight_suppress,
        "flat_threshold": args.flat_threshold,
        "flat_transition": args.flat_transition,
        "edge_threshold": args.edge_threshold,
        "edge_transition": args.edge_transition,
        "min_strength": args.min_strength,
        "max_strength": args.max_strength,
        "blur_sigma": args.blur_sigma,
        "reopen_strength": args.reopen_strength,
        "reopen_shadow_weight": args.reopen_shadow_weight,
        "reopen_structure_suppress": args.reopen_structure_suppress,
        "reopen_min": args.reopen_min,
        "reopen_max": args.reopen_max,
        "reopen_shadow_threshold": args.reopen_shadow_threshold,
        "reopen_shadow_transition": args.reopen_shadow_transition,
        "limiter_strength": args.limiter_strength,
        "limiter_min": args.limiter_min,
        "limiter_flat_threshold": args.limiter_flat_threshold,
        "limiter_flat_transition": args.limiter_flat_transition,
        "limiter_shadow_threshold": args.limiter_shadow_threshold,
        "limiter_shadow_transition": args.limiter_shadow_transition,
        "limiter_structure_suppress": args.limiter_structure_suppress,
    }
    smooth_params = {
        "luma_strength": args.luma_strength,
        "chroma_strength": args.chroma_strength,
        "luma_sigma": args.luma_sigma,
        "chroma_sigma": args.chroma_sigma,
    }
    out, stats, masks = apply_region_aware_gate(reference, current, gate, smooth_params=smooth_params, **params)

    name = args.name or f"{input_path.stem}_region_aware_flat_gate"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    if not args.no_tiff:
        write_tiff(out_dir / f"{name}.tiff", out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    for mask_name, mask in masks.items():
        Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(
            mask_dir / f"{name}_{mask_name}.png"
        )
    meta = {
        "reference": str(reference_path),
        "input": str(input_path),
        "gate": str(gate_path),
        "outputs": {
            "exr": str(exr_path),
            "preview": str(preview_path),
            "masks": str(mask_dir),
        },
        "params": params,
        "smooth_params": smooth_params,
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(exr_path), "filter": stats}, indent=2))


if __name__ == "__main__":
    main()
