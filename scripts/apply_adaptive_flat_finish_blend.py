"""Blend conservative and stronger flat-cleanup finishes adaptively.

The stronger flat cleanup can improve sky/noise fields but is not universally
better. This script keeps the conservative finish as the base and blends in the
stronger finish only where the reference/current evidence says the region is
flat, low-saturation, non-edge, and not protected by the learned detail gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, uniform_filter

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def _display(image: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb_np(np.clip(_safe_rgb(image), 0.0, None)), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def _read_gate(path: str | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.float32)
    img = Image.open(Path(path).expanduser()).convert("L")
    if img.size != (shape[1], shape[0]):
        raise ValueError(f"gate size mismatch: gate={img.size}, image={(shape[1], shape[0])}")
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def saturation(display: np.ndarray) -> np.ndarray:
    mx = np.max(display, axis=2)
    mn = np.min(display, axis=2)
    return ((mx - mn) / np.maximum(mx, 1.0e-6)).astype(np.float32, copy=False)


def build_blend(
    reference_linear: np.ndarray,
    base_linear: np.ndarray,
    strong_gate: np.ndarray,
    detail_gate: np.ndarray,
    *,
    flat_threshold: float,
    flat_transition: float,
    edge_threshold: float,
    edge_transition: float,
    saturation_threshold: float,
    saturation_transition: float,
    detail_protect: float,
    strong_gate_power: float,
    blend_strength: float,
    blur: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    ref = _display(reference_linear)
    base = _display(base_linear)
    y = luma(base, LUMA_SRGB)
    ref_y = luma(ref, LUMA_SRGB)
    sat = saturation(base)
    ref_structure = gaussian_filter(ref_y, sigma=0.8, mode="reflect")
    base_structure = gaussian_filter(y, sigma=0.9, mode="reflect")
    ref_detail = np.abs(ref_structure - uniform_filter(ref_structure, size=11, mode="reflect"))
    base_detail = np.abs(base_structure - uniform_filter(base_structure, size=11, mode="reflect"))
    edge = gaussian_gradient_magnitude(base_structure, sigma=0.9, mode="reflect")
    flat = sigmoid01((float(flat_threshold) - np.maximum(ref_detail, base_detail)) / max(float(flat_transition), 1.0e-6))
    non_edge = sigmoid01((float(edge_threshold) - edge) / max(float(edge_transition), 1.0e-6))
    low_sat = sigmoid01((float(saturation_threshold) - sat) / max(float(saturation_transition), 1.0e-6))
    detail_block = np.clip(detail_gate * float(detail_protect), 0.0, 1.0)
    strong = np.power(np.clip(strong_gate, 0.0, 1.0), float(strong_gate_power))
    blend = np.clip(flat * non_edge * low_sat * strong * (1.0 - detail_block) * float(blend_strength), 0.0, 1.0)
    if blur > 0:
        blend = gaussian_filter(blend.astype(np.float32, copy=False), sigma=float(blur), mode="reflect")
    masks = {
        "blend": blend,
        "flat": flat,
        "non_edge": non_edge,
        "low_sat": low_sat,
        "strong_gate": strong_gate,
        "detail_gate": detail_gate,
        "detail_block": detail_block,
    }
    stats = {f"{name}_mean": float(np.mean(mask)) for name, mask in masks.items()}
    stats.update({f"{name}_p95": float(np.quantile(mask, 0.95)) for name, mask in masks.items()})
    return blend.astype(np.float32, copy=False), stats, masks


def blend_display(base_linear: np.ndarray, strong_linear: np.ndarray, blend: np.ndarray) -> np.ndarray:
    base = _display(base_linear)
    strong = _display(strong_linear)
    out_display = base * (1.0 - blend[..., None]) + strong * blend[..., None]
    return srgb_to_linear_np(np.clip(out_display, 0.0, 1.0)).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptively blend stronger flat cleanup into a conservative finish.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--strong", required=True)
    parser.add_argument("--strong-gate", required=True)
    parser.add_argument("--detail-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--flat-threshold", type=float, default=0.018)
    parser.add_argument("--flat-transition", type=float, default=0.009)
    parser.add_argument("--edge-threshold", type=float, default=0.020)
    parser.add_argument("--edge-transition", type=float, default=0.010)
    parser.add_argument("--saturation-threshold", type=float, default=0.58)
    parser.add_argument("--saturation-transition", type=float, default=0.15)
    parser.add_argument("--detail-protect", type=float, default=1.25)
    parser.add_argument("--strong-gate-power", type=float, default=0.80)
    parser.add_argument("--blend-strength", type=float, default=0.92)
    parser.add_argument("--blur", type=float, default=1.1)
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    base_path = Path(args.base).expanduser()
    strong_path = Path(args.strong).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_path.stem}_adaptive_flat_finish"

    reference = read_image(reference_path)
    base = read_image(base_path)
    strong = read_image(strong_path)
    if not (reference.shape[:2] == base.shape[:2] == strong.shape[:2]):
        raise ValueError(f"shape mismatch: reference={reference.shape}, base={base.shape}, strong={strong.shape}")
    strong_gate = _read_gate(args.strong_gate, base.shape[:2])
    detail_gate = _read_gate(args.detail_gate, base.shape[:2])
    blend, stats, masks = build_blend(
        reference,
        base,
        strong_gate,
        detail_gate,
        flat_threshold=args.flat_threshold,
        flat_transition=args.flat_transition,
        edge_threshold=args.edge_threshold,
        edge_transition=args.edge_transition,
        saturation_threshold=args.saturation_threshold,
        saturation_transition=args.saturation_transition,
        detail_protect=args.detail_protect,
        strong_gate_power=args.strong_gate_power,
        blend_strength=args.blend_strength,
        blur=args.blur,
    )
    out = blend_display(base, strong, blend)

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    mask_outputs = {}
    for mask_name, mask in masks.items():
        path = out_dir / f"{name}_{mask_name}.png"
        Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)
        mask_outputs[mask_name] = str(path)
    meta = {
        "reference": str(reference_path),
        "base": str(base_path),
        "strong": str(strong_path),
        "strong_gate": args.strong_gate,
        "detail_gate": args.detail_gate,
        "outputs": {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path), "masks": mask_outputs},
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
