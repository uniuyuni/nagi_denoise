"""Apply stronger flat-region cleanup while protecting learned detail.

This is the companion stage to the learned luma-detail gate. The intended order
is:

1. reconstruct/pass coherent detail with the luma detail gate,
2. clean residual flat-region luma/chroma grain,
3. keep gate/coherence/texture regions protected so hair, branches, and fabric
   do not become softer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from apply_region_aware_luma_cleanup import make_coherent_structure_mask, make_skin_mask, make_texture_mask
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
        raise ValueError(f"detail gate size mismatch: gate={img.size}, image={(shape[1], shape[0])}")
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def saturation(display: np.ndarray) -> np.ndarray:
    mx = np.max(display, axis=2)
    mn = np.min(display, axis=2)
    return ((mx - mn) / np.maximum(mx, 1.0e-6)).astype(np.float32, copy=False)


def build_cleanup_gate(
    reference_linear: np.ndarray,
    current_linear: np.ndarray,
    detail_gate: np.ndarray,
    *,
    flat_threshold: float,
    flat_transition: float,
    edge_threshold: float,
    edge_transition: float,
    coherent_protect: float,
    texture_protect: float,
    detail_gate_protect: float,
    auto_detail_protect: float,
    auto_detail_threshold: float,
    auto_detail_transition: float,
    skin_protect: float,
    highlight_threshold: float,
    highlight_transition: float,
    gate_blur: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    reference = _display(reference_linear)
    current = _display(current_linear)
    current_y = luma(current, LUMA_SRGB)
    current_y_linear = luma(_safe_rgb(current_linear), LUMA_LINEAR)

    ref_texture = make_texture_mask(reference, texture_threshold=0.008, texture_transition=0.014)
    cur_texture = make_texture_mask(current, texture_threshold=0.016, texture_transition=0.014)
    coherent = make_coherent_structure_mask(
        reference,
        coherence_threshold=0.38,
        coherence_transition=0.17,
        energy_threshold=0.005,
        energy_transition=0.006,
    )
    structure = gaussian_filter(current_y, sigma=1.0, mode="reflect")
    local_detail = np.abs(structure - gaussian_filter(structure, sigma=3.0, mode="reflect"))
    edge = gaussian_gradient_magnitude(structure, sigma=0.9, mode="reflect")
    auto_detail_energy = np.maximum(local_detail, edge * 0.72)
    auto_detail_gate = sigmoid01(
        (auto_detail_energy - float(auto_detail_threshold)) / max(float(auto_detail_transition), 1.0e-6)
    )
    flat = sigmoid01((float(flat_threshold) - local_detail) / max(float(flat_transition), 1.0e-6))
    non_edge = sigmoid01((float(edge_threshold) - edge) / max(float(edge_transition), 1.0e-6))
    highlight = sigmoid01(
        (current_y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6)
    )
    skin = make_skin_mask(current, blur_sigma=1.4)
    sat = saturation(current)
    low_sat_flat = sigmoid01((0.55 - sat) / 0.16)

    protect = np.clip(
        coherent * float(coherent_protect)
        + np.maximum(ref_texture, cur_texture) * float(texture_protect)
        + detail_gate * float(detail_gate_protect)
        + auto_detail_gate * float(auto_detail_protect)
        + skin * float(skin_protect),
        0.0,
        1.0,
    )
    gate = np.clip(flat * non_edge * low_sat_flat * (1.0 - protect) * (1.0 - highlight), 0.0, 1.0)
    if gate_blur > 0:
        gate = gaussian_filter(gate.astype(np.float32, copy=False), sigma=float(gate_blur), mode="reflect")
    masks = {
        "gate": gate,
        "flat": flat,
        "non_edge": non_edge,
        "low_sat_flat": low_sat_flat,
        "coherent": coherent,
        "texture": np.maximum(ref_texture, cur_texture),
        "detail_gate": detail_gate,
        "auto_detail_gate": auto_detail_gate,
        "skin": skin,
        "highlight": highlight,
        "protect": protect,
    }
    stats = {f"{name}_mean": float(np.mean(mask)) for name, mask in masks.items()}
    stats.update({f"{name}_p95": float(np.quantile(mask, 0.95)) for name, mask in masks.items()})
    return gate.astype(np.float32, copy=False), stats, masks


def apply_cleanup(
    reference_linear: np.ndarray,
    current_linear: np.ndarray,
    detail_gate: np.ndarray,
    *,
    luma_strength: float,
    chroma_strength: float,
    luma_sigma: float,
    chroma_sigma: float,
    flat_threshold: float,
    flat_transition: float,
    edge_threshold: float,
    edge_transition: float,
    coherent_protect: float,
    texture_protect: float,
    detail_gate_protect: float,
    auto_detail_protect: float,
    auto_detail_threshold: float,
    auto_detail_transition: float,
    skin_protect: float,
    highlight_threshold: float,
    highlight_transition: float,
    hdr_restore_threshold: float,
    hdr_restore_transition: float,
    gate_blur: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    current = _display(current_linear)
    current_y = luma(current, LUMA_SRGB)
    chroma = current - current_y[..., None]
    gate, stats, masks = build_cleanup_gate(
        reference_linear,
        current_linear,
        detail_gate,
        flat_threshold=flat_threshold,
        flat_transition=flat_transition,
        edge_threshold=edge_threshold,
        edge_transition=edge_transition,
        coherent_protect=coherent_protect,
        texture_protect=texture_protect,
        detail_gate_protect=detail_gate_protect,
        auto_detail_protect=auto_detail_protect,
        auto_detail_threshold=auto_detail_threshold,
        auto_detail_transition=auto_detail_transition,
        skin_protect=skin_protect,
        highlight_threshold=highlight_threshold,
        highlight_transition=highlight_transition,
        gate_blur=gate_blur,
    )

    luma_low = gaussian_filter(current_y, sigma=float(luma_sigma), mode="reflect")
    chroma_low = gaussian_filter(chroma, sigma=(float(chroma_sigma), float(chroma_sigma), 0.0), mode="reflect")
    luma_blend = np.clip(gate * float(luma_strength), 0.0, 1.0)
    chroma_blend = np.clip(gate * float(chroma_strength), 0.0, 1.0)[..., None]
    out_y = current_y * (1.0 - luma_blend) + luma_low * luma_blend
    out_chroma = chroma * (1.0 - chroma_blend) + chroma_low * chroma_blend
    out_display = np.clip(out_y[..., None] + out_chroma, 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    current_rgb = _safe_rgb(current_linear)
    peak = np.max(current_rgb, axis=2)
    hdr_restore = sigmoid01(
        (peak - float(hdr_restore_threshold)) / max(float(hdr_restore_transition), 1.0e-6)
    )
    out = out * (1.0 - hdr_restore[..., None]) + current_rgb * hdr_restore[..., None]
    masks["hdr_restore"] = hdr_restore.astype(np.float32, copy=False)
    stats.update(
        {
            "luma_strength": float(luma_strength),
            "chroma_strength": float(chroma_strength),
            "luma_sigma": float(luma_sigma),
            "chroma_sigma": float(chroma_sigma),
            "hdr_restore_mean": float(np.mean(hdr_restore)),
            "hdr_restore_p95": float(np.quantile(hdr_restore, 0.95)),
        }
    )
    return out, stats, masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Detail-protected flat cleanup.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--detail-gate", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--luma-strength", type=float, default=0.72)
    parser.add_argument("--chroma-strength", type=float, default=0.86)
    parser.add_argument("--luma-sigma", type=float, default=1.65)
    parser.add_argument("--chroma-sigma", type=float, default=2.10)
    parser.add_argument("--flat-threshold", type=float, default=0.020)
    parser.add_argument("--flat-transition", type=float, default=0.010)
    parser.add_argument("--edge-threshold", type=float, default=0.026)
    parser.add_argument("--edge-transition", type=float, default=0.012)
    parser.add_argument("--coherent-protect", type=float, default=0.98)
    parser.add_argument("--texture-protect", type=float, default=0.82)
    parser.add_argument("--detail-gate-protect", type=float, default=1.20)
    parser.add_argument("--auto-detail-protect", type=float, default=0.0)
    parser.add_argument("--auto-detail-threshold", type=float, default=0.018)
    parser.add_argument("--auto-detail-transition", type=float, default=0.010)
    parser.add_argument("--skin-protect", type=float, default=0.55)
    parser.add_argument("--highlight-threshold", type=float, default=1.0)
    parser.add_argument("--highlight-transition", type=float, default=0.25)
    parser.add_argument("--hdr-restore-threshold", type=float, default=0.92)
    parser.add_argument("--hdr-restore-transition", type=float, default=0.24)
    parser.add_argument("--gate-blur", type=float, default=0.90)
    parser.add_argument("--no-tiff", action="store_true")
    args = parser.parse_args()

    reference_path = Path(args.reference).expanduser()
    input_path = Path(args.input).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_detail_protected_flat_cleanup"

    reference = read_image(reference_path)
    current = read_image(input_path)
    if reference.shape[:2] != current.shape[:2]:
        raise ValueError(f"shape mismatch: reference={reference.shape}, input={current.shape}")
    detail_gate = _read_gate(args.detail_gate, current.shape[:2])
    out, stats, masks = apply_cleanup(
        reference,
        current,
        detail_gate,
        luma_strength=args.luma_strength,
        chroma_strength=args.chroma_strength,
        luma_sigma=args.luma_sigma,
        chroma_sigma=args.chroma_sigma,
        flat_threshold=args.flat_threshold,
        flat_transition=args.flat_transition,
        edge_threshold=args.edge_threshold,
        edge_transition=args.edge_transition,
        coherent_protect=args.coherent_protect,
        texture_protect=args.texture_protect,
        detail_gate_protect=args.detail_gate_protect,
        auto_detail_protect=args.auto_detail_protect,
        auto_detail_threshold=args.auto_detail_threshold,
        auto_detail_transition=args.auto_detail_transition,
        skin_protect=args.skin_protect,
        highlight_threshold=args.highlight_threshold,
        highlight_transition=args.highlight_transition,
        hdr_restore_threshold=args.hdr_restore_threshold,
        hdr_restore_transition=args.hdr_restore_transition,
        gate_blur=args.gate_blur,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    if not args.no_tiff:
        write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    mask_outputs = {}
    for mask_name, mask in masks.items():
        path = out_dir / f"{name}_{mask_name}.png"
        Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)
        mask_outputs[mask_name] = str(path)

    meta = {
        "reference": str(reference_path),
        "input": str(input_path),
        "detail_gate": args.detail_gate,
        "outputs": {
            "exr": str(exr_path),
            "tiff": None if args.no_tiff else str(tiff_path),
            "preview": str(preview_path),
            "masks": mask_outputs,
        },
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "filter": stats,
        "input_stats": image_stats(current),
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
