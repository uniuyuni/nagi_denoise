"""Graft only guarded SCUNet luma structure for blue-shadow scenes.

This is a blue-shadow-safe preset candidate. It keeps current chroma and most of
current luma, then borrows only band-limited SCUNet-vs-current luma structure in
coherent edge/texture regions. The intent is to improve Ice-like edge retention
without returning SCUNet's flat-region luma tail noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_flat_chroma_smoother import LUMA_SRGB, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import image_stats, make_preview, read_image
from train_scunet_selector import RUN_ROOT, SCENES, display, display_chroma_ratio, hf_abs, make_coherent_structure_mask


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def graft_structure(
    noisy_linear: np.ndarray,
    current_linear: np.ndarray,
    scunet_linear: np.ndarray,
    *,
    strength: float,
    fine_sigma: float,
    coarse_sigma: float,
    gate_threshold: float,
    gate_transition: float,
    coherent_weight: float,
    texture_weight: float,
    correction_limit: float,
    tail_suppress: float,
    gate_blur: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    noisy_d = display(noisy_linear)
    current = np.clip(_safe_rgb(current_linear), 0.0, None)
    scunet = np.clip(_safe_rgb(scunet_linear), 0.0, None)
    current_d = display(current)
    scunet_d = display(scunet)
    current_y = luma(current_d, LUMA_SRGB)
    scunet_y = luma(scunet_d, LUMA_SRGB)

    current_band = gaussian_filter(current_y, sigma=float(fine_sigma), mode="reflect") - gaussian_filter(
        current_y, sigma=float(coarse_sigma), mode="reflect"
    )
    scunet_band = gaussian_filter(scunet_y, sigma=float(fine_sigma), mode="reflect") - gaussian_filter(
        scunet_y, sigma=float(coarse_sigma), mode="reflect"
    )
    correction = scunet_band - current_band
    correction = np.clip(correction, -float(correction_limit), float(correction_limit))

    edge = np.maximum(
        gaussian_gradient_magnitude(current_y, sigma=0.9, mode="reflect"),
        gaussian_gradient_magnitude(scunet_y, sigma=0.9, mode="reflect"),
    )
    edge_gate = sigmoid01((edge - float(gate_threshold)) / max(float(gate_transition), 1.0e-6))
    coherent = make_coherent_structure_mask(
        noisy_d,
        coherence_threshold=0.38,
        coherence_transition=0.18,
        energy_threshold=0.0048,
        energy_transition=0.0055,
    )
    texture = sigmoid01((np.maximum(np.abs(current_band), np.abs(scunet_band)) - 0.006) / 0.010)
    current_hf = hf_abs(current_y, 0.8)
    scunet_hf = hf_abs(scunet_y, 0.8)
    tail = sigmoid01((scunet_hf - current_hf - 0.0025) / 0.0075)
    gate = edge_gate * (1.0 + float(coherent_weight) * coherent + float(texture_weight) * texture)
    gate *= 1.0 - np.clip(float(tail_suppress), 0.0, 1.0) * tail * (1.0 - coherent)
    gate = np.clip(gate, 0.0, 1.0)
    if gate_blur > 0:
        gate = gaussian_filter(gate.astype(np.float32, copy=False), sigma=float(gate_blur), mode="reflect")
    gate = np.clip(gate, 0.0, 1.0).astype(np.float32, copy=False)

    out_y = np.clip(current_y + float(strength) * gate * correction, 0.0, 1.0)
    out_display = np.clip(display_chroma_ratio(current_d) * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    peak = np.max(current, axis=2)
    hdr = np.clip((peak - 0.82) / 0.25, 0.0, 1.0)
    hdr = (hdr * hdr * (3.0 - 2.0 * hdr)).astype(np.float32, copy=False)
    out = out * (1.0 - hdr[..., None]) + current * hdr[..., None]
    stats = {
        "gate_mean": float(np.mean(gate)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "tail_mean": float(np.mean(tail)),
        "correction_abs_mean": float(np.mean(np.abs(correction))),
        "correction_abs_p99": float(np.quantile(np.abs(correction), 0.99)),
    }
    return out.astype(np.float32, copy=False), stats, gate


def apply(args: argparse.Namespace) -> None:
    scene = SCENES[args.scene]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noisy = read_image(scene.noisy)
    current = read_image(scene.current)
    scunet = read_image(scene.scunet)
    out, stats, gate = graft_structure(
        noisy,
        current,
        scunet,
        strength=args.strength,
        fine_sigma=args.fine_sigma,
        coarse_sigma=args.coarse_sigma,
        gate_threshold=args.gate_threshold,
        gate_transition=args.gate_transition,
        coherent_weight=args.coherent_weight,
        texture_weight=args.texture_weight,
        correction_limit=args.correction_limit,
        tail_suppress=args.tail_suppress,
        gate_blur=args.gate_blur,
    )
    name = args.name or f"{scene.name}_blue_shadow_structure_graft"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    meta = {
        "scene": scene.name,
        "inputs": {"noisy": str(scene.noisy), "current": str(scene.current), "scunet": str(scene.scunet)},
        "outputs": {"exr": str(exr_path), "preview": str(preview_path), "gate": str(gate_path)},
        "params": vars(args),
        "graft_stats": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply blue-shadow-safe SCUNet structure graft.")
    parser.add_argument("--scene", default="k5_ice", choices=sorted(SCENES))
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "blue_shadow_structure_graft_v1"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--strength", type=float, default=0.80)
    parser.add_argument("--fine-sigma", type=float, default=0.65)
    parser.add_argument("--coarse-sigma", type=float, default=2.20)
    parser.add_argument("--gate-threshold", type=float, default=0.018)
    parser.add_argument("--gate-transition", type=float, default=0.018)
    parser.add_argument("--coherent-weight", type=float, default=0.55)
    parser.add_argument("--texture-weight", type=float, default=0.25)
    parser.add_argument("--correction-limit", type=float, default=0.045)
    parser.add_argument("--tail-suppress", type=float, default=0.65)
    parser.add_argument("--gate-blur", type=float, default=0.60)
    args = parser.parse_args()
    apply(args)


if __name__ == "__main__":
    main()
