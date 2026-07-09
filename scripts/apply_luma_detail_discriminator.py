"""Gate luma-rebuilder detail with a structure/noise discriminator.

This is a deterministic probe before training a learned discriminator. It takes
a luma-rebuilder result and suppresses its luma delta where the local evidence
looks like stochastic grain instead of coherent detail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from apply_region_aware_luma_cleanup import make_coherent_structure_mask, make_skin_mask, make_texture_mask
from perfect_nr_detail_guard import write_exr
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


def saturation(display: np.ndarray) -> np.ndarray:
    mx = np.max(display, axis=2)
    mn = np.min(display, axis=2)
    return ((mx - mn) / np.maximum(mx, 1.0e-6)).astype(np.float32, copy=False)


def highpass(x: np.ndarray, sigma: float) -> np.ndarray:
    return (x - gaussian_filter(x, sigma=float(sigma), mode="reflect")).astype(np.float32, copy=False)


def soft_texture(display: np.ndarray, threshold: float, transition: float) -> np.ndarray:
    y = luma(display, LUMA_SRGB)
    structure = gaussian_filter(y, sigma=0.7, mode="reflect")
    edge = gaussian_gradient_magnitude(structure, sigma=0.8, mode="reflect")
    contrast = np.abs(structure - uniform_filter(structure, size=11, mode="reflect"))
    return np.clip(
        np.maximum(
            sigmoid01((edge - float(threshold)) / max(float(transition), 1.0e-6)),
            sigmoid01((contrast - float(threshold)) / max(float(transition), 1.0e-6)),
        ),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)


def apply_luma_delta(current_linear: np.ndarray, delta: np.ndarray) -> np.ndarray:
    current = _display(current_linear)
    current_y = luma(current, LUMA_SRGB)
    out_y = np.clip(current_y + delta, 0.0, 1.0)
    chroma = current / np.maximum(current_y[..., None], 1.0e-6)
    out_display = np.clip(chroma * out_y[..., None], 0.0, 1.0)
    dark = current_y < 1.0e-5
    out_display[dark] = current[dark]
    return srgb_to_linear_np(out_display).astype(np.float32, copy=False)


def build_gate(
    noisy_linear: np.ndarray,
    current_linear: np.ndarray,
    base_linear: np.ndarray,
    rebuild_linear: np.ndarray,
    *,
    structure_gain: float,
    coherent_weight: float,
    base_texture_weight: float,
    rebuild_texture_weight: float,
    dark_flat_suppress: float,
    skin_suppress: float,
    grain_suppress: float,
    gate_blur: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    noisy = _display(noisy_linear)
    current = _display(current_linear)
    base = _display(base_linear)
    rebuild = _display(rebuild_linear)
    current_y = luma(current, LUMA_SRGB)
    sat = saturation(current)

    coherent = make_coherent_structure_mask(
        noisy,
        coherence_threshold=0.40,
        coherence_transition=0.18,
        energy_threshold=0.006,
        energy_transition=0.006,
    )
    base_texture = soft_texture(base, threshold=0.018, transition=0.014)
    rebuild_texture = soft_texture(rebuild, threshold=0.018, transition=0.014)
    noisy_texture = soft_texture(noisy, threshold=0.010, transition=0.016)
    structure = np.clip(
        np.maximum.reduce(
            [
                coherent * float(coherent_weight),
                base_texture * float(base_texture_weight),
                rebuild_texture * noisy_texture * float(rebuild_texture_weight),
            ]
        )
        * float(structure_gain),
        0.0,
        1.0,
    )

    skin = make_skin_mask(current, blur_sigma=1.4)
    dark = sigmoid01((0.34 - current_y) / 0.08)
    low_sat = sigmoid01((0.42 - sat) / 0.12)
    dark_flat = np.clip(dark * low_sat * (1.0 - base_texture), 0.0, 1.0)
    noisy_hf = np.abs(highpass(luma(noisy, LUMA_SRGB), 0.75))
    rebuild_hf = np.abs(highpass(luma(rebuild, LUMA_SRGB), 0.75))
    grain = np.clip(sigmoid01((noisy_hf - 0.018) / 0.010) * (1.0 - coherent) * (1.0 - base_texture), 0.0, 1.0)
    suppress = np.clip(
        dark_flat * float(dark_flat_suppress)
        + skin * float(skin_suppress)
        + grain * float(grain_suppress),
        0.0,
        1.0,
    )
    gate = np.clip(structure * (1.0 - suppress), 0.0, 1.0).astype(np.float32, copy=False)
    if gate_blur > 0:
        gate = gaussian_filter(gate, sigma=float(gate_blur), mode="reflect")
    masks = {
        "gate": gate,
        "structure": structure,
        "coherent": coherent,
        "base_texture": base_texture,
        "rebuild_texture": rebuild_texture,
        "dark_flat": dark_flat,
        "skin": skin,
        "grain": grain,
    }
    stats = {f"{name}_mean": float(np.mean(mask)) for name, mask in masks.items()}
    stats.update({f"{name}_p95": float(np.quantile(mask, 0.95)) for name, mask in masks.items()})
    stats["rebuild_hf_mean"] = float(np.mean(rebuild_hf))
    return gate, stats, masks


def apply(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noisy = read_image(args.noisy)
    current = read_image(args.current)
    base = read_image(args.base)
    rebuild = read_image(args.rebuild)
    result = read_image(args.result)
    if not (current.shape[:2] == result.shape[:2] == noisy.shape[:2] == base.shape[:2] == rebuild.shape[:2]):
        raise ValueError("all inputs must have matching height/width")

    current_y = luma(_display(current), LUMA_SRGB)
    result_y = luma(_display(result), LUMA_SRGB)
    raw_delta = result_y - current_y
    gate, stats, masks = build_gate(
        noisy,
        current,
        base,
        rebuild,
        structure_gain=args.structure_gain,
        coherent_weight=args.coherent_weight,
        base_texture_weight=args.base_texture_weight,
        rebuild_texture_weight=args.rebuild_texture_weight,
        dark_flat_suppress=args.dark_flat_suppress,
        skin_suppress=args.skin_suppress,
        grain_suppress=args.grain_suppress,
        gate_blur=args.gate_blur,
    )
    gated_delta = raw_delta * (args.floor + (1.0 - args.floor) * gate) * float(args.strength)
    out = apply_luma_delta(current, gated_delta)

    name = args.name or f"{Path(args.result).stem}_detail_discriminated"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    delta_path = out_dir / f"{name}_delta.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip((gated_delta / 0.12) * 127.0 + 128.0, 0, 255).astype(np.uint8)).save(delta_path)
    mask_outputs = {}
    for mask_name, mask in masks.items():
        path = out_dir / f"{name}_{mask_name}.png"
        Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)
        mask_outputs[mask_name] = str(path)

    stats.update(
        {
            "raw_delta_abs_mean": float(np.mean(np.abs(raw_delta))),
            "raw_delta_abs_p95": float(np.quantile(np.abs(raw_delta), 0.95)),
            "gated_delta_abs_mean": float(np.mean(np.abs(gated_delta))),
            "gated_delta_abs_p95": float(np.quantile(np.abs(gated_delta), 0.95)),
        }
    )
    meta = {
        "inputs": {
            "noisy": str(Path(args.noisy)),
            "current": str(Path(args.current)),
            "base": str(Path(args.base)),
            "rebuild": str(Path(args.rebuild)),
            "result": str(Path(args.result)),
        },
        "outputs": {"exr": str(exr_path), "preview": str(preview_path), "delta": str(delta_path), "masks": mask_outputs},
        "params": vars(args),
        "filter": stats,
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {exr_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate luma-rebuilder delta with a structure/noise discriminator.")
    parser.add_argument("--noisy", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--rebuild", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--floor", type=float, default=0.18)
    parser.add_argument("--structure-gain", type=float, default=1.18)
    parser.add_argument("--coherent-weight", type=float, default=1.00)
    parser.add_argument("--base-texture-weight", type=float, default=0.95)
    parser.add_argument("--rebuild-texture-weight", type=float, default=0.75)
    parser.add_argument("--dark-flat-suppress", type=float, default=0.92)
    parser.add_argument("--skin-suppress", type=float, default=0.32)
    parser.add_argument("--grain-suppress", type=float, default=0.82)
    parser.add_argument("--gate-blur", type=float, default=0.65)
    args = parser.parse_args()
    apply(args)


if __name__ == "__main__":
    main()
