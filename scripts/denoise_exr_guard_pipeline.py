"""Run the current practical Guard NR pipeline on EXR/TIFF images.

Pipeline:
1. ChromaGuard strong chroma NR.
2. LumaGuard safe guided-luma smoothing.
3. Final chroma polish to suppress color speckle made visible by luma cleanup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "nagi_nr" / "src"))

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smooth_chroma, smoothstep, srgb_to_linear_np
from apply_guided_luma_smoother import guided_filter_gray, make_luma_gate
from denoise_exr_chromaguard import apply_chroma_nr_with_gate, load_guard, predict_gate
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


CHROMA_PRESETS = {
    "balanced": {"gate_gain": 1.45, "gate_bias": 0.0, "strength": 1.0, "chroma_sigma": 2.0},
    "strong": {"gate_gain": 1.55, "gate_bias": 0.0, "strength": 1.0, "chroma_sigma": 2.0},
}

LUMA_PRESETS = {
    "balanced": {"gate_gain": 1.15, "gate_bias": 0.0, "strength": 0.75, "radius": 5, "eps": 0.006},
    "strong": {"gate_gain": 1.35, "gate_bias": 0.0, "strength": 0.95, "radius": 5, "eps": 0.006},
}

POLISH_PRESETS = {
    "off": None,
    "strong": {
        "strength": 1.0,
        "chroma_sigma": 2.0,
        "detail_sigma": 1.2,
        "threshold": 0.018,
        "transition": 0.010,
    },
    "xstrong": {
        "strength": 1.0,
        "chroma_sigma": 3.0,
        "detail_sigma": 1.2,
        "threshold": 0.024,
        "transition": 0.012,
    },
}


def apply_lumaguard_stage(
    image: np.ndarray,
    guide: np.ndarray,
    gate: np.ndarray,
    *,
    strength: float,
    radius: int,
    eps: float,
    safety_gate_gain: float,
    use_safety_gate: bool,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    base = np.nan_to_num(image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    guide_base = np.nan_to_num(guide[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    display = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    guide_display = np.clip(linear_to_srgb_np(guide_base), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)
    guide_y = luma(guide_display, LUMA_SRGB)

    if use_safety_gate:
        guide_y_linear = luma(np.clip(guide_base, 0.0, None), LUMA_LINEAR)
        safety_gate = make_luma_gate(
            guide_y,
            guide_y_linear,
            structure_sigma=1.2,
            detail_sigma=2.6,
            detail_threshold=0.010,
            detail_transition=0.006,
            edge_sigma=1.0,
            edge_threshold=0.018,
            edge_transition=0.010,
            highlight_threshold=1.0,
            highlight_transition=0.25,
        )
        gate = np.minimum(gate, np.clip(safety_gate * float(safety_gate_gain), 0.0, 1.0))

    structure = gaussian_filter(guide_y, sigma=1.0, mode="reflect")
    y_smooth = guided_filter_gray(structure, y, radius=int(radius), eps=float(eps))
    blend = np.clip(gate * float(strength), 0.0, 1.0)
    out_y = y * (1.0 - blend) + y_smooth * blend
    chroma = display / np.maximum(y[..., None], 1.0e-6)
    out_display = np.clip(chroma * np.maximum(out_y[..., None], 0.0), 0.0, 1.0)
    out = srgb_to_linear_np(out_display)

    y_linear = luma(base, LUMA_LINEAR)
    peak_linear = np.max(base, axis=2)
    hdr_signal = np.maximum(y_linear - 0.85, peak_linear - 0.95)
    hdr_restore = smoothstep(hdr_signal / 0.25)
    out = out * (1.0 - hdr_restore[..., None]) + base * hdr_restore[..., None]
    stats = {
        "strength": float(strength),
        "radius": int(radius),
        "eps": float(eps),
        "safety_gate_enabled": bool(use_safety_gate),
        "safety_gate_gain": float(safety_gate_gain),
        "gate_mean": float(np.mean(gate)),
        "gate_p50": float(np.quantile(gate, 0.50)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
    }
    return out.astype(np.float32, copy=False), stats, gate.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Guard NR pipeline.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--chromaguard-weights", required=True)
    parser.add_argument("--lumaguard-weights", required=True)
    parser.add_argument("--output-dir", default="runs/perfect_nr/guard_pipeline")
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--chroma-preset", choices=sorted(CHROMA_PRESETS), default="strong")
    parser.add_argument("--luma-preset", choices=sorted(LUMA_PRESETS), default="strong")
    parser.add_argument("--polish-preset", choices=sorted(POLISH_PRESETS), default="strong")
    parser.add_argument("--no-luma-safety-gate", action="store_true")
    parser.add_argument("--luma-safety-gate-gain", type=float, default=1.15)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--preview-exposure", type=float, default=1.0)
    parser.add_argument("--preview-tone", choices=["reinhard", "clip"], default="reinhard")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem.replace(' ', '_')}_guard_pipeline"
    device = torch.device(args.device)

    image = read_image(input_path)
    chroma_model = load_guard(Path(args.chromaguard_weights).expanduser(), device)
    luma_model = load_guard(Path(args.lumaguard_weights).expanduser(), device)

    chroma_params = dict(CHROMA_PRESETS[args.chroma_preset])
    chroma_gate = predict_gate(chroma_model, image, device=device, tile_size=args.tile_size, overlap=args.tile_overlap)
    chroma_gate = np.clip(chroma_gate * chroma_params["gate_gain"] + chroma_params["gate_bias"], 0.0, 1.0).astype(
        np.float32, copy=False
    )
    chroma_out = apply_chroma_nr_with_gate(
        image,
        chroma_gate,
        strength=chroma_params["strength"],
        chroma_sigma=chroma_params["chroma_sigma"],
        hdr_restore_peak_threshold=0.95,
        hdr_restore_threshold=0.85,
        hdr_restore_transition=0.25,
    )

    luma_params = dict(LUMA_PRESETS[args.luma_preset])
    luma_gate = predict_gate(luma_model, image, device=device, tile_size=args.tile_size, overlap=args.tile_overlap)
    luma_gate = np.clip(luma_gate * luma_params["gate_gain"] + luma_params["gate_bias"], 0.0, 1.0).astype(
        np.float32, copy=False
    )
    luma_out, luma_stats, luma_gate = apply_lumaguard_stage(
        chroma_out,
        image,
        luma_gate,
        strength=luma_params["strength"],
        radius=int(luma_params["radius"]),
        eps=float(luma_params["eps"]),
        safety_gate_gain=float(args.luma_safety_gate_gain),
        use_safety_gate=not args.no_luma_safety_gate,
    )

    polish_params = POLISH_PRESETS[args.polish_preset]
    if polish_params is None:
        output = luma_out
        polish_stats = None
        polish_gate = None
    else:
        output, polish_stats, polish_gate = smooth_chroma(
            luma_out,
            **polish_params,
            highlight_threshold=1.0,
            highlight_transition=0.2,
            hdr_restore_peak_threshold=0.95,
            hdr_restore_threshold=0.85,
            hdr_restore_transition=0.25,
        )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    chroma_gate_path = out_dir / f"{name}_chromaguard_gate.png"
    luma_gate_path = out_dir / f"{name}_lumaguard_gate.png"
    polish_gate_path = out_dir / f"{name}_polish_gate.png"
    meta_path = out_dir / f"{name}.json"

    write_exr(exr_path, output)
    write_tiff(tiff_path, output)
    Image.fromarray(make_preview(output, exposure=args.preview_exposure, tone=args.preview_tone)).save(preview_path)
    Image.fromarray(np.clip(chroma_gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(chroma_gate_path)
    Image.fromarray(np.clip(luma_gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(luma_gate_path)
    if polish_gate is not None:
        Image.fromarray(np.clip(polish_gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(polish_gate_path)

    meta = {
        "input": str(input_path),
        "weights": {
            "chromaguard": str(Path(args.chromaguard_weights).expanduser()),
            "lumaguard": str(Path(args.lumaguard_weights).expanduser()),
        },
        "params": {
            "chroma_preset": args.chroma_preset,
            "chroma": chroma_params,
            "luma_preset": args.luma_preset,
            "luma": luma_params,
            "polish_preset": args.polish_preset,
            "polish": polish_params,
        },
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "chromaguard_gate": str(chroma_gate_path),
            "lumaguard_gate": str(luma_gate_path),
            "polish_gate": str(polish_gate_path) if polish_gate is not None else None,
        },
        "gates": {
            "chromaguard": {
                "mean": float(np.mean(chroma_gate)),
                "p50": float(np.quantile(chroma_gate, 0.50)),
                "p90": float(np.quantile(chroma_gate, 0.90)),
                "p99": float(np.quantile(chroma_gate, 0.99)),
            },
            "lumaguard": luma_stats,
            "polish": polish_stats,
        },
        "input_stats": image_stats(image),
        "output_stats": image_stats(output),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
