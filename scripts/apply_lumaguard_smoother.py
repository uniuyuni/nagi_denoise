"""Apply LumaGuard AI-gated guided luma smoothing."""
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

from nagi_nr.chromaguard import ChromaGuard
from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, smoothstep, srgb_to_linear_np
from apply_guided_luma_smoother import guided_filter_gray, make_luma_gate
from denoise_exr_chromaguard import predict_gate
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


PRESETS = {
    "conservative": {"gate_gain": 1.0, "gate_bias": 0.0, "strength": 0.55, "radius": 4, "eps": 0.006},
    "balanced": {"gate_gain": 1.15, "gate_bias": 0.0, "strength": 0.75, "radius": 5, "eps": 0.006},
    "strong": {"gate_gain": 1.35, "gate_bias": 0.0, "strength": 0.95, "radius": 5, "eps": 0.006},
}


def load_guard(weights: Path, device: torch.device) -> ChromaGuard:
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model_cfg = dict((ckpt.get("config") or {}).get("model") or {})
    model = ChromaGuard(**model_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply LumaGuard guided luma smoothing.")
    parser.add_argument("--input", required=True, help="Chroma-denoised image to smooth.")
    parser.add_argument("--guide-input", required=True, help="Original noisy image for model/gating features.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="balanced")
    parser.add_argument("--gate-gain", type=float, default=None)
    parser.add_argument("--gate-bias", type=float, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--no-safety-gate", action="store_true")
    parser.add_argument("--safety-gate-gain", type=float, default=1.15)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    guide_path = Path(args.guide_input).expanduser()
    weights = Path(args.weights).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem}_lumaguard"
    preset = dict(PRESETS[args.preset])
    if args.gate_gain is not None:
        preset["gate_gain"] = float(args.gate_gain)
    if args.gate_bias is not None:
        preset["gate_bias"] = float(args.gate_bias)
    if args.strength is not None:
        preset["strength"] = float(args.strength)
    if args.radius is not None:
        preset["radius"] = int(args.radius)
    if args.eps is not None:
        preset["eps"] = float(args.eps)

    image = read_image(input_path)
    guide = read_image(guide_path)
    if image.shape[:2] != guide.shape[:2]:
        raise ValueError(f"shape mismatch: input={image.shape}, guide={guide.shape}")

    device = torch.device(args.device)
    model = load_guard(weights, device)
    gate = predict_gate(model, guide, device=device, tile_size=int(args.tile_size), overlap=int(args.tile_overlap))
    gate = np.clip(gate * preset["gate_gain"] + preset["gate_bias"], 0.0, 1.0).astype(np.float32, copy=False)

    base = np.nan_to_num(image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    guide_base = np.nan_to_num(guide[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    display = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    guide_display = np.clip(linear_to_srgb_np(guide_base), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)
    guide_y = luma(guide_display, LUMA_SRGB)
    guide_y_linear = luma(np.clip(guide_base, 0.0, None), LUMA_LINEAR)
    if not args.no_safety_gate:
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
        gate = np.minimum(gate, np.clip(safety_gate * float(args.safety_gate_gain), 0.0, 1.0))
    structure = gaussian_filter(guide_y, sigma=1.0, mode="reflect")
    y_smooth = guided_filter_gray(structure, y, radius=int(preset["radius"]), eps=float(preset["eps"]))
    blend = np.clip(gate * float(preset["strength"]), 0.0, 1.0)[..., None]
    out_y = y * (1.0 - blend[..., 0]) + y_smooth * blend[..., 0]
    chroma = display / np.maximum(y[..., None], 1.0e-6)
    out_display = np.clip(chroma * np.maximum(out_y[..., None], 0.0), 0.0, 1.0)
    out = srgb_to_linear_np(out_display)

    y_linear = luma(base, LUMA_LINEAR)
    peak_linear = np.max(base, axis=2)
    hdr_signal = np.maximum(y_linear - 0.85, peak_linear - 0.95)
    hdr_restore = smoothstep(hdr_signal / 0.25)
    out = out * (1.0 - hdr_restore[..., None]) + base * hdr_restore[..., None]
    stats = {
        "strength": float(preset["strength"]),
        "radius": int(preset["radius"]),
        "eps": float(preset["eps"]),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p99": float(np.quantile(hdr_restore, 0.99)),
    }

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    meta = {
        "input": str(input_path),
        "guide_input": str(guide_path),
        "weights": str(weights),
        "outputs": {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path), "gate": str(gate_path)},
        "params": {"preset": args.preset, **preset},
        "safety_gate": {
            "enabled": not args.no_safety_gate,
            "gain": float(args.safety_gate_gain),
        },
        "gate": {
            "mean": float(np.mean(gate)),
            "p50": float(np.quantile(gate, 0.50)),
            "p90": float(np.quantile(gate, 0.90)),
            "p99": float(np.quantile(gate, 0.99)),
        },
        "smoother": stats,
        "input_stats": image_stats(image),
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
