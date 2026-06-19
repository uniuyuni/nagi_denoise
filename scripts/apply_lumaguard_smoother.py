"""Apply LumaGuard AI-gated guided luma smoothing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "nagi_nr" / "src"))

from nagi_nr.chromaguard import ChromaGuard
from apply_guided_luma_smoother import apply_guided_luma_smoothing
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

    out, stats, _heuristic_gate = apply_guided_luma_smoothing(
        image,
        guide,
        strength=float(preset["strength"]),
        radius=int(preset["radius"]),
        eps=float(preset["eps"]),
        guide_sigma=1.0,
        structure_sigma=1.2,
        detail_sigma=2.6,
        detail_threshold=1.0,
        detail_transition=1.0,
        edge_sigma=1.0,
        edge_threshold=1.0,
        edge_transition=1.0,
        highlight_threshold=1.0,
        highlight_transition=0.25,
        hdr_restore_peak_threshold=0.95,
        hdr_restore_threshold=0.85,
        hdr_restore_transition=0.25,
    )

    # Re-apply the same guided smoothing internals with the learned gate by
    # blending between the ungated input and the fully-smoothed candidate.
    full, _stats_full, _ = apply_guided_luma_smoothing(
        image,
        guide,
        strength=1.0,
        radius=int(preset["radius"]),
        eps=float(preset["eps"]),
        guide_sigma=1.0,
        structure_sigma=1.2,
        detail_sigma=2.6,
        detail_threshold=1.0,
        detail_transition=1.0,
        edge_sigma=1.0,
        edge_threshold=1.0,
        edge_transition=1.0,
        highlight_threshold=1.0,
        highlight_transition=0.25,
        hdr_restore_peak_threshold=0.95,
        hdr_restore_threshold=0.85,
        hdr_restore_transition=0.25,
    )
    blend = np.clip(gate * float(preset["strength"]), 0.0, 1.0)[..., None]
    out = image[..., :3].astype(np.float32, copy=False) * (1.0 - blend) + full * blend

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
