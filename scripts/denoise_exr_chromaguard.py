"""Run ChromaGuard AI-gated HDR-safe chroma NR on EXR/TIFF images."""
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
from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, luma, linear_to_srgb_np, smoothstep, srgb_to_linear_np
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


def load_guard(weights: Path, device: torch.device) -> ChromaGuard:
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model_cfg = dict((ckpt.get("config") or {}).get("model") or {})
    model = ChromaGuard(**model_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def predict_gate(model: ChromaGuard, image: np.ndarray, device: torch.device, tile_size: int, overlap: int) -> np.ndarray:
    rgb = np.nan_to_num(image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    h, w = rgb.shape[:2]
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device=device)
        with torch.inference_mode():
            gate = model(x)
        return gate.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    if overlap * 2 >= tile_size:
        raise ValueError("tile overlap must be less than half of tile size")

    def starts(length: int) -> list[int]:
        stride = tile_size - overlap
        if length <= tile_size:
            return [0]
        out = list(range(0, max(length - tile_size, 0), stride))
        last = length - tile_size
        if not out or out[-1] != last:
            out.append(last)
        return out

    def window(th: int, tw: int, y0: int, x0: int) -> np.ndarray:
        wy = np.ones(th, dtype=np.float32)
        wx = np.ones(tw, dtype=np.float32)
        oy = min(overlap, th // 2)
        ox = min(overlap, tw // 2)
        if y0 > 0 and oy > 0:
            wy[:oy] = np.linspace(0.0, 1.0, oy, endpoint=False, dtype=np.float32)
        if y0 + th < h and oy > 0:
            wy[-oy:] = np.linspace(1.0, 0.0, oy, endpoint=False, dtype=np.float32)
        if x0 > 0 and ox > 0:
            wx[:ox] = np.linspace(0.0, 1.0, ox, endpoint=False, dtype=np.float32)
        if x0 + tw < w and ox > 0:
            wx[-ox:] = np.linspace(1.0, 0.0, ox, endpoint=False, dtype=np.float32)
        return wy[:, None] * wx[None, :]

    accum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    with torch.inference_mode():
        for y0 in starts(h):
            for x0 in starts(w):
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                tile = rgb[y0:y1, x0:x1]
                x = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).unsqueeze(0).to(device=device)
                pred = model(x).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
                win = window(y1 - y0, x1 - x0, y0, x0)
                accum[y0:y1, x0:x1] += pred * win
                weight[y0:y1, x0:x1] += win
    return accum / np.maximum(weight, 1.0e-6)


def apply_chroma_nr_with_gate(
    image: np.ndarray,
    gate: np.ndarray,
    *,
    strength: float,
    chroma_sigma: float,
    hdr_restore_peak_threshold: float,
    hdr_restore_threshold: float,
    hdr_restore_transition: float,
) -> np.ndarray:
    base = np.nan_to_num(image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    display = np.clip(linear_to_srgb_np(base), 0.0, 1.0)
    y = luma(display, LUMA_SRGB)[..., None]
    chroma = display - y
    low_chroma = gaussian_filter(chroma, sigma=(float(chroma_sigma), float(chroma_sigma), 0.0), mode="reflect")
    blend = np.clip(gate * float(strength), 0.0, 1.0)[..., None]
    out_display = y + chroma * (1.0 - blend) + low_chroma * blend
    out = srgb_to_linear_np(out_display)

    y_linear = luma(base, LUMA_LINEAR)
    peak_linear = np.max(base, axis=2)
    hdr_signal = np.maximum(y_linear - float(hdr_restore_threshold), peak_linear - float(hdr_restore_peak_threshold))
    hdr_restore = smoothstep(hdr_signal / max(float(hdr_restore_transition), 1.0e-6))
    return (out * (1.0 - hdr_restore[..., None]) + base * hdr_restore[..., None]).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChromaGuard AI-gated chroma NR.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", default="runs/perfect_nr/chromaguard")
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--chroma-sigma", type=float, default=2.0)
    parser.add_argument("--gate-gain", type=float, default=1.0)
    parser.add_argument("--gate-bias", type=float, default=0.0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--hdr-restore-peak-threshold", type=float, default=0.95)
    parser.add_argument("--hdr-restore-threshold", type=float, default=0.85)
    parser.add_argument("--hdr-restore-transition", type=float, default=0.25)
    parser.add_argument("--preview-exposure", type=float, default=1.0)
    parser.add_argument("--preview-tone", choices=["reinhard", "clip"], default="reinhard")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    weights = Path(args.weights).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem.replace(' ', '_')}_chromaguard"
    device = torch.device(args.device)

    image = read_image(input_path)
    model = load_guard(weights, device=device)
    gate = predict_gate(model, image, device=device, tile_size=int(args.tile_size), overlap=int(args.tile_overlap))
    gate = np.clip(gate * float(args.gate_gain) + float(args.gate_bias), 0.0, 1.0).astype(np.float32, copy=False)
    output = apply_chroma_nr_with_gate(
        image,
        gate,
        strength=float(args.strength),
        chroma_sigma=float(args.chroma_sigma),
        hdr_restore_peak_threshold=float(args.hdr_restore_peak_threshold),
        hdr_restore_threshold=float(args.hdr_restore_threshold),
        hdr_restore_transition=float(args.hdr_restore_transition),
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}.json"

    write_exr(exr_path, output)
    write_tiff(tiff_path, output)
    Image.fromarray(make_preview(output, exposure=args.preview_exposure, tone=args.preview_tone)).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    meta = {
        "input": str(input_path),
        "weights": str(weights),
        "outputs": {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path), "gate": str(gate_path)},
        "params": {
            "strength": float(args.strength),
            "chroma_sigma": float(args.chroma_sigma),
            "gate_gain": float(args.gate_gain),
            "gate_bias": float(args.gate_bias),
        },
        "gate": {
            "mean": float(np.mean(gate)),
            "p50": float(np.quantile(gate, 0.50)),
            "p90": float(np.quantile(gate, 0.90)),
            "p99": float(np.quantile(gate, 0.99)),
        },
        "input_stats": image_stats(image),
        "output_stats": image_stats(output),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
