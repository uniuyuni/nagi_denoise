"""Run a NagiPerfect checkpoint on an EXR/TIFF image and write diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "nagi_nr" / "src"))

from nagi_nr.nagiperfect import NagiPerfect, build_nagiperfect_preset
from apply_chroma_speckle_filter import PRESETS as CHROMA_SPECKLE_PRESETS
from apply_chroma_speckle_filter import apply_chroma_speckle_filter
from apply_luma_hf_shrink_filter import PRESETS as LUMA_HF_PRESETS
from apply_luma_hf_shrink_filter import apply_luma_hf_shrink
from perfect_nr_probe import image_stats, make_preview, read_image
from perfect_nr_detail_guard import write_exr, write_tiff


POST_CHROMA_OVERSHRINK_PRESETS = {
    "off": 1.0,
    "balanced": 1.6,
    "quality": 2.0,
}

POST_LUMA_HF_PRESETS = {
    "off": None,
    "balanced": "xstrong",
    "quality": "ultra",
}

POST_CHROMA_SPECKLE_PRESETS = {
    "off": None,
    "balanced": "mild",
    "quality": "quality",
    "xstrong": "xstrong",
}


def linear_to_srgb_hdr(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    is_low = x <= 0.0031308
    low = x * 12.92
    base = np.where(is_low, 1.0, np.maximum(x, 0.0))
    high = 1.055 * np.power(base, 1.0 / 2.4) - 0.055
    return np.where(is_low, low, high).astype(np.float32, copy=False)


def srgb_to_linear_hdr(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    low = x / 12.92
    high = np.power((np.maximum(x, 0.0) + 0.055) / 1.055, 2.4)
    return np.where(x <= 0.04045, low, high).astype(np.float32, copy=False)


def post_chroma_overshrink(
    image: np.ndarray,
    gate: np.ndarray,
    *,
    strength: float,
    kernel_size: int,
) -> np.ndarray:
    extra = max(float(strength) - 1.0, 0.0)
    if extra <= 0.0:
        return image
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    display = linear_to_srgb_hdr(np.clip(image, 0.0, None))
    y = (display[..., 0:1] * 0.299 + display[..., 1:2] * 0.587 + display[..., 2:3] * 0.114).astype(np.float32)
    chroma = display - y
    smooth = np.empty_like(chroma)
    for channel in range(3):
        smooth[..., channel] = uniform_filter(chroma[..., channel], size=k, mode="reflect")
    high = chroma - smooth
    corrected_display = np.maximum(display - extra * np.clip(gate, 0.0, 1.0)[..., None] * high, 0.0)
    return np.clip(srgb_to_linear_hdr(corrected_display), 0.0, None)


def load_model(weights: Path, device: torch.device, state_key: str) -> NagiPerfect:
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = dict((cfg.get("model") or {}))
    model_cfg.pop("kind", None)
    preset = model_cfg.pop("preset", None)
    if preset:
        model = build_nagiperfect_preset(str(preset))
        if model_cfg:
            merged = {
                "img_channels": model.img_channels,
                "width": model.width,
                "enc_blk_nums": model.enc_blk_nums,
                "middle_blk_num": model.middle_blk_num,
                "dec_blk_nums": model.dec_blk_nums,
                "asinh_k": model.asinh_k,
                "detail_scale": model.detail_scale,
                "compressed_output_clamp": model.compressed_output_clamp,
                "confidence_bias": model._confidence_bias,
                "highlight_protect_threshold": model.highlight_protect_threshold,
                "highlight_protect_transition": model.highlight_protect_transition,
                "highlight_protect_strength": model.highlight_protect_strength,
                "chroma_branch": model.chroma_branch,
                "chroma_branch_scale": model.chroma_branch_scale,
                "chroma_gate_threshold": model.chroma_gate_threshold,
                "chroma_gate_transition": model.chroma_gate_transition,
                "chroma_gate_kernel_size": model.chroma_gate_kernel_size,
                "chroma_gate_highlight_threshold": model.chroma_gate_highlight_threshold,
                "chroma_gate_highlight_transition": model.chroma_gate_highlight_transition,
                "chroma_gate_strength": model.chroma_gate_strength,
                "chroma_branch_width": model.chroma_branch_width,
                "chroma_branch_blocks": model.chroma_branch_blocks,
                "chroma_branch_use_input": model.chroma_branch_use_input,
                "chroma_smooth_branch": model.chroma_smooth_branch,
                "chroma_smooth_strength": model.chroma_smooth_strength,
                "chroma_smooth_kernel_size": model.chroma_smooth_kernel_size,
                "chroma_smooth_gate_bias": model.chroma_smooth_gate_bias,
                "chroma_smooth_gate_scale": model.chroma_smooth_gate_scale,
                "chroma_cleanup_branch": model.chroma_cleanup_branch,
                "chroma_cleanup_mode": model.chroma_cleanup_mode,
                "chroma_cleanup_strength": model.chroma_cleanup_strength,
                "chroma_cleanup_kernel_size": model.chroma_cleanup_kernel_size,
                "chroma_cleanup_width": model.chroma_cleanup_width,
                "chroma_cleanup_blocks": model.chroma_cleanup_blocks,
                "luma_smooth_branch": model.luma_smooth_branch,
                "luma_smooth_strength": model.luma_smooth_strength,
                "luma_smooth_kernel_size": model.luma_smooth_kernel_size,
                "luma_smooth_gate_bias": model.luma_smooth_gate_bias,
                "luma_smooth_gate_scale": model.luma_smooth_gate_scale,
            }
            merged.update(model_cfg)
            model = NagiPerfect(**merged)
    else:
        model = NagiPerfect(**model_cfg)

    state = ckpt.get(state_key)
    if state is None:
        raise KeyError(f"checkpoint has no state key {state_key!r}")
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def run_full_image(model: NagiPerfect, image: np.ndarray, device: torch.device) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rgb = np.nan_to_num(image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device=device)
    with torch.inference_mode():
        aux = model(x, return_aux=True)
    out = aux["output"].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    extras = {
        "detail": aux["detail"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32),
        "detail_confidence": aux["detail_confidence"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32),
        "detail_applied": aux["detail_applied"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32),
        "highlight_guard": aux["highlight_guard"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32),
    }
    for key in ("chroma_smooth_gate", "luma_smooth_gate"):
        if key in aux:
            extras[key] = aux[key].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    if "chroma_cleanup_gate" in aux:
        extras["chroma_cleanup_gate"] = aux["chroma_cleanup_gate"].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return out, extras


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size, 0), stride))
    last = length - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def _blend_window(height: int, width: int, y0: int, x0: int, full_h: int, full_w: int, overlap: int) -> np.ndarray:
    wy = np.ones(height, dtype=np.float32)
    wx = np.ones(width, dtype=np.float32)
    if overlap > 0:
        oy = min(overlap, height // 2)
        ox = min(overlap, width // 2)
        if y0 > 0 and oy > 0:
            wy[:oy] = np.linspace(0.0, 1.0, oy, endpoint=False, dtype=np.float32)
        if y0 + height < full_h and oy > 0:
            wy[-oy:] = np.linspace(1.0, 0.0, oy, endpoint=False, dtype=np.float32)
        if x0 > 0 and ox > 0:
            wx[:ox] = np.linspace(0.0, 1.0, ox, endpoint=False, dtype=np.float32)
        if x0 + width < full_w and ox > 0:
            wx[-ox:] = np.linspace(1.0, 0.0, ox, endpoint=False, dtype=np.float32)
    return wy[:, None] * wx[None, :]


def _highlight_guard_numpy(
    image: np.ndarray,
    threshold: float,
    transition: float,
    strength: float,
) -> np.ndarray:
    strength = float(strength)
    if strength <= 0.0:
        return np.zeros(image.shape[:2], dtype=np.float32)
    rgb = np.maximum(image[..., :3].astype(np.float32, copy=False), 0.0)
    y = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    if transition <= 0.0:
        mask = (y > threshold).astype(np.float32)
    else:
        z = np.clip((y - threshold) / max(float(transition), 1.0e-6), -80.0, 80.0)
        mask = 1.0 / (1.0 + np.exp(-z))
    return (mask * min(max(strength, 0.0), 1.0)).astype(np.float32, copy=False)


def run_tiled_image(
    model: NagiPerfect,
    image: np.ndarray,
    device: torch.device,
    tile_size: int,
    overlap: int,
    diagnostics: bool = True,
    progress: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rgb = np.nan_to_num(image[..., :3].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    h, w = rgb.shape[:2]
    tile_size = int(tile_size)
    overlap = max(0, int(overlap))
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        if diagnostics:
            return run_full_image(model, image, device=device)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device=device)
        with torch.inference_mode():
            out_t = model(x)
        out = out_t.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        extras = {
            "highlight_guard": _highlight_guard_numpy(
                rgb,
                threshold=model.highlight_protect_threshold,
                transition=model.highlight_protect_transition,
                strength=model.highlight_protect_strength,
            )
        }
        return out, extras
    if overlap * 2 >= tile_size:
        raise ValueError("tile overlap must be less than half of tile size")

    stride = tile_size - overlap
    y_starts = _tile_starts(h, tile_size, stride)
    x_starts = _tile_starts(w, tile_size, stride)
    accum = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    extras_accum = {
        "detail": np.zeros((h, w), dtype=np.float32),
        "detail_confidence": np.zeros((h, w), dtype=np.float32),
        "detail_applied": np.zeros((h, w), dtype=np.float32),
        "highlight_guard": np.zeros((h, w), dtype=np.float32),
    } if diagnostics else {}
    if diagnostics and getattr(model, "chroma_smooth_branch", False):
        extras_accum["chroma_smooth_gate"] = np.zeros((h, w), dtype=np.float32)
    if diagnostics and getattr(model, "chroma_cleanup_branch", False):
        extras_accum["chroma_cleanup_gate"] = np.zeros((h, w), dtype=np.float32)
    if diagnostics and getattr(model, "luma_smooth_branch", False):
        extras_accum["luma_smooth_gate"] = np.zeros((h, w), dtype=np.float32)
    total = len(y_starts) * len(x_starts)
    done = 0

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile = rgb[y0:y1, x0:x1]
            x = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).unsqueeze(0).to(device=device)
            with torch.inference_mode():
                pred = model(x, return_aux=diagnostics)
            if diagnostics:
                aux = pred
                out_t = aux["output"]
            else:
                aux = {}
                out_t = pred
            out = out_t.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            win = _blend_window(y1 - y0, x1 - x0, y0, x0, h, w, overlap)
            win3 = win[..., None]
            accum[y0:y1, x0:x1] += out * win3
            weight[y0:y1, x0:x1] += win3
            for key in extras_accum:
                if key not in aux:
                    continue
                arr = aux[key].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
                extras_accum[key][y0:y1, x0:x1] += arr * win
            done += 1
            if progress:
                print(f"tile {done}/{total} y={y0}:{y1} x={x0}:{x1}", flush=True)

    weight = np.maximum(weight, 1.0e-6)
    output = accum / weight
    if diagnostics:
        extras = {key: value / weight[..., 0] for key, value in extras_accum.items()}
    else:
        extras = {
            "highlight_guard": _highlight_guard_numpy(
                rgb,
                threshold=model.highlight_protect_threshold,
                transition=model.highlight_protect_transition,
                strength=model.highlight_protect_strength,
            )
        }
    return output, extras


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise EXR/TIFF with NagiPerfect.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", default="runs/perfect_nr/nagiperfect_exr")
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--state-key", default="state_dict")
    parser.add_argument("--preview-exposure", type=float, default=1.0)
    parser.add_argument("--preview-tone", choices=["reinhard", "clip"], default="reinhard")
    parser.add_argument("--highlight-protect-threshold", type=float, default=None)
    parser.add_argument("--highlight-protect-transition", type=float, default=None)
    parser.add_argument("--highlight-protect-strength", type=float, default=None)
    parser.add_argument("--chroma-smooth-strength", type=float, default=None)
    parser.add_argument("--chroma-smooth-gate-bias", type=float, default=None)
    parser.add_argument(
        "--post-chroma-overshrink-preset",
        choices=sorted(POST_CHROMA_OVERSHRINK_PRESETS),
        default="off",
        help="Post chroma cleanup preset. 'quality' is the current best real-photo setting.",
    )
    parser.add_argument(
        "--post-chroma-overshrink-strength",
        type=float,
        default=None,
        help="Override the post chroma cleanup strength. Values above 1.0 enable cleanup.",
    )
    parser.add_argument("--post-chroma-overshrink-kernel-size", type=int, default=9)
    parser.add_argument(
        "--post-chroma-speckle-preset",
        choices=sorted(POST_CHROMA_SPECKLE_PRESETS),
        default="off",
        help="Post multi-axis display-chroma speckle guard preset.",
    )
    parser.add_argument(
        "--post-luma-hf-preset",
        choices=sorted(POST_LUMA_HF_PRESETS),
        default="off",
        help="Post display-luma grain cleanup preset. 'quality' maps to the validated ultra shrink.",
    )
    parser.add_argument("--luma-smooth-strength", type=float, default=None)
    parser.add_argument("--luma-smooth-gate-bias", type=float, default=None)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--fast", action="store_true", help="Skip auxiliary diagnostic tensors for faster inference.")
    parser.add_argument("--quiet-tiles", action="store_true", help="Do not print per-tile progress.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    weights = Path(args.weights).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or input_path.stem.replace(" ", "_")
    device = torch.device(args.device)

    image = read_image(input_path)
    model = load_model(weights, device=device, state_key=args.state_key)
    if args.highlight_protect_threshold is not None:
        model.highlight_protect_threshold = float(args.highlight_protect_threshold)
    if args.highlight_protect_transition is not None:
        model.highlight_protect_transition = float(args.highlight_protect_transition)
    if args.highlight_protect_strength is not None:
        model.highlight_protect_strength = float(args.highlight_protect_strength)
    if args.chroma_smooth_strength is not None and hasattr(model, "chroma_smooth_strength"):
        model.chroma_smooth_strength = float(args.chroma_smooth_strength)
    if args.chroma_smooth_gate_bias is not None and getattr(model, "chroma_smooth_head", None) is not None:
        with torch.no_grad():
            model.chroma_smooth_head.bias.fill_(float(args.chroma_smooth_gate_bias))
    if args.luma_smooth_strength is not None and hasattr(model, "luma_smooth_strength"):
        model.luma_smooth_strength = float(args.luma_smooth_strength)
    if args.luma_smooth_gate_bias is not None and getattr(model, "luma_smooth_head", None) is not None:
        with torch.no_grad():
            model.luma_smooth_head.bias.fill_(float(args.luma_smooth_gate_bias))
    post_chroma_overshrink_strength = float(
        args.post_chroma_overshrink_strength
        if args.post_chroma_overshrink_strength is not None
        else POST_CHROMA_OVERSHRINK_PRESETS[args.post_chroma_overshrink_preset]
    )
    need_diagnostics = (not bool(args.fast)) or post_chroma_overshrink_strength > 1.0
    output, extras = run_tiled_image(
        model,
        image,
        device=device,
        tile_size=int(args.tile_size),
        overlap=int(args.tile_overlap),
        diagnostics=need_diagnostics,
        progress=not bool(args.quiet_tiles),
    )
    if post_chroma_overshrink_strength > 1.0:
        if "chroma_smooth_gate" not in extras:
            raise RuntimeError("post chroma overshrink requires chroma_smooth_gate diagnostics")
        output = post_chroma_overshrink(
            output,
            extras["chroma_smooth_gate"],
            strength=post_chroma_overshrink_strength,
            kernel_size=int(args.post_chroma_overshrink_kernel_size),
        )
    post_chroma_speckle_stats = None
    post_chroma_speckle_blend = None
    post_chroma_speckle_source = POST_CHROMA_SPECKLE_PRESETS[args.post_chroma_speckle_preset]
    if post_chroma_speckle_source is not None:
        params = dict(CHROMA_SPECKLE_PRESETS[post_chroma_speckle_source])
        output, post_chroma_speckle_stats, post_chroma_speckle_blend = apply_chroma_speckle_filter(
            output,
            image,
            strength=float(params["strength"]),
            chroma_sigma=float(params["chroma_sigma"]),
            median_size=int(params["median_size"]),
            speckle_threshold=float(params["speckle_threshold"]),
            speckle_transition=float(params["speckle_transition"]),
            local_sigma=float(params["local_sigma"]),
            local_gain=float(params["local_gain"]),
            axis_boost=float(params["axis_boost"]),
            axis_threshold=float(params["axis_threshold"]),
            axis_transition=float(params["axis_transition"]),
            magenta_boost=float(params["magenta_boost"]),
            magenta_threshold=float(params["magenta_threshold"]),
            magenta_transition=float(params["magenta_transition"]),
            structure_sigma=1.2,
            detail_sigma=2.8,
            detail_threshold=0.018,
            detail_transition=0.010,
            edge_sigma=1.0,
            edge_threshold=0.030,
            edge_transition=0.015,
            highlight_threshold=float(params["highlight_threshold"]),
            highlight_transition=float(params["highlight_transition"]),
            hdr_restore_peak_threshold=float(params["hdr_restore_peak_threshold"]),
            hdr_restore_threshold=float(params["hdr_restore_threshold"]),
            hdr_restore_transition=float(params["hdr_restore_transition"]),
        )
    post_luma_hf_stats = None
    post_luma_hf_gate = None
    post_luma_hf_source = POST_LUMA_HF_PRESETS[args.post_luma_hf_preset]
    if post_luma_hf_source is not None:
        params = dict(LUMA_HF_PRESETS[post_luma_hf_source])
        output, post_luma_hf_stats, post_luma_hf_gate = apply_luma_hf_shrink(
            output,
            image,
            strength=float(params["strength"]),
            low_sigma=float(params["low_sigma"]),
            shrink_threshold=float(params["shrink_threshold"]),
            detail_preserve_threshold=float(params["detail_preserve_threshold"]),
            detail_preserve_transition=float(params["detail_preserve_transition"]),
            shadow_boost=float(params["shadow_boost"]),
            shadow_threshold=0.18,
            shadow_transition=0.08,
            structure_sigma=1.2,
            detail_sigma=2.8,
            detail_threshold=0.018,
            detail_transition=0.010,
            edge_sigma=1.0,
            edge_threshold=0.030,
            edge_transition=0.015,
            highlight_threshold=1.0,
            highlight_transition=0.25,
            hdr_restore_peak_threshold=0.95,
            hdr_restore_threshold=0.85,
            hdr_restore_transition=0.25,
        )

    exr_path = out_dir / f"{name}_nagiperfect.exr"
    tiff_path = out_dir / f"{name}_nagiperfect.tiff"
    preview_path = out_dir / f"{name}_nagiperfect_preview.png"
    confidence_path = out_dir / f"{name}_detail_confidence.png"
    chroma_smooth_gate_path = out_dir / f"{name}_chroma_smooth_gate.png"
    chroma_cleanup_gate_path = out_dir / f"{name}_chroma_cleanup_gate.png"
    luma_smooth_gate_path = out_dir / f"{name}_luma_smooth_gate.png"
    chroma_speckle_blend_path = out_dir / f"{name}_post_chroma_speckle_blend.png"
    luma_hf_gate_path = out_dir / f"{name}_post_luma_hf_gate.png"
    json_path = out_dir / f"{name}_nagiperfect.json"

    write_exr(exr_path, output)
    write_tiff(tiff_path, output)
    Image.fromarray(make_preview(output, exposure=args.preview_exposure, tone=args.preview_tone)).save(preview_path)
    if "detail_confidence" in extras:
        Image.fromarray((np.clip(extras["detail_confidence"], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(
            confidence_path
        )
    if "chroma_smooth_gate" in extras:
        Image.fromarray((np.clip(extras["chroma_smooth_gate"], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(
            chroma_smooth_gate_path
        )
    if "chroma_cleanup_gate" in extras:
        Image.fromarray((np.clip(extras["chroma_cleanup_gate"], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(
            chroma_cleanup_gate_path
        )
    if "luma_smooth_gate" in extras:
        Image.fromarray((np.clip(extras["luma_smooth_gate"], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(
            luma_smooth_gate_path
        )
    if post_luma_hf_gate is not None:
        Image.fromarray((np.clip(post_luma_hf_gate, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(
            luma_hf_gate_path
        )
    if post_chroma_speckle_blend is not None:
        Image.fromarray((np.clip(post_chroma_speckle_blend, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(
            chroma_speckle_blend_path
        )

    meta = {
        "input": str(input_path),
        "weights": str(weights),
        "state_key": args.state_key,
        "device": str(device),
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "detail_confidence": str(confidence_path) if "detail_confidence" in extras else None,
            "chroma_smooth_gate": str(chroma_smooth_gate_path) if "chroma_smooth_gate" in extras else None,
            "chroma_cleanup_gate": str(chroma_cleanup_gate_path) if "chroma_cleanup_gate" in extras else None,
            "luma_smooth_gate": str(luma_smooth_gate_path) if "luma_smooth_gate" in extras else None,
            "post_chroma_speckle_blend": str(chroma_speckle_blend_path) if post_chroma_speckle_blend is not None else None,
            "post_luma_hf_gate": str(luma_hf_gate_path) if post_luma_hf_gate is not None else None,
        },
        "input_stats": image_stats(image),
        "output_stats": image_stats(output),
        "highlight_protect": {
            "threshold": model.highlight_protect_threshold,
            "transition": model.highlight_protect_transition,
            "strength": model.highlight_protect_strength,
            "mask_mean": float(np.mean(extras["highlight_guard"])),
            "mask_p95": float(np.percentile(extras["highlight_guard"], 95)),
            "mask_p99": float(np.percentile(extras["highlight_guard"], 99)),
        },
        "chroma_smooth": {
            "enabled": bool(getattr(model, "chroma_smooth_branch", False)),
            "strength": float(getattr(model, "chroma_smooth_strength", 0.0)),
            "kernel_size": int(getattr(model, "chroma_smooth_kernel_size", 0)),
            "gate_bias_override": args.chroma_smooth_gate_bias,
            "post_overshrink_preset": args.post_chroma_overshrink_preset,
            "post_overshrink_strength": post_chroma_overshrink_strength,
            "post_overshrink_kernel_size": int(args.post_chroma_overshrink_kernel_size),
        },
        "chroma_cleanup": {
            "enabled": bool(getattr(model, "chroma_cleanup_branch", False)),
            "mode": str(getattr(model, "chroma_cleanup_mode", "additive_chroma_delta")),
            "strength": float(getattr(model, "chroma_cleanup_strength", 0.0)),
            "kernel_size": int(getattr(model, "chroma_cleanup_kernel_size", 0)),
            "width": int(getattr(model, "chroma_cleanup_width", 0)),
            "blocks": int(getattr(model, "chroma_cleanup_blocks", 0)),
        },
        "luma_smooth": {
            "enabled": bool(getattr(model, "luma_smooth_branch", False)),
            "strength": float(getattr(model, "luma_smooth_strength", 0.0)),
            "kernel_size": int(getattr(model, "luma_smooth_kernel_size", 0)),
            "gate_bias_override": args.luma_smooth_gate_bias,
        },
        "post_luma_hf": {
            "preset": args.post_luma_hf_preset,
            "source_preset": post_luma_hf_source,
            "stats": post_luma_hf_stats,
        },
        "post_chroma_speckle": {
            "preset": args.post_chroma_speckle_preset,
            "source_preset": post_chroma_speckle_source,
            "stats": post_chroma_speckle_stats,
        },
        "tiling": {
            "tile_size": int(args.tile_size),
            "tile_overlap": int(args.tile_overlap),
            "fast": bool(args.fast),
        },
    }
    if "detail_confidence" in extras:
        meta["detail_confidence_mean"] = float(np.mean(extras["detail_confidence"]))
    if "detail_applied" in extras:
        meta["detail_applied_abs_mean"] = float(np.mean(np.abs(extras["detail_applied"])))
    for key in ("chroma_smooth_gate", "chroma_cleanup_gate", "luma_smooth_gate"):
        if key in extras:
            meta[key] = {
                "mean": float(np.mean(extras[key])),
                "p50": float(np.percentile(extras[key], 50)),
                "p90": float(np.percentile(extras[key], 90)),
                "p99": float(np.percentile(extras[key], 99)),
                "max": float(np.max(extras[key])),
            }
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"wrote {exr_path}")
    print(f"wrote {preview_path}")


if __name__ == "__main__":
    main()
