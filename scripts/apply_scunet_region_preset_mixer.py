"""Region-aware mixer for SCUNet-derived NR presets.

The coarse preset chooser is intentionally conservative, but a real photo can
contain hair/detail, flat sky, and blue-shadow risk at the same time. This
script runs the known useful presets and blends them with analytic full-res
region weights.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_scunet_preset_chooser import (  # noqa: E402
    DEFAULT_V2_SELECTOR,
    DEFAULT_V3_SELECTOR,
    RUN_ROOT,
    SCENES,
    apply_blue_shadow_graft_preset,
    apply_selector_preset,
    choose_preset,
    downsample,
    resolve_inputs,
    scene_metrics,
)
from apply_luma_tail_speckle_filter import sigmoid01  # noqa: E402
from perfect_nr_detail_guard import write_exr  # noqa: E402
from perfect_nr_probe import image_stats, make_preview, read_image  # noqa: E402
from train_scunet_selector import (  # noqa: E402
    LUMA_SRGB,
    choose_device,
    display,
    hf_abs,
    luma,
    make_coherent_structure_mask,
    make_texture_mask,
    signed_blue_magenta_risk,
)


def normalized_highpass(y: np.ndarray, sigma: float) -> np.ndarray:
    hp = np.abs(y - gaussian_filter(y, sigma=float(sigma), mode="reflect"))
    return hp.astype(np.float32, copy=False)


def region_weights(
    noisy: np.ndarray,
    current: np.ndarray,
    scunet: np.ndarray,
    *,
    base_preset: str,
    strategy: str,
) -> tuple[np.ndarray, dict[str, float]]:
    noisy_d = display(noisy)
    current_d = display(current)
    scunet_d = display(scunet)
    current_y = luma(current_d, LUMA_SRGB)
    scunet_y = luma(scunet_d, LUMA_SRGB)

    current_hf = hf_abs(current_y, 0.8)
    scunet_hf = hf_abs(scunet_y, 0.8)
    texture = make_texture_mask(current_d, texture_threshold=0.014, texture_transition=0.014)
    coherent = make_coherent_structure_mask(
        noisy_d,
        coherence_threshold=0.38,
        coherence_transition=0.18,
        energy_threshold=0.0048,
        energy_transition=0.0055,
    )
    signed = signed_blue_magenta_risk(scunet_d)
    shadow = sigmoid01((0.50 - current_y) / 0.16)
    tail = sigmoid01((scunet_hf - current_hf - 0.0025) / 0.0075)
    blue_evidence = signed * shadow * np.maximum(tail, np.maximum(coherent, texture))

    luma_benefit = sigmoid01((current_hf - scunet_hf - 0.0015) / 0.006)
    detail_evidence = np.maximum(texture, coherent)

    if strategy == "exploratory":
        blue_w = sigmoid01((blue_evidence - 0.20) / 0.12)
        hair_w = sigmoid01((detail_evidence - 0.52) / 0.16) * sigmoid01((0.30 - signed) / 0.14)
        hair_w *= 1.0 - 0.85 * blue_w
        flat_w = luma_benefit * sigmoid01((0.55 - detail_evidence) / 0.18)
        sky_w = np.maximum(0.0, 1.0 - np.maximum(hair_w, blue_w)) * sigmoid01((flat_w - 0.18) / 0.18)
    elif strategy == "conservative":
        blue_w = sigmoid01((blue_evidence - 0.38) / 0.09)
        hair_w = sigmoid01((detail_evidence - 0.70) / 0.10) * sigmoid01((0.20 - signed) / 0.08)
        hair_w *= 1.0 - 0.95 * blue_w
        sky_w = sigmoid01((luma_benefit - 0.66) / 0.11) * sigmoid01((0.18 - detail_evidence) / 0.08)
        sky_w *= 1.0 - 0.95 * blue_w
        if base_preset == "hair_luma":
            hair_w *= 0.0
            sky_w *= 0.0
            blue_w *= 0.65
        elif base_preset == "sky_luma":
            sky_w *= 0.0
            hair_w *= 0.55
            blue_w *= 0.55
        elif base_preset == "blue_shadow_safe":
            blue_w *= 0.0
            hair_w *= 0.15
            sky_w *= 0.15
        else:
            raise ValueError(f"unknown base preset: {base_preset!r}")
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    weights = np.stack([sky_w, hair_w, blue_w], axis=2).astype(np.float32, copy=False)
    weights = gaussian_filter(weights, sigma=(1.2, 1.2, 0.0), mode="reflect")
    weights = np.clip(weights, 0.0, 1.0)
    total = np.maximum(np.max(weights, axis=2, keepdims=True), 1.0)
    weights = weights / total

    stats = {}
    for idx, label in enumerate(("sky", "hair", "blue")):
        w = weights[..., idx]
        stats[f"{label}_mean"] = float(np.mean(w))
        stats[f"{label}_p90"] = float(np.quantile(w, 0.90))
        stats[f"{label}_p99"] = float(np.quantile(w, 0.99))
        stats[f"{label}_area50"] = float(np.mean(w > 0.50))
    stats["signed_mean"] = float(np.mean(signed))
    stats["blue_evidence_p95"] = float(np.quantile(blue_evidence, 0.95))
    stats["detail_evidence_mean"] = float(np.mean(detail_evidence))
    stats["luma_benefit_mean"] = float(np.mean(luma_benefit))
    return weights.astype(np.float32, copy=False), stats


def blend_presets(
    current: np.ndarray,
    sky: np.ndarray,
    hair: np.ndarray,
    blue: np.ndarray,
    weights: np.ndarray,
    *,
    base_preset: str,
    strength: float,
) -> np.ndarray:
    sky_w = weights[..., 0:1]
    hair_w = weights[..., 1:2]
    blue_w = weights[..., 2:3]
    if base_preset == "hair_luma":
        base = hair.copy()
    elif base_preset == "sky_luma":
        base = sky.copy()
    elif base_preset == "blue_shadow_safe":
        base = blue.copy()
    else:
        raise ValueError(f"unknown base preset: {base_preset!r}")
    base = base + (sky - base) * (sky_w * strength)
    base = base + (hair - base) * (hair_w * strength)
    base = base + (blue - base) * (blue_w * strength)
    return np.clip(base, 0.0, None).astype(np.float32, copy=False)


def apply(args: argparse.Namespace) -> None:
    scene_name, noisy_path, current_path, scunet_path = resolve_inputs(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    noisy = read_image(noisy_path)
    current = read_image(current_path)
    scunet = read_image(scunet_path)
    metrics = scene_metrics(
        downsample(noisy, args.metric_max_side),
        downsample(current, args.metric_max_side),
        downsample(scunet, args.metric_max_side),
    )
    base_preset, base_reason = choose_preset(metrics)
    if args.force_base_preset:
        base_preset = args.force_base_preset
        base_reason = "forced by CLI"

    sky, sky_stats = apply_selector_preset(
        Path(args.v3_selector),
        device,
        noisy,
        current,
        scunet,
        tile=args.tile,
        overlap=args.overlap,
        strength=2.4,
        gate_gamma=0.75,
        risk_inhibit=0.0,
        tail_inhibit=0.0,
        edge_inhibit=0.80,
    )
    hair, hair_stats = apply_selector_preset(
        Path(args.v2_selector),
        device,
        noisy,
        current,
        scunet,
        tile=args.tile,
        overlap=args.overlap,
        strength=2.4,
        gate_gamma=0.75,
        risk_inhibit=0.0,
        tail_inhibit=0.0,
        edge_inhibit=0.0,
    )
    blue, blue_stats = apply_blue_shadow_graft_preset(noisy, current, scunet)
    weights, weight_stats = region_weights(
        noisy,
        current,
        scunet,
        base_preset=base_preset,
        strategy=args.strategy,
    )
    out = blend_presets(current, sky, hair, blue, weights, base_preset=base_preset, strength=args.mix_strength)

    name = args.name or f"{scene_name}_scunet_region_preset_mixer"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=args.preview_exposure, tone=args.preview_tone)).save(preview_path)
    meta = {
        "scene": scene_name,
        "base_preset": base_preset,
        "base_reason": base_reason,
        "strategy": args.strategy,
        "inputs": {"noisy": str(noisy_path), "current": str(current_path), "scunet": str(scunet_path)},
        "outputs": {"exr": str(exr_path), "preview": str(preview_path)},
        "scene_metrics": metrics,
        "weight_stats": weight_stats,
        "preset_stats": {"sky": sky_stats, "hair": hair_stats, "blue": blue_stats},
        "output_stats": image_stats(out),
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend SCUNet presets with full-res region weights.")
    parser.add_argument("--scene", default=None, choices=sorted(SCENES))
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--noisy", default=None)
    parser.add_argument("--current", default=None)
    parser.add_argument("--scunet", default=None)
    parser.add_argument("--output-dir", default=str(RUN_ROOT / "scunet_region_preset_mixer_v1_outputs"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--mix-strength", type=float, default=1.0)
    parser.add_argument("--strategy", choices=["conservative", "exploratory"], default="conservative")
    parser.add_argument("--force-base-preset", choices=["hair_luma", "sky_luma", "blue_shadow_safe"], default=None)
    parser.add_argument("--metric-max-side", type=int, default=1400)
    parser.add_argument("--preview-exposure", type=float, default=1.0)
    parser.add_argument("--preview-tone", choices=["reinhard", "clip"], default="reinhard")
    parser.add_argument("--v2-selector", default=str(DEFAULT_V2_SELECTOR))
    parser.add_argument("--v3-selector", default=str(DEFAULT_V3_SELECTOR))
    args = parser.parse_args()
    apply(args)


if __name__ == "__main__":
    main()
