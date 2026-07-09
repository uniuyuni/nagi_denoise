"""Evaluate region-aware flat gate strength on fixed ROIs without writing images.

This is a lightweight design probe. It reads the reference/current image and a
learned flat-cleanup gate, builds the region-aware strength map for one or more
presets, and reports ROI-level effective gate statistics. It helps decide
whether a preset is leaking cleanup into hair/branches/people before spending
full-frame output time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_region_aware_flat_gate import build_strength_map
from apply_scunet_preset_chooser import REGION_AWARE_FLAT_GATE_PRESETS
from perfect_nr_probe import read_image
from roi_noise_eval import DEFAULT_ROIS, ROI_KIND
from train_flat_cleanup_gate import align_feature_channels, choose_device, load_model, make_features_and_gate


def parse_scene_paths(scene: str, gate_dir: Path | None = None, gate_suffix: str = "flat_cleanup_gate_v12_native_pilot_v1") -> tuple[Path, Path, Path]:
    root = Path("runs/refiner_pilot_stage11_hybrid_best")
    data = Path("/Users/uniuyuni/ProjectData/test_photos")
    current_root = root / "scunet_preset_chooser_v12_flat_cleanup_auto_outputs"
    gate_root = gate_dir if gate_dir is not None else root / "flat_cleanup_gate_v12_native_pilot_v1_outputs"
    mapping = {
        "xt5_occi": (
            data / "X-T5 Occi noisy.EXR",
            current_root / "xt5_occi_scunet_preset_chooser_v12_auto.exr",
            gate_root / f"xt5_occi_{gate_suffix}_gate.png",
        ),
        "k5_dance": (
            data / "K-5 Dance noisy.EXR",
            current_root / "k5_dance_scunet_preset_chooser_v12_auto.exr",
            gate_root / f"k5_dance_{gate_suffix}_gate.png",
        ),
        "k5_ice": (
            data / "K-5 Ice noisy.EXR",
            current_root / "k5_ice_scunet_preset_chooser_v12_auto.exr",
            gate_root / f"k5_ice_{gate_suffix}_gate.png",
        ),
    }
    return mapping[scene]


def read_gate(path: Path, shape: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != (shape[1], shape[0]):
        raise ValueError(f"gate size mismatch: {img.size} != {(shape[1], shape[0])}")
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def crop_center(arr: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    x0 = max(0, min(w - size, int(x) - size // 2))
    y0 = max(0, min(h - size, int(y) - size // 2))
    return arr[y0 : y0 + size, x0 : x0 + size]


def stat_block(arr: np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=np.float32)
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
    }


def build_reopen_map(
    masks: dict[str, np.ndarray],
    *,
    reopen_strength: float,
    reopen_shadow_weight: float,
    reopen_structure_suppress: float,
    reopen_min: float,
    reopen_max: float,
    reopen_shadow_threshold: float,
    reopen_shadow_transition: float,
) -> np.ndarray:
    if reopen_strength <= 0:
        return np.ones_like(masks["flat"], dtype=np.float32)
    shadow_gate = np.clip(
        (masks["shadow_flat"] - float(reopen_shadow_threshold)) / max(float(reopen_shadow_transition), 1.0e-6),
        0.0,
        1.0,
    )
    sky_flat = np.clip(
        masks["flat"]
        * shadow_gate
        * (masks["shadow_flat"] * float(reopen_shadow_weight) + masks["flat_target"] * (1.0 - float(reopen_shadow_weight))),
        0.0,
        1.0,
    )
    safe = np.clip(1.0 - masks["structure_protect"] * float(reopen_structure_suppress), 0.0, 1.0)
    reopen = 1.0 + float(reopen_strength) * sky_flat * safe
    return np.clip(reopen, float(reopen_min), float(reopen_max)).astype(np.float32, copy=False)


def summarize_scene(
    scene: str,
    presets: list[str],
    gate_dir: Path | None = None,
    gate_suffix: str = "flat_cleanup_gate_v12_native_pilot_v1",
    checkpoint: Path | None = None,
    device_name: str = "cpu",
    reopen_strength: float = 0.0,
    reopen_shadow_weight: float = 0.85,
    reopen_structure_suppress: float = 1.0,
    reopen_min: float = 1.0,
    reopen_max: float = 1.45,
    reopen_shadow_threshold: float = 0.0,
    reopen_shadow_transition: float = 0.12,
) -> dict[str, object]:
    reference_path, current_path, gate_path = parse_scene_paths(scene, gate_dir=gate_dir, gate_suffix=gate_suffix)
    reference = read_image(reference_path)
    current = read_image(current_path)
    if checkpoint is None:
        gate = read_gate(gate_path, current.shape[:2])
        model = None
        device = None
    else:
        gate = np.zeros(current.shape[:2], dtype=np.float32)
        device = choose_device(device_name)
        model = load_model(checkpoint, device)
    detail_gate_path = Path(str(current_path).replace(".exr", "_detail_gate.png"))
    if not detail_gate_path.exists():
        detail_gate_path = current_path.with_name(current_path.stem + "_detail_gate.png")
    detail_gate = read_gate(detail_gate_path, current.shape[:2]) if detail_gate_path.exists() else np.zeros(current.shape[:2], dtype=np.float32)
    report: dict[str, object] = {
        "scene": scene,
        "paths": {"reference": str(reference_path), "current": str(current_path), "gate": str(gate_path), "checkpoint": None if checkpoint is None else str(checkpoint)},
        "rois": [{"name": n, "x": x, "y": y, "size": s, "kind": ROI_KIND.get(n, "mixed")} for n, x, y, s in DEFAULT_ROIS[scene]],
        "reopen": {
            "strength": float(reopen_strength),
            "shadow_weight": float(reopen_shadow_weight),
            "structure_suppress": float(reopen_structure_suppress),
            "min": float(reopen_min),
            "max": float(reopen_max),
            "shadow_threshold": float(reopen_shadow_threshold),
            "shadow_transition": float(reopen_shadow_transition),
        },
        "presets": [],
    }
    for preset in presets:
        raw = dict(REGION_AWARE_FLAT_GATE_PRESETS[preset])
        raw.pop("smooth_params", None)
        preset_reopen = {
            "reopen_strength": float(raw.pop("reopen_strength", reopen_strength)),
            "reopen_shadow_weight": float(raw.pop("reopen_shadow_weight", reopen_shadow_weight)),
            "reopen_structure_suppress": float(raw.pop("reopen_structure_suppress", reopen_structure_suppress)),
            "reopen_min": float(raw.pop("reopen_min", reopen_min)),
            "reopen_max": float(raw.pop("reopen_max", reopen_max)),
            "reopen_shadow_threshold": float(raw.pop("reopen_shadow_threshold", reopen_shadow_threshold)),
            "reopen_shadow_transition": float(raw.pop("reopen_shadow_transition", reopen_shadow_transition)),
        }
        item: dict[str, object] = {
            "preset": preset,
            "roi_results": [],
        }
        for roi_name, x, y, size in DEFAULT_ROIS[scene]:
            ref_crop = crop_center(reference, x, y, size)
            cur_crop = crop_center(current, x, y, size)
            if model is None:
                gate_crop = crop_center(gate, x, y, size)
            else:
                detail_crop = crop_center(detail_gate, x, y, size)
                feats, _, _, _ = make_features_and_gate(ref_crop, cur_crop, detail_crop)
                feats = align_feature_channels(feats, getattr(model, "feature_channels", feats.shape[2]))
                inp = torch.from_numpy(np.transpose(feats, (2, 0, 1))[None]).to(device)
                with torch.inference_mode():
                    pred = model(inp).detach().cpu().numpy()[0, 0]
                gate_crop = pred.astype(np.float32, copy=False)
            strength, stats, masks = build_strength_map(ref_crop, cur_crop, **raw)
            reopen = build_reopen_map(
                masks,
                reopen_strength=preset_reopen["reopen_strength"],
                reopen_shadow_weight=preset_reopen["reopen_shadow_weight"],
                reopen_structure_suppress=preset_reopen["reopen_structure_suppress"],
                reopen_min=preset_reopen["reopen_min"],
                reopen_max=preset_reopen["reopen_max"],
                reopen_shadow_threshold=preset_reopen["reopen_shadow_threshold"],
                reopen_shadow_transition=preset_reopen["reopen_shadow_transition"],
            )
            effective = np.clip(gate_crop * strength * reopen, 0.0, 1.0).astype(np.float32, copy=False)
            roi = {
                "roi": roi_name,
                "kind": ROI_KIND.get(roi_name, "mixed"),
                "gate": stat_block(gate_crop),
                "strength": stat_block(strength),
                "effective_gate": stat_block(effective),
                "flat": stat_block(masks["flat"]),
                "structure_protect": stat_block(masks["structure_protect"]),
                "shadow_flat": stat_block(masks["shadow_flat"]),
                "reopen": stat_block(reopen),
                "build_stats": stats,
            }
            item["roi_results"].append(roi)
        report["presets"].append(item)
    return report


def write_markdown(report: dict[str, object], path: Path) -> None:
    lines = ["# Region Gate ROI Evaluation", "", f"Scene: `{report['scene']}`", ""]
    for preset in report["presets"]:  # type: ignore[index]
        lines.extend([
            f"## {preset['preset']}",
            "",
            "| ROI | kind | gate mean | strength mean | reopen mean | effective mean | effective p95 | flat mean | structure mean | shadow flat mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for roi in preset["roi_results"]:  # type: ignore[index]
            lines.append(
                "| {roi} | {kind} | {gm:.4f} | {sm:.4f} | {rm:.4f} | {em:.4f} | {ep95:.4f} | {fm:.4f} | {stm:.4f} | {shm:.4f} |".format(
                    roi=roi["roi"],
                    kind=roi["kind"],
                    gm=roi["gate"]["mean"],
                    sm=roi["strength"]["mean"],
                    rm=roi["reopen"]["mean"],
                    em=roi["effective_gate"]["mean"],
                    ep95=roi["effective_gate"]["p95"],
                    fm=roi["flat"]["mean"],
                    stm=roi["structure_protect"]["mean"],
                    shm=roi["shadow_flat"]["mean"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate region-aware gate behavior on fixed ROIs.")
    parser.add_argument("--scene", choices=sorted(DEFAULT_ROIS), required=True)
    parser.add_argument("--preset", action="append", default=[], choices=sorted(REGION_AWARE_FLAT_GATE_PRESETS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-dir", default=None)
    parser.add_argument("--gate-suffix", default="flat_cleanup_gate_v12_native_pilot_v1")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reopen-strength", type=float, default=0.0)
    parser.add_argument("--reopen-shadow-weight", type=float, default=0.85)
    parser.add_argument("--reopen-structure-suppress", type=float, default=1.0)
    parser.add_argument("--reopen-min", type=float, default=1.0)
    parser.add_argument("--reopen-max", type=float, default=1.45)
    parser.add_argument("--reopen-shadow-threshold", type=float, default=0.0)
    parser.add_argument("--reopen-shadow-transition", type=float, default=0.12)
    args = parser.parse_args()
    presets = args.preset or ["quality_v3", "dark_sky_strict"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_dir = Path(args.gate_dir) if args.gate_dir else None
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    report = summarize_scene(
        args.scene,
        presets,
        gate_dir=gate_dir,
        gate_suffix=args.gate_suffix,
        checkpoint=checkpoint,
        device_name=args.device,
        reopen_strength=args.reopen_strength,
        reopen_shadow_weight=args.reopen_shadow_weight,
        reopen_structure_suppress=args.reopen_structure_suppress,
        reopen_min=args.reopen_min,
        reopen_max=args.reopen_max,
        reopen_shadow_threshold=args.reopen_shadow_threshold,
        reopen_shadow_transition=args.reopen_shadow_transition,
    )
    json_path = out_dir / f"{args.scene}_region_gate_roi_eval.json"
    md_path = out_dir / f"{args.scene}_region_gate_roi_eval.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
