"""Build PL-informed frequency-split pseudo teachers for Perfect NR.

PhotoLab output is useful as a smoothness reference, but not as an absolute
teacher: its tone, color, and highlight rendering can differ from the Nagi
pipeline. This script keeps Nagi's local low-frequency look, borrows only PL's
local residual, and blends it into flat regions while protecting coherent
structure from the current best output and the noisy reference.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from .flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from .luma_tail_speckle import sigmoid01
from .region_aware_luma_cleanup import make_coherent_structure_mask, make_skin_mask, make_texture_mask
from .detail_guard import write_exr, write_tiff
from .probe import image_stats, make_preview, read_image


ROOT = Path(__file__).resolve().parents[2]
TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
# Frozen quality baseline from the pre-NagiV2 pipeline (see runs/baseline_v12/).
# Read-only reference for comparisons; Phase 2 pseudo-teacher bases below now
# point at NagiV2 (+ deterministic chroma pass) outputs instead.
RUN_ROOT = ROOT / "runs/baseline_v12"
BEST = RUN_ROOT
# Phase 2: NagiV2-L + flat_chroma_smoother outputs, used as the pseudo-teacher
# base for all weak-teacher scenes (see runs/nagi_v2_l_weak/base/).
NAGI_V2_WEAK_BASE = ROOT / "runs/nagi_v2_l_weak/base"


@dataclass(frozen=True)
class Scene:
    name: str
    noisy: Path
    base: Path
    pl: Path
    rois: tuple[tuple[str, int, int], ...]
    pl_offset_xy: tuple[int, int] = (0, 0)


SCENES: dict[str, Scene] = {
    "xt5_occi": Scene(
        "xt5_occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        NAGI_V2_WEAK_BASE / "xt5_occi_nagi_v2_chroma.exr",
        TEST_PHOTOS / "X-T5 Occi PL deepprimeXD.tif",
        (
            ("face_center", 2120, 1260),
            ("hair_detail", 2420, 1040),
            ("root", 512, 5632),
            ("noise_dark", 3072, 3600),
        ),
    ),
    "k5_dance": Scene(
        "k5_dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        NAGI_V2_WEAK_BASE / "k5_dance_nagi_v2_chroma.exr",
        TEST_PHOTOS / "K-5 Dance PL deepprimeXD.tif",
        (
            ("sky_existing", 4096, 0),
            ("sky_center", 2300, 320),
            ("dancer_center", 2800, 1200),
            ("house_detail", 260, 1180),
            ("snow_ground", 2100, 2500),
        ),
    ),
    "k5_ice": Scene(
        "k5_ice",
        TEST_PHOTOS / "K-5 Ice noisy.EXR",
        NAGI_V2_WEAK_BASE / "k5_ice_nagi_v2_chroma.exr",
        TEST_PHOTOS / "K-5 Ice PL deepprimeXD.tif",
        (
            ("ice_center", 2100, 1180),
            ("blue_shadow", 2700, 900),
            ("edge_detail", 1700, 1450),
        ),
    ),
    "xt5_cat": Scene(
        "xt5_cat",
        TEST_PHOTOS / "X-T5 Cat noisy.EXR",
        NAGI_V2_WEAK_BASE / "xt5_cat_nagi_v2_chroma.exr",
        TEST_PHOTOS / "X-T5 Cat PL deepprimeXD.tif",
        (
            ("fur_detail", 1808, 556),
            ("dark_noise", 1200, 900),
            ("whisker", 2100, 620),
        ),
        (1808, 556),
    ),
}


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def _display(image: np.ndarray, *, assume_linear: bool) -> np.ndarray:
    rgb = _safe_rgb(image)
    if assume_linear:
        rgb = linear_to_srgb_np(np.clip(rgb, 0.0, None))
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def saturation(display: np.ndarray) -> np.ndarray:
    mx = np.max(display, axis=2)
    mn = np.min(display, axis=2)
    return ((mx - mn) / np.maximum(mx, 1.0e-6)).astype(np.float32, copy=False)


def match_pl_local_residual(base_display: np.ndarray, pl_display: np.ndarray, sigma: float) -> np.ndarray:
    base_low = gaussian_filter(base_display, sigma=(float(sigma), float(sigma), 0.0), mode="reflect")
    pl_low = gaussian_filter(pl_display, sigma=(float(sigma), float(sigma), 0.0), mode="reflect")
    return np.clip(base_low + (pl_display - pl_low), 0.0, 1.0).astype(np.float32, copy=False)


def chroma_pl_safety_gate(
    base_display: np.ndarray,
    pl_matched: np.ndarray,
    *,
    hf_sigma: float,
    low_sigma: float,
    benefit_threshold: float,
    benefit_transition: float,
    color_threshold: float,
    color_transition: float,
) -> tuple[np.ndarray, dict[str, float]]:
    base_y = luma(base_display, LUMA_SRGB)
    pl_y = luma(pl_matched, LUMA_SRGB)
    base_chroma = base_display - base_y[..., None]
    pl_chroma = pl_matched - pl_y[..., None]
    base_hf = np.mean(
        np.abs(base_chroma - gaussian_filter(base_chroma, sigma=(float(hf_sigma), float(hf_sigma), 0.0), mode="reflect")),
        axis=2,
    )
    pl_hf = np.mean(
        np.abs(pl_chroma - gaussian_filter(pl_chroma, sigma=(float(hf_sigma), float(hf_sigma), 0.0), mode="reflect")),
        axis=2,
    )
    benefit = sigmoid01(
        (base_hf - pl_hf - float(benefit_threshold)) / max(float(benefit_transition), 1.0e-6)
    )
    base_low = gaussian_filter(base_chroma, sigma=(float(low_sigma), float(low_sigma), 0.0), mode="reflect")
    pl_low = gaussian_filter(pl_chroma, sigma=(float(low_sigma), float(low_sigma), 0.0), mode="reflect")
    color_delta = np.mean(np.abs(base_low - pl_low), axis=2)
    color_agree = sigmoid01((float(color_threshold) - color_delta) / max(float(color_transition), 1.0e-6))
    gate = np.clip(benefit * color_agree, 0.0, 1.0).astype(np.float32, copy=False)
    stats = {
        "pl_chroma_gate_mean": float(np.mean(gate)),
        "pl_chroma_gate_p50": float(np.quantile(gate, 0.50)),
        "pl_chroma_gate_p95": float(np.quantile(gate, 0.95)),
        "pl_chroma_benefit_mean": float(np.mean(benefit)),
        "pl_chroma_color_agree_mean": float(np.mean(color_agree)),
        "pl_chroma_color_delta_p95": float(np.quantile(color_delta, 0.95)),
    }
    return gate, stats


def build_flat_blend(
    noisy_display: np.ndarray,
    base_display: np.ndarray,
    *,
    flat_strength: float,
    structure_protect: float,
    skin_flat_bonus: float,
    low_sat_bonus: float,
    gate_blur: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    noisy_texture = make_texture_mask(noisy_display, texture_threshold=0.008, texture_transition=0.014)
    base_texture = make_texture_mask(base_display, texture_threshold=0.016, texture_transition=0.014)
    coherent = make_coherent_structure_mask(
        noisy_display,
        coherence_threshold=0.40,
        coherence_transition=0.18,
        energy_threshold=0.006,
        energy_transition=0.006,
    )
    structure = np.clip(np.maximum.reduce([base_texture, noisy_texture * base_texture * 1.45, coherent * 0.90]), 0.0, 1.0)
    base_y = luma(base_display, LUMA_SRGB)
    sat = saturation(base_display)
    low_sat = sigmoid01((0.50 - sat) / 0.14)
    dark_or_mid = sigmoid01((0.82 - base_y) / 0.16)
    skin = make_skin_mask(base_display, blur_sigma=1.4)
    flat = np.clip((1.0 - structure * float(structure_protect)), 0.0, 1.0)
    flat *= np.clip(0.70 + low_sat * float(low_sat_bonus) + skin * float(skin_flat_bonus), 0.0, 1.35)
    flat *= np.clip(0.80 + 0.30 * dark_or_mid, 0.0, 1.10)
    blend = np.clip(flat * float(flat_strength), 0.0, 1.0)
    if gate_blur > 0:
        blend = gaussian_filter(blend.astype(np.float32, copy=False), sigma=float(gate_blur), mode="reflect")
    stats = {
        "blend_mean": float(np.mean(blend)),
        "blend_p50": float(np.quantile(blend, 0.50)),
        "blend_p95": float(np.quantile(blend, 0.95)),
        "structure_mean": float(np.mean(structure)),
        "structure_p95": float(np.quantile(structure, 0.95)),
        "skin_mean": float(np.mean(skin)),
        "low_sat_mean": float(np.mean(low_sat)),
    }
    masks = {"blend": blend, "structure": structure, "skin": skin, "low_sat": low_sat}
    return blend.astype(np.float32, copy=False), stats, masks


def build_pseudo_teacher(
    noisy_linear: np.ndarray,
    base_linear: np.ndarray,
    pl_display: np.ndarray,
    *,
    tone_sigma: float,
    flat_strength: float,
    structure_protect: float,
    skin_flat_bonus: float,
    low_sat_bonus: float,
    gate_blur: float,
    luma_strength: float,
    chroma_strength: float,
    chroma_safety_strength: float,
    chroma_color_threshold: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    noisy_display = _display(noisy_linear, assume_linear=True)
    base_display = _display(base_linear, assume_linear=True)
    pl_display = _display(pl_display, assume_linear=False)
    if noisy_display.shape != base_display.shape or base_display.shape != pl_display.shape:
        raise ValueError(
            f"shape mismatch noisy={noisy_display.shape} base={base_display.shape} pl={pl_display.shape}"
        )

    pl_matched = match_pl_local_residual(base_display, pl_display, sigma=tone_sigma)
    blend, stats, masks = build_flat_blend(
        noisy_display,
        base_display,
        flat_strength=flat_strength,
        structure_protect=structure_protect,
        skin_flat_bonus=skin_flat_bonus,
        low_sat_bonus=low_sat_bonus,
        gate_blur=gate_blur,
    )
    base_y = luma(base_display, LUMA_SRGB)
    pl_y = luma(pl_matched, LUMA_SRGB)
    base_chroma = base_display - base_y[..., None]
    pl_chroma = pl_matched - pl_y[..., None]
    chroma_gate, chroma_gate_stats = chroma_pl_safety_gate(
        base_display,
        pl_matched,
        hf_sigma=1.2,
        low_sigma=8.0,
        benefit_threshold=0.0008,
        benefit_transition=0.0035,
        color_threshold=chroma_color_threshold,
        color_transition=0.018,
    )
    luma_blend = np.clip(blend * float(luma_strength), 0.0, 1.0)
    safety = 1.0 - float(chroma_safety_strength) + chroma_gate * float(chroma_safety_strength)
    chroma_blend = np.clip(blend * float(chroma_strength) * safety, 0.0, 1.0)
    out_y = base_y * (1.0 - luma_blend) + pl_y * luma_blend
    out_chroma = base_chroma * (1.0 - chroma_blend[..., None]) + pl_chroma * chroma_blend[..., None]
    out_display = np.clip(out_y[..., None] + out_chroma, 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    delta = out_display - base_display
    stats.update(
        {
            "tone_sigma": float(tone_sigma),
            "flat_strength": float(flat_strength),
            "luma_strength": float(luma_strength),
            "chroma_strength": float(chroma_strength),
            "structure_protect": float(structure_protect),
            "chroma_safety_strength": float(chroma_safety_strength),
            "chroma_color_threshold": float(chroma_color_threshold),
            "delta_abs_mean": float(np.mean(np.abs(delta))),
            "delta_abs_p95": float(np.quantile(np.abs(delta), 0.95)),
            "delta_abs_p99": float(np.quantile(np.abs(delta), 0.99)),
        }
    )
    stats.update(chroma_gate_stats)
    masks["pl_matched"] = pl_matched
    masks["chroma_gate"] = chroma_gate
    return out, stats, masks


def crop_to_match(arr: np.ndarray, shape: tuple[int, int], offset_xy: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if arr.shape[:2] == (height, width):
        return arr
    x, y = offset_xy
    if x < 0 or y < 0 or y + height > arr.shape[0] or x + width > arr.shape[1]:
        raise ValueError(f"cannot crop {arr.shape[:2]} to {(height, width)} at offset {(x, y)}")
    return arr[y : y + height, x : x + width]


def save_compare(scene: Scene, result: Path, out_dir: Path, crop_size: int, scale: int) -> list[str]:
    sources = [
        ("noisy", scene.noisy),
        ("base", scene.base),
        ("pseudo", result),
        ("pl_ref", scene.pl),
    ]
    out_paths: list[str] = []
    label_h = 26
    for roi_name, x, y in scene.rois:
        crops = []
        for _, path in sources:
            arr = read_image(path)
            crop = arr[y : y + crop_size, x : x + crop_size]
            img = Image.fromarray(make_preview(crop, exposure=1.0, tone="reinhard"))
            crops.append(img.resize((crop_size * scale, crop_size * scale), Image.Resampling.NEAREST))
        w = crop_size * scale
        h = crop_size * scale
        canvas = Image.new("RGB", (w * len(crops), h + label_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for idx, ((label, _), img) in enumerate(zip(sources, crops, strict=True)):
            canvas.paste(img, (idx * w, label_h))
            draw.text((idx * w + 6, 6), label, fill=(235, 235, 235))
        path = out_dir / f"{scene.name}_pseudo_teacher_compare_{roi_name}_{scale}x.png"
        canvas.save(path)
        out_paths.append(str(path))
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PL-informed frequency-split pseudo teacher.")
    parser.add_argument("--scene", required=True, choices=sorted(SCENES))
    parser.add_argument("--output-dir", default=str(ROOT / "runs/nagi_v2_l_weak/pseudo_teacher"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--tone-sigma", type=float, default=28.0)
    parser.add_argument("--flat-strength", type=float, default=0.72)
    parser.add_argument("--structure-protect", type=float, default=1.10)
    parser.add_argument("--skin-flat-bonus", type=float, default=0.20)
    parser.add_argument("--low-sat-bonus", type=float, default=0.34)
    parser.add_argument("--gate-blur", type=float, default=1.0)
    parser.add_argument("--luma-strength", type=float, default=0.25)
    parser.add_argument("--chroma-strength", type=float, default=1.0)
    parser.add_argument("--chroma-safety-strength", type=float, default=1.0)
    parser.add_argument("--chroma-color-threshold", type=float, default=0.035)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    scene = SCENES[args.scene]
    missing = [p for p in (scene.noisy, scene.base, scene.pl) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{scene.name}_frequency_split_pseudo_teacher"

    noisy = read_image(scene.noisy)
    base = read_image(scene.base)
    pl = crop_to_match(read_image(scene.pl), base.shape[:2], scene.pl_offset_xy)
    out, stats, masks = build_pseudo_teacher(
        noisy,
        base,
        pl,
        tone_sigma=args.tone_sigma,
        flat_strength=args.flat_strength,
        structure_protect=args.structure_protect,
        skin_flat_bonus=args.skin_flat_bonus,
        low_sat_bonus=args.low_sat_bonus,
        gate_blur=args.gate_blur,
        luma_strength=args.luma_strength,
        chroma_strength=args.chroma_strength,
        chroma_safety_strength=args.chroma_safety_strength,
        chroma_color_threshold=args.chroma_color_threshold,
    )

    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    mask_paths = {}
    for mask_name, mask in masks.items():
        if mask_name == "pl_matched":
            path = out_dir / f"{name}_pl_matched_preview.png"
            Image.fromarray(make_preview(srgb_to_linear_np(mask), exposure=1.0, tone="reinhard")).save(path)
        else:
            path = out_dir / f"{name}_{mask_name}.png"
            Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)
        mask_paths[mask_name] = str(path)

    compare_paths = save_compare(scene, exr_path, out_dir, args.crop_size, args.scale) if args.compare else []
    meta = {
        "scene": scene.name,
        "inputs": {"noisy": str(scene.noisy), "base": str(scene.base), "pl": str(scene.pl)},
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "masks": mask_paths,
            "compare": compare_paths,
        },
        "stats": stats,
        "input_stats": image_stats(base),
        "output_stats": image_stats(out),
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    meta_path = out_dir / f"{name}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
