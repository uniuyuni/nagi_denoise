"""Phase 5 Stage B (Core ML) numerical validation.

Compares the exported Core ML NagiV2-L graphs (see
`scripts/export_coreml_nagi_v2.py`) against the PyTorch-MPS reference on the
SAME 768x768 input tiles, drawn from real scenes -- not synthetic noise --
including the historical Core ML failure ROIs called out in the task:

  * thin structures: X-T5 Occi hair (x=2420,y=1040), X-T5 Cat whiskers
    (x=2100,y=620)
  * saturated / HDR: K-5 Ice blue shadows (x=2700,y=900), K-5 Dance (HDR
    peak ~10)

For every tile this reports max / mean / p99.9 absolute per-pixel diff
against the PyTorch-MPS reference, run through BOTH exported precisions
(fp16, fp32), each under both `cpu_and_gpu` and `all` compute units (4
Core ML configurations per tile). It also renders visual crops (input /
PT ref / Core ML / abs-diff heatmap) for the failure-mode ROIs into
`runs/phase5_speed/coreml/` so a human can eyeball thin-line corruption or
local colour breakage directly, not just trust an aggregate number.

Usage:
    pixi run python scripts/validate_coreml_nagi_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import coremltools as ct  # noqa: E402

from nagi_denoise.devices import resolve_device  # noqa: E402
from nagi_denoise.infer import Denoiser  # noqa: E402
from nagi_denoise.pipeline.denoise import PRODUCTION_WEIGHTS  # noqa: E402
from nagi_denoise.pipeline.probe import read_image  # noqa: E402
from nagi_denoise.transforms import linear_to_srgb  # noqa: E402

TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
COREML_DIR = REPO_ROOT / "runs" / "phase5_speed" / "coreml"
TILE = 768

# (label, scene file, tile top-left x, tile top-left y, is a failure-mode ROI?)
# Tile top-lefts are chosen so the named ROI point sits inside the tile, and
# the tile stays within image bounds. A handful of generic (non-ROI) tiles
# are added across scenes for aggregate coverage -> >=20 tiles total.
SCENES = {
    "occi": TEST_PHOTOS / "X-T5 Occi noisy.EXR",
    "cat": TEST_PHOTOS / "X-T5 Cat noisy.EXR",
    "ice": TEST_PHOTOS / "K-5 Ice noisy.EXR",
    "dance": TEST_PHOTOS / "K-5 Dance noisy.EXR",
    "room": TEST_PHOTOS / "X-T5 Room.EXR",
    "bird": TEST_PHOTOS / "Z7 bird noisy.EXR",
    "night": TEST_PHOTOS / "Z7 night noisy.EXR",
}


def _tile_origin(cx: int, cy: int, w: int, h: int, tile: int = TILE) -> tuple[int, int]:
    x0 = max(0, min(w - tile, cx - tile // 2))
    y0 = max(0, min(h - tile, cy - tile // 2))
    return x0, y0


def build_tile_plan() -> list[dict]:
    plan = []
    shapes = {}
    for key, path in SCENES.items():
        img = read_image(path)
        shapes[key] = img.shape[:2]  # (H, W)

    def add(label, scene, cx, cy, roi):
        h, w = shapes[scene]
        x0, y0 = _tile_origin(cx, cy, w, h)
        plan.append({"label": label, "scene": scene, "x0": x0, "y0": y0, "roi": roi, "center": (cx, cy)})

    # Historical failure-mode ROIs (exact task coordinates).
    add("occi_hair", "occi", 2420, 1040, "thin_structure")
    add("cat_whisker", "cat", 2100, 620, "thin_structure")
    add("ice_blue_shadow", "ice", 2700, 900, "saturated")
    add("dance_hdr", "dance", 2800, 1200, "hdr")

    # Additional coverage tiles (generic, spread across scenes/positions) to
    # reach >=20 tiles total for the aggregate error stats.
    extra = [
        ("occi_a", "occi", 500, 500),
        ("occi_b", "occi", 3500, 2000),
        ("occi_c", "occi", 1000, 2400),
        ("cat_fur", "cat", 1808, 556),
        ("cat_dark", "cat", 1200, 900),
        ("cat_b", "cat", 3000, 1800),
        ("ice_center", "ice", 2100, 1180),
        ("ice_edge", "ice", 1700, 1450),
        ("ice_b", "ice", 500, 2000),
        ("dance_sky", "dance", 2300, 320),
        ("dance_house", "dance", 260, 1180),
        ("dance_snow", "dance", 2100, 2500),
        ("room_a", "room", 1500, 1000),
        ("room_b", "room", 3500, 2200),
        ("room_c", "room", 800, 2800),
        ("bird_a", "bird", 1500, 1000),
        ("bird_b", "bird", 3000, 1500),
        ("night_a", "night", 1500, 1000),
        ("night_b", "night", 3000, 2000),
    ]
    for label, scene, cx, cy in extra:
        add(label, scene, cx, cy, "generic")

    return plan


def load_pt_model() -> Denoiser:
    device = resolve_device("mps")
    dn = Denoiser.load(str(PRODUCTION_WEIGHTS), device=device)
    assert float(dn.model.highlight_protect_strength) == 0.0, (
        "PT reference must run with the guard disarmed to match the exported "
        "CoreML graph (guard is applied post-hoc, see export script docstring)"
    )
    return dn


def pt_forward(dn: Denoiser, tile_np: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(tile_np).permute(2, 0, 1).unsqueeze(0).float().to(dn.device)
    with torch.inference_mode():
        y = dn.model(x, return_aux=False)
    torch.mps.synchronize() if dn.device.type == "mps" else None
    return y.detach().to("cpu").permute(0, 2, 3, 1)[0].numpy().astype(np.float32, copy=False)


def cml_forward(model: "ct.models.MLModel", tile_np: np.ndarray) -> np.ndarray:
    x = tile_np.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
    out = model.predict({"tile_in": x})["tile_out"]
    return out[0].transpose(1, 2, 0).astype(np.float32, copy=False)


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    finite = np.isfinite(d)
    if not finite.all():
        n_bad = int((~finite).sum())
        d = np.where(finite, d, 0.0)
    else:
        n_bad = 0
    return {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p999": float(np.percentile(d, 99.9)),
        "n_nonfinite": n_bad,
    }


def to_png_u8(linear_rgb: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    x = np.clip(linear_rgb * exposure, 0.0, None)
    disp = linear_to_srgb(torch.from_numpy(x)).clamp(0.0, 1.0).numpy()
    return (disp * 255.0 + 0.5).astype(np.uint8)


def diff_heatmap_u8(diff_map: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Simple black -> red -> yellow -> white heatmap for a scalar diff map."""
    d = diff_map.astype(np.float32)
    vmax = float(d.max()) if vmax is None else vmax
    vmax = max(vmax, 1e-6)
    t = np.clip(d / vmax, 0.0, 1.0)
    r = np.clip(t * 3.0, 0.0, 1.0)
    g = np.clip(t * 3.0 - 1.0, 0.0, 1.0)
    b = np.clip(t * 3.0 - 2.0, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def save_roi_visuals(out_dir: Path, label: str, inp: np.ndarray, pt: np.ndarray, cml: np.ndarray, precision: str, compute_units: str, crop: int = 192) -> None:
    h, w = inp.shape[:2]
    cy0, cx0 = h // 2 - crop // 2, w // 2 - crop // 2
    sl = (slice(cy0, cy0 + crop), slice(cx0, cx0 + crop))
    inp_c, pt_c, cml_c = inp[sl], pt[sl], cml[sl]
    diff_c = np.abs(pt_c.astype(np.float64) - cml_c.astype(np.float64)).mean(axis=-1)

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_png_u8(inp_c)).save(out_dir / f"{label}_input.png")
    Image.fromarray(to_png_u8(pt_c)).save(out_dir / f"{label}_pt_ref.png")
    Image.fromarray(to_png_u8(cml_c)).save(out_dir / f"{label}_coreml_{precision}_{compute_units}.png")
    Image.fromarray(diff_heatmap_u8(diff_c)).save(out_dir / f"{label}_diffheat_{precision}_{compute_units}.png")


def main() -> None:
    plan = build_tile_plan()
    print(f"tile plan: {len(plan)} tiles")

    dn = load_pt_model()

    variants = []
    for precision in ("fp32", "fp16"):
        pkg = COREML_DIR / f"nagi_v2_l_ft2_t{TILE}_{precision}.mlpackage"
        for cu_name, cu in (("cpu_and_gpu", ct.ComputeUnit.CPU_AND_GPU), ("all", ct.ComputeUnit.ALL)):
            print(f"loading {pkg.name} compute_units={cu_name} ...")
            model = ct.models.MLModel(str(pkg), compute_units=cu)
            variants.append({"precision": precision, "compute_units": cu_name, "model": model})

    scene_cache: dict[str, np.ndarray] = {}
    results = []
    visual_dir = COREML_DIR / "validation_crops"

    for item in plan:
        scene = item["scene"]
        if scene not in scene_cache:
            scene_cache[scene] = read_image(SCENES[scene])
        img = scene_cache[scene]
        x0, y0 = item["x0"], item["y0"]
        tile_np = np.ascontiguousarray(img[y0 : y0 + TILE, x0 : x0 + TILE, :3])

        pt_out = pt_forward(dn, tile_np)
        pt_max_abs = float(np.abs(pt_out).max())

        for v in variants:
            cml_out = cml_forward(v["model"], tile_np)
            stats = diff_stats(pt_out, cml_out)
            cml_max_abs = float(np.abs(cml_out).max())
            row = {
                "label": item["label"],
                "roi": item["roi"],
                "scene": scene,
                "precision": v["precision"],
                "compute_units": v["compute_units"],
                "pt_max_abs_value": pt_max_abs,
                "cml_max_abs_value": cml_max_abs,
                **stats,
            }
            results.append(row)
            flag = " <-- LARGE" if stats["max"] > 0.02 else ""
            print(
                f"{item['label']:16s} {v['precision']:4s} {v['compute_units']:11s} "
                f"max={stats['max']:.5f} mean={stats['mean']:.6f} p999={stats['p999']:.5f} "
                f"nonfinite={stats['n_nonfinite']}{flag}"
            )

            if item["roi"] != "generic":
                save_roi_visuals(visual_dir, item["label"], tile_np, pt_out, cml_out, v["precision"], v["compute_units"])

    out_json = COREML_DIR / "validation_report.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")

    # Summary per (precision, compute_units).
    print("\n=== summary (max over all tiles) ===")
    for precision in ("fp32", "fp16"):
        for cu_name in ("cpu_and_gpu", "all"):
            rows = [r for r in results if r["precision"] == precision and r["compute_units"] == cu_name]
            if not rows:
                continue
            worst_max = max(r["max"] for r in rows)
            worst_mean = max(r["mean"] for r in rows)
            worst_p999 = max(r["p999"] for r in rows)
            total_nonfinite = sum(r["n_nonfinite"] for r in rows)
            print(
                f"{precision:4s} {cu_name:11s}: worst_max_diff={worst_max:.5f} "
                f"worst_mean_diff={worst_mean:.6f} worst_p999_diff={worst_p999:.5f} "
                f"total_nonfinite_px={total_nonfinite}"
            )


if __name__ == "__main__":
    main()
