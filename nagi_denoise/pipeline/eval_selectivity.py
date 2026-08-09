"""Detail/noise selectivity metric for Phase 2B checkpoint evaluation.

This is the decisive acceptance metric for the NagiV2-L correlated-noise
fine-tune (see task notes for the full derivation). Given a real noisy EXR
and a candidate NagiV2 checkpoint, it:

  1. Runs the model over the full image (tiled), applies the deterministic
     chroma pass (``flat_chroma_smoother.smooth_chroma``), and extracts a
     crop at the requested ROI.
  2. Builds a structure mask from the NOISY crop (model-independent) via
     ``region_aware_luma_cleanup.make_coherent_structure_mask``.
  3. Computes HF = |luma - gaussian(luma, 1.2)| for both the noisy crop and
     the candidate crop, averaged inside (on) vs outside (off) the mask.
  4. retention_on  = HF_on(candidate)  / HF_on(noisy)
     retention_off = HF_off(candidate) / HF_off(noisy)
     selectivity   = (retention_on - retention_off) * 100  [percentage points]

Higher selectivity is better: it means the model removed more noise in flat
regions (low retention_off) while keeping more detail in structured regions
(high retention_on). A run that lowers both retentions uniformly does NOT
improve selectivity and is not progress.

Usage:
    pixi run python -m nagi_denoise.pipeline.eval_selectivity \\
        --noisy "/Users/uniuyuni/ProjectData/test_photos/X-T5 Occi noisy.EXR" \\
        --weights runs/nagi_v2_l_ft2/nagi_v2_l_ft2_ckpt_0002000.pt \\
        --x 2420 --y 1040 --size 256 --device mps \\
        --output-dir runs/phase2b_compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .denoise_exr import load_model, run_tiled_image
from .flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, smooth_chroma
from .probe import read_image
from .region_aware_luma_cleanup import make_coherent_structure_mask


# Reference chroma-pass params (must match the values the reference table in
# the task notes was measured with).
DEFAULT_CHROMA_KWARGS = dict(
    strength=1.0,
    chroma_sigma=4.5,
    detail_sigma=1.2,
    threshold=0.022,
    transition=0.010,
    highlight_threshold=1.0,
    highlight_transition=0.2,
    hdr_restore_peak_threshold=0.95,
    hdr_restore_threshold=0.85,
    hdr_restore_transition=0.25,
)

# Reference structure-mask params.
DEFAULT_MASK_KWARGS = dict(
    coherence_threshold=0.40,
    coherence_transition=0.18,
    energy_threshold=0.006,
    energy_transition=0.006,
)

HF_SIGMA = 1.2


def crop_center(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, min(w - size, int(x) - size // 2))
    y0 = max(0, min(h - size, int(y) - size // 2))
    return image[y0 : y0 + size, x0 : x0 + size]


def to_display(linear_rgb: np.ndarray) -> np.ndarray:
    return linear_to_srgb_np(np.clip(linear_rgb[..., :3], 0.0, None)).astype(np.float32, copy=False)


def hf_map(display_rgb: np.ndarray, sigma: float = HF_SIGMA) -> np.ndarray:
    y = luma(display_rgb, LUMA_SRGB)
    return np.abs(y - gaussian_filter(y, sigma=sigma, mode="reflect")).astype(np.float32, copy=False)


def selectivity_metrics(
    noisy_crop_linear: np.ndarray,
    candidate_crop_linear: np.ndarray,
    mask_kwargs: dict | None = None,
    hf_sigma: float = HF_SIGMA,
) -> dict:
    """Compute retention_on/off and selectivity for one crop pair.

    Both crops are linear-light HWC arrays of the same shape. The structure
    mask is built from ``noisy_crop_linear`` only (model-independent).
    """
    mask_kwargs = dict(DEFAULT_MASK_KWARGS if mask_kwargs is None else mask_kwargs)
    noisy_disp = to_display(noisy_crop_linear)
    cand_disp = to_display(candidate_crop_linear)

    mask = make_coherent_structure_mask(noisy_disp, **mask_kwargs)
    mask_off = 1.0 - mask

    hf_noisy = hf_map(noisy_disp, sigma=hf_sigma)
    hf_cand = hf_map(cand_disp, sigma=hf_sigma)

    def weighted_mean(hf: np.ndarray, w: np.ndarray) -> float:
        denom = float(w.sum())
        return float((hf * w).sum() / denom) if denom > 1.0e-8 else 0.0

    hf_on_noisy = weighted_mean(hf_noisy, mask)
    hf_off_noisy = weighted_mean(hf_noisy, mask_off)
    hf_on_cand = weighted_mean(hf_cand, mask)
    hf_off_cand = weighted_mean(hf_cand, mask_off)

    retention_on = hf_on_cand / hf_on_noisy if hf_on_noisy > 1.0e-8 else float("nan")
    retention_off = hf_off_cand / hf_off_noisy if hf_off_noisy > 1.0e-8 else float("nan")
    selectivity = (retention_on - retention_off) * 100.0

    return {
        "mask_mean": float(mask.mean()),
        "hf_on_noisy": hf_on_noisy,
        "hf_off_noisy": hf_off_noisy,
        "hf_on_candidate": hf_on_cand,
        "hf_off_candidate": hf_off_cand,
        "retention_on": retention_on,
        "retention_off": retention_off,
        "selectivity_pt": selectivity,
    }


def run_candidate(
    noisy_image: np.ndarray,
    weights: Path,
    device: torch.device,
    state_key: str = "state_dict",
    tile_size: int = 1024,
    tile_overlap: int = 64,
    chroma_kwargs: dict | None = None,
) -> np.ndarray:
    """Run NagiV2 + deterministic chroma pass over the full image; returns
    linear-light HWC output."""
    model = load_model(weights, device=device, state_key=state_key)
    out, _extras = run_tiled_image(
        model, noisy_image, device=device, tile_size=tile_size, overlap=tile_overlap, diagnostics=False
    )
    chroma_kwargs = dict(DEFAULT_CHROMA_KWARGS if chroma_kwargs is None else chroma_kwargs)
    out_chroma, _stats, _gate = smooth_chroma(out, **chroma_kwargs)
    return out_chroma


def main() -> None:
    parser = argparse.ArgumentParser(description="Detail/noise selectivity metric for a NagiV2 checkpoint.")
    parser.add_argument("--noisy", required=True, help="Path to noisy EXR/TIFF.")
    parser.add_argument("--weights", required=True, help="NagiV2 checkpoint (.pt).")
    parser.add_argument("--state-key", default="state_dict", help="'state_dict' = EMA weights (default).")
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--output-dir", default="runs/phase2b_compare")
    parser.add_argument("--name", default=None, help="Label for this candidate in the report (default: weights stem).")
    args = parser.parse_args()

    device = torch.device(args.device)
    noisy_path = Path(args.noisy).expanduser()
    weights_path = Path(args.weights).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or weights_path.stem

    noisy_full = read_image(noisy_path)
    candidate_full = run_candidate(
        noisy_full, weights_path, device=device, state_key=args.state_key,
        tile_size=args.tile_size, tile_overlap=args.tile_overlap,
    )

    noisy_crop = crop_center(noisy_full, args.x, args.y, args.size)
    cand_crop = crop_center(candidate_full, args.x, args.y, args.size)

    metrics = selectivity_metrics(noisy_crop, cand_crop)
    report = {
        "noisy": str(noisy_path),
        "weights": str(weights_path),
        "candidate_name": name,
        "roi": {"x": args.x, "y": args.y, "size": args.size},
        **metrics,
    }
    print(json.dumps(report, indent=2))

    json_path = out_dir / f"selectivity_{name}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
