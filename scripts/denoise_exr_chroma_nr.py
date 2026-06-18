"""Fast HDR-safe chroma-only NR for real EXR/TIFF photos.

This is the practical path distilled from the X-T5 cat diagnostics: suppress
display-space chroma grain in flat regions, keep luma/detail mostly untouched,
and restore true HDR highlights after the display-space operation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from apply_flat_chroma_smoother import smooth_chroma
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import image_stats, make_preview, read_image


PRESETS = {
    "mild": {
        "strength": 0.85,
        "chroma_sigma": 1.6,
        "detail_sigma": 1.2,
        "threshold": 0.014,
        "transition": 0.008,
    },
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply practical HDR-safe chroma-only NR.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="runs/perfect_nr/chroma_nr")
    parser.add_argument("--name", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="strong")
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--chroma-sigma", type=float, default=None)
    parser.add_argument("--detail-sigma", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--transition", type=float, default=None)
    parser.add_argument("--highlight-threshold", type=float, default=1.0)
    parser.add_argument("--highlight-transition", type=float, default=0.2)
    parser.add_argument("--hdr-restore-peak-threshold", type=float, default=0.95)
    parser.add_argument("--hdr-restore-threshold", type=float, default=0.85)
    parser.add_argument("--hdr-restore-transition", type=float, default=0.25)
    parser.add_argument("--preview-exposure", type=float, default=1.0)
    parser.add_argument("--preview-tone", choices=["reinhard", "clip"], default="reinhard")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{input_path.stem.replace(' ', '_')}_chroma_{args.preset}"

    params = dict(PRESETS[args.preset])
    for key, attr in (
        ("strength", "strength"),
        ("chroma_sigma", "chroma_sigma"),
        ("detail_sigma", "detail_sigma"),
        ("threshold", "threshold"),
        ("transition", "transition"),
    ):
        value = getattr(args, attr)
        if value is not None:
            params[key] = float(value)

    image = read_image(input_path)
    output, stats, gate = smooth_chroma(
        image,
        strength=float(params["strength"]),
        chroma_sigma=float(params["chroma_sigma"]),
        detail_sigma=float(params["detail_sigma"]),
        threshold=float(params["threshold"]),
        transition=float(params["transition"]),
        highlight_threshold=float(args.highlight_threshold),
        highlight_transition=float(args.highlight_transition),
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
        "preset": args.preset,
        "params": {
            **params,
            "highlight_threshold": float(args.highlight_threshold),
            "highlight_transition": float(args.highlight_transition),
            "hdr_restore_peak_threshold": float(args.hdr_restore_peak_threshold),
            "hdr_restore_threshold": float(args.hdr_restore_threshold),
            "hdr_restore_transition": float(args.hdr_restore_transition),
        },
        "outputs": {
            "exr": str(exr_path),
            "tiff": str(tiff_path),
            "preview": str(preview_path),
            "gate": str(gate_path),
        },
        "smoother": stats,
        "input_stats": image_stats(image),
        "output_stats": image_stats(output),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
