"""Detect NAFNet/Core ML tile artifacts in denoised TIFF outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def load_rgb(path: Path) -> np.ndarray:
    arr = tifffile.imread(path).astype(np.float32)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 2.0:
        arr = arr / 65535.0
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise RuntimeError(f"expected RGB image at {path}, got {arr.shape}")
    return np.clip(arr[..., :3], 0.0, 1.0)


def chroma_metrics(tile: np.ndarray) -> dict[str, float]:
    y = tile.mean(axis=2, keepdims=True)
    chroma = tile - y
    dx = np.abs(chroma[:, 1:] - chroma[:, :-1]).mean() if tile.shape[1] > 1 else 0.0
    dy = np.abs(chroma[1:] - chroma[:-1]).mean() if tile.shape[0] > 1 else 0.0
    return {
        "chroma_abs": float(np.abs(chroma).mean()),
        "chroma_hf": float((dx + dy) * 0.5),
        "sat_any": float((tile >= 0.99).any(axis=2).mean()),
        "mean": float(tile.mean()),
        "p50": float(np.percentile(tile, 50)),
        "p99": float(np.percentile(tile, 99)),
    }


def is_artifact(inp: dict[str, float], out: dict[str, float], args: argparse.Namespace) -> bool:
    hf_bad = out["chroma_hf"] >= args.hf_threshold and out["chroma_hf"] >= inp["chroma_hf"] * args.hf_gain
    chroma_bad = out["chroma_abs"] >= args.chroma_threshold and out["chroma_abs"] >= inp["chroma_abs"] * args.chroma_gain
    saturation_bad = out["sat_any"] >= args.sat_threshold and out["sat_any"] >= inp["sat_any"] + args.sat_delta
    return (hf_bad and chroma_bad) or (hf_bad and saturation_bad)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect periodic chroma artifact tiles.")
    ap.add_argument("--input", required=True, help="Input/reference sRGB16 TIFF.")
    ap.add_argument("--output", required=True, help="Denoised sRGB16 TIFF to inspect.")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--hf-threshold", type=float, default=0.12)
    ap.add_argument("--hf-gain", type=float, default=4.0)
    ap.add_argument("--chroma-threshold", type=float, default=0.12)
    ap.add_argument("--chroma-gain", type=float, default=3.0)
    ap.add_argument("--sat-threshold", type=float, default=0.20)
    ap.add_argument("--sat-delta", type=float, default=0.10)
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    inp = load_rgb(Path(args.input))
    out = load_rgb(Path(args.output))
    if inp.shape != out.shape:
        raise RuntimeError(f"shape mismatch: input={inp.shape}, output={out.shape}")

    rows: list[dict] = []
    h, w = out.shape[:2]
    for y in range(0, h, args.tile):
        for x in range(0, w, args.tile):
            inp_tile = inp[y : min(y + args.tile, h), x : min(x + args.tile, w)]
            out_tile = out[y : min(y + args.tile, h), x : min(x + args.tile, w)]
            im = chroma_metrics(inp_tile)
            om = chroma_metrics(out_tile)
            if is_artifact(im, om, args):
                rows.append(
                    {
                        "x": x,
                        "y": y,
                        "w": int(out_tile.shape[1]),
                        "h": int(out_tile.shape[0]),
                        "input": im,
                        "output": om,
                    }
                )

    rows.sort(key=lambda r: r["output"]["chroma_hf"], reverse=True)
    result = {
        "input": args.input,
        "output": args.output,
        "tile": args.tile,
        "artifact_tiles": rows,
        "count": len(rows),
    }
    print(json.dumps(result, indent=2))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.md:
        lines = [
            "# Core ML Artifact Tile Detection",
            "",
            f"Input: `{args.input}`",
            f"Output: `{args.output}`",
            f"Tile: `{args.tile}`",
            f"Detected: {len(rows)}",
            "",
            "| x | y | chroma_hf in | chroma_hf out | chroma_abs out | sat_any out |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in rows:
            lines.append(
                "| {x} | {y} | {ihf:.4f} | {ohf:.4f} | {oca:.4f} | {osat:.4f} |".format(
                    x=r["x"],
                    y=r["y"],
                    ihf=r["input"]["chroma_hf"],
                    ohf=r["output"]["chroma_hf"],
                    oca=r["output"]["chroma_abs"],
                    osat=r["output"]["sat_any"],
                )
            )
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
