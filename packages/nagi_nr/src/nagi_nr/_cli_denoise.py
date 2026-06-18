"""CLI to denoise a single image with a trained Nagi NR checkpoint.

Entry point: ``nagi-denoise``.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .infer import Denoiser


def load_image(path: Path) -> tuple[torch.Tensor, str]:
    """Return (CHW float32 tensor, color space hint).

    PNG/JPG/TIFF (8 or 16 bit) -> sRGB in [0,1].
    EXR/HDR float -> linear (loaded via OpenCV if available).
    """
    ext = path.suffix.lower()
    if ext in {".exr", ".hdr"}:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("Install opencv-python to read .exr/.hdr") from e
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if arr is None:
            raise RuntimeError(f"Could not load {path}")
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            arr = arr[..., ::-1]  # BGR -> RGB
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
        return t, "linear"

    with Image.open(path) as im:
        if im.mode in {"I;16", "I"}:
            arr = np.asarray(im, dtype=np.uint16).astype(np.float32) / 65535.0
            arr = np.stack([arr] * 3, axis=-1) if arr.ndim == 2 else arr
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), "srgb"
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), "srgb"


def save_image(t: torch.Tensor, path: Path, input_space: str) -> None:
    arr = t.clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
    if path.suffix.lower() in {".exr", ".hdr", ".tif", ".tiff"}:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("Install opencv-python to write HDR formats") from e
        cv2.imwrite(str(path), arr[..., ::-1].astype(np.float32))
        return
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8).save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(prog="nagi-denoise", description="Denoise an image with Nagi NR.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument(
        "--input-space",
        default="auto",
        choices=["auto", "srgb", "linear"],
        help="auto: PNG/JPG=sRGB, EXR/HDR=linear",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    img, hint = load_image(in_path)
    space = hint if args.input_space == "auto" else args.input_space

    dn = Denoiser.load(args.weights, device=args.device)
    out = dn(img, input_space=space, tile=args.tile, overlap=args.overlap)

    save_image(out, out_path, space)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
