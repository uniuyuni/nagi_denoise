"""Render equal-size crop comparisons for real-photo NR candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .probe import make_preview, read_image


def parse_panel(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value).expanduser()
        return path.stem, path
    label, path = value.split("=", 1)
    return label.strip(), Path(path).expanduser()


def parse_roi(value: str) -> tuple[str, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError(f"roi must be name,x,y,size: {value!r}")
    name, x, y, size = parts
    return name.strip(), int(x), int(y), int(size)


def crop(image, x: int, y: int, size: int):
    h, w = image.shape[:2]
    x0 = max(0, min(w - size, int(x) - size // 2))
    y0 = max(0, min(h - size, int(y) - size // 2))
    return image[y0 : y0 + size, x0 : x0 + size]


def render(path: Path, panels, *, scale: int, exposure: float, tone: str) -> None:
    previews = []
    for label, image in panels:
        preview = Image.fromarray(make_preview(image, exposure=exposure, tone=tone))
        if scale != 1:
            preview = preview.resize((preview.width * scale, preview.height * scale), Image.Resampling.NEAREST)
        previews.append((label, preview))
    width, height = previews[0][1].size
    label_h = 28
    canvas = Image.new("RGB", (width * len(previews), height + label_h), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    for i, (label, preview) in enumerate(previews):
        canvas.paste(preview, (i * width, label_h))
        draw.text((i * width + 6, 7), label, fill=(235, 235, 235))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render crop comparisons for candidate images.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="compare")
    parser.add_argument("--panel", action="append", required=True, help="LABEL=PATH. Repeatable.")
    parser.add_argument("--roi", action="append", required=True, help="name,x,y,size. Repeatable.")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--tone", choices=["reinhard", "clip"], default="reinhard")
    args = parser.parse_args()

    panels = [(label, read_image(path)) for label, path in (parse_panel(p) for p in args.panel)]
    out_dir = Path(args.output_dir)
    for name, x, y, size in (parse_roi(r) for r in args.roi):
        render(
            out_dir / f"{args.prefix}_{name}_compare.png",
            [(label, crop(image, x, y, size)) for label, image in panels],
            scale=args.scale,
            exposure=args.exposure,
            tone=args.tone,
        )
    print(out_dir)


if __name__ == "__main__":
    main()
