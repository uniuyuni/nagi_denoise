"""Distill PL-informed frequency-split chroma teachers into a PL-free branch.

The target comes from ``build_frequency_split_pseudo_teacher.py``. The model
sees only noisy/base images and predicts a bounded display-chroma residual. It
does not predict luma, so the pilot cannot invent PL tone/detail changes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from apply_region_aware_luma_cleanup import make_texture_mask
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import make_preview, read_image


ROOT = Path(__file__).resolve().parents[1]
TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
RUN_ROOT = ROOT / "runs/refiner_pilot_stage11_hybrid_best"
TEACHER_ROOT = RUN_ROOT / "frequency_split_pseudo_teacher_v3_chroma_safe"

FEATURE_CHANNELS = 18
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class Scene:
    name: str
    noisy: Path
    base: Path
    teacher: Path
    rois: tuple[tuple[str, int, int], ...]


SCENES: dict[str, Scene] = {
    "k5_dance": Scene(
        "k5_dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        RUN_ROOT / "detail_protected_flat_cleanup_v3_more_flat/k5_dance_gate_v2_flat_cleanup_v3_more_flat.exr",
        TEACHER_ROOT / "k5_dance_freqsplit_teacher_v3_chroma_safe.exr",
        (
            ("sky_existing", 4096, 0),
            ("sky_center", 2300, 320),
            ("dancer_center", 2800, 1200),
            ("house_detail", 260, 1180),
        ),
    ),
    "k5_ice": Scene(
        "k5_ice",
        TEST_PHOTOS / "K-5 Ice noisy.EXR",
        RUN_ROOT / "final_v4_red115_blue120_detailguard_mild/k5_ice_final_v4_red115_blue120_detailguard_mild.exr",
        TEACHER_ROOT / "k5_ice_freqsplit_teacher_v3_chroma_safe.exr",
        (
            ("ice_center", 2100, 1180),
            ("blue_shadow", 2700, 900),
            ("edge_detail", 1700, 1450),
        ),
    ),
    "xt5_cat": Scene(
        "xt5_cat",
        TEST_PHOTOS / "X-T5 Cat noisy.EXR",
        RUN_ROOT / "final_v4_red115_blue120_detailguard_mild/xt5_cat_final_v4_red115_blue120_detailguard_mild.exr",
        TEACHER_ROOT / "xt5_cat_freqsplit_teacher_v3_chroma_safe.exr",
        (
            ("fur_detail", 1808, 556),
            ("dark_noise", 1200, 900),
            ("whisker", 1900, 620),
        ),
    ),
    "xt5_occi": Scene(
        "xt5_occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        RUN_ROOT / "detail_protected_flat_cleanup_v3_more_flat/xt5_occi_gate_v2_flat_cleanup_v3_more_flat.exr",
        TEACHER_ROOT / "xt5_occi_freqsplit_teacher_v3_chroma_safe.exr",
        (
            ("face_center", 2120, 1260),
            ("hair_detail", 2420, 1040),
            ("root", 512, 5632),
        ),
    ),
}


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def _display(image: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb_np(np.clip(_safe_rgb(image), 0.0, None)), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def saturation(display: np.ndarray) -> np.ndarray:
    mx = np.max(display, axis=2)
    mn = np.min(display, axis=2)
    return ((mx - mn) / np.maximum(mx, 1.0e-6)).astype(np.float32, copy=False)


def chroma(display: np.ndarray) -> np.ndarray:
    y = luma(display, LUMA_SRGB)
    return (display - y[..., None]).astype(np.float32, copy=False)


def chroma_hf(display: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    c = chroma(display)
    return np.mean(np.abs(c - gaussian_filter(c, sigma=(float(sigma), float(sigma), 0.0), mode="reflect")), axis=2)


def project_chroma_delta(delta: np.ndarray) -> np.ndarray:
    y = np.sum(delta * LUMA_WEIGHTS.reshape(1, 1, 3), axis=2, keepdims=True)
    return (delta - y).astype(np.float32, copy=False)


def make_features_and_target(noisy_linear: np.ndarray, base_linear: np.ndarray, teacher_linear: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    noisy = _display(noisy_linear)
    base = _display(base_linear)
    teacher = _display(teacher_linear)
    if noisy.shape != base.shape or base.shape != teacher.shape:
        raise ValueError(f"shape mismatch noisy={noisy.shape} base={base.shape} teacher={teacher.shape}")

    base_y = luma(base, LUMA_SRGB)
    noisy_y = luma(noisy, LUMA_SRGB)
    sat = saturation(base)
    base_chroma_hf = chroma_hf(base)
    noisy_chroma_hf = chroma_hf(noisy)
    texture = make_texture_mask(base, texture_threshold=0.016, texture_transition=0.014)
    low_sat = sigmoid01((0.52 - sat) / 0.14)
    dark = sigmoid01((0.55 - base_y) / 0.16)
    target = project_chroma_delta(teacher - base)
    target_mag = np.mean(np.abs(target), axis=2)
    improve_hint = sigmoid01((base_chroma_hf - chroma_hf(teacher) - 0.0004) / 0.0025)
    flat_hint = np.clip((1.0 - texture) * low_sat, 0.0, 1.0)
    weight = np.clip(0.10 + target_mag / 0.012 + improve_hint * flat_hint * 1.40, 0.10, 2.5).astype(
        np.float32, copy=False
    )
    feats = np.concatenate(
        [
            noisy,
            base,
            noisy - base,
            noisy_y[..., None],
            base_y[..., None],
            sat[..., None],
            base_chroma_hf[..., None],
            noisy_chroma_hf[..., None],
            texture[..., None],
            low_sat[..., None],
            dark[..., None],
            flat_hint[..., None],
        ],
        axis=2,
    ).astype(np.float32, copy=False)
    if feats.shape[2] != FEATURE_CHANNELS:
        raise AssertionError(f"feature channel mismatch {feats.shape[2]} != {FEATURE_CHANNELS}")
    stats = {
        "target_abs_mean": float(np.mean(np.abs(target))),
        "target_abs_p95": float(np.quantile(np.abs(target), 0.95)),
        "weight_mean": float(np.mean(weight)),
        "improve_hint_mean": float(np.mean(improve_hint)),
    }
    return feats, target.astype(np.float32, copy=False), weight[..., None], stats


class Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(F.gelu(self.pw1(self.dw(x)))) * self.scale


class ChromaDistiller(nn.Module):
    def __init__(self, width: int = 28, blocks: int = 4, max_delta: float = 0.030) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.head = nn.Conv2d(FEATURE_CHANNELS, width, 3, padding=1)
        self.body = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.tail(self.body(F.gelu(self.head(features))))) * self.max_delta
        w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=raw.dtype, device=raw.device).view(1, 3, 1, 1)
        return raw - (raw * w).sum(1, keepdim=True)


class CropDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes: list[Scene],
        patch_size: int,
        context: int,
        samples: int,
        roi_probability: float,
        seed: int,
        stats_samples: int,
    ) -> None:
        self.patch_size = int(patch_size)
        self.context = int(context)
        self.samples = int(samples)
        self.roi_probability = float(roi_probability)
        self.rng = random.Random(seed)
        self.items = []
        for scene in scenes:
            missing = [p for p in (scene.noisy, scene.base, scene.teacher) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{scene.name} missing: {missing}")
            self.items.append(
                {
                    "scene": scene,
                    "noisy": read_image(scene.noisy),
                    "base": read_image(scene.base),
                    "teacher": read_image(scene.teacher),
                    "stats": {},
                }
            )
        for item in self.items:
            item["stats"] = self._estimate_stats(item, stats_samples)

    def __len__(self) -> int:
        return self.samples

    def _sample_xy(self, item: dict) -> tuple[int, int]:
        height, width = item["base"].shape[:2]
        patch = self.patch_size
        if self.rng.random() < self.roi_probability and item["scene"].rois:
            _, rx, ry = self.rng.choice(item["scene"].rois)
            jitter = max(16, patch // 2)
            x = rx + self.rng.randrange(-jitter, jitter + 1)
            y = ry + self.rng.randrange(-jitter, jitter + 1)
        else:
            x = self.rng.randrange(0, max(1, width - patch + 1))
            y = self.rng.randrange(0, max(1, height - patch + 1))
        return min(max(0, x), max(0, width - patch)), min(max(0, y), max(0, height - patch))

    def _crop(self, arr: np.ndarray, x: int, y: int) -> np.ndarray:
        h, w = arr.shape[:2]
        x0 = max(0, x - self.context)
        y0 = max(0, y - self.context)
        x1 = min(w, x + self.patch_size + self.context)
        y1 = min(h, y + self.patch_size + self.context)
        inner = (y - y0, x - x0, y - y0 + self.patch_size, x - x0 + self.patch_size)
        return arr[y0:y1, x0:x1], inner

    def _make_patch(self, item: dict, x: int, y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        noisy, inner = self._crop(item["noisy"], x, y)
        base, _ = self._crop(item["base"], x, y)
        teacher, _ = self._crop(item["teacher"], x, y)
        feats, target, weight, _ = make_features_and_target(noisy, base, teacher)
        y0, x0, y1, x1 = inner
        return feats[y0:y1, x0:x1], target[y0:y1, x0:x1], weight[y0:y1, x0:x1]

    def _estimate_stats(self, item: dict, stats_samples: int) -> dict[str, float]:
        vals = []
        for _ in range(max(1, int(stats_samples))):
            x, y = self._sample_xy(item)
            _, target, weight = self._make_patch(item, x, y)
            vals.append((float(np.mean(np.abs(target))), float(np.mean(weight))))
        return {
            "target_abs_mean": float(np.mean([v[0] for v in vals])),
            "weight_mean": float(np.mean([v[1] for v in vals])),
        }

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        item = self.rng.choice(self.items)
        x, y = self._sample_xy(item)
        feats, target, weight = self._make_patch(item, x, y)
        return {
            "features": torch.from_numpy(np.ascontiguousarray(np.transpose(feats, (2, 0, 1)))),
            "target": torch.from_numpy(np.ascontiguousarray(np.transpose(target, (2, 0, 1)))),
            "weight": torch.from_numpy(np.ascontiguousarray(np.transpose(weight, (2, 0, 1)))),
        }


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = [SCENES[name] for name in args.scenes.split(",") if name]
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ds = CropDataset(
        scenes,
        patch_size=args.patch_size,
        context=args.context,
        samples=args.steps * args.batch_size,
        roi_probability=args.roi_probability,
        seed=args.seed,
        stats_samples=args.stats_samples,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = ChromaDistiller(width=args.width, blocks=args.blocks, max_delta=args.max_delta).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    meta = {
        "scenes": [s.name for s in scenes],
        "scene_stats": {item["scene"].name: item["stats"] for item in ds.items},
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "feature_channels": FEATURE_CHANNELS,
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_path = out_dir / "stdout.log"
    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== train frequency chroma distiller steps={args.steps} device={device} ===\n")
        for step, batch in enumerate(dl, start=1):
            if step > args.steps:
                break
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)
            pred = model(features)
            loss_main = (F.smooth_l1_loss(pred, target, beta=0.006, reduction="none") * weight).mean()
            loss_zero = torch.mean(torch.abs(pred)) * args.zero_weight
            loss = loss_main + loss_zero
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0:
                elapsed = time.monotonic() - start
                msg = (
                    f"step {step:05d}/{args.steps} loss={float(loss.detach()):.6f} "
                    f"main={float(loss_main.detach()):.6f} zero={float(loss_zero.detach()):.6f} "
                    f"pred_abs={float(torch.mean(torch.abs(pred)).detach()):.5f} "
                    f"target_abs={float(torch.mean(torch.abs(target)).detach()):.5f} "
                    f"{elapsed / step:.3f}s/it"
                )
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / f"frequency_chroma_distiller_step_{step:06d}.pt", model, args, step)
        save_checkpoint(out_dir / "frequency_chroma_distiller_final.pt", model, args, args.steps)
        log.write(f"wrote {out_dir / 'frequency_chroma_distiller_final.pt'}\n")
    print(f"wrote {out_dir / 'frequency_chroma_distiller_final.pt'}")


def save_checkpoint(path: Path, model: ChromaDistiller, args: argparse.Namespace, step: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "width": args.width,
            "blocks": args.blocks,
            "max_delta": args.max_delta,
            "feature_channels": FEATURE_CHANNELS,
            "step": int(step),
            "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> ChromaDistiller:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ChromaDistiller(width=int(ckpt["width"]), blocks=int(ckpt["blocks"]), max_delta=float(ckpt["max_delta"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_tiled(model: ChromaDistiller, device: torch.device, noisy: np.ndarray, base: np.ndarray, *, tile: int, overlap: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = base.shape[:2]
    delta_acc = np.zeros((height, width, 3), dtype=np.float32)
    weight_acc = np.zeros((height, width, 1), dtype=np.float32)
    stride = max(1, int(tile) - int(overlap) * 2)
    for y0 in range(0, height, stride):
        for x0 in range(0, width, stride):
            x1 = min(width, x0 + tile)
            y1 = min(height, y0 + tile)
            px0 = max(0, x0 - overlap)
            py0 = max(0, y0 - overlap)
            px1 = min(width, x1 + overlap)
            py1 = min(height, y1 + overlap)
            dummy_teacher = base[py0:py1, px0:px1]
            feats, _, _, _ = make_features_and_target(noisy[py0:py1, px0:px1], base[py0:py1, px0:px1], dummy_teacher)
            inp = torch.from_numpy(np.transpose(feats, (2, 0, 1))[None]).to(device)
            pred = model(inp).detach().cpu().numpy()[0].transpose(1, 2, 0)
            cy0 = y0 - py0
            cx0 = x0 - px0
            cy1 = cy0 + (y1 - y0)
            cx1 = cx0 + (x1 - x0)
            delta_acc[y0:y1, x0:x1] += pred[cy0:cy1, cx0:cx1]
            weight_acc[y0:y1, x0:x1] += 1.0
    delta = delta_acc / np.maximum(weight_acc, 1.0e-6)
    base_display = _display(base)
    out = srgb_to_linear_np(np.clip(base_display + delta, 0.0, 1.0)).astype(np.float32, copy=False)
    return out, delta.astype(np.float32, copy=False)


def apply(args: argparse.Namespace) -> None:
    scene = SCENES[args.scene]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    noisy = read_image(scene.noisy)
    base = read_image(scene.base)
    out, delta = predict_tiled(model, device, noisy, base, tile=args.tile, overlap=args.overlap)
    name = args.name or f"{scene.name}_frequency_chroma_distilled"
    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    delta_path = out_dir / f"{name}_delta_preview.png"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    delta_vis = np.clip(delta / 0.030 * 0.5 + 0.5, 0.0, 1.0)
    Image.fromarray((delta_vis * 255.0 + 0.5).astype(np.uint8)).save(delta_path)
    meta = {
        "scene": scene.name,
        "checkpoint": str(Path(args.checkpoint)),
        "outputs": {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path), "delta": str(delta_path)},
        "delta_abs_mean": float(np.mean(np.abs(delta))),
        "delta_abs_p95": float(np.quantile(np.abs(delta), 0.95)),
        "delta_abs_p99": float(np.quantile(np.abs(delta), 0.99)),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def crop_display(path: Path, x: int, y: int, size: int, scale: int) -> Image.Image:
    arr = read_image(path)
    crop = arr[y : y + size, x : x + size]
    img = Image.fromarray(make_preview(crop, exposure=1.0, tone="reinhard"))
    return img.resize((size * scale, size * scale), Image.Resampling.NEAREST)


def compare(args: argparse.Namespace) -> None:
    scene = SCENES[args.scene]
    result = Path(args.result)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [("noisy", scene.noisy), ("base", scene.base), ("distilled", result), ("teacher", scene.teacher)]
    label_h = 26
    for roi_name, x, y in scene.rois:
        crops = [crop_display(path, x, y, args.crop_size, args.scale) for _, path in sources]
        w = args.crop_size * args.scale
        h = args.crop_size * args.scale
        canvas = Image.new("RGB", (w * len(crops), h + label_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for idx, ((label, _), img) in enumerate(zip(sources, crops, strict=True)):
            canvas.paste(img, (idx * w, label_h))
            draw.text((idx * w + 6, 6), label, fill=(235, 235, 235))
        path = out_dir / f"{scene.name}_frequency_chroma_distilled_compare_{roi_name}_{args.scale}x.png"
        canvas.save(path)
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply frequency-split chroma distiller.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--output-dir", default=str(RUN_ROOT / "frequency_chroma_distiller_pilot"))
    p_train.add_argument("--scenes", default="k5_dance,k5_ice,xt5_cat")
    p_train.add_argument("--steps", type=int, default=360)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--patch-size", type=int, default=160)
    p_train.add_argument("--context", type=int, default=48)
    p_train.add_argument("--width", type=int, default=28)
    p_train.add_argument("--blocks", type=int, default=4)
    p_train.add_argument("--max-delta", type=float, default=0.030)
    p_train.add_argument("--lr", type=float, default=2.0e-4)
    p_train.add_argument("--weight-decay", type=float, default=1.0e-4)
    p_train.add_argument("--zero-weight", type=float, default=0.018)
    p_train.add_argument("--roi-probability", type=float, default=0.82)
    p_train.add_argument("--stats-samples", type=int, default=10)
    p_train.add_argument("--seed", type=int, default=9449)
    p_train.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])
    p_train.add_argument("--log-every", type=int, default=40)
    p_train.add_argument("--save-every", type=int, default=180)
    p_train.set_defaults(func=train)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--scene", required=True, choices=sorted(SCENES))
    p_apply.add_argument("--output-dir", default=str(RUN_ROOT / "frequency_chroma_distiller_pilot_outputs"))
    p_apply.add_argument("--name", default=None)
    p_apply.add_argument("--tile", type=int, default=768)
    p_apply.add_argument("--overlap", type=int, default=48)
    p_apply.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])
    p_apply.set_defaults(func=apply)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--scene", required=True, choices=sorted(SCENES))
    p_compare.add_argument("--result", required=True)
    p_compare.add_argument("--output-dir", default=str(RUN_ROOT / "frequency_chroma_distiller_pilot_outputs"))
    p_compare.add_argument("--crop-size", type=int, default=512)
    p_compare.add_argument("--scale", type=int, default=2)
    p_compare.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
