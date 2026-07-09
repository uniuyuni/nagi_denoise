"""Train/apply a small learned flat cleanup residual branch.

This branch runs after the current best hand-tuned finish. It predicts a bounded
display-RGB residual only for flat/noisy regions, while detail-gate regions are
weighted toward zero residual. The target is a stronger protected-flat cleanup,
not a full PL image teacher.
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
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, uniform_filter
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_detail_protected_flat_cleanup import apply_cleanup
from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import make_preview, read_image


ROOT = Path(__file__).resolve().parents[1]
TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
RUN_ROOT = ROOT / "runs/refiner_pilot_stage11_hybrid_best"
CURRENT_ROOT = RUN_ROOT / "detail_protected_flat_cleanup_v3_more_flat"
DETAIL_GATE_ROOT = RUN_ROOT / "luma_detail_gate_pilot_v2_strict_outputs"

FEATURE_CHANNELS = 21
TARGET_PARAMS = {
    "luma_strength": 1.0,
    "chroma_strength": 1.0,
    "luma_sigma": 3.15,
    "chroma_sigma": 4.00,
    "flat_threshold": 0.034,
    "flat_transition": 0.010,
    "edge_threshold": 0.020,
    "edge_transition": 0.012,
    "coherent_protect": 1.14,
    "texture_protect": 0.36,
    "detail_gate_protect": 1.75,
    "auto_detail_protect": 0.0,
    "auto_detail_threshold": 0.018,
    "auto_detail_transition": 0.010,
    "skin_protect": 0.86,
    "highlight_threshold": 1.0,
    "highlight_transition": 0.25,
    "hdr_restore_threshold": 0.92,
    "hdr_restore_transition": 0.24,
    "gate_blur": 1.15,
}


@dataclass(frozen=True)
class Scene:
    name: str
    reference: Path
    current: Path
    detail_gate: Path


SCENES: tuple[Scene, ...] = (
    Scene(
        "xt5_occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        CURRENT_ROOT / "xt5_occi_gate_v2_flat_cleanup_v3_more_flat.exr",
        DETAIL_GATE_ROOT / "xt5_occi_luma_detail_gate_v2_strict_gate.png",
    ),
    Scene(
        "k5_dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        CURRENT_ROOT / "k5_dance_gate_v2_flat_cleanup_v3_more_flat.exr",
        DETAIL_GATE_ROOT / "k5_dance_luma_detail_gate_v2_strict_gate.png",
    ),
)
SCENE_BY_NAME = {scene.name: scene for scene in SCENES}

ROI_TOP_LEFT: dict[str, list[tuple[str, int, int]]] = {
    "xt5_occi": [
        ("hair_detail", 2420, 1040),
        ("root", 512, 5632),
        ("face_center", 2120, 1260),
        ("noise_dark", 3072, 3600),
    ],
    "k5_dance": [
        ("sky_existing", 4096, 0),
        ("sky_center", 2300, 320),
        ("dancer_center", 2800, 1200),
        ("house_detail", 260, 1180),
        ("snow_ground", 2100, 2500),
    ],
}

ROI_BIAS: dict[str, tuple[float, float]] = {
    "sky_existing": (1.45, 0.55),
    "sky_center": (1.45, 0.55),
    "snow_ground": (1.10, 0.80),
    "noise_dark": (1.25, 0.70),
    "face_center": (0.80, 1.25),
    "hair_detail": (0.55, 1.55),
    "root": (0.60, 1.45),
    "dancer_center": (0.62, 1.42),
    "house_detail": (0.62, 1.42),
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


def _read_gate(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0).astype(np.float32, copy=False)


def saturation(display: np.ndarray) -> np.ndarray:
    mx = np.max(display, axis=2)
    mn = np.min(display, axis=2)
    return ((mx - mn) / np.maximum(mx, 1.0e-6)).astype(np.float32, copy=False)


def highpass(x: np.ndarray, sigma: float) -> np.ndarray:
    return (x - gaussian_filter(x, sigma=float(sigma), mode="reflect")).astype(np.float32, copy=False)


def crop_with_context(arr: np.ndarray, x: int, y: int, patch: int, context: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = arr.shape[:2]
    x0 = max(0, int(x) - int(context))
    y0 = max(0, int(y) - int(context))
    x1 = min(w, int(x) + int(patch) + int(context))
    y1 = min(h, int(y) + int(patch) + int(context))
    return arr[y0:y1, x0:x1], (x - x0, y - y0, x - x0 + patch, y - y0 + patch)


def make_features_and_target(
    reference_linear: np.ndarray,
    current_linear: np.ndarray,
    detail_gate: np.ndarray,
    *,
    roi_name: str | None = None,
    roi_bias_strength: float = 0.0,
    max_delta: float = 0.045,
    target_gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    reference = _display(reference_linear)
    current = _display(current_linear)
    if reference.shape != current.shape:
        raise ValueError(f"shape mismatch reference={reference.shape} current={current.shape}")
    if detail_gate.shape != current.shape[:2]:
        raise ValueError(f"detail gate mismatch detail_gate={detail_gate.shape} current={current.shape}")

    target_linear, target_stats, target_masks = apply_cleanup(reference_linear, current_linear, detail_gate, **TARGET_PARAMS)
    target = _display(target_linear)
    ref_y = luma(reference, LUMA_SRGB)
    cur_y = luma(current, LUMA_SRGB)
    target_y = luma(target, LUMA_SRGB)
    sat = saturation(current)
    detail = np.maximum(
        np.abs(ref_y - uniform_filter(ref_y, size=11, mode="reflect")),
        np.abs(cur_y - uniform_filter(cur_y, size=11, mode="reflect")),
    )
    edge = gaussian_gradient_magnitude(gaussian_filter(cur_y, sigma=0.9, mode="reflect"), sigma=0.9, mode="reflect")
    flat_conf = sigmoid01((0.026 - detail) / 0.010) * sigmoid01((0.024 - edge) / 0.012)
    low_sat = sigmoid01((0.58 - sat) / 0.15)
    detail_block = np.clip(detail_gate * 1.35, 0.0, 1.0)
    cleanup_weight = np.clip(flat_conf * low_sat * (1.0 - detail_block), 0.0, 1.0)
    if roi_name in ROI_BIAS and roi_bias_strength > 0:
        clean_mul, protect_mul = ROI_BIAS[roi_name]
        cleanup_weight = np.clip(cleanup_weight * (1.0 + (clean_mul - 1.0) * roi_bias_strength), 0.0, 1.0)
        if protect_mul > 1.0:
            cleanup_weight *= max(0.0, 1.0 - (protect_mul - 1.0) * 0.22 * roi_bias_strength)
    target_shape = np.power(np.clip(cleanup_weight, 0.0, 1.0), 0.55)
    raw_delta = np.clip((target - current) * float(target_gain), -float(max_delta), float(max_delta))
    target_delta = (raw_delta * target_shape[..., None]).astype(np.float32, copy=False)
    weight = np.clip(0.16 + cleanup_weight * 1.70 + detail_gate * 0.75, 0.16, 2.0).astype(np.float32, copy=False)

    ref_hf = highpass(ref_y, 0.75)
    cur_hf = highpass(cur_y, 0.75)
    target_hf = highpass(target_y, 0.75)
    feats = np.concatenate(
        [
            reference,
            current,
            reference - current,
            ref_y[..., None],
            cur_y[..., None],
            sat[..., None],
            detail_gate[..., None],
            flat_conf[..., None],
            low_sat[..., None],
            cleanup_weight[..., None],
            ref_hf[..., None],
            cur_hf[..., None],
            target_hf[..., None],
            target_masks["gate"][..., None],
            target_masks["protect"][..., None],
        ],
        axis=2,
    ).astype(np.float32, copy=False)
    if feats.shape[2] != FEATURE_CHANNELS:
        raise AssertionError(f"feature channel mismatch: {feats.shape[2]} != {FEATURE_CHANNELS}")
    stats = {
        "target_delta_abs_mean": float(np.mean(np.abs(target_delta))),
        "target_delta_abs_p95": float(np.quantile(np.abs(target_delta), 0.95)),
        "cleanup_weight_mean": float(np.mean(cleanup_weight)),
        "target_shape_mean": float(np.mean(target_shape)),
        "target_gate_mean": float(target_stats["gate_mean"]),
    }
    return feats, target_delta, weight[..., None], stats


class FlatCleanupDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes: list[Scene],
        patch_size: int,
        context: int,
        samples: int,
        seed: int,
        roi_probability: float,
        roi_bias_strength: float,
        max_delta: float,
        target_gain: float,
        stats_samples: int,
    ) -> None:
        self.patch_size = int(patch_size)
        self.context = int(context)
        self.samples = int(samples)
        self.rng = random.Random(seed)
        self.roi_probability = float(roi_probability)
        self.roi_bias_strength = float(roi_bias_strength)
        self.max_delta = float(max_delta)
        self.target_gain = float(target_gain)
        self.items = []
        for scene in scenes:
            missing = [p for p in (scene.reference, scene.current, scene.detail_gate) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{scene.name} missing: {missing}")
            self.items.append(
                {
                    "scene": scene,
                    "reference": read_image(scene.reference),
                    "current": read_image(scene.current),
                    "detail_gate": _read_gate(scene.detail_gate),
                    "stats": {},
                }
            )
        for item in self.items:
            item["stats"] = self._estimate_stats(item, stats_samples)

    def __len__(self) -> int:
        return self.samples

    def _sample_xy(self, scene_name: str, width: int, height: int) -> tuple[int, int, str | None]:
        patch = self.patch_size
        roi_name: str | None = None
        if self.rng.random() < self.roi_probability and ROI_TOP_LEFT.get(scene_name):
            roi_name, rx, ry = self.rng.choice(ROI_TOP_LEFT[scene_name])
            jitter = max(16, patch // 2)
            x = rx + self.rng.randrange(-jitter, jitter + 1)
            y = ry + self.rng.randrange(-jitter, jitter + 1)
        else:
            x = self.rng.randrange(0, max(1, width - patch + 1))
            y = self.rng.randrange(0, max(1, height - patch + 1))
        return min(max(0, x), max(0, width - patch)), min(max(0, y), max(0, height - patch)), roi_name

    def _make_patch(self, item: dict, x: int, y: int, roi_name: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        patch = self.patch_size
        ref, inner = crop_with_context(item["reference"], x, y, patch, self.context)
        cur, _ = crop_with_context(item["current"], x, y, patch, self.context)
        gate, _ = crop_with_context(item["detail_gate"], x, y, patch, self.context)
        feats, delta, weight, _ = make_features_and_target(
            ref,
            cur,
            gate,
            roi_name=roi_name,
            roi_bias_strength=self.roi_bias_strength,
            max_delta=self.max_delta,
            target_gain=self.target_gain,
        )
        ix0, iy0, ix1, iy1 = inner
        return feats[iy0:iy1, ix0:ix1], delta[iy0:iy1, ix0:ix1], weight[iy0:iy1, ix0:ix1]

    def _estimate_stats(self, item: dict, stats_samples: int) -> dict[str, float | int | dict[str, int]]:
        count = max(1, int(stats_samples))
        height, width = item["current"].shape[:2]
        sum_abs = 0.0
        sum_weight = 0.0
        roi_counts: dict[str, int] = {}
        for _ in range(count):
            x, y, roi_name = self._sample_xy(item["scene"].name, width, height)
            _, delta, weight = self._make_patch(item, x, y, roi_name)
            sum_abs += float(np.mean(np.abs(delta)))
            sum_weight += float(np.mean(weight))
            if roi_name is not None:
                roi_counts[roi_name] = roi_counts.get(roi_name, 0) + 1
        return {"target_delta_abs_mean": sum_abs / count, "weight_mean": sum_weight / count, "roi_counts": roi_counts}

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        item = self.rng.choice(self.items)
        height, width = item["current"].shape[:2]
        x, y, roi_name = self._sample_xy(item["scene"].name, width, height)
        feats, delta, weight = self._make_patch(item, x, y, roi_name)
        return {
            "features": torch.from_numpy(np.ascontiguousarray(np.transpose(feats, (2, 0, 1)))),
            "delta": torch.from_numpy(np.ascontiguousarray(np.transpose(delta, (2, 0, 1)))),
            "weight": torch.from_numpy(np.ascontiguousarray(np.transpose(weight, (2, 0, 1)))),
        }


class Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(F.gelu(self.pw1(self.dw(x)))) * self.scale


class FlatCleanupBranch(nn.Module):
    def __init__(self, width: int = 28, blocks: int = 4, max_delta: float = 0.035) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.head = nn.Conv2d(FEATURE_CHANNELS, width, 3, padding=1)
        self.body = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.tail(self.body(F.gelu(self.head(features))))) * self.max_delta


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = [SCENE_BY_NAME[name] for name in args.scenes.split(",") if name]
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ds = FlatCleanupDataset(
        scenes,
        patch_size=args.patch_size,
        context=args.context,
        samples=args.steps * args.batch_size,
        seed=args.seed,
        roi_probability=args.roi_probability,
        roi_bias_strength=args.roi_bias_strength,
        max_delta=args.max_delta,
        target_gain=args.target_gain,
        stats_samples=args.stats_samples,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = FlatCleanupBranch(width=args.width, blocks=args.blocks, max_delta=args.max_delta).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    meta = {
        "scenes": [scene.name for scene in scenes],
        "scene_stats": {item["scene"].name: item["stats"] for item in ds.items},
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "target_params": TARGET_PARAMS,
        "feature_channels": FEATURE_CHANNELS,
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_path = out_dir / "stdout.log"
    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== train flat-cleanup-branch steps={args.steps} device={device} ===\n")
        for step, batch in enumerate(dl, start=1):
            if step > args.steps:
                break
            features = batch["features"].to(device)
            target = batch["delta"].to(device)
            weight = batch["weight"].to(device)
            pred = model(features)
            loss_delta = (F.smooth_l1_loss(pred, target, beta=0.008, reduction="none") * weight).mean()
            loss_tv = total_variation(pred) * args.tv_weight
            loss_sparse = torch.mean(torch.abs(pred)) * args.sparsity_weight
            loss = loss_delta + loss_tv + loss_sparse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0:
                elapsed = time.monotonic() - start
                msg = (
                    f"step {step:05d}/{args.steps} loss={float(loss.detach()):.6f} "
                    f"delta={float(loss_delta.detach()):.6f} tv={float(loss_tv.detach()):.6f} "
                    f"sparse={float(loss_sparse.detach()):.6f} pred_abs={float(torch.mean(torch.abs(pred)).detach()):.5f} "
                    f"target_abs={float(torch.mean(torch.abs(target)).detach()):.5f} {elapsed / step:.3f}s/it"
                )
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / f"flat_cleanup_branch_step_{step:06d}.pt", model, args, step)
        save_checkpoint(out_dir / "flat_cleanup_branch_final.pt", model, args, args.steps)
        log.write(f"wrote {out_dir / 'flat_cleanup_branch_final.pt'}\n")
    print(f"wrote {out_dir / 'flat_cleanup_branch_final.pt'}")


def save_checkpoint(path: Path, model: FlatCleanupBranch, args: argparse.Namespace, step: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "width": args.width,
            "blocks": args.blocks,
            "max_delta": args.max_delta,
            "feature_channels": FEATURE_CHANNELS,
            "step": step,
            "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> FlatCleanupBranch:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FlatCleanupBranch(width=int(ckpt["width"]), blocks=int(ckpt["blocks"]), max_delta=float(ckpt["max_delta"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_tiled(
    model: nn.Module,
    device: torch.device,
    scene: Scene,
    *,
    tile: int,
    overlap: int,
    strength: float,
    target_strength: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    reference = read_image(scene.reference)
    current = read_image(scene.current)
    detail_gate = _read_gate(scene.detail_gate)
    h, w = current.shape[:2]
    delta_acc = np.zeros((h, w, 3), dtype=np.float32)
    weight_acc = np.zeros((h, w, 1), dtype=np.float32)
    stride = max(1, int(tile) - int(overlap) * 2)
    total = len(range(0, h, stride)) * len(range(0, w, stride))
    done = 0
    start = time.monotonic()
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(w, x0 + tile)
            y1 = min(h, y0 + tile)
            px0 = max(0, x0 - overlap)
            py0 = max(0, y0 - overlap)
            px1 = min(w, x1 + overlap)
            py1 = min(h, y1 + overlap)
            feats, _, _, _ = make_features_and_target(
                reference[py0:py1, px0:px1],
                current[py0:py1, px0:px1],
                detail_gate[py0:py1, px0:px1],
                max_delta=target_strength,
            )
            inp = torch.from_numpy(np.transpose(feats, (2, 0, 1))[None]).to(device)
            pred = model(inp).detach().cpu().numpy()[0].transpose(1, 2, 0) * float(strength)
            cy0 = y0 - py0
            cx0 = x0 - px0
            cy1 = cy0 + (y1 - y0)
            cx1 = cx0 + (x1 - x0)
            delta_acc[y0:y1, x0:x1] += pred[cy0:cy1, cx0:cx1]
            weight_acc[y0:y1, x0:x1] += 1.0
            done += 1
            if done == 1 or done % 32 == 0 or done == total:
                print(f"tile {done:04d}/{total} {(time.monotonic() - start) / done:.3f}s/tile", flush=True)
    delta = delta_acc / np.maximum(weight_acc, 1.0e-6)
    current_display = _display(current)
    out_display = np.clip(current_display + delta, 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    current_rgb = _safe_rgb(current)
    peak = np.max(current_rgb, axis=2)
    hdr_restore = sigmoid01((peak - 0.92) / 0.24)
    out = (out * (1.0 - hdr_restore[..., None]) + current_rgb * hdr_restore[..., None]).astype(
        np.float32, copy=False
    )
    stats = {
        "delta_abs_mean": float(np.mean(np.abs(delta))),
        "delta_abs_p95": float(np.quantile(np.abs(delta), 0.95)),
        "delta_abs_p99": float(np.quantile(np.abs(delta), 0.99)),
        "hdr_restore_mean": float(np.mean(hdr_restore)),
        "hdr_restore_p95": float(np.quantile(hdr_restore, 0.95)),
        "elapsed_sec": float(time.monotonic() - start),
    }
    return out, delta, stats


def apply(args: argparse.Namespace) -> None:
    scene = SCENE_BY_NAME[args.scene]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    out, delta, stats = predict_tiled(
        model,
        device,
        scene,
        tile=args.tile,
        overlap=args.overlap,
        strength=args.strength,
        target_strength=args.target_strength,
    )
    name = args.name or f"{scene.name}_flat_cleanup_branch"
    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    delta_path = out_dir / f"{name}_delta.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    write_tiff(tiff_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip((np.mean(delta, axis=2) / 0.040) * 127.0 + 128.0, 0, 255).astype(np.uint8)).save(delta_path)
    meta = {
        "scene": scene.name,
        "checkpoint": str(Path(args.checkpoint)),
        "outputs": {"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path), "delta": str(delta_path)},
        "stats": stats,
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply learned flat cleanup residual branch.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--scenes", default="xt5_occi,k5_dance")
    p_train.add_argument("--steps", type=int, default=360)
    p_train.add_argument("--batch-size", type=int, default=3)
    p_train.add_argument("--patch-size", type=int, default=192)
    p_train.add_argument("--context", type=int, default=64)
    p_train.add_argument("--width", type=int, default=28)
    p_train.add_argument("--blocks", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1.4e-4)
    p_train.add_argument("--weight-decay", type=float, default=1.0e-4)
    p_train.add_argument("--max-delta", type=float, default=0.035)
    p_train.add_argument("--target-gain", type=float, default=1.0)
    p_train.add_argument("--tv-weight", type=float, default=0.012)
    p_train.add_argument("--sparsity-weight", type=float, default=0.004)
    p_train.add_argument("--roi-probability", type=float, default=0.94)
    p_train.add_argument("--roi-bias-strength", type=float, default=0.85)
    p_train.add_argument("--stats-samples", type=int, default=8)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--seed", type=int, default=27182)
    p_train.add_argument("--log-every", type=int, default=60)
    p_train.add_argument("--save-every", type=int, default=180)
    p_train.set_defaults(func=train)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--scene", required=True, choices=sorted(SCENE_BY_NAME))
    p_apply.add_argument("--output-dir", required=True)
    p_apply.add_argument("--name", default=None)
    p_apply.add_argument("--tile", type=int, default=768)
    p_apply.add_argument("--overlap", type=int, default=64)
    p_apply.add_argument("--strength", type=float, default=1.0)
    p_apply.add_argument("--target-strength", type=float, default=0.035)
    p_apply.add_argument("--device", default="cpu")
    p_apply.set_defaults(func=apply)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
