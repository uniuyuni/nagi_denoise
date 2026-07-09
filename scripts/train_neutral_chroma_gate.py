"""Train/apply a small gate for neutral chroma-dot cleanup.

The deterministic neutral chroma pass is useful but hand-thresholded. This
pilot learns only the blend gate between the v8 base and the deterministic
neutral cleanup candidate, keeping the actual chroma correction deterministic.
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
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, median_filter
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma
from apply_luma_hf_shrink_filter import sigmoid01
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import image_stats, make_preview, read_image


ROOT = Path(__file__).resolve().parents[1]
TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
RUN_ROOT = ROOT / "runs/refiner_pilot_stage11_hybrid_best"
V8_ROOT = RUN_ROOT / "signed_chroma_after_v7_probe_bm_strong"
NEUTRAL_ROOT = RUN_ROOT / "neutral_chroma_dot_after_v8_probe"

FEATURE_CHANNELS = 25


@dataclass(frozen=True)
class Scene:
    name: str
    reference: Path
    base: Path
    neutral: Path
    rois: tuple[tuple[str, int, int], ...]


SCENES: dict[str, Scene] = {
    "xt5_occi": Scene(
        "xt5_occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        V8_ROOT / "xt5_occi_v8_signed_chroma_bm_strong.exr",
        NEUTRAL_ROOT / "xt5_occi_v9_neutral_chroma_dot_strong.exr",
        (("face_hair", 1780, 1140), ("bangs", 2170, 850), ("body_shadow", 2900, 3720)),
    ),
    "k5_dance": Scene(
        "k5_dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        V8_ROOT / "k5_dance_v8_signed_chroma_bm_strong.exr",
        NEUTRAL_ROOT / "k5_dance_v9_neutral_chroma_dot_strong.exr",
        (("sky_center", 2300, 320), ("dancer_center", 2800, 1200), ("snow_ground", 2100, 2500)),
    ),
    "k5_ice": Scene(
        "k5_ice",
        TEST_PHOTOS / "K-5 Ice noisy.EXR",
        V8_ROOT / "k5_ice_v8_signed_chroma_bm_strong.exr",
        NEUTRAL_ROOT / "k5_ice_v9_neutral_chroma_dot_strong.exr",
        (("blue_shadow", 2700, 900), ("edge_detail", 1700, 1450), ("ice_center", 2100, 1180)),
    ),
}


ROI_BIAS: dict[str, tuple[float, float]] = {
    "sky_center": (1.18, 0.88),
    "snow_ground": (1.12, 0.92),
    "body_shadow": (1.08, 0.95),
    "blue_shadow": (1.00, 1.00),
    "ice_center": (0.80, 1.18),
    "edge_detail": (0.68, 1.45),
    "face_hair": (0.76, 1.32),
    "bangs": (0.72, 1.38),
    "dancer_center": (0.82, 1.20),
}


def _safe_rgb(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def display(image: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb_np(np.clip(_safe_rgb(image), 0.0, None)), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def saturation(rgb: np.ndarray) -> np.ndarray:
    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    return ((mx - mn) / np.maximum(mx, 1.0e-6)).astype(np.float32, copy=False)


def highpass_abs(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.abs(x - gaussian_filter(x, sigma=float(sigma), mode="reflect")).astype(np.float32, copy=False)


def chroma(rgb: np.ndarray) -> np.ndarray:
    y = luma(rgb, LUMA_SRGB)
    return (rgb - y[..., None]).astype(np.float32, copy=False)


def chroma_impulse(rgb: np.ndarray) -> np.ndarray:
    c = chroma(rgb)
    rg = c[..., 0] - c[..., 1]
    by = c[..., 2] - 0.5 * (c[..., 0] + c[..., 1])
    rg_imp = rg - median_filter(rg, size=3, mode="reflect")
    by_imp = by - median_filter(by, size=3, mode="reflect")
    return np.sqrt(0.5 * (rg_imp * rg_imp + by_imp * by_imp)).astype(np.float32, copy=False)


def blue_struct_signal(rgb: np.ndarray) -> np.ndarray:
    c = chroma(rgb)
    low = gaussian_filter(c, sigma=(2.0, 2.0, 0.0), mode="reflect")
    low_blue = low[..., 2] - 0.5 * (low[..., 0] + low[..., 1])
    low_mag = np.sqrt(np.sum(low * low, axis=2))
    return np.clip(sigmoid01((low_blue - 0.050) / 0.025) * sigmoid01((low_mag - 0.065) / 0.030), 0.0, 1.0)


def crop_with_context(arr: np.ndarray, x: int, y: int, patch: int, context: int) -> np.ndarray:
    h, w = arr.shape[:2]
    x0 = max(0, int(x) - int(context))
    y0 = max(0, int(y) - int(context))
    x1 = min(w, int(x) + int(patch) + int(context))
    y1 = min(h, int(y) + int(patch) + int(context))
    return arr[y0:y1, x0:x1]


def make_features_and_target(
    reference_linear: np.ndarray,
    base_linear: np.ndarray,
    neutral_linear: np.ndarray,
    *,
    roi_name: str | None,
    roi_bias_strength: float,
    target_gain: float,
    target_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    ref = display(reference_linear)
    base = display(base_linear)
    neutral = display(neutral_linear)
    if ref.shape != base.shape or base.shape != neutral.shape:
        raise ValueError(f"shape mismatch ref={ref.shape} base={base.shape} neutral={neutral.shape}")

    ref_y = luma(ref, LUMA_SRGB)
    base_y = luma(base, LUMA_SRGB)
    base_c = chroma(base)
    ref_c = chroma(ref)
    low_base_c = gaussian_filter(base_c, sigma=(2.0, 2.0, 0.0), mode="reflect")
    low_chroma_mag = np.sqrt(np.sum(low_base_c * low_base_c, axis=2))
    neutral_safe = sigmoid01((0.085 - low_chroma_mag) / 0.030)
    edge = gaussian_gradient_magnitude(gaussian_filter(base_y, sigma=0.9, mode="reflect"), sigma=0.9, mode="reflect")
    flat = sigmoid01((0.024 - highpass_abs(base_y, 1.0)) / 0.010) * sigmoid01((0.030 - edge) / 0.015)
    shadow_mid = sigmoid01((0.78 - base_y) / 0.18)
    highlight = sigmoid01((base_y - 0.86) / 0.10)
    blue_struct = blue_struct_signal(base)
    base_imp = chroma_impulse(base)
    neutral_imp = chroma_impulse(neutral)
    benefit = sigmoid01((base_imp - neutral_imp - 0.00045) / 0.0018)
    delta = np.mean(np.abs(neutral - base), axis=2)
    active = sigmoid01((delta - 0.00025) / 0.00055)
    target = np.clip(
        neutral_safe * flat * shadow_mid * (1.0 - highlight) * (1.0 - 0.78 * blue_struct) * (0.20 + 0.80 * benefit) * active,
        0.0,
        1.0,
    )
    if roi_name in ROI_BIAS and roi_bias_strength > 0.0:
        open_mul, protect_mul = ROI_BIAS[roi_name]
        target = np.clip(target * (1.0 + (open_mul - 1.0) * roi_bias_strength), 0.0, 1.0)
        if protect_mul > 1.0:
            target *= max(0.0, 1.0 - (protect_mul - 1.0) * 0.22 * roi_bias_strength)
    if target_power != 1.0:
        target = np.power(np.clip(target, 0.0, 1.0), float(target_power))
    if target_gain != 1.0:
        target = np.clip(target * float(target_gain), 0.0, 1.0)

    sat = saturation(base)
    ref_sat = saturation(ref)
    base_hf = highpass_abs(base_y, 0.75)
    ref_hf = highpass_abs(ref_y, 0.75)
    magenta = 0.5 * (base[..., 0] + base[..., 2]) - base[..., 1]
    blue = base[..., 2] - 0.5 * (base[..., 0] + base[..., 1])
    magenta_imp = np.maximum(magenta - median_filter(magenta, size=3, mode="reflect"), 0.0)
    blue_imp = np.maximum(blue - median_filter(blue, size=3, mode="reflect"), 0.0)
    feats = np.concatenate(
        [
            base,
            ref - base,
            neutral - base,
            base_y[..., None],
            ref_y[..., None],
            sat[..., None],
            ref_sat[..., None],
            neutral_safe[..., None],
            flat[..., None],
            edge[..., None],
            blue_struct[..., None],
            base_imp[..., None],
            neutral_imp[..., None],
            benefit[..., None],
            magenta_imp[..., None],
            blue_imp[..., None],
            low_chroma_mag[..., None],
            base_hf[..., None],
            ref_hf[..., None],
        ],
        axis=2,
    ).astype(np.float32, copy=False)
    if feats.shape[2] != FEATURE_CHANNELS:
        raise AssertionError(f"feature channel mismatch: {feats.shape[2]} != {FEATURE_CHANNELS}")
    weight = np.clip(0.18 + target * 1.75 + blue_struct * 0.65 + edge * 4.0, 0.18, 2.4).astype(np.float32, copy=False)
    stats = {
        "target_mean": float(np.mean(target)),
        "target_p95": float(np.quantile(target, 0.95)),
        "neutral_safe_mean": float(np.mean(neutral_safe)),
        "blue_struct_mean": float(np.mean(blue_struct)),
        "benefit_mean": float(np.mean(benefit)),
    }
    return feats, target[..., None].astype(np.float32, copy=False), weight[..., None], stats


class GateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes: list[Scene],
        *,
        patch_size: int,
        context: int,
        samples: int,
        seed: int,
        roi_probability: float,
        roi_bias_strength: float,
        target_gain: float,
        target_power: float,
        stats_samples: int,
    ) -> None:
        self.patch_size = int(patch_size)
        self.context = int(context)
        self.samples = int(samples)
        self.rng = random.Random(seed)
        self.roi_probability = float(roi_probability)
        self.roi_bias_strength = float(roi_bias_strength)
        self.target_gain = float(target_gain)
        self.target_power = float(target_power)
        self.items = []
        for scene in scenes:
            missing = [p for p in (scene.reference, scene.base, scene.neutral) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"missing inputs for {scene.name}: {missing}")
            item = {
                "scene": scene,
                "reference": read_image(scene.reference),
                "base": read_image(scene.base),
                "neutral": read_image(scene.neutral),
            }
            self.items.append(item)
        self.stats = self._estimate_stats(max(1, int(stats_samples)))

    def __len__(self) -> int:
        return self.samples

    def _sample_xy(self, scene: Scene, width: int, height: int) -> tuple[int, int, str | None]:
        patch = self.patch_size
        if self.rng.random() < self.roi_probability and scene.rois:
            roi_name, x, y = self.rng.choice(scene.rois)
            j = patch // 3
            xx = min(max(0, int(x) + self.rng.randint(-j, j)), max(0, width - patch))
            yy = min(max(0, int(y) + self.rng.randint(-j, j)), max(0, height - patch))
            return xx, yy, roi_name
        return self.rng.randint(0, max(0, width - patch)), self.rng.randint(0, max(0, height - patch)), None

    def _make_patch(self, item: dict, x: int, y: int, roi_name: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref = crop_with_context(item["reference"], x, y, self.patch_size, self.context)
        base = crop_with_context(item["base"], x, y, self.patch_size, self.context)
        neutral = crop_with_context(item["neutral"], x, y, self.patch_size, self.context)
        feats, target, weight, _ = make_features_and_target(
            ref,
            base,
            neutral,
            roi_name=roi_name,
            roi_bias_strength=self.roi_bias_strength,
            target_gain=self.target_gain,
            target_power=self.target_power,
        )
        c = self.context
        p = self.patch_size
        return feats[c : c + p, c : c + p], target[c : c + p, c : c + p], weight[c : c + p, c : c + p]

    def _estimate_stats(self, count: int) -> dict:
        target_sum = 0.0
        weight_sum = 0.0
        roi_counts: dict[str, int] = {}
        for _ in range(count):
            item = self.rng.choice(self.items)
            h, w = item["base"].shape[:2]
            x, y, roi_name = self._sample_xy(item["scene"], w, h)
            _, target, weight = self._make_patch(item, x, y, roi_name)
            target_sum += float(np.mean(target))
            weight_sum += float(np.mean(weight))
            roi_counts[str(roi_name)] = roi_counts.get(str(roi_name), 0) + 1
        return {"target_mean": target_sum / count, "weight_mean": weight_sum / count, "roi_counts": roi_counts}

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        item = self.rng.choice(self.items)
        h, w = item["base"].shape[:2]
        x, y, roi_name = self._sample_xy(item["scene"], w, h)
        feats, target, weight = self._make_patch(item, x, y, roi_name)
        return {
            "features": torch.from_numpy(np.ascontiguousarray(np.transpose(feats, (2, 0, 1)))),
            "gate": torch.from_numpy(np.ascontiguousarray(np.transpose(target, (2, 0, 1)))),
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


class NeutralChromaGate(nn.Module):
    def __init__(self, width: int = 20, blocks: int = 3) -> None:
        super().__init__()
        self.head = nn.Conv2d(FEATURE_CHANNELS, width, 3, padding=1)
        self.body = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.tail(self.body(F.gelu(self.head(features)))))


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def smoothness_loss(gate: torch.Tensor) -> torch.Tensor:
    return torch.abs(gate[:, :, :, 1:] - gate[:, :, :, :-1]).mean() + torch.abs(gate[:, :, 1:, :] - gate[:, :, :-1, :]).mean()


def save_checkpoint(path: Path, model: NeutralChromaGate, args: argparse.Namespace, step: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "width": int(args.width),
            "blocks": int(args.blocks),
            "feature_channels": FEATURE_CHANNELS,
            "step": int(step),
            "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> NeutralChromaGate:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = NeutralChromaGate(width=int(ckpt["width"]), blocks=int(ckpt["blocks"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = [SCENES[name] for name in args.scenes.split(",") if name]
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ds = GateDataset(
        scenes,
        patch_size=args.patch_size,
        context=args.context,
        samples=args.steps * args.batch_size,
        seed=args.seed,
        roi_probability=args.roi_probability,
        roi_bias_strength=args.roi_bias_strength,
        target_gain=args.target_gain,
        target_power=args.target_power,
        stats_samples=args.stats_samples,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = NeutralChromaGate(width=args.width, blocks=args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    meta = {
        "scenes": [scene.name for scene in scenes],
        "dataset_stats": ds.stats,
        "feature_channels": FEATURE_CHANNELS,
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_path = out_dir / "stdout.log"
    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== train neutral-chroma-gate steps={args.steps} device={device} ===\n")
        for step, batch in enumerate(dl, start=1):
            if step > args.steps:
                break
            features = batch["features"].to(device)
            target = batch["gate"].to(device)
            weight = batch["weight"].to(device)
            pred = model(features)
            loss_gate = (F.smooth_l1_loss(pred, target, beta=0.040, reduction="none") * weight).mean()
            loss_smooth = smoothness_loss(pred) * float(args.smooth_weight)
            loss_mean = torch.abs(pred.mean() - target.mean()) * float(args.mean_weight)
            blue_struct = features[:, 16:17]
            loss_blue_close = torch.mean(pred.square() * blue_struct) * float(args.blue_close_weight)
            loss = loss_gate + loss_smooth + loss_mean + loss_blue_close
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0:
                elapsed = time.monotonic() - start
                msg = (
                    f"step {step:05d}/{args.steps} loss={float(loss.detach()):.6f} "
                    f"gate={float(loss_gate.detach()):.6f} smooth={float(loss_smooth.detach()):.6f} "
                    f"mean={float(loss_mean.detach()):.6f} blue={float(loss_blue_close.detach()):.6f} "
                    f"pred_mean={float(pred.mean().detach()):.4f} "
                    f"target_mean={float(target.mean().detach()):.4f} {elapsed / step:.3f}s/it"
                )
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / f"neutral_chroma_gate_step_{step:06d}.pt", model, args, step)
        save_checkpoint(out_dir / "neutral_chroma_gate_final.pt", model, args, args.steps)
        log.write(f"wrote {out_dir / 'neutral_chroma_gate_final.pt'}\n")
    print(f"wrote {out_dir / 'neutral_chroma_gate_final.pt'}")


@torch.inference_mode()
def predict_gate_tiled(
    model: nn.Module,
    device: torch.device,
    reference: np.ndarray,
    base: np.ndarray,
    neutral: np.ndarray,
    *,
    tile: int,
    overlap: int,
) -> tuple[np.ndarray, dict[str, float]]:
    h, w = base.shape[:2]
    gate_acc = np.zeros((h, w, 1), dtype=np.float32)
    weight_acc = np.zeros((h, w, 1), dtype=np.float32)
    stride = max(1, int(tile) - int(overlap) * 2)
    total = len(range(0, h, stride)) * len(range(0, w, stride))
    done = 0
    start = time.monotonic()
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(w, x0 + int(tile))
            y1 = min(h, y0 + int(tile))
            px0 = max(0, x0 - int(overlap))
            py0 = max(0, y0 - int(overlap))
            px1 = min(w, x1 + int(overlap))
            py1 = min(h, y1 + int(overlap))
            feats, _, _, _ = make_features_and_target(
                reference[py0:py1, px0:px1],
                base[py0:py1, px0:px1],
                neutral[py0:py1, px0:px1],
                roi_name=None,
                roi_bias_strength=0.0,
                target_gain=1.0,
                target_power=1.0,
            )
            inp = torch.from_numpy(np.transpose(feats, (2, 0, 1))[None]).to(device)
            pred = model(inp).detach().cpu().numpy()[0].transpose(1, 2, 0)
            cy0 = y0 - py0
            cx0 = x0 - px0
            cy1 = cy0 + (y1 - y0)
            cx1 = cx0 + (x1 - x0)
            gate_acc[y0:y1, x0:x1] += pred[cy0:cy1, cx0:cx1]
            weight_acc[y0:y1, x0:x1] += 1.0
            done += 1
            if done == 1 or done % 32 == 0 or done == total:
                print(f"tile {done:04d}/{total} {(time.monotonic() - start) / done:.3f}s/tile", flush=True)
    gate = gate_acc[..., 0] / np.maximum(weight_acc[..., 0], 1.0e-6)
    stats = {
        "gate_mean": float(np.mean(gate)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "gate_p99": float(np.quantile(gate, 0.99)),
        "elapsed_sec": float(time.monotonic() - start),
    }
    return gate.astype(np.float32, copy=False), stats


def apply(args: argparse.Namespace) -> None:
    scene = SCENES[args.scene]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    reference = read_image(scene.reference)
    base = read_image(scene.base)
    neutral = read_image(scene.neutral)
    gate, stats = predict_gate_tiled(model, device, reference, base, neutral, tile=args.tile, overlap=args.overlap)
    if args.gate_gamma != 1.0:
        gate = np.power(np.clip(gate, 0.0, 1.0), float(args.gate_gamma))
    gate = np.clip(gate * float(args.strength), 0.0, 1.0).astype(np.float32, copy=False)
    out = base * (1.0 - gate[..., None]) + neutral * gate[..., None]
    name = args.name or f"{scene.name}_neutral_chroma_gate"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    stats.update({"strength": float(args.strength), "gate_gamma": float(args.gate_gamma)})
    meta = {
        "scene": scene.name,
        "checkpoint": str(Path(args.checkpoint)),
        "outputs": {"exr": str(exr_path), "preview": str(preview_path), "gate": str(gate_path)},
        "stats": stats,
        "output_stats": image_stats(out),
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply learned neutral chroma cleanup gate.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--scenes", default="xt5_occi,k5_dance,k5_ice")
    p_train.add_argument("--steps", type=int, default=360)
    p_train.add_argument("--batch-size", type=int, default=3)
    p_train.add_argument("--patch-size", type=int, default=192)
    p_train.add_argument("--context", type=int, default=64)
    p_train.add_argument("--width", type=int, default=20)
    p_train.add_argument("--blocks", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=1.4e-4)
    p_train.add_argument("--weight-decay", type=float, default=1.0e-4)
    p_train.add_argument("--smooth-weight", type=float, default=0.025)
    p_train.add_argument("--mean-weight", type=float, default=0.040)
    p_train.add_argument("--blue-close-weight", type=float, default=0.0)
    p_train.add_argument("--target-gain", type=float, default=1.0)
    p_train.add_argument("--target-power", type=float, default=1.0)
    p_train.add_argument("--roi-probability", type=float, default=0.92)
    p_train.add_argument("--roi-bias-strength", type=float, default=0.65)
    p_train.add_argument("--stats-samples", type=int, default=12)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--seed", type=int, default=27182)
    p_train.add_argument("--log-every", type=int, default=60)
    p_train.add_argument("--save-every", type=int, default=180)
    p_train.set_defaults(func=train)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--scene", required=True, choices=sorted(SCENES))
    p_apply.add_argument("--output-dir", required=True)
    p_apply.add_argument("--name", default=None)
    p_apply.add_argument("--tile", type=int, default=768)
    p_apply.add_argument("--overlap", type=int, default=64)
    p_apply.add_argument("--strength", type=float, default=1.0)
    p_apply.add_argument("--gate-gamma", type=float, default=1.0)
    p_apply.add_argument("--device", default="cpu")
    p_apply.set_defaults(func=apply)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
