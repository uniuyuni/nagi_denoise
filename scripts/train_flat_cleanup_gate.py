"""Train/apply a learned gate for explicit flat cleanup.

Unlike ``train_flat_cleanup_branch.py``, this model does not predict RGB
residuals. It predicts a scalar cleanup gate, then a deterministic luma/chroma
smoother applies the actual cleanup. This keeps the learned part focused on the
hard question: where is it safe to smooth more?
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

from apply_detail_protected_flat_cleanup import build_cleanup_gate
from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import make_preview, read_image


ROOT = Path(__file__).resolve().parents[1]
TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
RUN_ROOT = ROOT / "runs/refiner_pilot_stage11_hybrid_best"
INPUT_ROOT = RUN_ROOT / "scunet_preset_chooser_v12_flat_cleanup_auto_outputs"

FEATURE_CHANNELS = 23
TARGET_PARAMS = {
    "flat_threshold": 0.034,
    "flat_transition": 0.010,
    "edge_threshold": 0.021,
    "edge_transition": 0.012,
    "coherent_protect": 1.08,
    "texture_protect": 0.30,
    "detail_gate_protect": 1.65,
    "auto_detail_protect": 0.38,
    "auto_detail_threshold": 0.022,
    "auto_detail_transition": 0.012,
    "skin_protect": 0.82,
    "highlight_threshold": 1.0,
    "highlight_transition": 0.25,
    "gate_blur": 1.00,
}
SMOOTH_PARAMS = {
    "luma_strength": 0.94,
    "chroma_strength": 0.97,
    "luma_sigma": 2.75,
    "chroma_sigma": 3.45,
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
        INPUT_ROOT / "xt5_occi_scunet_preset_chooser_v12_auto.exr",
        INPUT_ROOT / "xt5_occi_scunet_preset_chooser_v12_auto_detail_gate.png",
    ),
    Scene(
        "k5_dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        INPUT_ROOT / "k5_dance_scunet_preset_chooser_v12_auto.exr",
        INPUT_ROOT / "k5_dance_scunet_preset_chooser_v12_auto_detail_gate.png",
    ),
    Scene(
        "k5_ice",
        TEST_PHOTOS / "K-5 Ice noisy.EXR",
        INPUT_ROOT / "k5_ice_scunet_preset_chooser_v12_auto.exr",
        INPUT_ROOT / "k5_ice_scunet_preset_chooser_v12_auto_detail_gate.png",
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
    "k5_ice": [
        ("blue_shadow", 1900, 900),
        ("ice_branch", 2420, 1200),
        ("dark_edge", 1400, 1520),
        ("flat_blue", 3000, 780),
    ],
}

ROI_BIAS: dict[str, tuple[float, float]] = {
    "sky_existing": (1.25, 0.75),
    "sky_center": (1.25, 0.75),
    "snow_ground": (1.05, 0.92),
    "noise_dark": (1.15, 0.82),
    "face_center": (0.92, 1.12),
    "hair_detail": (0.70, 1.35),
    "root": (0.72, 1.30),
    "dancer_center": (0.75, 1.28),
    "house_detail": (0.75, 1.28),
    "blue_shadow": (0.96, 1.06),
    "ice_branch": (0.74, 1.30),
    "dark_edge": (0.78, 1.24),
    "flat_blue": (1.18, 0.84),
}

ROI_SUPPRESS_SCALE: dict[str, tuple[float, float]] = {
    "sky_existing": (0.35, 0.35),
    "sky_center": (0.35, 0.35),
    "snow_ground": (0.70, 0.65),
    "noise_dark": (0.55, 0.50),
    "flat_blue": (0.60, 0.58),
    "blue_shadow": (0.90, 0.95),
    "face_center": (1.10, 1.15),
    "hair_detail": (1.25, 1.35),
    "root": (1.20, 1.25),
    "dancer_center": (1.18, 1.25),
    "house_detail": (1.18, 1.25),
    "ice_branch": (1.22, 1.30),
    "dark_edge": (1.15, 1.25),
}

ROI_WEIGHT_MUL: dict[str, float] = {
    "sky_existing": 1.55,
    "sky_center": 1.55,
    "noise_dark": 1.35,
    "flat_blue": 1.30,
    "snow_ground": 1.12,
    "hair_detail": 1.25,
    "root": 1.22,
    "face_center": 1.18,
    "dancer_center": 1.18,
    "house_detail": 1.18,
    "ice_branch": 1.22,
    "dark_edge": 1.18,
    "blue_shadow": 1.05,
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


def _read_gate(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    if not path.exists():
        if shape is None:
            raise FileNotFoundError(path)
        return np.zeros(shape, dtype=np.float32)
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


def make_features_and_gate(
    reference_linear: np.ndarray,
    current_linear: np.ndarray,
    detail_gate: np.ndarray,
    *,
    roi_name: str | None = None,
    roi_bias_strength: float = 0.0,
    target_detail_suppress: float = 0.0,
    target_edge_suppress: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    reference = _display(reference_linear)
    current = _display(current_linear)
    if reference.shape != current.shape:
        raise ValueError(f"shape mismatch reference={reference.shape} current={current.shape}")
    if detail_gate.shape != current.shape[:2]:
        raise ValueError(f"detail gate mismatch detail_gate={detail_gate.shape} current={current.shape}")

    target_gate, target_stats, masks = build_cleanup_gate(
        reference_linear,
        current_linear,
        detail_gate,
        **TARGET_PARAMS,
    )
    if roi_name in ROI_BIAS and roi_bias_strength > 0:
        open_mul, protect_mul = ROI_BIAS[roi_name]
        target_gate = np.clip(target_gate * (1.0 + (open_mul - 1.0) * roi_bias_strength), 0.0, 1.0)
        if protect_mul > 1.0:
            target_gate *= max(0.0, 1.0 - (protect_mul - 1.0) * 0.18 * roi_bias_strength)

    ref_y = luma(reference, LUMA_SRGB)
    cur_y = luma(current, LUMA_SRGB)
    sat = saturation(current)
    ref_detail = np.abs(ref_y - uniform_filter(ref_y, size=11, mode="reflect"))
    cur_detail = np.abs(cur_y - uniform_filter(cur_y, size=11, mode="reflect"))
    edge = gaussian_gradient_magnitude(gaussian_filter(cur_y, sigma=0.9, mode="reflect"), sigma=0.9, mode="reflect")
    if target_detail_suppress > 0 or target_edge_suppress > 0:
        edge_block = sigmoid01((edge - 0.018) / 0.008)
        detail_scale, edge_scale = ROI_SUPPRESS_SCALE.get(roi_name or "", (1.0, 1.0))
        suppress = np.clip(
            detail_gate * float(target_detail_suppress) * float(detail_scale)
            + edge_block * float(target_edge_suppress) * float(edge_scale),
            0.0,
            1.0,
        )
        target_gate = np.clip(target_gate * (1.0 - suppress), 0.0, 1.0)
    low_sat = sigmoid01((0.58 - sat) / 0.15)
    flat_hint = sigmoid01((0.028 - np.maximum(ref_detail, cur_detail)) / 0.010) * sigmoid01((0.024 - edge) / 0.012)
    shadow_hint = sigmoid01((0.30 - cur_y) / 0.10)
    sky_flat_hint = np.clip(flat_hint * low_sat * shadow_hint, 0.0, 1.0).astype(np.float32, copy=False)
    ref_hf = highpass(ref_y, 0.75)
    cur_hf = highpass(cur_y, 0.75)
    chroma = current - cur_y[..., None]
    chroma_hf = np.mean(np.abs(chroma - gaussian_filter(chroma, sigma=(1.0, 1.0, 0.0), mode="reflect")), axis=2)

    weight = np.clip(0.15 + target_gate * 1.55 + detail_gate * 0.75, 0.15, 2.0)
    weight *= float(ROI_WEIGHT_MUL.get(roi_name or "", 1.0))
    weight = np.clip(weight, 0.15, 3.0).astype(np.float32, copy=False)
    feats = np.concatenate(
        [
            reference,
            current,
            reference - current,
            ref_y[..., None],
            cur_y[..., None],
            sat[..., None],
            detail_gate[..., None],
            flat_hint[..., None],
            low_sat[..., None],
            sky_flat_hint[..., None],
            edge[..., None],
            ref_hf[..., None],
            cur_hf[..., None],
            chroma_hf[..., None],
            masks["coherent"][..., None],
            masks["texture"][..., None],
            masks["protect"][..., None],
        ],
        axis=2,
    ).astype(np.float32, copy=False)
    if feats.shape[2] != FEATURE_CHANNELS:
        raise AssertionError(f"feature channel mismatch: {feats.shape[2]} != {FEATURE_CHANNELS}")
    stats = {
        "target_gate_mean": float(np.mean(target_gate)),
        "target_gate_p95": float(np.quantile(target_gate, 0.95)),
        "weight_mean": float(np.mean(weight)),
        "deterministic_gate_mean": float(target_stats["gate_mean"]),
        "target_detail_suppress": float(target_detail_suppress),
        "target_edge_suppress": float(target_edge_suppress),
    }
    return feats, target_gate[..., None].astype(np.float32, copy=False), weight[..., None], stats


def align_feature_channels(feats: np.ndarray, expected_channels: int) -> np.ndarray:
    if feats.shape[2] == int(expected_channels):
        return feats
    if feats.shape[2] == 23 and int(expected_channels) == 22:
        return np.concatenate([feats[:, :, :15], feats[:, :, 16:]], axis=2).astype(np.float32, copy=False)
    raise ValueError(f"feature channel mismatch: feats={feats.shape[2]} expected={expected_channels}")


class GateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes: list[Scene],
        patch_size: int,
        context: int,
        samples: int,
        seed: int,
        roi_probability: float,
        roi_bias_strength: float,
        target_detail_suppress: float,
        target_edge_suppress: float,
        stats_samples: int,
    ) -> None:
        self.patch_size = int(patch_size)
        self.context = int(context)
        self.samples = int(samples)
        self.rng = random.Random(seed)
        self.roi_probability = float(roi_probability)
        self.roi_bias_strength = float(roi_bias_strength)
        self.target_detail_suppress = float(target_detail_suppress)
        self.target_edge_suppress = float(target_edge_suppress)
        self.items = []
        for scene in scenes:
            missing = [p for p in (scene.reference, scene.current) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{scene.name} missing: {missing}")
            current = read_image(scene.current)
            self.items.append(
                {
                    "scene": scene,
                    "reference": read_image(scene.reference),
                    "current": current,
                    "detail_gate": _read_gate(scene.detail_gate, current.shape[:2]),
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
        feats, target, weight, _ = make_features_and_gate(
            ref,
            cur,
            gate,
            roi_name=roi_name,
            roi_bias_strength=self.roi_bias_strength,
            target_detail_suppress=self.target_detail_suppress,
            target_edge_suppress=self.target_edge_suppress,
        )
        ix0, iy0, ix1, iy1 = inner
        return feats[iy0:iy1, ix0:ix1], target[iy0:iy1, ix0:ix1], weight[iy0:iy1, ix0:ix1]

    def _estimate_stats(self, item: dict, stats_samples: int) -> dict[str, float | int | dict[str, int]]:
        count = max(1, int(stats_samples))
        height, width = item["current"].shape[:2]
        gate_sum = 0.0
        weight_sum = 0.0
        roi_counts: dict[str, int] = {}
        for _ in range(count):
            x, y, roi_name = self._sample_xy(item["scene"].name, width, height)
            _, gate, weight = self._make_patch(item, x, y, roi_name)
            gate_sum += float(np.mean(gate))
            weight_sum += float(np.mean(weight))
            if roi_name is not None:
                roi_counts[roi_name] = roi_counts.get(roi_name, 0) + 1
        return {"target_gate_mean": gate_sum / count, "weight_mean": weight_sum / count, "roi_counts": roi_counts}

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        item = self.rng.choice(self.items)
        height, width = item["current"].shape[:2]
        x, y, roi_name = self._sample_xy(item["scene"].name, width, height)
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


class FlatCleanupGate(nn.Module):
    def __init__(self, width: int = 24, blocks: int = 4, feature_channels: int = FEATURE_CHANNELS) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.head = nn.Conv2d(self.feature_channels, width, 3, padding=1)
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


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = [SCENE_BY_NAME[name] for name in args.scenes.split(",") if name]
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
        target_detail_suppress=args.target_detail_suppress,
        target_edge_suppress=args.target_edge_suppress,
        stats_samples=args.stats_samples,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = FlatCleanupGate(width=args.width, blocks=args.blocks, feature_channels=FEATURE_CHANNELS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    meta = {
        "scenes": [scene.name for scene in scenes],
        "scene_stats": {item["scene"].name: item["stats"] for item in ds.items},
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "target_params": TARGET_PARAMS,
        "smooth_params": SMOOTH_PARAMS,
        "roi_bias": ROI_BIAS,
        "roi_suppress_scale": ROI_SUPPRESS_SCALE,
        "roi_weight_mul": ROI_WEIGHT_MUL,
        "feature_channels": FEATURE_CHANNELS,
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_path = out_dir / "stdout.log"
    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== train flat-cleanup-gate steps={args.steps} device={device} ===\n")
        for step, batch in enumerate(dl, start=1):
            if step > args.steps:
                break
            features = batch["features"].to(device)
            target = batch["gate"].to(device)
            weight = batch["weight"].to(device)
            pred = model(features)
            loss_gate = (F.smooth_l1_loss(pred, target, beta=0.05, reduction="none") * weight).mean()
            loss_smooth = smoothness_loss(pred) * args.smooth_weight
            loss_floor = torch.mean(F.relu(args.min_gate_mean - pred.mean(dim=(2, 3))) ** 2) * args.floor_weight
            loss = loss_gate + loss_smooth + loss_floor
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0:
                elapsed = time.monotonic() - start
                msg = (
                    f"step {step:05d}/{args.steps} loss={float(loss.detach()):.6f} "
                    f"gate={float(loss_gate.detach()):.6f} smooth={float(loss_smooth.detach()):.6f} "
                    f"floor={float(loss_floor.detach()):.6f} pred_mean={float(pred.mean().detach()):.4f} "
                    f"target_mean={float(target.mean().detach()):.4f} {elapsed / step:.3f}s/it"
                )
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / f"flat_cleanup_gate_step_{step:06d}.pt", model, args, step)
        save_checkpoint(out_dir / "flat_cleanup_gate_final.pt", model, args, args.steps)
        log.write(f"wrote {out_dir / 'flat_cleanup_gate_final.pt'}\n")
    print(f"wrote {out_dir / 'flat_cleanup_gate_final.pt'}")


def save_checkpoint(path: Path, model: FlatCleanupGate, args: argparse.Namespace, step: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "width": args.width,
            "blocks": args.blocks,
            "feature_channels": FEATURE_CHANNELS,
            "step": step,
            "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> FlatCleanupGate:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FlatCleanupGate(
        width=int(ckpt["width"]),
        blocks=int(ckpt["blocks"]),
        feature_channels=int(ckpt.get("feature_channels", FEATURE_CHANNELS)),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def apply_smoother_with_gate(current_linear: np.ndarray, gate: np.ndarray, *, strength: float) -> np.ndarray:
    current = _display(current_linear)
    y = luma(current, LUMA_SRGB)
    chroma = current - y[..., None]
    y_low = gaussian_filter(y, sigma=SMOOTH_PARAMS["luma_sigma"], mode="reflect")
    chroma_low = gaussian_filter(chroma, sigma=(SMOOTH_PARAMS["chroma_sigma"], SMOOTH_PARAMS["chroma_sigma"], 0.0), mode="reflect")
    luma_blend = np.clip(gate * SMOOTH_PARAMS["luma_strength"] * float(strength), 0.0, 1.0)
    chroma_blend = np.clip(gate * SMOOTH_PARAMS["chroma_strength"] * float(strength), 0.0, 1.0)[..., None]
    out_y = y * (1.0 - luma_blend) + y_low * luma_blend
    out_chroma = chroma * (1.0 - chroma_blend) + chroma_low * chroma_blend
    out_display = np.clip(out_y[..., None] + out_chroma, 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    current_rgb = _safe_rgb(current_linear)
    peak = np.max(current_rgb, axis=2)
    hdr_restore = sigmoid01((peak - 0.92) / 0.24)
    return (out * (1.0 - hdr_restore[..., None]) + current_rgb * hdr_restore[..., None]).astype(
        np.float32, copy=False
    )


@torch.inference_mode()
def predict_gate_tiled(
    model: nn.Module,
    device: torch.device,
    scene: Scene,
    *,
    tile: int,
    overlap: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    reference = read_image(scene.reference)
    current = read_image(scene.current)
    detail_gate = _read_gate(scene.detail_gate, current.shape[:2])
    h, w = current.shape[:2]
    gate_acc = np.zeros((h, w, 1), dtype=np.float32)
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
            feats, _, _, _ = make_features_and_gate(
                reference[py0:py1, px0:px1],
                current[py0:py1, px0:px1],
                detail_gate[py0:py1, px0:px1],
            )
            feats = align_feature_channels(feats, getattr(model, "feature_channels", FEATURE_CHANNELS))
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
        "gate_p95": float(np.quantile(gate, 0.95)),
        "elapsed_sec": float(time.monotonic() - start),
    }
    return current, gate.astype(np.float32, copy=False), stats


def apply(args: argparse.Namespace) -> None:
    scene = SCENE_BY_NAME[args.scene]
    apply_scene(args, scene)


def apply_scene(args: argparse.Namespace, scene: Scene) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    current, gate, stats = predict_gate_tiled(model, device, scene, tile=args.tile, overlap=args.overlap)
    if args.detail_suppress > 0:
        detail_gate = _read_gate(scene.detail_gate, current.shape[:2])
        gate = np.clip(gate * (1.0 - detail_gate * float(args.detail_suppress)), 0.0, 1.0)
        stats["detail_suppress"] = float(args.detail_suppress)
        stats["suppressed_gate_mean"] = float(np.mean(gate))
        stats["suppressed_gate_p95"] = float(np.quantile(gate, 0.95))
    name = args.name or f"{scene.name}_flat_cleanup_gate"
    exr_path = out_dir / f"{name}.exr"
    tiff_path = out_dir / f"{name}.tiff"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}.json"
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    outputs = {"gate": str(gate_path)}
    if not args.gate_only:
        out = apply_smoother_with_gate(current, gate, strength=args.strength)
        write_exr(exr_path, out)
        write_tiff(tiff_path, out)
        Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
        outputs.update({"exr": str(exr_path), "tiff": str(tiff_path), "preview": str(preview_path)})
    stats["strength"] = float(args.strength)
    meta = {
        "scene": scene.name,
        "checkpoint": str(Path(args.checkpoint)),
        "outputs": outputs,
        "stats": stats,
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def apply_custom(args: argparse.Namespace) -> None:
    if not args.reference or not args.current:
        raise SystemExit("apply-custom requires --reference and --current")
    detail_gate = Path(args.detail_gate) if args.detail_gate else Path("__missing_detail_gate__.png")
    scene = Scene(
        args.scene_name or Path(args.current).stem,
        Path(args.reference).expanduser(),
        Path(args.current).expanduser(),
        detail_gate.expanduser(),
    )
    apply_scene(args, scene)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply learned flat cleanup gate.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--scenes", default="xt5_occi,k5_dance")
    p_train.add_argument("--steps", type=int, default=320)
    p_train.add_argument("--batch-size", type=int, default=3)
    p_train.add_argument("--patch-size", type=int, default=192)
    p_train.add_argument("--context", type=int, default=64)
    p_train.add_argument("--width", type=int, default=24)
    p_train.add_argument("--blocks", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1.4e-4)
    p_train.add_argument("--weight-decay", type=float, default=1.0e-4)
    p_train.add_argument("--smooth-weight", type=float, default=0.030)
    p_train.add_argument("--floor-weight", type=float, default=0.010)
    p_train.add_argument("--min-gate-mean", type=float, default=0.06)
    p_train.add_argument("--roi-probability", type=float, default=0.94)
    p_train.add_argument("--roi-bias-strength", type=float, default=0.70)
    p_train.add_argument("--target-detail-suppress", type=float, default=0.0)
    p_train.add_argument("--target-edge-suppress", type=float, default=0.0)
    p_train.add_argument("--stats-samples", type=int, default=8)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--seed", type=int, default=16180)
    p_train.add_argument("--log-every", type=int, default=60)
    p_train.add_argument("--save-every", type=int, default=160)
    p_train.set_defaults(func=train)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--scene", required=True, choices=sorted(SCENE_BY_NAME))
    p_apply.add_argument("--output-dir", required=True)
    p_apply.add_argument("--name", default=None)
    p_apply.add_argument("--tile", type=int, default=768)
    p_apply.add_argument("--overlap", type=int, default=64)
    p_apply.add_argument("--strength", type=float, default=1.0)
    p_apply.add_argument("--detail-suppress", type=float, default=0.0)
    p_apply.add_argument("--gate-only", action="store_true")
    p_apply.add_argument("--device", default="cpu")
    p_apply.set_defaults(func=apply)

    p_apply_custom = sub.add_parser("apply-custom")
    p_apply_custom.add_argument("--checkpoint", required=True)
    p_apply_custom.add_argument("--reference", required=True)
    p_apply_custom.add_argument("--current", required=True)
    p_apply_custom.add_argument("--detail-gate", default=None)
    p_apply_custom.add_argument("--scene-name", default=None)
    p_apply_custom.add_argument("--output-dir", required=True)
    p_apply_custom.add_argument("--name", default=None)
    p_apply_custom.add_argument("--tile", type=int, default=768)
    p_apply_custom.add_argument("--overlap", type=int, default=64)
    p_apply_custom.add_argument("--strength", type=float, default=1.0)
    p_apply_custom.add_argument("--detail-suppress", type=float, default=0.0)
    p_apply_custom.add_argument("--gate-only", action="store_true")
    p_apply_custom.add_argument("--device", default="cpu")
    p_apply_custom.set_defaults(func=apply_custom)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
