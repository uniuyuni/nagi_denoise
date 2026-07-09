"""Train/apply a learned gate for luma-rebuilder detail.

This distills the deterministic luma detail discriminator into a tiny CNN. The
model predicts where the luma-rebuilder residual should pass through, keeping
the current selector output everywhere else.
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
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_flat_chroma_smoother import LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_detail_discriminator import apply_luma_delta, build_gate
from apply_region_aware_luma_cleanup import make_coherent_structure_mask, make_skin_mask, make_texture_mask
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import make_preview, read_image


ROOT = Path(__file__).resolve().parents[1]
TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
RUN_ROOT = ROOT / "runs/refiner_pilot_stage11_hybrid_best"
V4 = RUN_ROOT / "final_v4_red115_blue120_detailguard_mild"
V7 = RUN_ROOT / "final_v7_luma_rebuild"
V8 = RUN_ROOT / "final_v8_region_rebuild"
V9 = RUN_ROOT / "final_v9_adaptive_region"
SELECTOR_V2 = RUN_ROOT / "blend_selector_pilot_v2_outputs"
REBUILDER_V3 = RUN_ROOT / "luma_rebuilder_pilot_v3_outputs"

FEATURE_CHANNELS = 31
STRICT_NOISE = {
    "floor": 0.12,
    "structure_gain": 1.22,
    "coherent_weight": 1.12,
    "base_texture_weight": 1.05,
    "rebuild_texture_weight": 0.62,
    "dark_flat_suppress": 1.15,
    "skin_suppress": 0.38,
    "grain_suppress": 1.18,
    "gate_blur": 0.75,
}
V1 = {
    "floor": 0.18,
    "structure_gain": 1.18,
    "coherent_weight": 1.00,
    "base_texture_weight": 0.95,
    "rebuild_texture_weight": 0.75,
    "dark_flat_suppress": 0.92,
    "skin_suppress": 0.32,
    "grain_suppress": 0.82,
    "gate_blur": 0.65,
}


@dataclass(frozen=True)
class GateScene:
    name: str
    noisy: Path
    current: Path
    base: Path
    rebuild: Path
    cleanup: Path
    result: Path


SCENES: tuple[GateScene, ...] = (
    GateScene(
        "xt5_occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        SELECTOR_V2 / "xt5_occi_blend_selector_v2.exr",
        V4 / "xt5_occi_final_v4_red115_blue120_detailguard_mild.exr",
        V7 / "xt5_occi_luma_rebuild.exr",
        V8 / "xt5_occi_rebuild_region_hybrid_clean.exr",
        REBUILDER_V3 / "xt5_occi_luma_rebuilder_v3_s1.exr",
    ),
    GateScene(
        "k5_dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        SELECTOR_V2 / "k5_dance_blend_selector_v2.exr",
        V4 / "k5_dance_final_v4_red115_blue120_detailguard_mild.exr",
        V7 / "k5_dance_luma_rebuild.exr",
        V9 / "k5_dance_rebuild_adaptive_sky.exr",
        REBUILDER_V3 / "k5_dance_luma_rebuilder_v3_s1.exr",
    ),
)
SCENE_BY_NAME = {scene.name: scene for scene in SCENES}

ROI_TOP_LEFT: dict[str, list[tuple[str, int, int]]] = {
    "xt5_occi": [
        ("face_center", 2120, 1260),
        ("hair_detail", 2420, 1040),
        ("cheek_hair", 2280, 1420),
        ("root", 512, 5632),
        ("noise_dark", 3072, 3600),
    ],
    "k5_dance": [
        ("sky_existing", 4096, 0),
        ("sky_center", 2300, 320),
        ("dancer_center", 2800, 1200),
        ("right_dancer", 3820, 1200),
        ("snow_ground", 2100, 2500),
        ("house_detail", 260, 1180),
    ],
}

ROI_GATE_BIAS: dict[str, tuple[float, float]] = {
    "face_center": (0.85, 1.20),
    "hair_detail": (1.20, 0.75),
    "cheek_hair": (1.16, 0.82),
    "root": (1.24, 0.70),
    "noise_dark": (0.62, 1.55),
    "sky_existing": (0.58, 1.72),
    "sky_center": (0.60, 1.62),
    "dancer_center": (1.18, 0.78),
    "right_dancer": (1.18, 0.78),
    "snow_ground": (0.82, 1.24),
    "house_detail": (1.20, 0.76),
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
    noisy_linear: np.ndarray,
    current_linear: np.ndarray,
    base_linear: np.ndarray,
    rebuild_linear: np.ndarray,
    cleanup_linear: np.ndarray,
    result_linear: np.ndarray,
    *,
    strict_mix: float,
    detail_mix: float,
    roi_name: str | None = None,
    roi_bias_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    noisy = _display(noisy_linear)
    current = _display(current_linear)
    base = _display(base_linear)
    rebuild = _display(rebuild_linear)
    cleanup = _display(cleanup_linear)
    result = _display(result_linear)
    if not (noisy.shape == current.shape == base.shape == rebuild.shape == cleanup.shape == result.shape):
        raise ValueError("all inputs must have matching shape")

    noisy_y = luma(noisy, LUMA_SRGB)
    current_y = luma(current, LUMA_SRGB)
    base_y = luma(base, LUMA_SRGB)
    rebuild_y = luma(rebuild, LUMA_SRGB)
    cleanup_y = luma(cleanup, LUMA_SRGB)
    result_y = luma(result, LUMA_SRGB)
    raw_delta = result_y - current_y
    sat = saturation(current)
    ref_texture = make_texture_mask(noisy, texture_threshold=0.006, texture_transition=0.012)
    base_texture = make_texture_mask(base, texture_threshold=0.018, texture_transition=0.014)
    coherent = make_coherent_structure_mask(
        noisy,
        coherence_threshold=0.40,
        coherence_transition=0.18,
        energy_threshold=0.006,
        energy_transition=0.006,
    )
    skin = make_skin_mask(current, blur_sigma=1.4)
    noisy_hf = np.abs(highpass(noisy_y, 0.75))
    current_hf = highpass(current_y, 0.75)
    rebuild_hf = highpass(rebuild_y, 0.75)
    result_hf = highpass(result_y, 0.75)

    strict_gate, _, _ = build_gate(
        noisy_linear,
        current_linear,
        base_linear,
        rebuild_linear,
        structure_gain=STRICT_NOISE["structure_gain"],
        coherent_weight=STRICT_NOISE["coherent_weight"],
        base_texture_weight=STRICT_NOISE["base_texture_weight"],
        rebuild_texture_weight=STRICT_NOISE["rebuild_texture_weight"],
        dark_flat_suppress=STRICT_NOISE["dark_flat_suppress"],
        skin_suppress=STRICT_NOISE["skin_suppress"],
        grain_suppress=STRICT_NOISE["grain_suppress"],
        gate_blur=STRICT_NOISE["gate_blur"],
    )
    v1_gate, _, _ = build_gate(
        noisy_linear,
        current_linear,
        base_linear,
        rebuild_linear,
        structure_gain=V1["structure_gain"],
        coherent_weight=V1["coherent_weight"],
        base_texture_weight=V1["base_texture_weight"],
        rebuild_texture_weight=V1["rebuild_texture_weight"],
        dark_flat_suppress=V1["dark_flat_suppress"],
        skin_suppress=V1["skin_suppress"],
        grain_suppress=V1["grain_suppress"],
        gate_blur=V1["gate_blur"],
    )
    detail_extra = np.clip(v1_gate - strict_gate, 0.0, 1.0) * np.clip(coherent + base_texture, 0.0, 1.0)
    target_gate = np.clip(strict_gate * float(strict_mix) + detail_extra * float(detail_mix), 0.0, 1.0)
    if roi_name in ROI_GATE_BIAS and roi_bias_strength > 0:
        gate_mul, suppress_mul = ROI_GATE_BIAS[roi_name]
        bias = 1.0 + (float(gate_mul) - 1.0) * float(roi_bias_strength)
        target_gate = np.clip(target_gate * bias, 0.0, 1.0)
        if suppress_mul > 1.0:
            suppress = 1.0 - (float(suppress_mul) - 1.0) * 0.18 * float(roi_bias_strength)
            target_gate = np.clip(target_gate * suppress, 0.0, 1.0)

    delta_weight = np.clip(0.20 + np.abs(raw_delta) / 0.030 + coherent * 0.85 + base_texture * 0.65, 0.20, 2.0)
    feats = np.concatenate(
        [
            noisy,
            current,
            base,
            rebuild,
            cleanup,
            noisy_y[..., None],
            current_y[..., None],
            base_y[..., None],
            rebuild_y[..., None],
            cleanup_y[..., None],
            result_y[..., None],
            raw_delta[..., None],
            sat[..., None],
            ref_texture[..., None],
            base_texture[..., None],
            coherent[..., None],
            skin[..., None],
            noisy_hf[..., None],
            current_hf[..., None],
            rebuild_hf[..., None],
            result_hf[..., None],
        ],
        axis=2,
    ).astype(np.float32, copy=False)
    if feats.shape[2] != FEATURE_CHANNELS:
        raise AssertionError(f"feature channel mismatch: {feats.shape[2]} != {FEATURE_CHANNELS}")
    stats = {
        "strict_gate_mean": float(np.mean(strict_gate)),
        "v1_gate_mean": float(np.mean(v1_gate)),
        "target_gate_mean": float(np.mean(target_gate)),
        "target_gate_p95": float(np.quantile(target_gate, 0.95)),
        "raw_delta_abs_mean": float(np.mean(np.abs(raw_delta))),
    }
    return feats, target_gate[..., None].astype(np.float32, copy=False), delta_weight[..., None].astype(np.float32, copy=False), stats


class GatePatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes: list[GateScene],
        patch_size: int,
        context: int,
        samples: int,
        seed: int,
        roi_probability: float,
        roi_bias_strength: float,
        strict_mix: float,
        detail_mix: float,
        stats_samples: int,
    ) -> None:
        self.patch_size = int(patch_size)
        self.context = int(context)
        self.samples = int(samples)
        self.rng = random.Random(seed)
        self.roi_probability = float(roi_probability)
        self.roi_bias_strength = float(roi_bias_strength)
        self.strict_mix = float(strict_mix)
        self.detail_mix = float(detail_mix)
        self.items = []
        for scene in scenes:
            missing = [
                p
                for p in (scene.noisy, scene.current, scene.base, scene.rebuild, scene.cleanup, scene.result)
                if not p.exists()
            ]
            if missing:
                raise FileNotFoundError(f"{scene.name} missing: {missing}")
            self.items.append(
                {
                    "scene": scene,
                    "noisy": read_image(scene.noisy),
                    "current": read_image(scene.current),
                    "base": read_image(scene.base),
                    "rebuild": read_image(scene.rebuild),
                    "cleanup": read_image(scene.cleanup),
                    "result": read_image(scene.result),
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
        noisy, inner = crop_with_context(item["noisy"], x, y, patch, self.context)
        current, _ = crop_with_context(item["current"], x, y, patch, self.context)
        base, _ = crop_with_context(item["base"], x, y, patch, self.context)
        rebuild, _ = crop_with_context(item["rebuild"], x, y, patch, self.context)
        cleanup, _ = crop_with_context(item["cleanup"], x, y, patch, self.context)
        result, _ = crop_with_context(item["result"], x, y, patch, self.context)
        feats, gate, weight, _ = make_features_and_gate(
            noisy,
            current,
            base,
            rebuild,
            cleanup,
            result,
            strict_mix=self.strict_mix,
            detail_mix=self.detail_mix,
            roi_name=roi_name,
            roi_bias_strength=self.roi_bias_strength,
        )
        ix0, iy0, ix1, iy1 = inner
        return feats[iy0:iy1, ix0:ix1], gate[iy0:iy1, ix0:ix1], weight[iy0:iy1, ix0:ix1]

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
        return {
            "target_gate_mean": gate_sum / count,
            "weight_mean": weight_sum / count,
            "stats_samples": count,
            "roi_counts": roi_counts,
        }

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        item = self.rng.choice(self.items)
        height, width = item["current"].shape[:2]
        x, y, roi_name = self._sample_xy(item["scene"].name, width, height)
        feats, gate, weight = self._make_patch(item, x, y, roi_name)
        return {
            "features": torch.from_numpy(np.ascontiguousarray(np.transpose(feats, (2, 0, 1)))),
            "gate": torch.from_numpy(np.ascontiguousarray(np.transpose(gate, (2, 0, 1)))),
            "weight": torch.from_numpy(np.ascontiguousarray(np.transpose(weight, (2, 0, 1)))),
        }


class GateBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(F.gelu(self.pw1(self.dw(x)))) * self.scale


class LumaDetailGate(nn.Module):
    def __init__(self, width: int = 24, blocks: int = 4) -> None:
        super().__init__()
        self.head = nn.Conv2d(FEATURE_CHANNELS, width, 3, padding=1)
        self.body = nn.Sequential(*[GateBlock(width) for _ in range(blocks)])
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
    ds = GatePatchDataset(
        scenes,
        patch_size=args.patch_size,
        context=args.context,
        samples=args.steps * args.batch_size,
        seed=args.seed,
        roi_probability=args.roi_probability,
        roi_bias_strength=args.roi_bias_strength,
        strict_mix=args.strict_mix,
        detail_mix=args.detail_mix,
        stats_samples=args.stats_samples,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = LumaDetailGate(width=args.width, blocks=args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    meta = {
        "scenes": [scene.name for scene in scenes],
        "scene_stats": {item["scene"].name: item["stats"] for item in ds.items},
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "feature_channels": FEATURE_CHANNELS,
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_path = out_dir / "stdout.log"
    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== train luma-detail-gate steps={args.steps} device={device} ===\n")
        for step, batch in enumerate(dl, start=1):
            if step > args.steps:
                break
            features = batch["features"].to(device)
            target = batch["gate"].to(device)
            weight = batch["weight"].to(device)
            pred = model(features)
            loss_gate = (F.smooth_l1_loss(pred, target, beta=0.045, reduction="none") * weight).mean()
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
                save_checkpoint(out_dir / f"luma_detail_gate_step_{step:06d}.pt", model, args, step)
        save_checkpoint(out_dir / "luma_detail_gate_final.pt", model, args, args.steps)
        log.write(f"wrote {out_dir / 'luma_detail_gate_final.pt'}\n")
    print(f"wrote {out_dir / 'luma_detail_gate_final.pt'}")


def save_checkpoint(path: Path, model: LumaDetailGate, args: argparse.Namespace, step: int) -> None:
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


def load_model(checkpoint: Path, device: torch.device) -> LumaDetailGate:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = LumaDetailGate(width=int(ckpt["width"]), blocks=int(ckpt["blocks"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_gate_tiled(
    model: nn.Module,
    device: torch.device,
    scene: GateScene,
    *,
    tile: int,
    overlap: int,
    strict_mix: float,
    detail_mix: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    noisy = read_image(scene.noisy)
    current = read_image(scene.current)
    base = read_image(scene.base)
    rebuild = read_image(scene.rebuild)
    cleanup = read_image(scene.cleanup)
    result = read_image(scene.result)
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
                noisy[py0:py1, px0:px1],
                current[py0:py1, px0:px1],
                base[py0:py1, px0:px1],
                rebuild[py0:py1, px0:px1],
                cleanup[py0:py1, px0:px1],
                result[py0:py1, px0:px1],
                strict_mix=strict_mix,
                detail_mix=detail_mix,
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
    gate = gate_acc / np.maximum(weight_acc, 1.0e-6)
    current_y = luma(_display(current), LUMA_SRGB)
    result_y = luma(_display(result), LUMA_SRGB)
    raw_delta = result_y - current_y
    stats = {
        "gate_mean": float(np.mean(gate)),
        "gate_p95": float(np.quantile(gate, 0.95)),
        "raw_delta_abs_mean": float(np.mean(np.abs(raw_delta))),
        "elapsed_sec": float(time.monotonic() - start),
    }
    return current, gate[..., 0], raw_delta, stats


def apply(args: argparse.Namespace) -> None:
    scene = SCENE_BY_NAME[args.scene]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    current, gate, raw_delta, stats = predict_gate_tiled(
        model,
        device,
        scene,
        tile=args.tile,
        overlap=args.overlap,
        strict_mix=args.strict_mix,
        detail_mix=args.detail_mix,
    )
    gated_delta = raw_delta * (float(args.floor) + (1.0 - float(args.floor)) * gate) * float(args.strength)
    out = apply_luma_delta(current, gated_delta)
    name = args.name or f"{scene.name}_luma_detail_gate"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    delta_path = out_dir / f"{name}_delta.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    Image.fromarray(np.clip((gated_delta / 0.12) * 127.0 + 128.0, 0, 255).astype(np.uint8)).save(delta_path)
    stats.update(
        {
            "floor": float(args.floor),
            "strength": float(args.strength),
            "gated_delta_abs_mean": float(np.mean(np.abs(gated_delta))),
            "gated_delta_abs_p95": float(np.quantile(np.abs(gated_delta), 0.95)),
        }
    )
    meta = {
        "scene": scene.name,
        "checkpoint": str(Path(args.checkpoint)),
        "outputs": {"exr": str(exr_path), "preview": str(preview_path), "gate": str(gate_path), "delta": str(delta_path)},
        "stats": stats,
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply a learned luma detail gate.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--scenes", default="xt5_occi,k5_dance")
    p_train.add_argument("--steps", type=int, default=300)
    p_train.add_argument("--batch-size", type=int, default=3)
    p_train.add_argument("--patch-size", type=int, default=192)
    p_train.add_argument("--context", type=int, default=64)
    p_train.add_argument("--width", type=int, default=24)
    p_train.add_argument("--blocks", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1.5e-4)
    p_train.add_argument("--weight-decay", type=float, default=1.0e-4)
    p_train.add_argument("--smooth-weight", type=float, default=0.025)
    p_train.add_argument("--floor-weight", type=float, default=0.012)
    p_train.add_argument("--min-gate-mean", type=float, default=0.08)
    p_train.add_argument("--strict-mix", type=float, default=1.0)
    p_train.add_argument("--detail-mix", type=float, default=0.55)
    p_train.add_argument("--roi-probability", type=float, default=0.92)
    p_train.add_argument("--roi-bias-strength", type=float, default=0.75)
    p_train.add_argument("--stats-samples", type=int, default=10)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--seed", type=int, default=31415)
    p_train.add_argument("--log-every", type=int, default=50)
    p_train.add_argument("--save-every", type=int, default=150)
    p_train.set_defaults(func=train)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--scene", required=True, choices=sorted(SCENE_BY_NAME))
    p_apply.add_argument("--output-dir", required=True)
    p_apply.add_argument("--name", default=None)
    p_apply.add_argument("--tile", type=int, default=768)
    p_apply.add_argument("--overlap", type=int, default=64)
    p_apply.add_argument("--floor", type=float, default=0.12)
    p_apply.add_argument("--strength", type=float, default=1.0)
    p_apply.add_argument("--strict-mix", type=float, default=1.0)
    p_apply.add_argument("--detail-mix", type=float, default=0.55)
    p_apply.add_argument("--device", default="cpu")
    p_apply.set_defaults(func=apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
