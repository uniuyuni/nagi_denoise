"""Train/apply a small luma reconstruction branch for Perfect NR.

This sits after the candidate blend selector. Chroma is kept from the current
selector output; the model only predicts a bounded display-luma residual. The
teacher is pseudo-generated from existing candidates:

* structure regions borrow luma from the rebuild candidate,
* flat dark/skin regions stay close to cleanup/current,
* a small high-frequency boost is allowed only where structure confidence is
  high.

It is deliberately conservative: this is the first step beyond candidate
blending, not a full denoiser rewrite.
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

FEATURE_CHANNELS = 25
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass(frozen=True)
class RebuildScene:
    name: str
    noisy: Path
    current: Path
    base: Path
    rebuild: Path
    cleanup: Path
    pl: Path | None = None


SCENES: tuple[RebuildScene, ...] = (
    RebuildScene(
        "xt5_occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        SELECTOR_V2 / "xt5_occi_blend_selector_v2.exr",
        V4 / "xt5_occi_final_v4_red115_blue120_detailguard_mild.exr",
        V7 / "xt5_occi_luma_rebuild.exr",
        V8 / "xt5_occi_rebuild_region_hybrid_clean.exr",
        TEST_PHOTOS / "X-T5 Occi PL deepprimeXD.tif",
    ),
    RebuildScene(
        "k5_dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        SELECTOR_V2 / "k5_dance_blend_selector_v2.exr",
        V4 / "k5_dance_final_v4_red115_blue120_detailguard_mild.exr",
        V7 / "k5_dance_luma_rebuild.exr",
        V9 / "k5_dance_rebuild_adaptive_sky.exr",
        TEST_PHOTOS / "K-5 Dance PL deepprimeXD.tif",
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


ROI_TARGET_BIAS: dict[str, tuple[float, float]] = {
    "face_center": (0.55, 1.25),
    "hair_detail": (1.85, 0.65),
    "cheek_hair": (1.55, 0.80),
    "root": (2.05, 0.55),
    "noise_dark": (0.25, 1.85),
    "sky_existing": (0.16, 2.25),
    "sky_center": (0.18, 2.10),
    "dancer_center": (1.70, 0.70),
    "right_dancer": (1.65, 0.72),
    "snow_ground": (0.55, 1.45),
    "house_detail": (1.80, 0.70),
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


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


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
    noisy_linear: np.ndarray,
    current_linear: np.ndarray,
    base_linear: np.ndarray,
    rebuild_linear: np.ndarray,
    cleanup_linear: np.ndarray,
    *,
    roi_name: str | None = None,
    roi_bias_strength: float = 0.0,
    structure_pull: float = 0.90,
    detail_boost: float = 0.70,
    synth_detail_boost: float = 0.35,
    mid_boost: float = 0.45,
    max_target_delta: float = 0.16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    noisy = _display(noisy_linear)
    current = _display(current_linear)
    base = _display(base_linear)
    rebuild = _display(rebuild_linear)
    cleanup = _display(cleanup_linear)
    if not (noisy.shape == current.shape == base.shape == rebuild.shape == cleanup.shape):
        raise ValueError(
            f"shape mismatch noisy={noisy.shape} current={current.shape} base={base.shape} "
            f"rebuild={rebuild.shape} cleanup={cleanup.shape}"
        )

    noisy_y = luma(noisy, LUMA_SRGB)
    current_y = luma(current, LUMA_SRGB)
    base_y = luma(base, LUMA_SRGB)
    rebuild_y = luma(rebuild, LUMA_SRGB)
    cleanup_y = luma(cleanup, LUMA_SRGB)
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
    texture_agreement = np.clip(ref_texture * base_texture, 0.0, 1.0)
    structure = np.clip(np.maximum.reduce([base_texture, texture_agreement * 1.35, coherent * 0.95]), 0.0, 1.0)
    dark = sigmoid01((0.34 - current_y) / 0.08)
    low_sat = sigmoid01((0.42 - sat) / 0.12)
    dark_flat = np.clip(dark * low_sat * (1.0 - base_texture), 0.0, 1.0)
    skin_flat = np.clip(skin * (1.0 - base_texture), 0.0, 1.0)
    clean_gate = np.clip(np.maximum(dark_flat, skin_flat * 0.80), 0.0, 1.0)

    if roi_name in ROI_TARGET_BIAS and roi_bias_strength > 0:
        structure_mul, clean_mul = ROI_TARGET_BIAS[roi_name]
        s_mix = 1.0 + (float(structure_mul) - 1.0) * float(roi_bias_strength)
        c_mix = 1.0 + (float(clean_mul) - 1.0) * float(roi_bias_strength)
        structure = np.clip(structure * s_mix, 0.0, 1.0)
        clean_gate = np.clip(clean_gate * c_mix, 0.0, 1.0)

    current_hf = highpass(current_y, 0.75)
    rebuild_hf = highpass(rebuild_y, 0.75)
    current_mid = gaussian_filter(current_y, sigma=0.9, mode="reflect") - gaussian_filter(
        current_y, sigma=3.2, mode="reflect"
    )
    rebuild_mid = gaussian_filter(rebuild_y, sigma=0.9, mode="reflect") - gaussian_filter(
        rebuild_y, sigma=3.2, mode="reflect"
    )
    structure_delta = (
        (rebuild_y - current_y) * float(structure_pull)
        + np.clip(rebuild_hf - current_hf, -0.060, 0.060) * float(detail_boost)
        + np.clip(rebuild_hf, -0.045, 0.045) * float(synth_detail_boost)
        + np.clip(rebuild_mid - current_mid, -0.075, 0.075) * float(mid_boost)
    )
    clean_delta = (cleanup_y - current_y) * (0.85 + 0.15 * clean_gate)
    target_delta_raw = structure_delta * structure * (1.0 - clean_gate * 0.55) + clean_delta * clean_gate
    target_y = current_y + target_delta_raw
    target_y = np.clip(current_y + np.clip(target_y - current_y, -float(max_target_delta), float(max_target_delta)), 0.0, 1.0)
    target_delta = (target_y - current_y).astype(np.float32, copy=False)
    weight = np.clip(0.15 + structure * 1.50 + clean_gate * 1.15, 0.15, 2.0).astype(np.float32, copy=False)

    feats = np.concatenate(
        [
            noisy,
            current,
            base,
            rebuild,
            cleanup,
            noisy_y[..., None],
            current_y[..., None],
            rebuild_y[..., None],
            cleanup_y[..., None],
            (rebuild_y - current_y)[..., None],
            (cleanup_y - current_y)[..., None],
            structure[..., None],
            clean_gate[..., None],
            skin[..., None],
            dark_flat[..., None],
        ],
        axis=2,
    ).astype(np.float32, copy=False)
    if feats.shape[2] != FEATURE_CHANNELS:
        raise AssertionError(f"feature channel mismatch: {feats.shape[2]} != {FEATURE_CHANNELS}")
    stats = {
        "structure_mean": float(np.mean(structure)),
        "clean_gate_mean": float(np.mean(clean_gate)),
        "target_delta_abs_mean": float(np.mean(np.abs(target_delta))),
        "target_delta_abs_p95": float(np.quantile(np.abs(target_delta), 0.95)),
    }
    return feats, target_delta[..., None], weight[..., None], stats


class LumaRebuildDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes: list[RebuildScene],
        patch_size: int,
        context: int,
        samples: int,
        seed: int,
        roi_probability: float,
        roi_bias_strength: float,
        structure_pull: float,
        detail_boost: float,
        synth_detail_boost: float,
        mid_boost: float,
        max_target_delta: float,
        stats_samples: int,
    ) -> None:
        self.patch_size = int(patch_size)
        self.context = int(context)
        self.samples = int(samples)
        self.rng = random.Random(seed)
        self.roi_probability = float(roi_probability)
        self.roi_bias_strength = float(roi_bias_strength)
        self.structure_pull = float(structure_pull)
        self.detail_boost = float(detail_boost)
        self.synth_detail_boost = float(synth_detail_boost)
        self.mid_boost = float(mid_boost)
        self.max_target_delta = float(max_target_delta)
        self.items = []
        for scene in scenes:
            missing = [p for p in (scene.noisy, scene.current, scene.base, scene.rebuild, scene.cleanup) if not p.exists()]
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
        feats, delta, weight, _ = make_features_and_target(
            noisy,
            current,
            base,
            rebuild,
            cleanup,
            roi_name=roi_name,
            roi_bias_strength=self.roi_bias_strength,
            structure_pull=self.structure_pull,
            detail_boost=self.detail_boost,
            synth_detail_boost=self.synth_detail_boost,
            mid_boost=self.mid_boost,
            max_target_delta=self.max_target_delta,
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
        return {
            "target_delta_abs_mean": sum_abs / count,
            "weight_mean": sum_weight / count,
            "stats_samples": count,
            "roi_counts": roi_counts,
        }

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


class RebuildBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(F.gelu(self.pw1(self.dw(x)))) * self.scale


class LumaRebuilder(nn.Module):
    def __init__(self, width: int = 32, blocks: int = 5, max_delta: float = 0.14) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.head = nn.Conv2d(FEATURE_CHANNELS, width, 3, padding=1)
        self.body = nn.Sequential(*[RebuildBlock(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, 1, 3, padding=1)

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
    ds = LumaRebuildDataset(
        scenes,
        patch_size=args.patch_size,
        context=args.context,
        samples=args.steps * args.batch_size,
        seed=args.seed,
        roi_probability=args.roi_probability,
        roi_bias_strength=args.roi_bias_strength,
        structure_pull=args.structure_pull,
        detail_boost=args.detail_boost,
        synth_detail_boost=args.synth_detail_boost,
        mid_boost=args.mid_boost,
        max_target_delta=args.max_target_delta,
        stats_samples=args.stats_samples,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = LumaRebuilder(width=args.width, blocks=args.blocks, max_delta=args.max_delta).to(device)
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
        log.write(f"\n=== train luma-rebuilder steps={args.steps} device={device} ===\n")
        for step, batch in enumerate(dl, start=1):
            if step > args.steps:
                break
            features = batch["features"].to(device)
            target = batch["delta"].to(device)
            weight = batch["weight"].to(device)
            pred = model(features)
            loss_delta = (F.smooth_l1_loss(pred, target, beta=0.018, reduction="none") * weight).mean()
            loss_tv = total_variation(pred) * args.tv_weight
            loss_sparse = torch.abs(pred).mean() * args.sparsity_weight
            loss = loss_delta + loss_tv + loss_sparse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0:
                elapsed = time.monotonic() - start
                msg = (
                    f"step {step:05d}/{args.steps} loss={float(loss.detach()):.6f} "
                    f"delta={float(loss_delta.detach()):.6f} tv={float(loss_tv.detach()):.6f} "
                    f"sparse={float(loss_sparse.detach()):.6f} "
                    f"pred_abs={float(torch.mean(torch.abs(pred)).detach()):.5f} "
                    f"target_abs={float(torch.mean(torch.abs(target)).detach()):.5f} "
                    f"{elapsed / step:.3f}s/it"
                )
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / f"luma_rebuilder_step_{step:06d}.pt", model, args, step)
        save_checkpoint(out_dir / "luma_rebuilder_final.pt", model, args, args.steps)
        log.write(f"wrote {out_dir / 'luma_rebuilder_final.pt'}\n")
    print(f"wrote {out_dir / 'luma_rebuilder_final.pt'}")


def save_checkpoint(path: Path, model: LumaRebuilder, args: argparse.Namespace, step: int) -> None:
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


def load_model(checkpoint: Path, device: torch.device) -> LumaRebuilder:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = LumaRebuilder(width=int(ckpt["width"]), blocks=int(ckpt["blocks"]), max_delta=float(ckpt["max_delta"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def apply_luma_delta(current_linear: np.ndarray, delta: np.ndarray) -> np.ndarray:
    current = _display(current_linear)
    current_y = luma(current, LUMA_SRGB)
    out_y = np.clip(current_y + delta, 0.0, 1.0)
    chroma = current / np.maximum(current_y[..., None], 1.0e-6)
    out_display = np.clip(chroma * out_y[..., None], 0.0, 1.0)
    dark = current_y < 1.0e-5
    out_display[dark] = current[dark]
    return srgb_to_linear_np(out_display).astype(np.float32, copy=False)


@torch.inference_mode()
def predict_delta_tiled(
    model: nn.Module,
    device: torch.device,
    scene: RebuildScene,
    *,
    tile: int,
    overlap: int,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    noisy = read_image(scene.noisy)
    current = read_image(scene.current)
    base = read_image(scene.base)
    rebuild = read_image(scene.rebuild)
    cleanup = read_image(scene.cleanup)
    h, w = current.shape[:2]
    delta_acc = np.zeros((h, w, 1), dtype=np.float32)
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
                noisy[py0:py1, px0:px1],
                current[py0:py1, px0:px1],
                base[py0:py1, px0:px1],
                rebuild[py0:py1, px0:px1],
                cleanup[py0:py1, px0:px1],
                detail_boost=0.0,
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
    out = apply_luma_delta(current, delta[..., 0])
    stats = {
        "delta_abs_mean": float(np.mean(np.abs(delta))),
        "delta_abs_p95": float(np.quantile(np.abs(delta), 0.95)),
        "delta_abs_p99": float(np.quantile(np.abs(delta), 0.99)),
        "elapsed_sec": float(time.monotonic() - start),
    }
    return out, delta[..., 0], stats


def apply(args: argparse.Namespace) -> None:
    scene = SCENE_BY_NAME[args.scene]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    out, delta, stats = predict_delta_tiled(
        model,
        device,
        scene,
        tile=args.tile,
        overlap=args.overlap,
        strength=args.strength,
    )
    name = args.name or f"{scene.name}_luma_rebuilder"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    delta_path = out_dir / f"{name}_delta.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    delta_vis = np.clip((delta / 0.12) * 127.0 + 128.0, 0, 255).astype(np.uint8)
    Image.fromarray(delta_vis).save(delta_path)
    meta = {
        "scene": scene.name,
        "checkpoint": str(Path(args.checkpoint)),
        "strength": args.strength,
        "outputs": {"exr": str(exr_path), "preview": str(preview_path), "delta": str(delta_path)},
        "stats": stats,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def crop_display(path: Path, x: int, y: int, size: int, scale: int) -> Image.Image:
    arr = read_image(path)
    crop = arr[y : y + size, x : x + size]
    img = Image.fromarray(make_preview(crop, exposure=1.0, tone="reinhard"))
    return img.resize((size * scale, size * scale), Image.Resampling.NEAREST)


def compare(args: argparse.Namespace) -> None:
    scene = SCENE_BY_NAME[args.scene]
    result = Path(args.result)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources: list[tuple[str, Path]] = [
        ("input", scene.noisy),
        ("selector_v2", scene.current),
        ("rebuild", scene.rebuild),
        ("cleanup", scene.cleanup),
        ("rebuilder", result),
    ]
    if scene.pl is not None and scene.pl.exists():
        sources.append(("pl_ref", scene.pl))
    size = int(args.crop_size)
    scale = int(args.scale)
    label_h = 26
    for roi_name, x, y in ROI_TOP_LEFT[scene.name]:
        crops = [crop_display(path, x, y, size, scale) for _, path in sources]
        w = size * scale
        h = size * scale
        canvas = Image.new("RGB", (w * len(crops), h + label_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for idx, ((label, _), img) in enumerate(zip(sources, crops, strict=True)):
            canvas.paste(img, (idx * w, label_h))
            draw.text((idx * w + 6, 6), label, fill=(235, 235, 235))
        out = out_dir / f"{scene.name}_luma_rebuilder_compare_{roi_name}_{scale}x.png"
        canvas.save(out)
        print(f"wrote {out}")


def scale_result(args: argparse.Namespace) -> None:
    scene = SCENE_BY_NAME[args.scene]
    result = read_image(args.result)
    current = read_image(scene.current)
    if result.shape[:2] != current.shape[:2]:
        raise ValueError(f"shape mismatch: current={current.shape}, result={result.shape}")
    current_display = _display(current)
    result_display = _display(result)
    current_y = luma(current_display, LUMA_SRGB)
    result_y = luma(result_display, LUMA_SRGB)
    scaled_delta = (result_y - current_y) * float(args.strength)
    out = apply_luma_delta(current, scaled_delta)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{scene.name}_luma_rebuilder_scaled_{args.strength:g}"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    delta_path = out_dir / f"{name}_delta.png"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    delta_vis = np.clip((scaled_delta / 0.12) * 127.0 + 128.0, 0, 255).astype(np.uint8)
    Image.fromarray(delta_vis).save(delta_path)
    meta = {
        "scene": scene.name,
        "source_result": str(Path(args.result)),
        "strength": float(args.strength),
        "outputs": {"exr": str(exr_path), "preview": str(preview_path), "delta": str(delta_path)},
        "stats": {
            "delta_abs_mean": float(np.mean(np.abs(scaled_delta))),
            "delta_abs_p95": float(np.quantile(np.abs(scaled_delta), 0.95)),
            "delta_abs_p99": float(np.quantile(np.abs(scaled_delta), 0.99)),
        },
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply a luma reconstruction branch.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--output-dir", default=str(RUN_ROOT / "luma_rebuilder_pilot"))
    p_train.add_argument("--scenes", default="xt5_occi,k5_dance")
    p_train.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p_train.add_argument("--steps", type=int, default=800)
    p_train.add_argument("--batch-size", type=int, default=3)
    p_train.add_argument("--patch-size", type=int, default=192)
    p_train.add_argument("--context", type=int, default=64)
    p_train.add_argument("--width", type=int, default=32)
    p_train.add_argument("--blocks", type=int, default=5)
    p_train.add_argument("--max-delta", type=float, default=0.14)
    p_train.add_argument("--max-target-delta", type=float, default=0.16)
    p_train.add_argument("--structure-pull", type=float, default=0.90)
    p_train.add_argument("--detail-boost", type=float, default=0.70)
    p_train.add_argument("--synth-detail-boost", type=float, default=0.35)
    p_train.add_argument("--mid-boost", type=float, default=0.45)
    p_train.add_argument("--roi-probability", type=float, default=0.92)
    p_train.add_argument("--roi-bias-strength", type=float, default=0.75)
    p_train.add_argument("--stats-samples", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=2.0e-4)
    p_train.add_argument("--weight-decay", type=float, default=1.0e-4)
    p_train.add_argument("--tv-weight", type=float, default=0.020)
    p_train.add_argument("--sparsity-weight", type=float, default=0.006)
    p_train.add_argument("--seed", type=int, default=5317)
    p_train.add_argument("--log-every", type=int, default=50)
    p_train.add_argument("--save-every", type=int, default=0)
    p_train.set_defaults(func=train)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--scene", required=True, choices=sorted(SCENE_BY_NAME))
    p_apply.add_argument("--output-dir", default=str(RUN_ROOT / "luma_rebuilder_outputs"))
    p_apply.add_argument("--name", default=None)
    p_apply.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])
    p_apply.add_argument("--tile", type=int, default=768)
    p_apply.add_argument("--overlap", type=int, default=48)
    p_apply.add_argument("--strength", type=float, default=1.0)
    p_apply.set_defaults(func=apply)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--scene", required=True, choices=sorted(SCENE_BY_NAME))
    p_compare.add_argument("--result", required=True)
    p_compare.add_argument("--output-dir", default=str(RUN_ROOT / "luma_rebuilder_outputs"))
    p_compare.add_argument("--crop-size", type=int, default=512)
    p_compare.add_argument("--scale", type=int, default=2)
    p_compare.set_defaults(func=compare)

    p_scale = sub.add_parser("scale")
    p_scale.add_argument("--scene", required=True, choices=sorted(SCENE_BY_NAME))
    p_scale.add_argument("--result", required=True)
    p_scale.add_argument("--output-dir", default=str(RUN_ROOT / "luma_rebuilder_outputs"))
    p_scale.add_argument("--name", default=None)
    p_scale.add_argument("--strength", type=float, default=0.55)
    p_scale.set_defaults(func=scale_result)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
