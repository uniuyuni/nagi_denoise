"""Train and apply a tiny learned selector for Perfect NR candidate blends.

The model does not denoise from scratch. It learns where to trust the current
candidate images: conservative base, structure rebuild, and region cleanup.
This keeps the experiment practical while moving the brittle hand-written masks
toward a learned region selector.
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
from scipy.ndimage import gaussian_filter, uniform_filter
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
CURRENT_BASE = RUN_ROOT / "luma_detail_gate_pilot_v2_strict_outputs"
CURRENT_CLEANUP = RUN_ROOT / "detail_protected_flat_cleanup_v3_more_flat"

FEATURE_CHANNELS = 25
CANDIDATE_NAMES = ("base", "rebuild", "cleanup")


@dataclass(frozen=True)
class BlendScene:
    name: str
    noisy: Path
    base: Path
    rebuild: Path
    cleanup: Path
    pl: Path | None = None


LEGACY_SCENES: tuple[BlendScene, ...] = (
    BlendScene(
        "xt5_occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        V4 / "xt5_occi_final_v4_red115_blue120_detailguard_mild.exr",
        V7 / "xt5_occi_luma_rebuild.exr",
        V8 / "xt5_occi_rebuild_region_hybrid_clean.exr",
        TEST_PHOTOS / "X-T5 Occi PL deepprimeXD.tif",
    ),
    BlendScene(
        "k5_dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        V4 / "k5_dance_final_v4_red115_blue120_detailguard_mild.exr",
        V7 / "k5_dance_luma_rebuild.exr",
        V9 / "k5_dance_rebuild_adaptive_sky.exr",
        TEST_PHOTOS / "K-5 Dance PL deepprimeXD.tif",
    ),
)

CURRENT_SCENES: tuple[BlendScene, ...] = (
    BlendScene(
        "xt5_occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        CURRENT_BASE / "xt5_occi_luma_detail_gate_v2_strict.exr",
        V7 / "xt5_occi_luma_rebuild.exr",
        CURRENT_CLEANUP / "xt5_occi_gate_v2_flat_cleanup_v3_more_flat.exr",
        TEST_PHOTOS / "X-T5 Occi PL deepprimeXD.tif",
    ),
    BlendScene(
        "k5_dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        CURRENT_BASE / "k5_dance_luma_detail_gate_v2_strict.exr",
        V7 / "k5_dance_luma_rebuild_safe.exr",
        CURRENT_CLEANUP / "k5_dance_gate_v2_flat_cleanup_v3_more_flat.exr",
        TEST_PHOTOS / "K-5 Dance PL deepprimeXD.tif",
    ),
)


def scene_map(candidate_set: str) -> dict[str, BlendScene]:
    if candidate_set == "legacy":
        scenes = LEGACY_SCENES
    elif candidate_set == "current":
        scenes = CURRENT_SCENES
    else:
        raise ValueError(f"unknown candidate set: {candidate_set}")
    return {scene.name: scene for scene in scenes}


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


ROI_WEIGHT_BIAS: dict[str, tuple[float, float, float]] = {
    "face_center": (0.90, 0.58, 1.85),
    "hair_detail": (0.44, 4.20, 0.42),
    "cheek_hair": (0.50, 3.40, 0.55),
    "root": (0.38, 4.80, 0.34),
    "noise_dark": (1.10, 0.10, 2.65),
    "sky_existing": (1.22, 0.13, 2.55),
    "sky_center": (1.18, 0.16, 2.35),
    "dancer_center": (0.50, 3.85, 0.52),
    "right_dancer": (0.50, 3.70, 0.55),
    "snow_ground": (0.95, 0.45, 1.80),
    "house_detail": (0.46, 4.10, 0.46),
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


def luma_detail(display: np.ndarray, sigma: float) -> np.ndarray:
    y = luma(display, LUMA_SRGB)
    return np.abs(y - gaussian_filter(y, sigma=float(sigma), mode="reflect")).astype(np.float32, copy=False)


def make_features_and_teacher(
    noisy_linear: np.ndarray,
    base_linear: np.ndarray,
    rebuild_linear: np.ndarray,
    cleanup_linear: np.ndarray,
    *,
    teacher_sharpness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    noisy = _display(noisy_linear)
    base = _display(base_linear)
    rebuild = _display(rebuild_linear)
    cleanup = _display(cleanup_linear)
    if noisy.shape != base.shape or base.shape != rebuild.shape or base.shape != cleanup.shape:
        raise ValueError(
            f"shape mismatch noisy={noisy.shape} base={base.shape} rebuild={rebuild.shape} cleanup={cleanup.shape}"
        )

    noisy_y = luma(noisy, LUMA_SRGB)
    base_y = luma(base, LUMA_SRGB)
    rebuild_y = luma(rebuild, LUMA_SRGB)
    cleanup_y = luma(cleanup, LUMA_SRGB)
    sat = saturation(base)
    ref_texture = make_texture_mask(noisy, texture_threshold=0.006, texture_transition=0.012)
    base_texture = make_texture_mask(base, texture_threshold=0.018, texture_transition=0.014)
    texture_agreement = np.clip(ref_texture * base_texture, 0.0, 1.0)
    coherent = make_coherent_structure_mask(
        noisy,
        coherence_threshold=0.40,
        coherence_transition=0.18,
        energy_threshold=0.006,
        energy_transition=0.006,
    )
    skin = make_skin_mask(base, blur_sigma=1.4)
    dark = sigmoid01((0.34 - base_y) / 0.08)
    low_sat = sigmoid01((0.42 - sat) / 0.12)
    dark_flat = np.clip(dark * low_sat * (1.0 - base_texture), 0.0, 1.0)
    skin_flat = np.clip(skin * (1.0 - base_texture), 0.0, 1.0)
    rebuild_delta = np.clip(np.abs(rebuild_y - base_y) / 0.18, 0.0, 1.0)
    cleanup_delta = np.clip(np.abs(cleanup_y - rebuild_y) / 0.12, 0.0, 1.0)
    noisy_fine = np.clip(luma_detail(noisy, 0.7) / 0.060, 0.0, 1.0)
    base_fine = np.clip(luma_detail(base, 0.7) / 0.045, 0.0, 1.0)

    # Pseudo-teacher policy:
    # - coherent/agreed texture wants rebuild detail,
    # - dark low-saturation flats and skin-like flats want cleanup/base restraint,
    # - base keeps an anchor so the selector cannot blindly copy the noisy rebuild.
    structure_confidence = np.clip(
        np.maximum.reduce([base_texture, texture_agreement * 1.35, coherent * 0.85]), 0.0, 1.0
    )
    flat_confidence = np.clip(1.0 - structure_confidence, 0.0, 1.0)
    rebuild_score = (
        0.16
        + 2.05 * texture_agreement
        + 1.15 * coherent
        + 0.85 * base_texture
        + 0.35 * ref_texture * structure_confidence
        - 0.92 * dark_flat * flat_confidence
        - 0.35 * skin_flat * flat_confidence
    )
    cleanup_score = (
        0.17
        + 1.18 * dark_flat * flat_confidence
        + 0.68 * skin_flat * flat_confidence
        + 0.12 * cleanup_delta
    )
    base_score = 0.18 + 0.44 * dark_flat * flat_confidence + 0.12 * flat_confidence + 0.10 * rebuild_delta
    scores = np.stack([base_score, rebuild_score, cleanup_score], axis=2).astype(np.float32, copy=False)
    scores = np.maximum(scores, 0.030)
    scores = np.power(scores, float(teacher_sharpness)).astype(np.float32, copy=False)
    weights = scores / np.maximum(np.sum(scores, axis=2, keepdims=True), 1.0e-6)

    feats = np.concatenate(
        [
            noisy,
            base,
            rebuild,
            cleanup,
            noisy_y[..., None],
            base_y[..., None],
            rebuild_y[..., None],
            cleanup_y[..., None],
            (rebuild_y - base_y)[..., None],
            (cleanup_y - rebuild_y)[..., None],
            sat[..., None],
            ref_texture[..., None],
            base_texture[..., None],
            texture_agreement[..., None],
            coherent[..., None],
            skin[..., None],
            dark_flat[..., None],
        ],
        axis=2,
    ).astype(np.float32, copy=False)
    if feats.shape[2] != FEATURE_CHANNELS:
        raise AssertionError(f"feature channel mismatch: {feats.shape[2]} != {FEATURE_CHANNELS}")

    stats = {
        "teacher_base_mean": float(np.mean(weights[..., 0])),
        "teacher_rebuild_mean": float(np.mean(weights[..., 1])),
        "teacher_cleanup_mean": float(np.mean(weights[..., 2])),
        "dark_flat_mean": float(np.mean(dark_flat)),
        "skin_flat_mean": float(np.mean(skin_flat)),
        "texture_agreement_mean": float(np.mean(texture_agreement)),
        "coherent_mean": float(np.mean(coherent)),
        "structure_confidence_mean": float(np.mean(structure_confidence)),
    }
    return feats, weights.astype(np.float32, copy=False), stats


def apply_roi_bias(weights: np.ndarray, roi_name: str | None, strength: float) -> np.ndarray:
    if roi_name is None or roi_name not in ROI_WEIGHT_BIAS or strength <= 0:
        return weights.astype(np.float32, copy=False)
    bias = np.array(ROI_WEIGHT_BIAS[roi_name], dtype=np.float32).reshape(1, 1, 3)
    mixed_bias = 1.0 + (bias - 1.0) * float(strength)
    out = np.maximum(weights.astype(np.float32, copy=False) * mixed_bias, 1.0e-5)
    out = out / np.maximum(np.sum(out, axis=2, keepdims=True), 1.0e-6)
    return out.astype(np.float32, copy=False)


def _crop_with_context(
    arr: np.ndarray,
    x: int,
    y: int,
    patch: int,
    context: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = arr.shape[:2]
    x0 = max(0, int(x) - int(context))
    y0 = max(0, int(y) - int(context))
    x1 = min(width, int(x) + int(patch) + int(context))
    y1 = min(height, int(y) + int(patch) + int(context))
    return arr[y0:y1, x0:x1], (x - x0, y - y0, x - x0 + patch, y - y0 + patch)


class SelectorBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw2(F.gelu(self.pw1(self.dw(x))))
        return x + y * self.scale


class BlendSelector(nn.Module):
    def __init__(self, width: int = 24, blocks: int = 4, in_channels: int = FEATURE_CHANNELS) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, width, 3, padding=1)
        self.body = nn.Sequential(*[SelectorBlock(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, len(CANDIDATE_NAMES), 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.tail(self.body(F.gelu(self.head(features))))
        return torch.softmax(logits, dim=1)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


class BlendPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes: list[BlendScene],
        patch_size: int,
        samples: int,
        seed: int,
        teacher_sharpness: float,
        roi_probability: float,
    ) -> None:
        self.patch_size = int(patch_size)
        self.samples = int(samples)
        self.rng = random.Random(seed)
        self.roi_probability = float(roi_probability)
        self.items = []
        for scene in scenes:
            missing = [p for p in (scene.noisy, scene.base, scene.rebuild, scene.cleanup) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{scene.name} missing: {missing}")
            feats, teacher, stats = make_features_and_teacher(
                read_image(scene.noisy),
                read_image(scene.base),
                read_image(scene.rebuild),
                read_image(scene.cleanup),
                teacher_sharpness=teacher_sharpness,
            )
            self.items.append({"scene": scene, "features": feats, "teacher": teacher, "stats": stats})

    def __len__(self) -> int:
        return self.samples

    def _sample_xy(self, scene_name: str, width: int, height: int) -> tuple[int, int]:
        patch = self.patch_size
        if self.rng.random() < self.roi_probability and ROI_TOP_LEFT.get(scene_name):
            _, rx, ry = self.rng.choice(ROI_TOP_LEFT[scene_name])
            jitter = max(16, patch // 2)
            x = rx + self.rng.randrange(-jitter, jitter + 1)
            y = ry + self.rng.randrange(-jitter, jitter + 1)
        else:
            x = self.rng.randrange(0, max(1, width - patch + 1))
            y = self.rng.randrange(0, max(1, height - patch + 1))
        return min(max(0, x), max(0, width - patch)), min(max(0, y), max(0, height - patch))

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        item = self.rng.choice(self.items)
        feats = item["features"]
        teacher = item["teacher"]
        height, width, _ = feats.shape
        x, y = self._sample_xy(item["scene"].name, width, height)
        patch = self.patch_size
        f = feats[y : y + patch, x : x + patch]
        t = teacher[y : y + patch, x : x + patch]
        return {
            "features": torch.from_numpy(np.ascontiguousarray(np.transpose(f, (2, 0, 1)))),
            "teacher": torch.from_numpy(np.ascontiguousarray(np.transpose(t, (2, 0, 1)))),
        }


class CropFeatureBlendPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes: list[BlendScene],
        patch_size: int,
        context: int,
        samples: int,
        seed: int,
        teacher_sharpness: float,
        roi_probability: float,
        roi_bias_strength: float,
        stats_samples: int,
    ) -> None:
        self.patch_size = int(patch_size)
        self.context = int(context)
        self.samples = int(samples)
        self.rng = random.Random(seed)
        self.teacher_sharpness = float(teacher_sharpness)
        self.roi_probability = float(roi_probability)
        self.roi_bias_strength = float(roi_bias_strength)
        self.items = []
        for scene in scenes:
            missing = [p for p in (scene.noisy, scene.base, scene.rebuild, scene.cleanup) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{scene.name} missing: {missing}")
            self.items.append(
                {
                    "scene": scene,
                    "noisy": read_image(scene.noisy),
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

    def _make_patch(self, item: dict, x: int, y: int, roi_name: str | None) -> tuple[np.ndarray, np.ndarray]:
        patch = self.patch_size
        noisy_crop, inner = _crop_with_context(item["noisy"], x, y, patch, self.context)
        base_crop, _ = _crop_with_context(item["base"], x, y, patch, self.context)
        rebuild_crop, _ = _crop_with_context(item["rebuild"], x, y, patch, self.context)
        cleanup_crop, _ = _crop_with_context(item["cleanup"], x, y, patch, self.context)
        feats, teacher, _ = make_features_and_teacher(
            noisy_crop,
            base_crop,
            rebuild_crop,
            cleanup_crop,
            teacher_sharpness=self.teacher_sharpness,
        )
        ix0, iy0, ix1, iy1 = inner
        feats = feats[iy0:iy1, ix0:ix1]
        teacher = teacher[iy0:iy1, ix0:ix1]
        teacher = apply_roi_bias(teacher, roi_name, self.roi_bias_strength)
        return feats, teacher

    def _estimate_stats(self, item: dict, stats_samples: int) -> dict[str, float]:
        count = max(1, int(stats_samples))
        sums = np.zeros(3, dtype=np.float64)
        roi_counts: dict[str, int] = {}
        height, width = item["base"].shape[:2]
        for _ in range(count):
            x, y, roi_name = self._sample_xy(item["scene"].name, width, height)
            _, teacher = self._make_patch(item, x, y, roi_name)
            sums += np.mean(teacher, axis=(0, 1))
            if roi_name is not None:
                roi_counts[roi_name] = roi_counts.get(roi_name, 0) + 1
        means = sums / float(count)
        return {
            "teacher_base_mean": float(means[0]),
            "teacher_rebuild_mean": float(means[1]),
            "teacher_cleanup_mean": float(means[2]),
            "stats_samples": int(count),
            "roi_counts": roi_counts,
        }

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        item = self.rng.choice(self.items)
        height, width = item["base"].shape[:2]
        x, y, roi_name = self._sample_xy(item["scene"].name, width, height)
        feats, teacher = self._make_patch(item, x, y, roi_name)
        return {
            "features": torch.from_numpy(np.ascontiguousarray(np.transpose(feats, (2, 0, 1)))),
            "teacher": torch.from_numpy(np.ascontiguousarray(np.transpose(teacher, (2, 0, 1)))),
        }


def smoothness_loss(weights: torch.Tensor) -> torch.Tensor:
    return (torch.abs(weights[:, :, :, 1:] - weights[:, :, :, :-1]).mean() + torch.abs(weights[:, :, 1:, :] - weights[:, :, :-1, :]).mean())


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes_by_name = scene_map(args.candidate_set)
    scenes = [scenes_by_name[name] for name in args.scenes.split(",") if name]
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.feature_mode == "full":
        ds = BlendPatchDataset(
            scenes,
            patch_size=args.patch_size,
            samples=args.steps * args.batch_size,
            seed=args.seed,
            teacher_sharpness=args.teacher_sharpness,
            roi_probability=args.roi_probability,
        )
    elif args.feature_mode == "crop":
        ds = CropFeatureBlendPatchDataset(
            scenes,
            patch_size=args.patch_size,
            context=args.context,
            samples=args.steps * args.batch_size,
            seed=args.seed,
            teacher_sharpness=args.teacher_sharpness,
            roi_probability=args.roi_probability,
            roi_bias_strength=args.roi_bias_strength,
            stats_samples=args.stats_samples,
        )
    else:
        raise ValueError(f"unknown feature mode: {args.feature_mode}")
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = BlendSelector(width=args.width, blocks=args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log_path = out_dir / "stdout.log"
    meta = {
        "scenes": [scene.name for scene in scenes],
        "candidate_set": args.candidate_set,
        "scene_stats": {item["scene"].name: item["stats"] for item in ds.items},
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "candidate_names": CANDIDATE_NAMES,
        "feature_channels": FEATURE_CHANNELS,
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== train blend-selector steps={args.steps} device={device} ===\n")
        for step, batch in enumerate(dl, start=1):
            if step > args.steps:
                break
            features = batch["features"].to(device)
            teacher = batch["teacher"].to(device)
            pred = model(features)
            loss_weight = F.smooth_l1_loss(pred, teacher, beta=0.05)
            loss_smooth = smoothness_loss(pred) * args.smooth_weight
            loss_entropy = -(pred * torch.log(torch.clamp(pred, min=1.0e-6))).sum(1).mean() * args.entropy_weight
            loss = loss_weight + loss_smooth + loss_entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step == 1 or step % args.log_every == 0:
                elapsed = time.monotonic() - start
                msg = (
                    f"step {step:05d}/{args.steps} loss={float(loss.detach()):.6f} "
                    f"weight={float(loss_weight.detach()):.6f} smooth={float(loss_smooth.detach()):.6f} "
                    f"entropy={float(loss_entropy.detach()):.6f} "
                    f"means={[round(float(v), 3) for v in pred.detach().mean(dim=(0, 2, 3)).cpu()]} "
                    f"{elapsed / step:.3f}s/it"
                )
                print(msg)
                log.write(msg + "\n")
                log.flush()
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / f"blend_selector_step_{step:06d}.pt", model, args, step)
        save_checkpoint(out_dir / "blend_selector_final.pt", model, args, args.steps)
        log.write(f"wrote {out_dir / 'blend_selector_final.pt'}\n")
    print(f"wrote {out_dir / 'blend_selector_final.pt'}")


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, step: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "width": args.width,
            "blocks": args.blocks,
            "feature_channels": FEATURE_CHANNELS,
            "candidate_names": CANDIDATE_NAMES,
            "step": step,
            "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        },
        path,
    )


def load_model(checkpoint: Path, device: torch.device) -> BlendSelector:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = BlendSelector(width=int(ckpt["width"]), blocks=int(ckpt["blocks"]), in_channels=int(ckpt["feature_channels"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def blend_candidates(weights: np.ndarray, base: np.ndarray, rebuild: np.ndarray, cleanup: np.ndarray) -> np.ndarray:
    base_s = _display(base)
    rebuild_s = _display(rebuild)
    cleanup_s = _display(cleanup)
    out_s = (
        base_s * weights[..., 0:1]
        + rebuild_s * weights[..., 1:2]
        + cleanup_s * weights[..., 2:3]
    )
    return srgb_to_linear_np(np.clip(out_s, 0.0, 1.0)).astype(np.float32, copy=False)


def structure_lock_weights(
    weights: np.ndarray,
    noisy_linear: np.ndarray,
    base_linear: np.ndarray,
    *,
    strength: float,
    cleanup_floor: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    if strength <= 0:
        return weights.astype(np.float32, copy=False), {}, np.zeros(weights.shape[:2], dtype=np.float32)
    noisy = _display(noisy_linear)
    base = _display(base_linear)
    ref_texture = make_texture_mask(noisy, texture_threshold=0.006, texture_transition=0.012)
    base_texture = make_texture_mask(base, texture_threshold=0.018, texture_transition=0.014)
    texture_agreement = np.clip(ref_texture * base_texture, 0.0, 1.0)
    coherent = make_coherent_structure_mask(
        noisy,
        coherence_threshold=0.40,
        coherence_transition=0.18,
        energy_threshold=0.006,
        energy_transition=0.006,
    )
    structure = np.clip(np.maximum.reduce([base_texture, texture_agreement * 1.35, coherent * 0.90]), 0.0, 1.0)
    lock = np.clip(structure * float(strength), 0.0, 1.0)
    cleanup_cap = np.clip(1.0 - lock * (1.0 - float(cleanup_floor)), float(cleanup_floor), 1.0)
    out = weights.astype(np.float32, copy=True)
    excess = np.maximum(out[..., 2] - cleanup_cap, 0.0)
    out[..., 2] -= excess
    rebuild_share = np.clip(structure, 0.0, 1.0)
    out[..., 1] += excess * rebuild_share
    out[..., 0] += excess * (1.0 - rebuild_share)
    out /= np.maximum(np.sum(out, axis=2, keepdims=True), 1.0e-6)
    stats = {
        "structure_lock_strength": float(strength),
        "structure_lock_mean": float(np.mean(lock)),
        "cleanup_cap_mean": float(np.mean(cleanup_cap)),
        "cleanup_excess_mean": float(np.mean(excess)),
    }
    return out.astype(np.float32, copy=False), stats, lock.astype(np.float32, copy=False)


@torch.inference_mode()
def predict_weights_tiled(
    model: nn.Module,
    device: torch.device,
    noisy: np.ndarray,
    base: np.ndarray,
    rebuild: np.ndarray,
    cleanup: np.ndarray,
    *,
    tile: int,
    overlap: int,
    teacher_sharpness: float,
) -> np.ndarray:
    height, width = base.shape[:2]
    output = np.zeros((height, width, len(CANDIDATE_NAMES)), dtype=np.float32)
    weight_sum = np.zeros((height, width, 1), dtype=np.float32)
    stride = max(1, int(tile) - int(overlap) * 2)
    for y0 in range(0, height, stride):
        for x0 in range(0, width, stride):
            x1 = min(width, x0 + tile)
            y1 = min(height, y0 + tile)
            px0 = max(0, x0 - overlap)
            py0 = max(0, y0 - overlap)
            px1 = min(width, x1 + overlap)
            py1 = min(height, y1 + overlap)
            feats, _, _ = make_features_and_teacher(
                noisy[py0:py1, px0:px1],
                base[py0:py1, px0:px1],
                rebuild[py0:py1, px0:px1],
                cleanup[py0:py1, px0:px1],
                teacher_sharpness=teacher_sharpness,
            )
            inp = torch.from_numpy(np.transpose(feats, (2, 0, 1))[None]).to(device)
            pred = model(inp).detach().cpu().numpy()[0].transpose(1, 2, 0)
            cy0 = y0 - py0
            cx0 = x0 - px0
            cy1 = cy0 + (y1 - y0)
            cx1 = cx0 + (x1 - x0)
            output[y0:y1, x0:x1] += pred[cy0:cy1, cx0:cx1]
            weight_sum[y0:y1, x0:x1] += 1.0
    return output / np.maximum(weight_sum, 1.0e-6)


def apply(args: argparse.Namespace) -> None:
    scene = scene_map(args.candidate_set)[args.scene]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    noisy = read_image(scene.noisy)
    base = read_image(scene.base)
    rebuild = read_image(scene.rebuild)
    cleanup = read_image(scene.cleanup)
    weights = predict_weights_tiled(
        model,
        device,
        noisy,
        base,
        rebuild,
        cleanup,
        tile=args.tile,
        overlap=args.overlap,
        teacher_sharpness=args.teacher_sharpness,
    )
    lock_stats = {}
    lock_mask = None
    if args.structure_lock_strength > 0:
        weights, lock_stats, lock_mask = structure_lock_weights(
            weights,
            noisy,
            base,
            strength=args.structure_lock_strength,
            cleanup_floor=args.structure_cleanup_floor,
        )
    out = blend_candidates(weights, base, rebuild, cleanup)
    name = args.name or f"{scene.name}_blend_selector"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    for idx, candidate in enumerate(CANDIDATE_NAMES):
        Image.fromarray(np.clip(weights[..., idx] * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(
            out_dir / f"{name}_weight_{candidate}.png"
        )
    if lock_mask is not None:
        Image.fromarray(np.clip(lock_mask * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(
            out_dir / f"{name}_structure_lock.png"
        )
    meta = {
        "scene": scene.name,
        "candidate_set": args.candidate_set,
        "checkpoint": str(Path(args.checkpoint)),
        "outputs": {"exr": str(exr_path), "preview": str(preview_path)},
        "weight_means": {candidate: float(np.mean(weights[..., idx])) for idx, candidate in enumerate(CANDIDATE_NAMES)},
        "weight_p95": {candidate: float(np.quantile(weights[..., idx], 0.95)) for idx, candidate in enumerate(CANDIDATE_NAMES)},
        "structure_lock": lock_stats,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def crop_display(path: Path, x: int, y: int, size: int, scale: int) -> Image.Image:
    arr = read_image(path)
    crop = arr[y : y + size, x : x + size]
    img = Image.fromarray(make_preview(crop, exposure=1.0, tone="reinhard"))
    return img.resize((size * scale, size * scale), Image.Resampling.NEAREST)


def compare(args: argparse.Namespace) -> None:
    scene = scene_map(args.candidate_set)[args.scene]
    result = Path(args.result)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rois = ROI_TOP_LEFT[scene.name]
    size = int(args.crop_size)
    scale = int(args.scale)
    sources: list[tuple[str, Path]] = [
        ("input", scene.noisy),
        ("base", scene.base),
        ("rebuild", scene.rebuild),
        ("cleanup", scene.cleanup),
        ("ai_selector", result),
    ]
    if scene.pl is not None and scene.pl.exists():
        sources.append(("pl_ref", scene.pl))
    label_h = 26
    for roi_name, x, y in rois:
        crops = [crop_display(path, x, y, size, scale) for _, path in sources]
        w = size * scale
        h = size * scale
        canvas = Image.new("RGB", (w * len(crops), h + label_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for idx, ((label, _), img) in enumerate(zip(sources, crops, strict=True)):
            canvas.paste(img, (idx * w, label_h))
            draw.text((idx * w + 6, 6), label, fill=(235, 235, 235))
        out = out_dir / f"{scene.name}_blend_selector_compare_{roi_name}_{scale}x.png"
        canvas.save(out)
        print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply a learned candidate blend selector.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--output-dir", default=str(RUN_ROOT / "blend_selector_pilot"))
    p_train.add_argument("--candidate-set", default="legacy", choices=["legacy", "current"])
    p_train.add_argument("--scenes", default="xt5_occi,k5_dance")
    p_train.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p_train.add_argument("--steps", type=int, default=800)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--patch-size", type=int, default=192)
    p_train.add_argument("--feature-mode", default="crop", choices=["crop", "full"])
    p_train.add_argument("--context", type=int, default=64)
    p_train.add_argument("--width", type=int, default=24)
    p_train.add_argument("--blocks", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=2.0e-4)
    p_train.add_argument("--weight-decay", type=float, default=1.0e-4)
    p_train.add_argument("--smooth-weight", type=float, default=0.018)
    p_train.add_argument("--entropy-weight", type=float, default=0.004)
    p_train.add_argument("--teacher-sharpness", type=float, default=1.20)
    p_train.add_argument("--roi-probability", type=float, default=0.82)
    p_train.add_argument("--roi-bias-strength", type=float, default=0.65)
    p_train.add_argument("--stats-samples", type=int, default=24)
    p_train.add_argument("--seed", type=int, default=4317)
    p_train.add_argument("--log-every", type=int, default=50)
    p_train.add_argument("--save-every", type=int, default=0)
    p_train.set_defaults(func=train)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--candidate-set", default="legacy", choices=["legacy", "current"])
    p_apply.add_argument("--scene", required=True, choices=sorted(scene_map("legacy")))
    p_apply.add_argument("--output-dir", default=str(RUN_ROOT / "blend_selector_pilot_outputs"))
    p_apply.add_argument("--name", default=None)
    p_apply.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])
    p_apply.add_argument("--tile", type=int, default=768)
    p_apply.add_argument("--overlap", type=int, default=48)
    p_apply.add_argument("--teacher-sharpness", type=float, default=1.20)
    p_apply.add_argument("--structure-lock-strength", type=float, default=0.0)
    p_apply.add_argument("--structure-cleanup-floor", type=float, default=0.20)
    p_apply.set_defaults(func=apply)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--candidate-set", default="legacy", choices=["legacy", "current"])
    p_compare.add_argument("--scene", required=True, choices=sorted(scene_map("legacy")))
    p_compare.add_argument("--result", required=True)
    p_compare.add_argument("--output-dir", default=str(RUN_ROOT / "blend_selector_pilot_outputs"))
    p_compare.add_argument("--crop-size", type=int, default=512)
    p_compare.add_argument("--scale", type=int, default=2)
    p_compare.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
