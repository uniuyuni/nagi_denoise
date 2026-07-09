"""Train/apply a tiny selector for SCUNet reconstruction candidates.

The model predicts a single display-space blend gate:

    output = current * (1 - gate) + scunet_candidate * gate

SCUNet is treated as a reconstruction candidate, not a target image. The
pseudo-label is built from local evidence: luma/chroma cleanup benefit,
coherent structure, color agreement, highlight safety, and chroma outlier risk.
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
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, median_filter, uniform_filter
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_flat_chroma_smoother import LUMA_LINEAR, LUMA_SRGB, linear_to_srgb_np, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from apply_region_aware_luma_cleanup import make_coherent_structure_mask, make_skin_mask, make_texture_mask
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import image_stats, make_preview, read_image


ROOT = Path(__file__).resolve().parents[1]
TEST_PHOTOS = Path("/Users/uniuyuni/ProjectData/test_photos")
RUN_ROOT = ROOT / "runs/refiner_pilot_stage11_hybrid_best"

FEATURE_CHANNELS = 26


@dataclass(frozen=True)
class ScunetScene:
    name: str
    noisy: Path
    current: Path
    scunet: Path
    rois: tuple[tuple[str, int, int], ...]


SCENES: dict[str, ScunetScene] = {
    "xt5_occi": ScunetScene(
        "xt5_occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        RUN_ROOT / "selective_pl_hair_detail_v4_tight_prox_occi/xt5_occi_selective_pl_hair_detail_v4_tight_prox.exr",
        RUN_ROOT / "scunet_global_hdr_chroma_cleanup_v1_occi/xt5_occi_scunet_global_hdr_chroma_cleanup_v1.exr",
        (
            ("face_hair", 1780, 1140),
            ("bangs", 2170, 850),
            ("top_hair", 2350, 520),
            ("body_shadow", 2900, 3720),
            ("root", 512, 5632),
        ),
    ),
    "k5_dance": ScunetScene(
        "k5_dance",
        TEST_PHOTOS / "K-5 Dance noisy.EXR",
        RUN_ROOT / "signed_chroma_outlier_v25_hdr_restore/k5_dance_v25_hdr_restore.exr",
        RUN_ROOT / "scunet_global_hdr_chroma_cleanup_v1_k5_dance/k5_dance_scunet_global_hdr_chroma_cleanup_v1.exr",
        (
            ("sky_center", 2300, 320),
            ("dancer_center", 2800, 1200),
            ("snow_ground", 2100, 2500),
            ("house_detail", 260, 1180),
        ),
    ),
    "k5_ice": ScunetScene(
        "k5_ice",
        TEST_PHOTOS / "K-5 Ice noisy.EXR",
        RUN_ROOT / "signed_chroma_outlier_v25_hdr_restore/k5_ice_v25_hdr_restore.exr",
        RUN_ROOT / "scunet_global_hdr_chroma_cleanup_v2_k5_ice/k5_ice_scunet_global_hdr_chroma_cleanup_v2.exr",
        (
            ("ice_center", 2100, 1180),
            ("blue_shadow", 2700, 900),
            ("edge_detail", 1700, 1450),
        ),
    ),
}


ROI_GATE_BIAS: dict[str, tuple[float, float]] = {
    "face_hair": (1.25, 0.78),
    "bangs": (1.28, 0.76),
    "top_hair": (1.15, 0.88),
    "body_shadow": (0.80, 1.20),
    "root": (1.20, 0.82),
    "sky_center": (1.28, 0.76),
    "dancer_center": (1.06, 0.95),
    "snow_ground": (1.12, 0.90),
    "house_detail": (1.04, 0.98),
    "ice_center": (0.90, 1.12),
    "blue_shadow": (0.55, 1.55),
    "edge_detail": (0.82, 1.25),
}


ROI_GATE_BIAS_V2: dict[str, tuple[float, float]] = {
    **ROI_GATE_BIAS,
    "face_hair": (1.55, 0.68),
    "bangs": (1.62, 0.66),
    "top_hair": (1.36, 0.78),
    "root": (1.42, 0.72),
    "sky_center": (1.48, 0.68),
    "snow_ground": (1.25, 0.82),
    "ice_center": (0.78, 1.28),
    "blue_shadow": (0.38, 1.85),
    "edge_detail": (0.72, 1.42),
}


ROI_GATE_BONUS_V2: dict[str, float] = {
    "face_hair": 0.06,
    "bangs": 0.08,
    "top_hair": 0.05,
    "root": 0.06,
    "sky_center": 0.14,
    "snow_ground": 0.05,
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


def smoothstep01(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def saturation(rgb: np.ndarray) -> np.ndarray:
    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    return ((mx - mn) / np.maximum(mx, 1.0e-6)).astype(np.float32, copy=False)


def hf_abs(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.abs(x - gaussian_filter(x, sigma=float(sigma), mode="reflect")).astype(np.float32, copy=False)


def chroma_hf(rgb: np.ndarray, sigma: float) -> np.ndarray:
    y = luma(rgb, LUMA_SRGB)
    chroma = rgb - y[..., None]
    low = gaussian_filter(chroma, sigma=(float(sigma), float(sigma), 0.0), mode="reflect")
    return np.mean(np.abs(chroma - low), axis=2).astype(np.float32, copy=False)


def chroma_outlier_risk(rgb: np.ndarray) -> np.ndarray:
    y = luma(rgb, LUMA_SRGB)
    chroma = rgb - y[..., None]
    med = median_filter(chroma, size=(7, 7, 1), mode="reflect")
    out = np.mean(np.abs(chroma - med), axis=2)
    sat = saturation(rgb)
    dark = sigmoid01((0.42 - y) / 0.12)
    return np.clip(sigmoid01((out - 0.010) / 0.006) * (0.35 + 0.65 * sat) * dark, 0.0, 1.0)


def signed_blue_magenta_risk(rgb: np.ndarray) -> np.ndarray:
    y = luma(rgb, LUMA_SRGB)
    sat = saturation(rgb)
    blue = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])
    magenta = 0.5 * (rgb[..., 0] + rgb[..., 2]) - rgb[..., 1]
    blue_out = np.abs(blue - median_filter(blue, size=7, mode="reflect"))
    magenta_out = np.abs(magenta - median_filter(magenta, size=7, mode="reflect"))
    signed_out = np.maximum(blue_out * 1.15, magenta_out)
    dark = sigmoid01((0.50 - y) / 0.14)
    colored_dot = sigmoid01((sat - 0.16) / 0.10)
    return np.clip(sigmoid01((signed_out - 0.0090) / 0.0052) * dark * (0.20 + 0.80 * colored_dot), 0.0, 1.0)


def crop_with_context(
    arr: np.ndarray,
    x: int,
    y: int,
    patch: int,
    context: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = arr.shape[:2]
    x0 = max(0, int(x) - int(context))
    y0 = max(0, int(y) - int(context))
    x1 = min(w, int(x) + int(patch) + int(context))
    y1 = min(h, int(y) + int(patch) + int(context))
    return arr[y0:y1, x0:x1], (int(x) - x0, int(y) - y0, int(x) - x0 + int(patch), int(y) - y0 + int(patch))


def apply_roi_bias(target: np.ndarray, roi_name: str | None, strength: float) -> np.ndarray:
    if roi_name is None or roi_name not in ROI_GATE_BIAS or strength <= 0:
        return target.astype(np.float32, copy=False)
    mul, suppress = ROI_GATE_BIAS[roi_name]
    out = target * (1.0 + (float(mul) - 1.0) * float(strength))
    if suppress > 1.0:
        out *= 1.0 - min(0.55, (float(suppress) - 1.0) * 0.28 * float(strength))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def apply_roi_bias_v2(target: np.ndarray, roi_name: str | None, strength: float) -> np.ndarray:
    if roi_name is None or roi_name not in ROI_GATE_BIAS_V2 or strength <= 0:
        return target.astype(np.float32, copy=False)
    mul, suppress = ROI_GATE_BIAS_V2[roi_name]
    out = target * (1.0 + (float(mul) - 1.0) * float(strength))
    if suppress > 1.0:
        out *= 1.0 - min(0.72, (float(suppress) - 1.0) * 0.34 * float(strength))
    if roi_name in ROI_GATE_BONUS_V2:
        bonus = float(ROI_GATE_BONUS_V2[roi_name]) * float(strength)
        out += bonus * (1.0 - out)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def make_features_and_target(
    noisy_linear: np.ndarray,
    current_linear: np.ndarray,
    scunet_linear: np.ndarray,
    *,
    roi_name: str | None = None,
    roi_bias_strength: float = 0.0,
    target_preset: str = "v1",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    noisy = display(noisy_linear)
    current = display(current_linear)
    scunet = display(scunet_linear)
    if noisy.shape != current.shape or current.shape != scunet.shape:
        raise ValueError(f"shape mismatch noisy={noisy.shape} current={current.shape} scunet={scunet.shape}")

    noisy_y = luma(noisy, LUMA_SRGB)
    current_y = luma(current, LUMA_SRGB)
    scunet_y = luma(scunet, LUMA_SRGB)
    current_luma_hf = hf_abs(current_y, 0.8)
    scunet_luma_hf = hf_abs(scunet_y, 0.8)
    noisy_luma_hf = hf_abs(noisy_y, 0.8)
    current_chroma_hf = chroma_hf(current, 1.0)
    scunet_chroma_hf = chroma_hf(scunet, 1.0)
    current_texture = make_texture_mask(current, texture_threshold=0.014, texture_transition=0.014)
    scunet_texture = make_texture_mask(scunet, texture_threshold=0.014, texture_transition=0.014)
    coherent = make_coherent_structure_mask(
        noisy,
        coherence_threshold=0.38,
        coherence_transition=0.18,
        energy_threshold=0.0048,
        energy_transition=0.0055,
    )
    skin = make_skin_mask(current, blur_sigma=1.4)
    sat = saturation(current)
    flat = np.clip(
        sigmoid01((0.026 - gaussian_gradient_magnitude(noisy_y, sigma=1.0, mode="reflect")) / 0.012)
        * sigmoid01((0.020 - noisy_luma_hf) / 0.012),
        0.0,
        1.0,
    )
    structure = np.clip(np.maximum.reduce([coherent, current_texture * 0.90, scunet_texture * 0.70]), 0.0, 1.0)
    luma_benefit = sigmoid01((current_luma_hf - scunet_luma_hf - 0.0015) / 0.006)
    chroma_benefit = sigmoid01((current_chroma_hf - scunet_chroma_hf - 0.0008) / 0.0045)
    scunet_structure_gain = sigmoid01((scunet_texture - current_texture + 0.06) / 0.16)
    low_current_detail = sigmoid01((0.025 - current_luma_hf) / 0.014)
    color_delta = np.mean(
        np.abs(
            gaussian_filter(scunet - current, sigma=(5.0, 5.0, 0.0), mode="reflect")
        ),
        axis=2,
    )
    color_agree = sigmoid01((0.070 - color_delta) / 0.035)
    risk = chroma_outlier_risk(scunet)
    signed_risk = signed_blue_magenta_risk(scunet)
    hdr_peak = np.max(np.clip(noisy_linear, 0.0, None), axis=2)
    hdr_risk = smoothstep01((hdr_peak - 0.88) / 0.28)

    if target_preset == "v1":
        target = (
            0.14
            + 0.34 * luma_benefit * flat
            + 0.30 * chroma_benefit * flat
            + 0.30 * structure * scunet_structure_gain
            + 0.18 * coherent * low_current_detail
            - 0.42 * risk
            - 0.20 * skin * (1.0 - structure)
            - 0.26 * hdr_risk * (1.0 - structure)
        )
        target = np.clip(target * (0.35 + 0.65 * color_agree), 0.0, 1.0)
        target = gaussian_filter(target.astype(np.float32, copy=False), sigma=0.9, mode="reflect")
        target = apply_roi_bias(target, roi_name, roi_bias_strength)
    elif target_preset in {"v2", "v3_luma", "v4_trust"}:
        scunet_has_detail = sigmoid01((scunet_luma_hf - 0.012) / 0.012)
        current_is_sleepy = sigmoid01((scunet_luma_hf - current_luma_hf + 0.004) / 0.012)
        coherent_rebuild = np.clip(
            np.maximum(coherent * 0.85, scunet_texture * 0.75) * scunet_has_detail * current_is_sleepy,
            0.0,
            1.0,
        )
        flat_luma_safe = np.clip(flat * luma_benefit * color_agree, 0.0, 1.0)
        sky_cleanup = np.clip(flat_luma_safe * (1.0 - 0.35 * risk) * (1.0 - 0.25 * signed_risk), 0.0, 1.0)
        risk_penalty = risk * (1.0 - 0.45 * flat_luma_safe)
        signed_penalty = signed_risk * (1.0 - 0.70 * flat_luma_safe)
        if target_preset == "v2":
            target = (
                0.12
                + 0.40 * luma_benefit * flat
                + 0.24 * chroma_benefit * flat
                + 0.46 * structure * scunet_structure_gain
                + 0.32 * coherent_rebuild
                + 0.26 * sky_cleanup
                - 0.40 * risk_penalty
                - 0.38 * signed_penalty
                - 0.16 * skin * (1.0 - structure)
                - 0.34 * hdr_risk * (1.0 - np.maximum(structure, coherent_rebuild))
            )
            target = np.clip(target * (0.42 + 0.58 * color_agree), 0.0, 1.0)
            target = gaussian_filter(target.astype(np.float32, copy=False), sigma=0.72, mode="reflect")
            target = apply_roi_bias_v2(target, roi_name, roi_bias_strength)
        elif target_preset == "v3_luma":
            scunet_luma_noise_risk = sigmoid01((scunet_luma_hf - current_luma_hf - 0.004) / 0.010)
            flat_noise_risk = np.clip(scunet_luma_noise_risk * flat * (1.0 - coherent_rebuild), 0.0, 1.0)
            target = (
                0.10
                + 0.58 * luma_benefit * flat
                + 0.54 * coherent_rebuild
                + 0.42 * structure * scunet_structure_gain
                + 0.28 * sky_cleanup
                - 0.42 * flat_noise_risk
                - 0.26 * signed_penalty
                - 0.12 * skin * (1.0 - np.maximum(structure, coherent_rebuild))
                - 0.30 * hdr_risk * (1.0 - np.maximum(structure, coherent_rebuild))
            )
            target = np.clip(target * (0.50 + 0.50 * color_agree), 0.0, 1.0)
            target = gaussian_filter(target.astype(np.float32, copy=False), sigma=0.68, mode="reflect")
            target = apply_roi_bias_v2(target, roi_name, roi_bias_strength)
        else:
            scunet_luma_noise_risk = sigmoid01((scunet_luma_hf - current_luma_hf - 0.003) / 0.008)
            flat_noise_risk = np.clip(scunet_luma_noise_risk * flat * (1.0 - 0.75 * coherent_rebuild), 0.0, 1.0)
            shadow = sigmoid01((0.50 - noisy_y) / 0.16)
            blue_shadow_tail = np.clip(signed_risk * shadow * flat_noise_risk * (1.0 - 0.55 * flat_luma_safe), 0.0, 1.0)
            blue_shadow_risk = np.clip(signed_risk * shadow * (1.0 - 0.35 * flat_luma_safe), 0.0, 1.0)
            trustworthy_rebuild = np.clip(
                coherent_rebuild
                * (0.45 + 0.55 * color_agree)
                * (1.0 - 0.38 * blue_shadow_risk)
                * (1.0 - 0.45 * blue_shadow_tail),
                0.0,
                1.0,
            )
            target = (
                0.10
                + 0.60 * luma_benefit * flat
                + 0.56 * trustworthy_rebuild
                + 0.42 * structure * scunet_structure_gain
                + 0.32 * sky_cleanup
                - 0.50 * flat_noise_risk
                - 0.72 * blue_shadow_tail
                - 0.34 * blue_shadow_risk
                - 0.20 * signed_penalty * shadow * (1.0 - flat_luma_safe)
                - 0.12 * skin * (1.0 - np.maximum(structure, trustworthy_rebuild))
                - 0.30 * hdr_risk * (1.0 - np.maximum(structure, trustworthy_rebuild))
            )
            target = np.clip(target * (0.50 + 0.50 * color_agree), 0.0, 1.0)
            target = gaussian_filter(target.astype(np.float32, copy=False), sigma=0.68, mode="reflect")
            target = apply_roi_bias_v2(target, roi_name, roi_bias_strength)
    else:
        raise ValueError(f"unknown target preset: {target_preset}")

    y_delta = scunet_y - current_y
    chroma_delta = np.mean(np.abs((scunet - scunet_y[..., None]) - (current - current_y[..., None])), axis=2)
    features = np.concatenate(
        [
            noisy,
            current,
            scunet,
            noisy_y[..., None],
            current_y[..., None],
            scunet_y[..., None],
            y_delta[..., None],
            chroma_delta[..., None],
            sat[..., None],
            current_luma_hf[..., None],
            scunet_luma_hf[..., None],
            current_chroma_hf[..., None],
            scunet_chroma_hf[..., None],
            current_texture[..., None],
            scunet_texture[..., None],
            coherent[..., None],
            flat[..., None],
            skin[..., None],
            risk[..., None],
            hdr_risk[..., None],
        ],
        axis=2,
    ).astype(np.float32, copy=False)
    if features.shape[2] != FEATURE_CHANNELS:
        raise AssertionError(f"feature channel mismatch: {features.shape[2]} != {FEATURE_CHANNELS}")
    stats = {
        "target_mean": float(np.mean(target)),
        "target_p95": float(np.quantile(target, 0.95)),
        "luma_benefit_mean": float(np.mean(luma_benefit)),
        "chroma_benefit_mean": float(np.mean(chroma_benefit)),
        "risk_mean": float(np.mean(risk)),
        "signed_risk_mean": float(np.mean(signed_risk)),
        "hdr_risk_mean": float(np.mean(hdr_risk)),
        "structure_mean": float(np.mean(structure)),
    }
    return features, target.astype(np.float32, copy=False), stats


class SelectorBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(F.gelu(self.pw1(self.dw(x)))) * self.scale


class ScunetSelector(nn.Module):
    def __init__(self, width: int = 18, blocks: int = 3) -> None:
        super().__init__()
        self.head = nn.Conv2d(FEATURE_CHANNELS, width, 3, padding=1)
        self.body = nn.Sequential(*[SelectorBlock(width) for _ in range(blocks)])
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


class ScunetSelectorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes: list[ScunetScene],
        *,
        patch_size: int,
        context: int,
        samples: int,
        roi_probability: float,
        roi_bias_strength: float,
        target_preset: str,
        seed: int,
        stats_samples: int,
    ) -> None:
        self.patch_size = int(patch_size)
        self.context = int(context)
        self.samples = int(samples)
        self.roi_probability = float(roi_probability)
        self.roi_bias_strength = float(roi_bias_strength)
        self.target_preset = str(target_preset)
        self.rng = random.Random(seed)
        self.items: list[dict] = []
        for scene in scenes:
            missing = [p for p in (scene.noisy, scene.current, scene.scunet) if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{scene.name} missing files: {missing}")
            self.items.append(
                {
                    "scene": scene,
                    "noisy": read_image(scene.noisy),
                    "current": read_image(scene.current),
                    "scunet": read_image(scene.scunet),
                    "stats": {},
                }
            )
        for item in self.items:
            item["stats"] = self._estimate_stats(item, stats_samples)

    def __len__(self) -> int:
        return self.samples

    def _sample_xy(self, scene: ScunetScene, width: int, height: int) -> tuple[int, int, str | None]:
        patch = self.patch_size
        roi_name: str | None = None
        if self.rng.random() < self.roi_probability and scene.rois:
            roi_name, rx, ry = self.rng.choice(scene.rois)
            jitter = max(16, patch // 2)
            x = rx + self.rng.randrange(-jitter, jitter + 1)
            y = ry + self.rng.randrange(-jitter, jitter + 1)
        else:
            x = self.rng.randrange(0, max(1, width - patch + 1))
            y = self.rng.randrange(0, max(1, height - patch + 1))
        return min(max(0, x), max(0, width - patch)), min(max(0, y), max(0, height - patch)), roi_name

    def _make_patch(self, item: dict, x: int, y: int, roi_name: str | None) -> tuple[np.ndarray, np.ndarray, dict]:
        patch = self.patch_size
        noisy_crop, inner = crop_with_context(item["noisy"], x, y, patch, self.context)
        current_crop, _ = crop_with_context(item["current"], x, y, patch, self.context)
        scunet_crop, _ = crop_with_context(item["scunet"], x, y, patch, self.context)
        features, target, stats = make_features_and_target(
            noisy_crop,
            current_crop,
            scunet_crop,
            roi_name=roi_name,
            roi_bias_strength=self.roi_bias_strength,
            target_preset=self.target_preset,
        )
        ix0, iy0, ix1, iy1 = inner
        return features[iy0:iy1, ix0:ix1], target[iy0:iy1, ix0:ix1], stats

    def _estimate_stats(self, item: dict, stats_samples: int) -> dict[str, float]:
        sums: dict[str, float] = {}
        count = max(1, int(stats_samples))
        h, w = item["current"].shape[:2]
        for _ in range(count):
            x, y, roi_name = self._sample_xy(item["scene"], w, h)
            _, _, stats = self._make_patch(item, x, y, roi_name)
            for key, value in stats.items():
                sums[key] = sums.get(key, 0.0) + float(value)
        return {key: value / float(count) for key, value in sums.items()} | {"stats_samples": int(count)}

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        item = self.rng.choice(self.items)
        h, w = item["current"].shape[:2]
        x, y, roi_name = self._sample_xy(item["scene"], w, h)
        features, target, _ = self._make_patch(item, x, y, roi_name)
        return {
            "features": torch.from_numpy(np.ascontiguousarray(np.transpose(features, (2, 0, 1)))),
            "target": torch.from_numpy(np.ascontiguousarray(target[None])),
        }


def smoothness_loss(gate: torch.Tensor) -> torch.Tensor:
    return (
        torch.abs(gate[:, :, :, 1:] - gate[:, :, :, :-1]).mean()
        + torch.abs(gate[:, :, 1:, :] - gate[:, :, :-1, :]).mean()
    )


def save_checkpoint(path: Path, model: ScunetSelector, args: argparse.Namespace, step: int) -> None:
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


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = [SCENES[name] for name in args.scenes.split(",") if name]
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ds = ScunetSelectorDataset(
        scenes,
        patch_size=args.patch_size,
        context=args.context,
        samples=args.steps * args.batch_size,
        roi_probability=args.roi_probability,
        roi_bias_strength=args.roi_bias_strength,
        target_preset=args.target_preset,
        seed=args.seed,
        stats_samples=args.stats_samples,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = ScunetSelector(width=args.width, blocks=args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    meta = {
        "scenes": [scene.name for scene in scenes],
        "scene_stats": {item["scene"].name: item["stats"] for item in ds.items},
        "feature_channels": FEATURE_CHANNELS,
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_path = out_dir / "stdout.log"
    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== train scunet-selector steps={args.steps} device={device} ===\n")
        for step, batch in enumerate(dl, start=1):
            if step > args.steps:
                break
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            pred = model(features)
            loss_fit = F.smooth_l1_loss(pred, target, beta=0.04)
            loss_smooth = smoothness_loss(pred) * float(args.smooth_weight)
            loss_mean = torch.abs(pred.mean() - target.mean()) * float(args.mean_weight)
            loss = loss_fit + loss_smooth + loss_mean
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0:
                elapsed = time.monotonic() - start
                msg = (
                    f"step {step:05d}/{args.steps} loss={float(loss.detach()):.6f} "
                    f"fit={float(loss_fit.detach()):.6f} smooth={float(loss_smooth.detach()):.6f} "
                    f"mean={float(loss_mean.detach()):.6f} pred_mean={float(pred.detach().mean()):.4f} "
                    f"target_mean={float(target.detach().mean()):.4f} {elapsed / step:.3f}s/it"
                )
                print(msg)
                log.write(msg + "\n")
                log.flush()
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / f"scunet_selector_step_{step:06d}.pt", model, args, step)
        save_checkpoint(out_dir / "scunet_selector_final.pt", model, args, args.steps)
        log.write(f"wrote {out_dir / 'scunet_selector_final.pt'}\n")
    print(f"wrote {out_dir / 'scunet_selector_final.pt'}")


def load_model(path: Path, device: torch.device) -> ScunetSelector:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ScunetSelector(width=int(ckpt["width"]), blocks=int(ckpt["blocks"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_gate_tiled(
    model: ScunetSelector,
    device: torch.device,
    noisy: np.ndarray,
    current: np.ndarray,
    scunet: np.ndarray,
    *,
    tile: int,
    overlap: int,
) -> np.ndarray:
    h, w = current.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    stride = max(1, int(tile) - int(overlap) * 2)
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(w, x0 + int(tile))
            y1 = min(h, y0 + int(tile))
            px0 = max(0, x0 - int(overlap))
            py0 = max(0, y0 - int(overlap))
            px1 = min(w, x1 + int(overlap))
            py1 = min(h, y1 + int(overlap))
            features, _, _ = make_features_and_target(
                noisy[py0:py1, px0:px1],
                current[py0:py1, px0:px1],
                scunet[py0:py1, px0:px1],
                target_preset="v1",
            )
            inp = torch.from_numpy(np.transpose(features, (2, 0, 1))[None]).to(device)
            pred = model(inp).detach().cpu().numpy()[0, 0]
            cy0 = y0 - py0
            cx0 = x0 - px0
            cy1 = cy0 + (y1 - y0)
            cx1 = cx0 + (x1 - x0)
            out[y0:y1, x0:x1] += pred[cy0:cy1, cx0:cx1]
            count[y0:y1, x0:x1] += 1.0
    return out / np.maximum(count, 1.0e-6)


def blend_display_with_hdr_restore(
    current_linear: np.ndarray,
    scunet_linear: np.ndarray,
    gate: np.ndarray,
    *,
    strength: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
) -> np.ndarray:
    current = np.clip(_safe_rgb(current_linear), 0.0, None)
    scunet = np.clip(_safe_rgb(scunet_linear), 0.0, None)
    current_d = display(current)
    scunet_d = display(scunet)
    blend = np.clip(gate * float(strength), 0.0, 1.0).astype(np.float32, copy=False)
    out_display = np.clip(current_d * (1.0 - blend[..., None]) + scunet_d * blend[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    peak = np.max(current, axis=2)
    hdr = smoothstep01((peak - float(hdr_peak_threshold)) / max(float(hdr_transition), 1.0e-6))
    return (out * (1.0 - hdr[..., None]) + current * hdr[..., None]).astype(np.float32, copy=False)


def display_chroma_ratio(rgb: np.ndarray) -> np.ndarray:
    y = luma(rgb, LUMA_SRGB)
    return (rgb / np.maximum(y[..., None], 1.0e-6)).astype(np.float32, copy=False)


def blend_luma_with_hdr_restore(
    current_linear: np.ndarray,
    scunet_linear: np.ndarray,
    gate: np.ndarray,
    *,
    strength: float,
    gate_gamma: float,
    chroma_source_mix: float,
    chroma_limit: float,
    risk_inhibit: float,
    tail_inhibit: float,
    edge_inhibit: float,
    edge_inhibit_mode: str,
    edge_threshold: float,
    edge_transition: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
) -> np.ndarray:
    current = np.clip(_safe_rgb(current_linear), 0.0, None)
    scunet = np.clip(_safe_rgb(scunet_linear), 0.0, None)
    current_d = display(current)
    scunet_d = display(scunet)
    current_y = luma(current_d, LUMA_SRGB)
    scunet_y = luma(scunet_d, LUMA_SRGB)
    gamma = max(float(gate_gamma), 1.0e-6)
    blend = np.clip(np.power(np.clip(gate, 0.0, 1.0), gamma) * float(strength), 0.0, 1.0).astype(
        np.float32, copy=False
    )
    if risk_inhibit > 0:
        risk = signed_blue_magenta_risk(scunet_d)
        blend *= 1.0 - np.clip(float(risk_inhibit), 0.0, 1.0) * risk
    if tail_inhibit > 0:
        current_hf = hf_abs(current_y, 0.8)
        scunet_hf = hf_abs(scunet_y, 0.8)
        scunet_luma_tail = sigmoid01((scunet_hf - current_hf - 0.0025) / 0.0075)
        shadow = sigmoid01((0.50 - current_y) / 0.16)
        blue_shadow_tail = np.clip(signed_blue_magenta_risk(scunet_d) * shadow * scunet_luma_tail, 0.0, 1.0)
        blend *= 1.0 - np.clip(float(tail_inhibit), 0.0, 1.0) * blue_shadow_tail
    if edge_inhibit > 0:
        edge = gaussian_gradient_magnitude(current_y, sigma=1.0, mode="reflect")
        edge_gate = sigmoid01((edge - float(edge_threshold)) / max(float(edge_transition), 1.0e-6))
        if edge_inhibit_mode == "detail_loss":
            current_hf = hf_abs(current_y, 0.8)
            scunet_hf = hf_abs(scunet_y, 0.8)
            detail_loss = sigmoid01((current_hf - scunet_hf - 0.0015) / 0.006)
            edge_gate *= detail_loss
        elif edge_inhibit_mode != "current":
            raise ValueError(f"unknown edge inhibit mode: {edge_inhibit_mode!r}")
        blend *= 1.0 - np.clip(float(edge_inhibit), 0.0, 1.0) * edge_gate
    out_y = np.clip(current_y * (1.0 - blend) + scunet_y * blend, 0.0, 1.0)
    current_chroma = display_chroma_ratio(current_d)
    scunet_chroma = display_chroma_ratio(scunet_d)
    chroma_mix = np.clip(float(chroma_source_mix), 0.0, 1.0)
    chroma = current_chroma * (1.0 - chroma_mix) + scunet_chroma * chroma_mix
    chroma = np.clip(chroma, 0.0, float(chroma_limit))
    out_display = np.clip(chroma * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    peak = np.max(current, axis=2)
    hdr = smoothstep01((peak - float(hdr_peak_threshold)) / max(float(hdr_transition), 1.0e-6))
    return (out * (1.0 - hdr[..., None]) + current * hdr[..., None]).astype(np.float32, copy=False)


def apply(args: argparse.Namespace) -> None:
    scene = SCENES[args.scene]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    noisy = read_image(scene.noisy)
    current = read_image(scene.current)
    scunet = read_image(scene.scunet)
    gate = predict_gate_tiled(model, device, noisy, current, scunet, tile=args.tile, overlap=args.overlap)
    if args.blend_mode == "rgb":
        out = blend_display_with_hdr_restore(
            current,
            scunet,
            gate,
            strength=args.strength,
            hdr_peak_threshold=args.hdr_peak_threshold,
            hdr_transition=args.hdr_transition,
        )
    elif args.blend_mode == "luma":
        out = blend_luma_with_hdr_restore(
            current,
            scunet,
            gate,
            strength=args.strength,
            gate_gamma=args.gate_gamma,
            chroma_source_mix=args.chroma_source_mix,
            chroma_limit=args.chroma_limit,
            risk_inhibit=args.risk_inhibit,
            tail_inhibit=args.tail_inhibit,
            edge_inhibit=args.edge_inhibit,
            edge_inhibit_mode=args.edge_inhibit_mode,
            edge_threshold=args.edge_threshold,
            edge_transition=args.edge_transition,
            hdr_peak_threshold=args.hdr_peak_threshold,
            hdr_transition=args.hdr_transition,
        )
    else:
        raise ValueError(f"unknown blend mode: {args.blend_mode!r}")
    name = args.name or f"{scene.name}_scunet_selector"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    gate_path = out_dir / f"{name}_gate.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    Image.fromarray(np.clip(gate * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(gate_path)
    meta = {
        "scene": scene.name,
        "checkpoint": str(Path(args.checkpoint)),
        "inputs": {"noisy": str(scene.noisy), "current": str(scene.current), "scunet": str(scene.scunet)},
        "outputs": {"exr": str(exr_path), "preview": str(preview_path), "gate": str(gate_path)},
        "gate": {
            "mean": float(np.mean(gate)),
            "p50": float(np.quantile(gate, 0.50)),
            "p90": float(np.quantile(gate, 0.90)),
            "p99": float(np.quantile(gate, 0.99)),
        },
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def crop_preview(path: Path, x: int, y: int, size: int, scale: int) -> Image.Image:
    arr = read_image(path)
    h, w = arr.shape[:2]
    x0 = min(max(0, int(x)), max(0, w - int(size)))
    y0 = min(max(0, int(y)), max(0, h - int(size)))
    img = Image.fromarray(make_preview(arr[y0 : y0 + int(size), x0 : x0 + int(size)], exposure=1.0, tone="reinhard"))
    return img.resize((int(size) * int(scale), int(size) * int(scale)), Image.Resampling.NEAREST)


def compare(args: argparse.Namespace) -> None:
    scene = SCENES[args.scene]
    result = Path(args.result)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = int(args.crop_size)
    scale = int(args.scale)
    sources = [("noisy", scene.noisy), ("current", scene.current), ("scunet", scene.scunet), ("selector", result)]
    label_h = 26
    for roi_name, x, y in scene.rois:
        crops = [crop_preview(path, x, y, size, scale) for _, path in sources]
        canvas = Image.new("RGB", (size * scale * len(crops), size * scale + label_h), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        for idx, ((label, _), img) in enumerate(zip(sources, crops, strict=True)):
            canvas.paste(img, (idx * size * scale, label_h))
            draw.text((idx * size * scale + 6, 6), label, fill=(235, 235, 235))
        path = out_dir / f"{scene.name}_scunet_selector_compare_{roi_name}_{scale}x.png"
        canvas.save(path)
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply a learned SCUNet reconstruction selector.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--output-dir", default=str(RUN_ROOT / "scunet_selector_pilot_v1"))
    p_train.add_argument("--scenes", default="xt5_occi,k5_dance,k5_ice")
    p_train.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])
    p_train.add_argument("--steps", type=int, default=600)
    p_train.add_argument("--batch-size", type=int, default=3)
    p_train.add_argument("--patch-size", type=int, default=192)
    p_train.add_argument("--context", type=int, default=64)
    p_train.add_argument("--width", type=int, default=18)
    p_train.add_argument("--blocks", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=2.0e-4)
    p_train.add_argument("--weight-decay", type=float, default=1.0e-4)
    p_train.add_argument("--smooth-weight", type=float, default=0.018)
    p_train.add_argument("--mean-weight", type=float, default=0.12)
    p_train.add_argument("--roi-probability", type=float, default=0.86)
    p_train.add_argument("--roi-bias-strength", type=float, default=0.70)
    p_train.add_argument("--target-preset", default="v1", choices=["v1", "v2", "v3_luma", "v4_trust"])
    p_train.add_argument("--stats-samples", type=int, default=16)
    p_train.add_argument("--seed", type=int, default=7361)
    p_train.add_argument("--log-every", type=int, default=50)
    p_train.add_argument("--save-every", type=int, default=0)
    p_train.set_defaults(func=train)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--scene", required=True, choices=sorted(SCENES))
    p_apply.add_argument("--output-dir", default=str(RUN_ROOT / "scunet_selector_pilot_v1_outputs"))
    p_apply.add_argument("--name", default=None)
    p_apply.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])
    p_apply.add_argument("--tile", type=int, default=768)
    p_apply.add_argument("--overlap", type=int, default=64)
    p_apply.add_argument("--blend-mode", choices=["rgb", "luma"], default="rgb")
    p_apply.add_argument("--strength", type=float, default=1.0)
    p_apply.add_argument("--gate-gamma", type=float, default=1.0)
    p_apply.add_argument("--chroma-source-mix", type=float, default=0.0)
    p_apply.add_argument("--chroma-limit", type=float, default=8.0)
    p_apply.add_argument("--risk-inhibit", type=float, default=0.0)
    p_apply.add_argument("--tail-inhibit", type=float, default=0.0)
    p_apply.add_argument("--edge-inhibit", type=float, default=0.0)
    p_apply.add_argument("--edge-inhibit-mode", choices=["current", "detail_loss"], default="current")
    p_apply.add_argument("--edge-threshold", type=float, default=0.026)
    p_apply.add_argument("--edge-transition", type=float, default=0.014)
    p_apply.add_argument("--hdr-peak-threshold", type=float, default=0.82)
    p_apply.add_argument("--hdr-transition", type=float, default=0.25)
    p_apply.set_defaults(func=apply)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--scene", required=True, choices=sorted(SCENES))
    p_compare.add_argument("--result", required=True)
    p_compare.add_argument("--output-dir", default=str(RUN_ROOT / "scunet_selector_pilot_v1_compare"))
    p_compare.add_argument("--crop-size", type=int, default=512)
    p_compare.add_argument("--scale", type=int, default=2)
    p_compare.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
