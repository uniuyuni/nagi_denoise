"""Train/apply a multi-output policy for SCUNet luma reconstruction.

The earlier selector predicts one scalar gate. That is not enough for the
current Perfect NR direction because different failure modes need different
decisions:

* borrow SCUNet luma where it improves structure or flat luma tails;
* preserve current edges when SCUNet makes them sleepy;
* reduce borrowing in blue/magenta shadow-risk regions.

This script keeps the existing feature pipeline and trains three output maps:

    luma_gate, edge_keep, risk_gate
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_flat_chroma_smoother import LUMA_SRGB, luma, srgb_to_linear_np
from apply_luma_tail_speckle_filter import sigmoid01
from perfect_nr_detail_guard import write_exr
from perfect_nr_probe import image_stats, make_preview, read_image
from train_scunet_selector import (
    FEATURE_CHANNELS,
    RUN_ROOT,
    SCENES,
    ScunetScene,
    SelectorBlock,
    apply_roi_bias_v2,
    chroma_outlier_risk,
    crop_with_context,
    display,
    display_chroma_ratio,
    hf_abs,
    load_model as load_selector_model,
    make_features_and_target,
    predict_gate_tiled as predict_selector_gate_tiled,
    signed_blue_magenta_risk,
    smoothstep01,
)


POLICY_CHANNELS = 3


def make_policy_targets(
    noisy_linear: np.ndarray,
    current_linear: np.ndarray,
    scunet_linear: np.ndarray,
    *,
    roi_name: str | None,
    roi_bias_strength: float,
    target_preset: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    features, _, base_stats = make_features_and_target(
        noisy_linear,
        current_linear,
        scunet_linear,
        roi_name=roi_name,
        roi_bias_strength=roi_bias_strength,
        target_preset="v3_luma",
    )

    noisy = features[..., 0:3]
    current = features[..., 3:6]
    scunet = features[..., 6:9]
    noisy_y = features[..., 9]
    current_y = features[..., 10]
    scunet_y = features[..., 11]
    current_luma_hf = features[..., 15]
    scunet_luma_hf = features[..., 16]
    current_texture = features[..., 19]
    scunet_texture = features[..., 20]
    coherent = features[..., 21]
    flat = features[..., 22]
    skin = features[..., 23]
    risk = features[..., 24]
    hdr_risk = features[..., 25]

    color_delta = np.mean(
        np.abs(gaussian_filter(scunet - current, sigma=(5.0, 5.0, 0.0), mode="reflect")),
        axis=2,
    )
    color_agree = sigmoid01((0.070 - color_delta) / 0.035)
    signed_risk = signed_blue_magenta_risk(scunet)
    combined_risk = np.clip(np.maximum(risk, signed_risk), 0.0, 1.0)

    luma_benefit = sigmoid01((current_luma_hf - scunet_luma_hf - 0.0015) / 0.006)
    scunet_structure_gain = sigmoid01((scunet_texture - current_texture + 0.06) / 0.16)
    scunet_has_detail = sigmoid01((scunet_luma_hf - 0.012) / 0.012)
    current_is_sleepy = sigmoid01((scunet_luma_hf - current_luma_hf + 0.004) / 0.012)
    coherent_rebuild = np.clip(
        np.maximum(coherent * 0.85, scunet_texture * 0.75) * scunet_has_detail * current_is_sleepy,
        0.0,
        1.0,
    )
    flat_luma_safe = np.clip(flat * luma_benefit * color_agree, 0.0, 1.0)
    scunet_luma_noise_risk = sigmoid01((scunet_luma_hf - current_luma_hf - 0.004) / 0.010)
    flat_noise_risk = np.clip(scunet_luma_noise_risk * flat * (1.0 - coherent_rebuild), 0.0, 1.0)

    current_edge = gaussian_gradient_magnitude(current_y, sigma=1.0, mode="reflect")
    edge_gate = sigmoid01((current_edge - 0.024) / 0.014)
    detail_loss = sigmoid01((current_luma_hf - scunet_luma_hf - 0.0015) / 0.006)

    # Risk should be high for colored shadow speckles, but not for clean dark
    # sky where SCUNet's luma smoothing is beneficial.
    shadow = sigmoid01((0.48 - noisy_y) / 0.16)
    if target_preset == "v1":
        luma_gate = (
            0.08
            + 0.62 * flat_luma_safe
            + 0.56 * coherent_rebuild
            + 0.36 * scunet_structure_gain * np.maximum(coherent, scunet_texture)
            - 0.48 * flat_noise_risk
            - 0.18 * combined_risk * (1.0 - flat_luma_safe)
            - 0.12 * skin * (1.0 - coherent_rebuild)
            - 0.30 * hdr_risk * (1.0 - coherent_rebuild)
        )
        edge_keep = np.clip(edge_gate * detail_loss, 0.0, 1.0)
        risk_gate = np.clip(combined_risk * shadow * (1.0 - 0.55 * flat_luma_safe), 0.0, 1.0)
    elif target_preset == "v2_guarded":
        clean_flat = np.clip(flat_luma_safe * (1.0 - 0.65 * combined_risk) * (1.0 - 0.70 * flat_noise_risk), 0.0, 1.0)
        detail_rebuild = np.clip(
            coherent_rebuild
            * (0.35 + 0.65 * color_agree)
            * (1.0 - 0.55 * combined_risk)
            * (1.0 - 0.45 * hdr_risk),
            0.0,
            1.0,
        )
        structure_gain_safe = np.clip(
            scunet_structure_gain
            * np.maximum(coherent, scunet_texture)
            * color_agree
            * (1.0 - 0.60 * combined_risk)
            * (1.0 - 0.60 * flat_noise_risk),
            0.0,
            1.0,
        )
        luma_gate = (
            0.025
            + 0.86 * clean_flat
            + 0.74 * detail_rebuild
            + 0.28 * structure_gain_safe
            - 0.74 * flat_noise_risk
            - 0.46 * combined_risk * shadow
            - 0.24 * skin * (1.0 - detail_rebuild)
            - 0.42 * hdr_risk * (1.0 - detail_rebuild)
        )
        # Protect every visible fine edge, not only edges where SCUNet is clearly
        # lower-frequency; v1 slept on Dance subject lines for this reason.
        edge_keep = np.clip(edge_gate * (0.48 + 0.52 * detail_loss), 0.0, 1.0)
        risk_gate = np.clip(
            1.55 * combined_risk * shadow
            + 0.82 * flat_noise_risk
            + 0.34 * hdr_risk * (1.0 - detail_rebuild)
            - 0.42 * clean_flat,
            0.0,
            1.0,
        )
    elif target_preset == "v3_balanced":
        clean_flat = np.clip(
            flat_luma_safe
            * (1.0 - 0.38 * combined_risk * shadow)
            * (1.0 - 0.54 * flat_noise_risk),
            0.0,
            1.0,
        )
        detail_rebuild = np.clip(
            coherent_rebuild
            * (0.42 + 0.58 * color_agree)
            * (1.0 - 0.34 * combined_risk * shadow)
            * (1.0 - 0.38 * hdr_risk),
            0.0,
            1.0,
        )
        structure_gain_safe = np.clip(
            scunet_structure_gain
            * np.maximum(coherent, scunet_texture)
            * color_agree
            * (1.0 - 0.42 * combined_risk * shadow)
            * (1.0 - 0.44 * flat_noise_risk),
            0.0,
            1.0,
        )
        risk_not_compensated = combined_risk * shadow * (1.0 - 0.72 * flat_luma_safe)
        luma_gate = (
            0.045
            + 0.82 * clean_flat
            + 0.72 * detail_rebuild
            + 0.30 * structure_gain_safe
            - 0.58 * flat_noise_risk
            - 0.36 * risk_not_compensated
            - 0.18 * skin * (1.0 - detail_rebuild)
            - 0.36 * hdr_risk * (1.0 - detail_rebuild)
        )
        edge_keep = np.clip(edge_gate * (0.38 + 0.62 * detail_loss), 0.0, 1.0)
        risk_gate = np.clip(
            1.18 * risk_not_compensated
            + 0.62 * flat_noise_risk
            + 0.24 * hdr_risk * (1.0 - detail_rebuild)
            - 0.32 * clean_flat,
            0.0,
            1.0,
        )
    else:
        raise ValueError(f"unknown target preset: {target_preset!r}")
    luma_gate = np.clip(luma_gate * (0.52 + 0.48 * color_agree), 0.0, 1.0)
    luma_gate = gaussian_filter(luma_gate.astype(np.float32, copy=False), sigma=0.68, mode="reflect")
    luma_gate = apply_roi_bias_v2(luma_gate, roi_name, roi_bias_strength)
    edge_keep = gaussian_filter(edge_keep.astype(np.float32, copy=False), sigma=0.55, mode="reflect")
    risk_gate = gaussian_filter(risk_gate.astype(np.float32, copy=False), sigma=0.60, mode="reflect")

    targets = np.stack([luma_gate, edge_keep, risk_gate], axis=2).astype(np.float32, copy=False)
    stats = {
        **base_stats,
        "luma_gate_mean": float(np.mean(luma_gate)),
        "edge_keep_mean": float(np.mean(edge_keep)),
        "risk_gate_mean": float(np.mean(risk_gate)),
        "luma_gate_p95": float(np.quantile(luma_gate, 0.95)),
        "edge_keep_p95": float(np.quantile(edge_keep, 0.95)),
        "risk_gate_p95": float(np.quantile(risk_gate, 0.95)),
    }
    return features, targets, stats


class ScunetPolicy(nn.Module):
    def __init__(self, width: int = 20, blocks: int = 4) -> None:
        super().__init__()
        self.head = nn.Conv2d(FEATURE_CHANNELS, width, 3, padding=1)
        self.body = nn.Sequential(*[SelectorBlock(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, POLICY_CHANNELS, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.tail(self.body(F.gelu(self.head(features)))))


class ScunetPolicyDataset(torch.utils.data.Dataset):
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
        features, targets, stats = make_policy_targets(
            noisy_crop,
            current_crop,
            scunet_crop,
            roi_name=roi_name,
            roi_bias_strength=self.roi_bias_strength,
            target_preset=self.target_preset,
        )
        ix0, iy0, ix1, iy1 = inner
        return features[iy0:iy1, ix0:ix1], targets[iy0:iy1, ix0:ix1], stats

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
        features, targets, _ = self._make_patch(item, x, y, roi_name)
        return {
            "features": torch.from_numpy(np.ascontiguousarray(np.transpose(features, (2, 0, 1)))),
            "target": torch.from_numpy(np.ascontiguousarray(np.transpose(targets, (2, 0, 1)))),
        }


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        + torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    )


def save_checkpoint(path: Path, model: ScunetPolicy, args: argparse.Namespace, step: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "width": int(args.width),
            "blocks": int(args.blocks),
            "feature_channels": FEATURE_CHANNELS,
            "policy_channels": POLICY_CHANNELS,
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
    ds = ScunetPolicyDataset(
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
    model = ScunetPolicy(width=args.width, blocks=args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    meta = {
        "scenes": [scene.name for scene in scenes],
        "scene_stats": {item["scene"].name: item["stats"] for item in ds.items},
        "feature_channels": FEATURE_CHANNELS,
        "policy_channels": POLICY_CHANNELS,
        "channels": ["luma_gate", "edge_keep", "risk_gate"],
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_path = out_dir / "stdout.log"
    start = time.monotonic()
    weights = torch.tensor([args.luma_weight, args.edge_weight, args.risk_weight], device=device).view(1, 3, 1, 1)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== train scunet-policy steps={args.steps} device={device} ===\n")
        for step, batch in enumerate(dl, start=1):
            if step > args.steps:
                break
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            pred = model(features)
            loss_fit = (F.smooth_l1_loss(pred, target, beta=0.04, reduction="none") * weights).mean()
            loss_smooth = smoothness_loss(pred) * float(args.smooth_weight)
            loss_mean = torch.abs(pred.mean(dim=(0, 2, 3)) - target.mean(dim=(0, 2, 3))).mean() * float(args.mean_weight)
            loss = loss_fit + loss_smooth + loss_mean
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0:
                elapsed = time.monotonic() - start
                pred_mean = pred.detach().mean(dim=(0, 2, 3)).cpu().numpy()
                target_mean = target.detach().mean(dim=(0, 2, 3)).cpu().numpy()
                msg = (
                    f"step {step:05d}/{args.steps} loss={float(loss.detach()):.6f} "
                    f"fit={float(loss_fit.detach()):.6f} smooth={float(loss_smooth.detach()):.6f} "
                    f"mean={float(loss_mean.detach()):.6f} "
                    f"pred=({pred_mean[0]:.4f},{pred_mean[1]:.4f},{pred_mean[2]:.4f}) "
                    f"target=({target_mean[0]:.4f},{target_mean[1]:.4f},{target_mean[2]:.4f}) "
                    f"{elapsed / step:.3f}s/it"
                )
                print(msg)
                log.write(msg + "\n")
                log.flush()
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / f"scunet_policy_step_{step:06d}.pt", model, args, step)
        save_checkpoint(out_dir / "scunet_policy_final.pt", model, args, args.steps)
        log.write(f"wrote {out_dir / 'scunet_policy_final.pt'}\n")
    print(f"wrote {out_dir / 'scunet_policy_final.pt'}")


def load_model(path: Path, device: torch.device) -> ScunetPolicy:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ScunetPolicy(width=int(ckpt["width"]), blocks=int(ckpt["blocks"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_policy_tiled(
    model: ScunetPolicy,
    device: torch.device,
    noisy: np.ndarray,
    current: np.ndarray,
    scunet: np.ndarray,
    *,
    tile: int,
    overlap: int,
) -> np.ndarray:
    h, w = current.shape[:2]
    out = np.zeros((h, w, POLICY_CHANNELS), dtype=np.float32)
    count = np.zeros((h, w, 1), dtype=np.float32)
    stride = max(1, int(tile) - int(overlap) * 2)
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(w, x0 + int(tile))
            y1 = min(h, y0 + int(tile))
            px0 = max(0, x0 - int(overlap))
            py0 = max(0, y0 - int(overlap))
            px1 = min(w, x1 + int(overlap))
            py1 = min(h, y1 + int(overlap))
            features, _, _ = make_policy_targets(
                noisy[py0:py1, px0:px1],
                current[py0:py1, px0:px1],
                scunet[py0:py1, px0:px1],
                roi_name=None,
                roi_bias_strength=0.0,
                target_preset="v1",
            )
            inp = torch.from_numpy(np.transpose(features, (2, 0, 1))[None]).to(device)
            pred = model(inp).detach().cpu().numpy()[0].transpose(1, 2, 0)
            cy0 = y0 - py0
            cx0 = x0 - px0
            cy1 = cy0 + (y1 - y0)
            cx1 = cx0 + (x1 - x0)
            out[y0:y1, x0:x1] += pred[cy0:cy1, cx0:cx1]
            count[y0:y1, x0:x1] += 1.0
    return out / np.maximum(count, 1.0e-6)


def blend_policy_luma(
    current_linear: np.ndarray,
    scunet_linear: np.ndarray,
    policy: np.ndarray,
    *,
    strength: float,
    gate_gamma: float,
    edge_inhibit: float,
    risk_inhibit: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
) -> np.ndarray:
    current = np.clip(np.asarray(current_linear, dtype=np.float32)[..., :3], 0.0, None)
    scunet = np.clip(np.asarray(scunet_linear, dtype=np.float32)[..., :3], 0.0, None)
    current_d = display(current)
    scunet_d = display(scunet)
    current_y = luma(current_d, LUMA_SRGB)
    scunet_y = luma(scunet_d, LUMA_SRGB)
    luma_gate = np.clip(policy[..., 0], 0.0, 1.0)
    edge_keep = np.clip(policy[..., 1], 0.0, 1.0)
    risk_gate = np.clip(policy[..., 2], 0.0, 1.0)
    blend = np.clip(np.power(luma_gate, max(float(gate_gamma), 1.0e-6)) * float(strength), 0.0, 1.0)
    blend *= 1.0 - np.clip(float(edge_inhibit), 0.0, 1.0) * edge_keep
    blend *= 1.0 - np.clip(float(risk_inhibit), 0.0, 1.0) * risk_gate
    out_y = np.clip(current_y * (1.0 - blend) + scunet_y * blend, 0.0, 1.0)
    out_display = np.clip(display_chroma_ratio(current_d) * out_y[..., None], 0.0, 1.0)
    out = srgb_to_linear_np(out_display).astype(np.float32, copy=False)
    peak = np.max(current, axis=2)
    hdr = smoothstep01((peak - float(hdr_peak_threshold)) / max(float(hdr_transition), 1.0e-6))
    return (out * (1.0 - hdr[..., None]) + current * hdr[..., None]).astype(np.float32, copy=False)


def blend_selector_policy_luma(
    current_linear: np.ndarray,
    scunet_linear: np.ndarray,
    selector_gate: np.ndarray,
    policy: np.ndarray,
    *,
    strength: float,
    gate_gamma: float,
    edge_inhibit: float,
    risk_inhibit: float,
    hdr_peak_threshold: float,
    hdr_transition: float,
) -> np.ndarray:
    current = np.clip(np.asarray(current_linear, dtype=np.float32)[..., :3], 0.0, None)
    scunet = np.clip(np.asarray(scunet_linear, dtype=np.float32)[..., :3], 0.0, None)
    current_d = display(current)
    scunet_d = display(scunet)
    current_y = luma(current_d, LUMA_SRGB)
    scunet_y = luma(scunet_d, LUMA_SRGB)
    edge_keep = np.clip(policy[..., 1], 0.0, 1.0)
    risk_gate = np.clip(policy[..., 2], 0.0, 1.0)
    blend = np.clip(
        np.power(np.clip(selector_gate, 0.0, 1.0), max(float(gate_gamma), 1.0e-6)) * float(strength),
        0.0,
        1.0,
    )
    blend *= 1.0 - np.clip(float(edge_inhibit), 0.0, 1.0) * edge_keep
    blend *= 1.0 - np.clip(float(risk_inhibit), 0.0, 1.0) * risk_gate
    out_y = np.clip(current_y * (1.0 - blend) + scunet_y * blend, 0.0, 1.0)
    out_display = np.clip(display_chroma_ratio(current_d) * out_y[..., None], 0.0, 1.0)
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
    policy = predict_policy_tiled(model, device, noisy, current, scunet, tile=args.tile, overlap=args.overlap)
    selector_gate = None
    if args.selector_checkpoint:
        selector_model = load_selector_model(Path(args.selector_checkpoint), device)
        selector_gate = predict_selector_gate_tiled(
            selector_model,
            device,
            noisy,
            current,
            scunet,
            tile=args.tile,
            overlap=args.overlap,
        )
        out = blend_selector_policy_luma(
            current,
            scunet,
            selector_gate,
            policy,
            strength=args.strength,
            gate_gamma=args.gate_gamma,
            edge_inhibit=args.edge_inhibit,
            risk_inhibit=args.risk_inhibit,
            hdr_peak_threshold=args.hdr_peak_threshold,
            hdr_transition=args.hdr_transition,
        )
    else:
        out = blend_policy_luma(
            current,
            scunet,
            policy,
            strength=args.strength,
            gate_gamma=args.gate_gamma,
            edge_inhibit=args.edge_inhibit,
            risk_inhibit=args.risk_inhibit,
            hdr_peak_threshold=args.hdr_peak_threshold,
            hdr_transition=args.hdr_transition,
        )
    name = args.name or f"{scene.name}_scunet_policy"
    exr_path = out_dir / f"{name}.exr"
    preview_path = out_dir / f"{name}_preview.png"
    meta_path = out_dir / f"{name}.json"
    write_exr(exr_path, out)
    Image.fromarray(make_preview(out, exposure=1.0, tone="reinhard")).save(preview_path)
    for idx, label in enumerate(("luma_gate", "edge_keep", "risk_gate")):
        Image.fromarray(np.clip(policy[..., idx] * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(
            out_dir / f"{name}_{label}.png"
        )
    meta = {
        "scene": scene.name,
        "checkpoint": str(Path(args.checkpoint)),
        "inputs": {"noisy": str(scene.noisy), "current": str(scene.current), "scunet": str(scene.scunet)},
        "outputs": {"exr": str(exr_path), "preview": str(preview_path)},
        "policy": {
            label: {
                "mean": float(np.mean(policy[..., idx])),
                "p50": float(np.quantile(policy[..., idx], 0.50)),
                "p90": float(np.quantile(policy[..., idx], 0.90)),
                "p99": float(np.quantile(policy[..., idx], 0.99)),
            }
            for idx, label in enumerate(("luma_gate", "edge_keep", "risk_gate"))
        },
        "selector_gate": None
        if selector_gate is None
        else {
            "mean": float(np.mean(selector_gate)),
            "p50": float(np.quantile(selector_gate, 0.50)),
            "p90": float(np.quantile(selector_gate, 0.90)),
            "p99": float(np.quantile(selector_gate, 0.99)),
        },
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "output_stats": image_stats(out),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply a multi-output SCUNet luma policy.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--output-dir", default=str(RUN_ROOT / "scunet_policy_pilot_v1"))
    p_train.add_argument("--scenes", default="xt5_occi,k5_dance,k5_ice")
    p_train.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])
    p_train.add_argument("--steps", type=int, default=600)
    p_train.add_argument("--batch-size", type=int, default=3)
    p_train.add_argument("--patch-size", type=int, default=192)
    p_train.add_argument("--context", type=int, default=64)
    p_train.add_argument("--width", type=int, default=20)
    p_train.add_argument("--blocks", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=2.0e-4)
    p_train.add_argument("--weight-decay", type=float, default=1.0e-4)
    p_train.add_argument("--smooth-weight", type=float, default=0.018)
    p_train.add_argument("--mean-weight", type=float, default=0.12)
    p_train.add_argument("--luma-weight", type=float, default=1.0)
    p_train.add_argument("--edge-weight", type=float, default=0.70)
    p_train.add_argument("--risk-weight", type=float, default=0.85)
    p_train.add_argument("--target-preset", default="v1", choices=["v1", "v2_guarded", "v3_balanced"])
    p_train.add_argument("--roi-probability", type=float, default=0.86)
    p_train.add_argument("--roi-bias-strength", type=float, default=0.70)
    p_train.add_argument("--stats-samples", type=int, default=16)
    p_train.add_argument("--seed", type=int, default=9463)
    p_train.add_argument("--log-every", type=int, default=50)
    p_train.add_argument("--save-every", type=int, default=0)
    p_train.set_defaults(func=train)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--selector-checkpoint", default=None)
    p_apply.add_argument("--scene", required=True, choices=sorted(SCENES))
    p_apply.add_argument("--output-dir", default=str(RUN_ROOT / "scunet_policy_pilot_v1_outputs"))
    p_apply.add_argument("--name", default=None)
    p_apply.add_argument("--device", default="cpu", choices=["auto", "mps", "cuda", "cpu"])
    p_apply.add_argument("--tile", type=int, default=768)
    p_apply.add_argument("--overlap", type=int, default=64)
    p_apply.add_argument("--strength", type=float, default=2.4)
    p_apply.add_argument("--gate-gamma", type=float, default=0.75)
    p_apply.add_argument("--edge-inhibit", type=float, default=0.80)
    p_apply.add_argument("--risk-inhibit", type=float, default=0.80)
    p_apply.add_argument("--hdr-peak-threshold", type=float, default=0.82)
    p_apply.add_argument("--hdr-transition", type=float, default=0.25)
    p_apply.set_defaults(func=apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
