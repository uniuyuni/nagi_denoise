"""Weak-teacher real-photo refiner pilot.

This is intentionally a small experiment, not a production denoiser.  It tests
whether a tiny CNN can learn the residual cleanup that the hand-written filters
are struggling with, while keeping hair/edge detail close to the current Nagi
output.  PhotoLab XD is used only as a weak target in low-detail regions.
"""
from __future__ import annotations

import argparse
import json
import pickle
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

from perfect_nr_detail_guard import write_exr, write_tiff
from perfect_nr_probe import read_image


ROOT = Path(__file__).resolve().parents[1]
TEST_PHOTOS = Path("/Users/uniuyuni/PythonProjects/test_photos")
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    noisy: Path
    current: Path
    teacher: Path
    teacher_offset_xy: tuple[int, int] = (0, 0)


SCENES: tuple[SceneSpec, ...] = (
    SceneSpec(
        "xt5_occi",
        TEST_PHOTOS / "X-T5 Occi noisy.EXR",
        ROOT / "runs/perfect_nr/luma_surface_tune/xt5_occi_iso6400_max_chroma_surface/xt5_occi_iso6400_max_chroma_surface.exr",
        TEST_PHOTOS / "X-T5 Occi PL deepprimeXD.tif",
    ),
    SceneSpec(
        "k5_ice",
        TEST_PHOTOS / "K-5 Ice noisy.EXR",
        ROOT / "runs/perfect_nr/luma_surface_tune/k5_ice_iso6400_max_chroma_surface/k5_ice_iso6400_max_chroma_surface.exr",
        TEST_PHOTOS / "K-5 Ice PL deepprimeXD.tif",
    ),
    SceneSpec(
        "xt5_cat",
        TEST_PHOTOS / "X-T5 Cat noisy.EXR",
        ROOT / "runs/perfect_nr/luma_surface_tune/xt5_cat_iso6400_max_chroma_surface/xt5_cat_iso6400_max_chroma_surface.exr",
        TEST_PHOTOS / "X-T5 Cat PL deepprimeXD.tif",
        teacher_offset_xy=(1808, 556),
    ),
)
SCENE_BY_NAME = {scene.name: scene for scene in SCENES}


ROI_TOP_LEFT: dict[str, list[tuple[str, int, int]]] = {
    "xt5_occi": [
        ("face_center", 2120, 1260),
        ("hair_detail", 2420, 1040),
        ("cheek_hair", 2280, 1420),
        ("skin_shadow", 3000, 1680),
        ("noise_dark", 3072, 3600),
    ],
    "k5_ice": [
        ("center", 2208, 1376),
        ("blue_shadow", 2450, 1750),
        ("lower_left", 1400, 2450),
        ("upper_mid", 2200, 900),
        ("dark_detail", 3050, 1850),
    ],
    "xt5_cat": [
        ("center", 986, 985),
        ("whisker", 950, 800),
        ("dark_fur", 1220, 1120),
        ("upper_mid", 900, 520),
        ("lower_left", 500, 1450),
    ],
}


def _safe_rgb(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return np.nan_to_num(x[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(_safe_rgb(x), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(_safe_rgb(x), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def luma(x: np.ndarray) -> np.ndarray:
    return np.sum(_safe_rgb(x) * LUMA.reshape(1, 1, 3), axis=2)


def detail_map(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    y = luma(x)
    return np.abs(y - gaussian_filter(y, sigma=float(sigma), mode="reflect")).astype(np.float32)


def chroma_hf_map(img: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    y = luma(img)[..., None]
    chroma = _safe_rgb(img) - y
    hf = chroma - gaussian_filter(chroma, sigma=(float(sigma), float(sigma), 0.0), mode="reflect")
    return np.sqrt(np.sum(hf * hf, axis=2)).astype(np.float32)


def dark_chroma_dot_gate(current: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    cur = _safe_rgb(current)
    noi = _safe_rgb(noisy)
    y = luma(cur)
    dark = np.clip((0.58 - y) / 0.42, 0.0, 1.0)
    dark = (dark * dark * (3.0 - 2.0 * dark)).astype(np.float32)
    chroma_noise = np.maximum(chroma_hf_map(cur), chroma_hf_map(noi))
    dot = np.clip((chroma_noise - 0.016) / 0.032, 0.0, 1.0).astype(np.float32)
    return (dark * dot).astype(np.float32)


def match_teacher_low_frequency(current: np.ndarray, teacher: np.ndarray, sigma: float = 28.0) -> np.ndarray:
    """Keep Nagi's local tone/color and borrow only PL's local residual texture.

    PL output can differ in rendering, color, ICC handling, and highlight rolloff.
    Those are not valid denoising supervision signals for this pilot.  This
    high-pass target asks the model to learn local cleanup without chasing PL's
    broader look.
    """
    cur = _safe_rgb(current)
    ref = _safe_rgb(teacher)
    cur_low = gaussian_filter(cur, sigma=(float(sigma), float(sigma), 0.0), mode="reflect")
    ref_low = gaussian_filter(ref, sigma=(float(sigma), float(sigma), 0.0), mode="reflect")
    return np.clip(cur_low + (ref - ref_low), 0.0, 1.0).astype(np.float32)


def crop_with_offset(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    x = int(np.clip(x, 0, max(0, w - size)))
    y = int(np.clip(y, 0, max(0, h - size)))
    return img[y : y + size, x : x + size]


def read_scene_arrays(spec: SceneSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    noisy = linear_to_srgb(read_image(spec.noisy))
    current = linear_to_srgb(read_image(spec.current))
    teacher = np.clip(read_image(spec.teacher), 0.0, 1.0).astype(np.float32, copy=False)
    return noisy, current, teacher


def build_crops(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old in crop_dir.glob("*.npz"):
            old.unlink()
    rng = random.Random(args.seed)
    written: list[dict] = []

    for spec in SCENES:
        print(f"loading {spec.name}")
        noisy, current, teacher = read_scene_arrays(spec)
        h, w = current.shape[:2]
        tx0, ty0 = spec.teacher_offset_xy
        rois = list(ROI_TOP_LEFT[spec.name])
        for i in range(args.random_per_scene):
            x = rng.randint(0, max(0, w - args.crop_size))
            y = rng.randint(0, max(0, h - args.crop_size))
            rois.append((f"random_{i:02d}", x, y))

        for label, x, y in rois:
            n = crop_with_offset(noisy, x, y, args.crop_size)
            c = crop_with_offset(current, x, y, args.crop_size)
            t_raw = crop_with_offset(teacher, x + tx0, y + ty0, args.crop_size)
            t = match_teacher_low_frequency(c, t_raw, sigma=args.teacher_match_sigma)
            if n.shape[:2] != (args.crop_size, args.crop_size) or t.shape[:2] != (args.crop_size, args.crop_size):
                continue
            d_current = detail_map(c, sigma=1.0)
            d_teacher = detail_map(t, sigma=1.0)
            # Low-detail weak target, but preserve structures that PL might soften.
            flat = np.exp(-np.maximum(d_current, d_teacher) / 0.018).astype(np.float32)
            edge = np.clip(np.maximum(d_current, d_teacher) / 0.030, 0.0, 1.0).astype(np.float32)
            delta = np.linalg.norm(t - c, axis=2).astype(np.float32)
            useful = np.clip(delta / 0.060, 0.0, 1.0)
            chroma_noise = np.maximum(chroma_hf_map(c), chroma_hf_map(n))
            noise_gate = np.clip((chroma_noise - 0.010) / 0.035, 0.0, 1.0).astype(np.float32)
            dark_dot_gate = dark_chroma_dot_gate(c, n)
            weak = np.clip(flat * useful, 0.0, 1.0).astype(np.float32)

            stem = f"{spec.name}_{label}_x{x}_y{y}"
            path = crop_dir / f"{stem}.npz"
            np.savez_compressed(
                path,
                noisy=n.astype(np.float16),
                current=c.astype(np.float16),
                teacher=t.astype(np.float16),
                teacher_raw=t_raw.astype(np.float16),
                weak=weak.astype(np.float16),
                protect=edge.astype(np.float16),
                noise_gate=noise_gate.astype(np.float16),
                dark_dot_gate=dark_dot_gate.astype(np.float16),
                scene=spec.name,
                label=label,
                x=np.int32(x),
                y=np.int32(y),
            )
            written.append(
                {
                    "path": str(path),
                    "scene": spec.name,
                    "label": label,
                    "x": x,
                    "y": y,
                    "weak_mean": float(np.mean(weak)),
                    "protect_mean": float(np.mean(edge)),
                    "noise_gate_mean": float(np.mean(noise_gate)),
                    "dark_dot_gate_mean": float(np.mean(dark_dot_gate)),
                    "teacher_delta_mean": float(np.mean(delta)),
                    "teacher_raw_delta_mean": float(np.mean(np.linalg.norm(t_raw - c, axis=2))),
                }
            )
    (out_dir / "crops_manifest.json").write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(written)} crops to {crop_dir}")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw2(self.act(self.pw1(y)))
        return x + y * self.scale


class PilotRefiner(nn.Module):
    def __init__(self, width: int = 32, blocks: int = 5, max_delta: float = 0.050, in_channels: int = 11) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.head = nn.Conv2d(in_channels, width, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, features: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        y = self.body(F.gelu(self.head(features)))
        delta = torch.tanh(self.tail(y)) * self.max_delta
        return torch.clamp(current + delta, 0.0, 1.0)


class GatedPilotRefiner(nn.Module):
    def __init__(self, width: int = 32, blocks: int = 5, max_delta: float = 0.080, in_channels: int = 11) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.head = nn.Conv2d(in_channels, width, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.delta_tail = nn.Conv2d(width, 3, 3, padding=1)
        self.gate_tail = nn.Conv2d(width, 1, 3, padding=1)

    def forward_with_aux(self, features: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y = self.body(F.gelu(self.head(features)))
        delta = torch.tanh(self.delta_tail(y)) * self.max_delta
        gate = torch.sigmoid(self.gate_tail(y))
        out = torch.clamp(current + delta * gate, 0.0, 1.0)
        return out, {"gate": gate, "delta": delta}

    def forward(self, features: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_aux(features, current)
        return out


class ChromaPilotRefiner(nn.Module):
    def __init__(self, width: int = 32, blocks: int = 5, max_delta: float = 0.100, in_channels: int = 15) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.head = nn.Conv2d(in_channels, width, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, 3, 3, padding=1)

    def forward_with_aux(self, features: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y = (current * torch.tensor([0.299, 0.587, 0.114], device=current.device, dtype=current.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
        raw_delta = torch.tanh(self.tail(self.body(F.gelu(self.head(features))))) * self.max_delta
        delta_mean = (raw_delta * torch.tensor([0.299, 0.587, 0.114], device=current.device, dtype=current.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
        chroma_delta = raw_delta - delta_mean
        out = torch.clamp(current + chroma_delta, 0.0, 1.0)
        out_y = (out * torch.tensor([0.299, 0.587, 0.114], device=out.device, dtype=out.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
        out = torch.clamp(out + (y - out_y), 0.0, 1.0)
        return out, {"delta": chroma_delta}

    def forward(self, features: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_aux(features, current)
        return out


def refine_with_aux(model: nn.Module, noisy: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    features = make_features(noisy, current, getattr(model, "feature_set", "basic"))
    if hasattr(model, "forward_with_aux"):
        return model.forward_with_aux(features, current)  # type: ignore[attr-defined]
    out = model(features, current)
    return out, {}


def make_features(noisy: torch.Tensor, current: torch.Tensor, feature_set: str = "basic") -> torch.Tensor:
    y = (current * torch.tensor([0.299, 0.587, 0.114], device=current.device, dtype=current.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
    chroma = current - y
    chroma_mag = torch.sqrt(torch.clamp((chroma * chroma).sum(1, keepdim=True), min=1.0e-8))
    residual = noisy - current
    features = [noisy, current, residual, y, chroma_mag]
    if feature_set == "texture":
        noisy_y = (noisy * torch.tensor([0.299, 0.587, 0.114], device=noisy.device, dtype=noisy.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
        noisy_chroma = noisy - noisy_y
        noisy_chroma_mag = torch.sqrt(torch.clamp((noisy_chroma * noisy_chroma).sum(1, keepdim=True), min=1.0e-8))
        residual_y = (residual * torch.tensor([0.299, 0.587, 0.114], device=residual.device, dtype=residual.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
        residual_chroma = residual - residual_y
        residual_chroma_mag = torch.sqrt(torch.clamp((residual_chroma * residual_chroma).sum(1, keepdim=True), min=1.0e-8))
        y_detail = torch.abs(y - F.avg_pool2d(y, kernel_size=7, stride=1, padding=3))
        chroma_hf = torch.abs(chroma_mag - F.avg_pool2d(chroma_mag, kernel_size=7, stride=1, padding=3))
        noisy_chroma_hf = torch.abs(noisy_chroma_mag - F.avg_pool2d(noisy_chroma_mag, kernel_size=7, stride=1, padding=3))
        features.extend([y_detail, chroma_hf, noisy_chroma_hf, residual_chroma_mag])
    elif feature_set != "basic":
        raise ValueError(f"unknown feature set: {feature_set}")
    return torch.cat(features, dim=1)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, crop_dir: Path, patch: int, samples_per_epoch: int, seed: int) -> None:
        self.paths = sorted(Path(crop_dir).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no crop npz files under {crop_dir}")
        self.patch = int(patch)
        self.samples_per_epoch = int(samples_per_epoch)
        self.rng = random.Random(seed)
        self.cache = [dict(np.load(p)) for p in self.paths]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.cache[self.rng.randrange(len(self.cache))]
        h, w = item["current"].shape[:2]
        x = self.rng.randrange(0, max(1, w - self.patch + 1))
        y = self.rng.randrange(0, max(1, h - self.patch + 1))

        def take(name: str, channels: bool) -> torch.Tensor:
            arr = item[name][y : y + self.patch, x : x + self.patch].astype(np.float32)
            if channels:
                arr = np.transpose(arr, (2, 0, 1))
            else:
                arr = arr[None, ...]
            return torch.from_numpy(np.ascontiguousarray(arr))

        return {
            "noisy": take("noisy", True),
            "current": take("current", True),
            "teacher": take("teacher", True),
            "weak": take("weak", False),
            "protect": take("protect", False),
            "noise_gate": take("noise_gate", False) if "noise_gate" in item else torch.zeros((1, self.patch, self.patch), dtype=torch.float32),
            "dark_dot_gate": take("dark_dot_gate", False) if "dark_dot_gate" in item else torch.zeros((1, self.patch, self.patch), dtype=torch.float32),
        }


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def feature_channels(feature_set: str) -> int:
    if feature_set == "basic":
        return 11
    if feature_set == "texture":
        return 15
    raise ValueError(f"unknown feature set: {feature_set}")


def create_model(kind: str, width: int, blocks: int, max_delta: float, feature_set: str = "basic") -> nn.Module:
    if kind == "residual":
        model: nn.Module = PilotRefiner(width=width, blocks=blocks, max_delta=max_delta, in_channels=feature_channels(feature_set))
        model.feature_set = feature_set  # type: ignore[attr-defined]
        return model
    if kind == "gated":
        model = GatedPilotRefiner(width=width, blocks=blocks, max_delta=max_delta, in_channels=feature_channels(feature_set))
        model.feature_set = feature_set  # type: ignore[attr-defined]
        return model
    if kind == "chroma":
        model = ChromaPilotRefiner(width=width, blocks=blocks, max_delta=max_delta, in_channels=feature_channels(feature_set))
        model.feature_set = feature_set  # type: ignore[attr-defined]
        return model
    raise ValueError(f"unknown model kind: {kind}")


def load_model(checkpoint: str | Path, device: torch.device) -> nn.Module:
    try:
        ckpt = torch.load(checkpoint, map_location="cpu")
    except pickle.UnpicklingError:
        # Backward compatibility for the first pilot checkpoint, which stored
        # argparse's subcommand function object in args.
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = create_model(
        str(ckpt.get("model_kind", "residual")),
        width=int(ckpt["width"]),
        blocks=int(ckpt["blocks"]),
        max_delta=float(ckpt["max_delta"]),
        feature_set=str(ckpt.get("feature_set", "basic")),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ds = CropDataset(Path(args.crop_dir), args.patch_size, args.iters * args.batch_size, args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    model = create_model(args.model_kind, width=args.width, blocks=args.blocks, max_delta=args.max_delta, feature_set=args.feature_set).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)

    log_path = out_dir / "stdout.log"
    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== refiner-pilot train iters={args.iters} device={device} ===\n")
        for step, batch in enumerate(loader, start=1):
            if step > args.iters:
                break
            noisy = batch["noisy"].to(device)
            current = batch["current"].to(device)
            teacher = batch["teacher"].to(device)
            weak = batch["weak"].to(device)
            protect = batch["protect"].to(device)
            noise_gate = batch["noise_gate"].to(device)
            dark_dot_gate = batch["dark_dot_gate"].to(device)
            pred, aux = refine_with_aux(model, noisy, current)

            weak_w = 0.08 + weak * args.weak_weight * (1.0 + noise_gate * args.noise_gate_weight + dark_dot_gate * args.dark_dot_weight)
            chroma_w = torch.clamp(
                weak + noise_gate * args.chroma_noise_weight + dark_dot_gate * args.dark_dot_chroma_weight,
                0.0,
                args.chroma_weight_clip,
            )
            protect_w = protect * args.protect_weight
            loss_teacher = (torch.abs(pred - teacher) * weak_w).mean()
            loss_identity = (torch.abs(pred - current) * (protect_w + args.identity_weight)).mean()
            loss_chroma = chroma_loss(pred, teacher, chroma_w) * args.chroma_weight
            loss_luma_identity = luma_identity_loss(pred, current, protect, args.luma_protect_weight) * args.luma_change_weight
            loss_gate = gate_loss(aux.get("gate"), weak, protect, noise_gate, args.gate_weight, args.gate_sparsity_weight)
            loss_tv = total_variation(pred - current) * args.tv_weight
            loss = args.teacher_rgb_weight * loss_teacher + loss_identity + loss_chroma + loss_luma_identity + loss_gate + loss_tv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step == 1 or step % args.log_every == 0:
                elapsed = time.monotonic() - start
                msg = (
                    f"step {step:05d}/{args.iters} loss={float(loss.detach()):.6f} "
                    f"teacher={float(loss_teacher.detach()):.6f} identity={float(loss_identity.detach()):.6f} "
                    f"chroma={float(loss_chroma.detach()):.6f} luma_id={float(loss_luma_identity.detach()):.6f} "
                    f"gate={float(loss_gate.detach()):.6f} "
                    f"tv={float(loss_tv.detach()):.6f} "
                    f"{elapsed / step:.3f}s/it"
                )
                print(msg)
                log.write(msg + "\n")
                log.flush()
        ckpt = {
            "model": model.state_dict(),
            "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
            "model_kind": args.model_kind,
            "feature_set": args.feature_set,
            "width": args.width,
            "blocks": args.blocks,
            "max_delta": args.max_delta,
        }
        ckpt_path = out_dir / "refiner_pilot_final.pt"
        torch.save(ckpt, ckpt_path)
        print(f"wrote {ckpt_path}")
        log.write(f"wrote {ckpt_path}\n")


def chroma_loss(pred: torch.Tensor, teacher: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    wp = torch.tensor([0.299, 0.587, 0.114], device=pred.device, dtype=pred.dtype).view(1, 3, 1, 1)
    yp = (pred * wp).sum(1, keepdim=True)
    yt = (teacher * wp).sum(1, keepdim=True)
    return (torch.abs((pred - yp) - (teacher - yt)) * weight).mean()


def luma_identity_loss(pred: torch.Tensor, current: torch.Tensor, protect: torch.Tensor, protect_weight: float) -> torch.Tensor:
    wp = torch.tensor([0.299, 0.587, 0.114], device=pred.device, dtype=pred.dtype).view(1, 3, 1, 1)
    yp = (pred * wp).sum(1, keepdim=True)
    yc = (current * wp).sum(1, keepdim=True)
    weight = 1.0 + protect * float(protect_weight)
    return (torch.abs(yp - yc) * weight).mean()


def gate_loss(
    gate: torch.Tensor | None,
    weak: torch.Tensor,
    protect: torch.Tensor,
    noise_gate: torch.Tensor,
    weight: float,
    sparsity_weight: float,
) -> torch.Tensor:
    if gate is None:
        return torch.zeros((), device=weak.device, dtype=weak.dtype)
    target = torch.clamp((weak * 0.75 + noise_gate * 0.45) * (1.0 - protect * 0.75), 0.0, 1.0)
    supervised = F.smooth_l1_loss(gate, target, reduction="mean") * float(weight)
    sparsity = (gate * (0.25 + protect * 1.50)).mean() * float(sparsity_weight)
    return supervised + sparsity


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))


@torch.no_grad()
def apply_crops(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.crop_dir).glob("*.npz"))
    rows = []
    for path in paths:
        item = dict(np.load(path))
        noisy = item["noisy"].astype(np.float32)
        current = item["current"].astype(np.float32)
        teacher = item["teacher"].astype(np.float32)
        inp_noisy = torch.from_numpy(np.transpose(noisy, (2, 0, 1))[None]).to(device)
        inp_current = torch.from_numpy(np.transpose(current, (2, 0, 1))[None]).to(device)
        pred, aux = refine_with_aux(model, inp_noisy, inp_current)
        out = np.transpose(pred.squeeze(0).cpu().numpy(), (1, 2, 0))
        stem = path.stem
        rows.append((stem, noisy, current, out, teacher))
        write_exr(out_dir / f"{stem}_refined.exr", srgb_to_linear(out))
        write_tiff(out_dir / f"{stem}_refined.tiff", srgb_to_linear(out))
        if "gate" in aux:
            gate = aux["gate"].squeeze().cpu().numpy()
            Image.fromarray((np.clip(gate, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(out_dir / f"{stem}_gate.png")
    make_contact_sheets(rows, out_dir, max_rows=args.max_rows)
    print(f"wrote refined crops and contact sheets to {out_dir}")


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_weight(height: int, width: int, top: int, left: int, full_h: int, full_w: int, overlap: int) -> np.ndarray:
    wy = np.ones((height,), dtype=np.float32)
    wx = np.ones((width,), dtype=np.float32)
    ramp_y = min(int(overlap), height // 2)
    ramp_x = min(int(overlap), width // 2)
    if ramp_y > 0 and top > 0:
        wy[:ramp_y] *= np.linspace(0.02, 1.0, ramp_y, dtype=np.float32)
    if ramp_y > 0 and top + height < full_h:
        wy[-ramp_y:] *= np.linspace(1.0, 0.02, ramp_y, dtype=np.float32)
    if ramp_x > 0 and left > 0:
        wx[:ramp_x] *= np.linspace(0.02, 1.0, ramp_x, dtype=np.float32)
    if ramp_x > 0 and left + width < full_w:
        wx[-ramp_x:] *= np.linspace(1.0, 0.02, ramp_x, dtype=np.float32)
    return (wy[:, None] * wx[None, :])[..., None]


def chroma_hf_p99(img: np.ndarray) -> float:
    y = luma(img)[..., None]
    chroma = _safe_rgb(img) - y
    hf = chroma - gaussian_filter(chroma, sigma=(1.2, 1.2, 0.0), mode="reflect")
    return float(np.quantile(np.sqrt(np.sum(hf * hf, axis=2)), 0.99))


def luma_hf_p99(img: np.ndarray) -> float:
    y = luma(img)
    hf = y - gaussian_filter(y, sigma=1.2, mode="reflect")
    return float(np.quantile(np.abs(hf), 0.99))


@torch.no_grad()
def apply_full(args: argparse.Namespace) -> None:
    spec = SCENE_BY_NAME.get(args.scene) if args.scene else None
    noisy_path = Path(args.noisy or (spec.noisy if spec else ""))
    current_path = Path(args.current or (spec.current if spec else ""))
    if not noisy_path or not current_path:
        raise ValueError("--scene or both --noisy/--current are required")

    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or (args.scene or current_path.stem.replace(" ", "_"))

    print(f"reading noisy: {noisy_path}")
    noisy = linear_to_srgb(read_image(noisy_path))
    print(f"reading current: {current_path}")
    current = linear_to_srgb(read_image(current_path))
    if noisy.shape != current.shape:
        raise ValueError(f"shape mismatch: noisy={noisy.shape} current={current.shape}")

    h, w = current.shape[:2]
    out_acc = np.zeros((h, w, 3), dtype=np.float32)
    weight_acc = np.zeros((h, w, 1), dtype=np.float32)
    ys = tile_starts(h, args.tile_size, args.tile_overlap)
    xs = tile_starts(w, args.tile_size, args.tile_overlap)
    total = len(ys) * len(xs)
    start = time.monotonic()
    tile_index = 0
    for y0 in ys:
        for x0 in xs:
            tile_index += 1
            n = noisy[y0 : y0 + args.tile_size, x0 : x0 + args.tile_size]
            c = current[y0 : y0 + args.tile_size, x0 : x0 + args.tile_size]
            inp_noisy = torch.from_numpy(np.transpose(np.ascontiguousarray(n), (2, 0, 1))[None]).to(device)
            inp_current = torch.from_numpy(np.transpose(np.ascontiguousarray(c), (2, 0, 1))[None]).to(device)
            pred, aux = refine_with_aux(model, inp_noisy, inp_current)
            refined = np.transpose(pred.squeeze(0).cpu().numpy(), (1, 2, 0))
            if args.strength != 1.0:
                refined = np.clip(c + (refined - c) * float(args.strength), 0.0, 1.0)
            ww = tile_weight(refined.shape[0], refined.shape[1], y0, x0, h, w, args.tile_overlap)
            out_acc[y0 : y0 + refined.shape[0], x0 : x0 + refined.shape[1]] += refined * ww
            weight_acc[y0 : y0 + refined.shape[0], x0 : x0 + refined.shape[1]] += ww
            if tile_index == 1 or tile_index % args.log_every == 0 or tile_index == total:
                elapsed = time.monotonic() - start
                print(f"tile {tile_index:04d}/{total} {elapsed / tile_index:.3f}s/tile")

    refined_srgb = np.clip(out_acc / np.maximum(weight_acc, 1.0e-6), 0.0, 1.0)
    refined_linear = srgb_to_linear(refined_srgb)
    exr_path = out_dir / f"{name}_refined.exr"
    tiff_path = out_dir / f"{name}_refined.tiff"
    preview_path = out_dir / f"{name}_refined_preview.png"
    meta_path = out_dir / f"{name}_refined_meta.json"
    write_exr(exr_path, refined_linear)
    write_tiff(tiff_path, refined_linear)
    Image.fromarray(to_u8(refined_srgb)).save(preview_path)

    elapsed = time.monotonic() - start
    meta = {
        "scene": args.scene,
        "checkpoint": str(args.checkpoint),
        "noisy": str(noisy_path),
        "current": str(current_path),
        "shape": [h, w, 3],
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_count": total,
        "device": str(device),
        "strength": args.strength,
        "elapsed_sec": elapsed,
        "sec_per_tile": elapsed / max(total, 1),
        "mean_abs_change_srgb": float(np.mean(np.abs(refined_srgb - current))),
        "current_chroma_hf_p99": chroma_hf_p99(current),
        "refined_chroma_hf_p99": chroma_hf_p99(refined_srgb),
        "current_luma_hf_p99": luma_hf_p99(current),
        "refined_luma_hf_p99": luma_hf_p99(refined_srgb),
    }
    if isinstance(model, GatedPilotRefiner):
        meta["model_kind"] = "gated"
    elif isinstance(model, ChromaPilotRefiner):
        meta["model_kind"] = "chroma"
    else:
        meta["model_kind"] = "residual"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"wrote {exr_path}")
    print(f"wrote {tiff_path}")
    print(f"wrote {preview_path}")


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(_safe_rgb(x), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def make_contact_sheets(rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], out_dir: Path, max_rows: int) -> None:
    font_h = 22
    pad = 8
    labels = ["noisy", "current", "refiner", "PL XD ref"]
    for chunk_index in range(0, len(rows), max_rows):
        chunk = rows[chunk_index : chunk_index + max_rows]
        size = chunk[0][1].shape[0]
        w = 4 * size + 5 * pad
        h = len(chunk) * (size + font_h + 2 * pad) + pad
        canvas = Image.new("RGB", (w, h), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        y = pad
        for stem, noisy, current, out, teacher in chunk:
            x = pad
            draw.text((x, y), stem, fill=(235, 235, 235))
            for label, img in zip(labels, [noisy, current, out, teacher], strict=True):
                draw.text((x, y + font_h - 16), label, fill=(210, 210, 210))
                canvas.paste(Image.fromarray(to_u8(img)), (x, y + font_h + pad))
                x += size + pad
            y += size + font_h + 2 * pad
        path = out_dir / f"contact_sheet_{chunk_index // max_rows:02d}.png"
        canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weak-teacher refiner pilot for real-photo NR.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-crops")
    p.add_argument("--output-dir", default="runs/refiner_pilot")
    p.add_argument("--crop-size", type=int, default=384)
    p.add_argument("--random-per-scene", type=int, default=8)
    p.add_argument("--teacher-match-sigma", type=float, default=28.0)
    p.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=build_crops)

    p = sub.add_parser("train")
    p.add_argument("--crop-dir", default="runs/refiner_pilot/crops")
    p.add_argument("--output-dir", default="runs/refiner_pilot/train")
    p.add_argument("--device", default="auto")
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=160)
    p.add_argument("--model-kind", choices=["residual", "gated", "chroma"], default="residual")
    p.add_argument("--feature-set", choices=["basic", "texture"], default="basic")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--max-delta", type=float, default=0.050)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--weak-weight", type=float, default=1.0)
    p.add_argument("--teacher-rgb-weight", type=float, default=1.0)
    p.add_argument("--noise-gate-weight", type=float, default=0.0)
    p.add_argument("--dark-dot-weight", type=float, default=0.0)
    p.add_argument("--chroma-noise-weight", type=float, default=0.0)
    p.add_argument("--dark-dot-chroma-weight", type=float, default=0.0)
    p.add_argument("--chroma-weight-clip", type=float, default=1.0)
    p.add_argument("--protect-weight", type=float, default=4.0)
    p.add_argument("--identity-weight", type=float, default=0.18)
    p.add_argument("--chroma-weight", type=float, default=1.2)
    p.add_argument("--luma-change-weight", type=float, default=0.0)
    p.add_argument("--luma-protect-weight", type=float, default=3.0)
    p.add_argument("--gate-weight", type=float, default=0.0)
    p.add_argument("--gate-sparsity-weight", type=float, default=0.0)
    p.add_argument("--tv-weight", type=float, default=0.035)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=train)

    p = sub.add_parser("apply-crops")
    p.add_argument("--crop-dir", default="runs/refiner_pilot/crops")
    p.add_argument("--checkpoint", default="runs/refiner_pilot/train/refiner_pilot_final.pt")
    p.add_argument("--output-dir", default="runs/refiner_pilot/eval_crops")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-rows", type=int, default=5)
    p.set_defaults(func=apply_crops)

    p = sub.add_parser("apply-full")
    p.add_argument("--scene", choices=sorted(SCENE_BY_NAME), default=None)
    p.add_argument("--noisy", default=None)
    p.add_argument("--current", default=None)
    p.add_argument("--checkpoint", default="runs/refiner_pilot/train_chroma_luma_guard/refiner_pilot_final.pt")
    p.add_argument("--output-dir", default="runs/refiner_pilot/full_apply")
    p.add_argument("--name", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--tile-overlap", type=int, default=96)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.set_defaults(func=apply_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
