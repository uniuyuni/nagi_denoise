"""ChromaGuard: tiny AI gate for HDR-safe chroma NR.

The denoising operation itself is deliberately simple and stable. The network's
job is to predict where the chroma-only smoother is allowed to act.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transforms import linear_to_srgb


def srgb_luma(x: torch.Tensor) -> torch.Tensor:
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x[:, :3] * weights).sum(dim=1, keepdim=True)


def linear_luma(x: torch.Tensor) -> torch.Tensor:
    weights = x.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (x[:, :3] * weights).sum(dim=1, keepdim=True)


def local_lowpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    pad = kernel_size // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)


def heuristic_chroma_gate(
    x_linear: torch.Tensor,
    *,
    detail_kernel_size: int = 9,
    threshold: float = 0.018,
    transition: float = 0.010,
    highlight_threshold: float = 1.0,
    highlight_transition: float = 0.2,
) -> torch.Tensor:
    """Teacher gate matching the practical chroma-NR heuristic."""

    x = x_linear.clamp_min(0.0)
    display = linear_to_srgb(x).clamp(0.0, 1.0)
    y = srgb_luma(display)
    detail = (y - local_lowpass(y, int(detail_kernel_size))).abs()
    flat = torch.sigmoid((float(threshold) - detail) / max(float(transition), 1.0e-6))
    y_linear = linear_luma(x)
    highlight = torch.sigmoid(
        (y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6)
    )
    return (flat * (1.0 - highlight)).clamp(0.0, 1.0)


def adaptive_chroma_gate(
    x_linear: torch.Tensor,
    *,
    detail_kernel_size: int = 9,
    mild_threshold: float = 0.014,
    strong_threshold: float = 0.018,
    xstrong_threshold: float = 0.024,
    transition: float = 0.010,
    highlight_threshold: float = 1.0,
    highlight_transition: float = 0.2,
    color_edge_threshold: float = 0.018,
    color_edge_transition: float = 0.010,
    chroma_threshold: float = 0.090,
    chroma_transition: float = 0.050,
    xstrong_gain: float = 1.10,
    mild_gain: float = 0.70,
) -> torch.Tensor:
    """Adaptive teacher that chooses mild/strong/xstrong chroma NR locally.

    Flat, low-saturation regions get the strongest chroma cleanup. Color
    boundaries and saturated regions are pulled toward mild to avoid bleeding
    real color detail.
    """

    x = x_linear.clamp_min(0.0)
    display = linear_to_srgb(x).clamp(0.0, 1.0)
    y = srgb_luma(display)
    chroma = display - y
    detail = (y - local_lowpass(y, int(detail_kernel_size))).abs()
    chroma_mag = chroma.abs().mean(dim=1, keepdim=True)
    chroma_detail = (chroma - local_lowpass(chroma, int(detail_kernel_size))).abs().mean(dim=1, keepdim=True)

    mild = torch.sigmoid((float(mild_threshold) - detail) / max(float(transition), 1.0e-6))
    strong = torch.sigmoid((float(strong_threshold) - detail) / max(float(transition), 1.0e-6))
    xstrong = torch.sigmoid((float(xstrong_threshold) - detail) / max(float(transition), 1.0e-6))

    color_edge = torch.sigmoid(
        (chroma_detail - float(color_edge_threshold)) / max(float(color_edge_transition), 1.0e-6)
    )
    saturated = torch.sigmoid((chroma_mag - float(chroma_threshold)) / max(float(chroma_transition), 1.0e-6))
    risky_color = torch.maximum(color_edge, saturated)

    safe_flat = (1.0 - risky_color) * xstrong * float(xstrong_gain)
    risky_flat = risky_color * mild * float(mild_gain)
    gate = safe_flat + risky_flat

    y_linear = linear_luma(x)
    highlight = torch.sigmoid(
        (y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6)
    )
    return (gate * (1.0 - highlight)).clamp(0.0, 1.0)


def adaptive_luma_gate(
    x_linear: torch.Tensor,
    *,
    structure_kernel_size: int = 9,
    detail_kernel_size: int = 19,
    detail_threshold: float = 0.010,
    detail_transition: float = 0.006,
    edge_threshold: float = 0.018,
    edge_transition: float = 0.010,
    highlight_threshold: float = 1.0,
    highlight_transition: float = 0.25,
) -> torch.Tensor:
    """Teacher gate for guided luma NR.

    This mirrors the practical ``luma_flat_strong`` diagnostic: allow luma
    smoothing in flat noisy areas, but back off around coherent edges, hairs,
    eyes, whiskers, and true highlights.
    """

    x = x_linear.clamp_min(0.0)
    display = linear_to_srgb(x).clamp(0.0, 1.0)
    y = srgb_luma(display)
    structure = local_lowpass(y, int(structure_kernel_size))
    detail = (structure - local_lowpass(structure, int(detail_kernel_size))).abs()

    dx = F.pad(structure[:, :, :, 1:] - structure[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
    dy = F.pad(structure[:, :, 1:, :] - structure[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
    edge = torch.sqrt(dx.pow(2) + dy.pow(2) + 1.0e-12)

    flat = torch.sigmoid((float(detail_threshold) - detail) / max(float(detail_transition), 1.0e-6))
    non_edge = torch.sigmoid((float(edge_threshold) - edge) / max(float(edge_transition), 1.0e-6))
    y_linear = linear_luma(x)
    highlight = torch.sigmoid(
        (y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6)
    )
    return (flat * non_edge * (1.0 - highlight)).clamp(0.0, 1.0)


def paired_luma_noise_gate(
    noisy_linear: torch.Tensor,
    clean_linear: torch.Tensor,
    *,
    structure_kernel_size: int = 9,
    detail_kernel_size: int = 13,
    noise_kernel_size: int = 5,
    noise_threshold: float = 0.006,
    noise_transition: float = 0.005,
    clean_detail_threshold: float = 0.010,
    clean_detail_transition: float = 0.006,
    edge_threshold: float = 0.018,
    edge_transition: float = 0.010,
    highlight_threshold: float = 1.0,
    highlight_transition: float = 0.25,
) -> torch.Tensor:
    """Paired teacher gate for visible luma noise.

    Unlike the heuristic luma teacher, this uses the clean target to separate
    true texture from noisy luma variation. The gate is high where noisy-clean
    residual is visible, while clean-side detail/edges/highlights suppress it.
    """

    noisy = noisy_linear.clamp_min(0.0)
    clean = clean_linear.clamp_min(0.0)
    noisy_display = linear_to_srgb(noisy).clamp(0.0, 1.0)
    clean_display = linear_to_srgb(clean).clamp(0.0, 1.0)
    noisy_y = srgb_luma(noisy_display)
    clean_y = srgb_luma(clean_display)

    residual = noisy_y - clean_y
    residual_hf = residual - local_lowpass(residual, int(noise_kernel_size))
    visible_noise = torch.maximum(residual.abs(), residual_hf.abs())
    noise = torch.sigmoid(
        (visible_noise - float(noise_threshold)) / max(float(noise_transition), 1.0e-6)
    )

    clean_structure = local_lowpass(clean_y, int(structure_kernel_size))
    clean_detail = (clean_structure - local_lowpass(clean_structure, int(detail_kernel_size))).abs()
    flat = torch.sigmoid(
        (float(clean_detail_threshold) - clean_detail) / max(float(clean_detail_transition), 1.0e-6)
    )

    dx = F.pad(
        clean_structure[:, :, :, 1:] - clean_structure[:, :, :, :-1],
        (0, 1, 0, 0),
        mode="replicate",
    )
    dy = F.pad(
        clean_structure[:, :, 1:, :] - clean_structure[:, :, :-1, :],
        (0, 0, 0, 1),
        mode="replicate",
    )
    edge = torch.sqrt(dx.pow(2) + dy.pow(2) + 1.0e-12)
    non_edge = torch.sigmoid((float(edge_threshold) - edge) / max(float(edge_transition), 1.0e-6))

    clean_y_linear = linear_luma(clean)
    highlight = torch.sigmoid(
        (clean_y_linear - float(highlight_threshold)) / max(float(highlight_transition), 1.0e-6)
    )
    return (noise * flat * non_edge * (1.0 - highlight)).clamp(0.0, 1.0)


class ChromaGuardBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) * self.scale


class ChromaGuard(nn.Module):
    """Small fully-convolutional strength-map predictor."""

    def __init__(self, width: int = 16, blocks: int = 4, gate_bias: float = 0.0):
        super().__init__()
        self.width = int(width)
        self.blocks = int(blocks)
        self.gate_bias = float(gate_bias)
        # Features: display RGB, display luma, chroma magnitude, luma detail,
        # linear luma. These make the tiny CNN easier to train.
        self.intro = nn.Conv2d(7, self.width, 3, padding=1)
        self.body = nn.Sequential(*[ChromaGuardBlock(self.width) for _ in range(self.blocks)])
        self.head = nn.Conv2d(self.width, 1, 3, padding=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, self.gate_bias)

    def features(self, x_linear: torch.Tensor) -> torch.Tensor:
        x = x_linear.clamp_min(0.0)
        display = linear_to_srgb(x).clamp(0.0, 1.0)
        y = srgb_luma(display)
        chroma_mag = (display - y).abs().mean(dim=1, keepdim=True)
        detail = (y - local_lowpass(y, 9)).abs()
        y_linear = linear_luma(x).clamp(0.0, 4.0) / 4.0
        return torch.cat([display, y, chroma_mag, detail, y_linear], dim=1)

    def forward(self, x_linear: torch.Tensor) -> torch.Tensor:
        feat = self.intro(self.features(x_linear))
        feat = self.body(feat)
        return torch.sigmoid(self.head(feat))
