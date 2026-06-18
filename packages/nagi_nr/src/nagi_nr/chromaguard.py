"""ChromaGuard: tiny AI gate for HDR-safe chroma NR.

The denoising operation itself is deliberately simple and stable. The network's
job is to predict where the chroma-only smoother is allowed to act.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import linear_to_srgb


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
