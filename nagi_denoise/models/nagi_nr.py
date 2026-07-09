"""Nagi NR — Lightweight HDR blind denoiser.

A 3-level U-shape network using NAFNet-style activation-free blocks.
Operates internally in compressed (asinh) HDR space; trains and infers in float32.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Per-channel LayerNorm for NCHW tensors (channel-last semantics)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """NAFNet's parameter-free gate: split along channels and multiply."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class SCA(nn.Module):
    """Simplified Channel Attention (no sigmoid)."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class NAFLiteBlock(nn.Module):
    """NAFNet-style block.

    Token-mixing sub-block (always on):
        LN -> 1x1 expand -> DWConv3x3 -> SimpleGate -> SCA -> 1x1, beta residual.

    Optional FFN (channel-mixing) sub-block, restoring the full NAFNet design:
        LN -> 1x1 expand -> SimpleGate -> 1x1, gamma residual.

    The "Lite" variant (ffn=False) drops the FFN to save params; setting
    ffn=True recovers the canonical NAFNet block. Both residuals are scaled by
    a learnable per-channel factor initialized to 0 for a safe identity start.
    """

    def __init__(self, channels: int, expansion: int = 2, ffn: bool = False, ffn_expansion: int = 2):
        super().__init__()
        inner = channels * expansion
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, inner, 1)
        self.dw = nn.Conv2d(inner, inner, 3, padding=1, groups=inner)
        self.gate = SimpleGate()  # inner -> inner/2 (== channels when expansion=2)
        self.sca = SCA(inner // 2)
        self.pw2 = nn.Conv2d(inner // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.use_ffn = bool(ffn)
        if self.use_ffn:
            f_inner = channels * ffn_expansion
            self.norm2 = LayerNorm2d(channels)
            self.ffn_pw1 = nn.Conv2d(channels, f_inner, 1)
            self.ffn_gate = SimpleGate()  # f_inner -> f_inner/2
            self.ffn_pw2 = nn.Conv2d(f_inner // 2, channels, 1)
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.gate(y)
        y = self.sca(y)
        y = self.pw2(y)
        x = x + y * self.beta

        if self.use_ffn:
            z = self.norm2(x)
            z = self.ffn_pw1(z)
            z = self.ffn_gate(z)
            z = self.ffn_pw2(z)
            x = x + z * self.gamma
        return x


def _stack(channels: int, n: int, ffn: bool = False) -> nn.Sequential:
    return nn.Sequential(*[NAFLiteBlock(channels, ffn=ffn) for _ in range(n)])


class NagiNR(nn.Module):
    """Blind HDR denoiser.

    Pipeline (in-graph):
        x (linear float32) -> asinh compress -> PixelUnshuffle(2)
        -> U-Net (3 levels) producing residual in compressed space
        -> PixelShuffle(2) -> add to compressed input -> sinh decompress -> y

    Args:
        in_channels: input channel count (3 for RGB).
        base_channels: width at the top level. Doubled at each downsample.
        num_blocks: NAFLiteBlock counts per stage in encoder->bottleneck->decoder order.
            Default (2, 2, 4, 2, 2): enc1, enc2, bottleneck, dec2, dec1.
        asinh_k: HDR compression curvature. Larger k = stronger compression.
    """

    SIZE_MULTIPLE = 8  # input H,W must be divisible by this

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_blocks: tuple = (2, 2, 4, 2, 2),
        asinh_k: float = 8.0,
        block_ffn: bool = False,
    ):
        super().__init__()
        if len(num_blocks) != 5:
            raise ValueError("num_blocks must be a 5-tuple (enc1, enc2, bot, dec2, dec1)")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = tuple(num_blocks)
        self.asinh_k = float(asinh_k)
        self.block_ffn = bool(block_ffn)
        # Constant normalization factor for compression (registered as buffer for export).
        self.register_buffer(
            "_asinh_norm",
            torch.tensor(math.asinh(self.asinh_k), dtype=torch.float32),
        )

        C = base_channels
        # Stem: pixel-unshuffle 2x then 1x1 to base width.
        self.unshuffle = nn.PixelUnshuffle(2)
        self.stem = nn.Conv2d(in_channels * 4, C, 1)

        ffn = self.block_ffn
        # Encoder
        self.enc1 = _stack(C, num_blocks[0], ffn=ffn)
        self.down1 = nn.Conv2d(C, C * 2, 2, stride=2)
        self.enc2 = _stack(C * 2, num_blocks[1], ffn=ffn)
        self.down2 = nn.Conv2d(C * 2, C * 4, 2, stride=2)

        # Bottleneck
        self.bottleneck = _stack(C * 4, num_blocks[2], ffn=ffn)

        # Decoder (PixelShuffle upsampling)
        self.up2 = nn.Sequential(
            nn.Conv2d(C * 4, C * 2 * 4, 1),
            nn.PixelShuffle(2),
        )
        self.dec2 = _stack(C * 2, num_blocks[3], ffn=ffn)
        self.up1 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 1),
            nn.PixelShuffle(2),
        )
        self.dec1 = _stack(C, num_blocks[4], ffn=ffn)

        # Head: produce residual in compressed space at H/2, then PixelShuffle to full res.
        self.head = nn.Conv2d(C, in_channels * 4, 1)
        self.shuffle = nn.PixelShuffle(2)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Force head bias 0 and weights tiny — start near identity (residual ~ 0).
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    # ---- HDR range compression (deterministic, in-graph) ----
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(x * self.asinh_k) / self._asinh_norm

    def decompress(self, x_c: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x_c * self._asinh_norm) / self.asinh_k

    # ---- Forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) linear float32. H, W must be divisible by 8."""
        H, W = x.shape[-2:]
        if H % self.SIZE_MULTIPLE or W % self.SIZE_MULTIPLE:
            raise ValueError(
                f"Spatial dims must be multiples of {self.SIZE_MULTIPLE}; got {H}x{W}. "
                "Use Denoiser.__call__ for arbitrary sizes (it pads internally)."
            )

        x_c = self.compress(x)

        f = self.unshuffle(x_c)
        f = self.stem(f)
        s1 = self.enc1(f)
        f = self.down1(s1)
        s2 = self.enc2(f)
        f = self.down2(s2)
        f = self.bottleneck(f)
        f = self.up2(f)
        f = self.dec2(f + s2)
        f = self.up1(f)
        f = self.dec1(f + s1)

        r = self.shuffle(self.head(f))  # residual in compressed space, full res
        y_c = x_c + r
        return self.decompress(y_c)

    # ---- Utility ----
    @torch.no_grad()
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
