"""Nagi-RealFast: practical sRGB denoiser for local MPS/Core ML experiments."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LayerNorm2d, SimpleGate


def _round_even(value: float) -> int:
    out = int(round(float(value)))
    out = max(2, out)
    return out if out % 2 == 0 else out + 1


class LocalDenoiseBlock(nn.Module):
    """Cheap local denoising block for full and half resolution stages."""

    def __init__(self, channels: int, expand: float = 1.25, residual_init: float = 0.0):
        super().__init__()
        hidden = _round_even(channels * expand)
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(hidden // 2, channels, 1, bias=True)
        self.beta = nn.Parameter(torch.full((1, channels, 1, 1), float(residual_init)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm(inp)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = self.conv2(x)
        return inp + x * self.beta


class ContextBlock(nn.Module):
    """Richer low-resolution block with the FFN branch kept where it is cheaper."""

    def __init__(
        self,
        channels: int,
        local_expand: float = 2.0,
        ffn_expand: float = 2.0,
        residual_init: float = 0.0,
    ):
        super().__init__()
        self.local = LocalDenoiseBlock(channels, expand=local_expand, residual_init=residual_init)
        hidden = _round_even(channels * ffn_expand)
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(hidden // 2, channels, 1, bias=True)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(residual_init)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        y = self.local(inp)
        x = self.norm(y)
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        return y + x * self.gamma


class NoiseGuideHead(nn.Module):
    """Small residual-gain guide head.

    The first three maps are kept available for future diagnostics. v0 only uses
    the fourth map as a residual gain, avoiding complicated routing before the
    first speed and learning gates.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
            nn.Conv2d(channels, max(8, channels // 4), 1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(8, channels // 4), 4, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _local_stack(channels: int, count: int, expand: float, residual_init: float) -> nn.Sequential:
    return nn.Sequential(
        *[LocalDenoiseBlock(channels, expand=expand, residual_init=residual_init) for _ in range(int(count))]
    )


def _context_stack(
    channels: int,
    count: int,
    local_expand: float,
    ffn_expand: float,
    residual_init: float,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            ContextBlock(
                channels,
                local_expand=local_expand,
                ffn_expand=ffn_expand,
                residual_init=residual_init,
            )
            for _ in range(int(count))
        ]
    )


def _hybrid_stack(
    channels: int,
    count: int,
    local_expand: float,
    ffn_expand: float,
    residual_init: float,
) -> nn.Sequential:
    count = int(count)
    if count <= 0:
        return nn.Sequential()
    blocks: list[nn.Module] = [
        LocalDenoiseBlock(channels, expand=local_expand, residual_init=residual_init)
        for _ in range(max(0, count - 1))
    ]
    blocks.append(
        ContextBlock(
            channels,
            local_expand=local_expand,
            ffn_expand=ffn_expand,
            residual_init=residual_init,
        )
    )
    return nn.Sequential(*blocks)


class NagiRealFast(nn.Module):
    """sRGB residual denoiser tuned for practical speed/quality gates."""

    def __init__(
        self,
        img_channels: int = 3,
        width: int = 48,
        enc_blk_nums: tuple[int, ...] = (2, 3, 4),
        middle_blk_num: int = 6,
        dec_blk_nums: tuple[int, ...] = (2, 2, 1),
        high_expand: float = 1.25,
        high_ffn_expand: float = 0.0,
        high_ffn_enc_stages: tuple[int, ...] = (),
        high_ffn_dec_stages: tuple[int, ...] = (),
        low_expand: float = 2.0,
        ffn_expand: float = 2.0,
        residual_init: float = 0.0,
        ending_init_std: float = 0.0,
    ):
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc_blk_nums and dec_blk_nums must have the same length")

        self.img_channels = int(img_channels)
        self.width = int(width)
        self.enc_blk_nums = tuple(int(x) for x in enc_blk_nums)
        self.middle_blk_num = int(middle_blk_num)
        self.dec_blk_nums = tuple(int(x) for x in dec_blk_nums)
        self.high_expand = float(high_expand)
        self.high_ffn_expand = float(high_ffn_expand)
        self.high_ffn_enc_stages = tuple(int(x) for x in high_ffn_enc_stages)
        self.high_ffn_dec_stages = tuple(int(x) for x in high_ffn_dec_stages)
        self.low_expand = float(low_expand)
        self.ffn_expand = float(ffn_expand)
        self.residual_init = float(residual_init)
        self.ending_init_std = float(ending_init_std)
        self.size_multiple = 2 ** len(self.enc_blk_nums)

        self.intro = nn.Conv2d(self.img_channels, self.width, 3, padding=1, bias=True)
        self.guide = NoiseGuideHead(self.width)
        self.ending = nn.Conv2d(self.width, self.img_channels, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = self.width
        for stage, count in enumerate(self.enc_blk_nums):
            if stage < 2 and stage in self.high_ffn_enc_stages and self.high_ffn_expand > 0:
                self.encoders.append(
                    _hybrid_stack(
                        channels,
                        count,
                        local_expand=self.high_expand,
                        ffn_expand=self.high_ffn_expand,
                        residual_init=self.residual_init,
                    )
                )
            elif stage < 2:
                self.encoders.append(
                    _local_stack(channels, count, expand=self.high_expand, residual_init=self.residual_init)
                )
            else:
                self.encoders.append(
                    _context_stack(
                        channels,
                        count,
                        local_expand=self.low_expand,
                        ffn_expand=self.ffn_expand,
                        residual_init=self.residual_init,
                    )
                )
            self.downs.append(nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1, bias=True))
            channels *= 2

        self.middle_blks = _context_stack(
            channels,
            self.middle_blk_num,
            local_expand=self.low_expand,
            ffn_expand=self.ffn_expand,
            residual_init=self.residual_init,
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for stage, count in enumerate(self.dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            if stage >= len(self.dec_blk_nums) - 2 and stage in self.high_ffn_dec_stages and self.high_ffn_expand > 0:
                self.decoders.append(
                    _hybrid_stack(
                        channels,
                        count,
                        local_expand=self.high_expand,
                        ffn_expand=self.high_ffn_expand,
                        residual_init=self.residual_init,
                    )
                )
            elif stage >= len(self.dec_blk_nums) - 2:
                self.decoders.append(
                    _local_stack(channels, count, expand=self.high_expand, residual_init=self.residual_init)
                )
            else:
                self.decoders.append(
                    _context_stack(
                        channels,
                        count,
                        local_expand=self.low_expand,
                        ffn_expand=self.ffn_expand,
                        residual_init=self.residual_init,
                    )
                )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.ending_init_std > 0:
            nn.init.trunc_normal_(self.ending.weight, std=self.ending_init_std)
        else:
            nn.init.zeros_(self.ending.weight)
        if self.ending.bias is not None:
            nn.init.zeros_(self.ending.bias)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        pad_h = (self.size_multiple - h % self.size_multiple) % self.size_multiple
        pad_w = (self.size_multiple - w % self.size_multiple) % self.size_multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        h, w = inp.shape[-2:]
        x, _, _ = self._pad(inp)
        residual_base = x

        x = self.intro(x)
        guide = self.guide(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = decoder(x + skip)

        residual = self.ending(x)
        residual_gain = torch.sigmoid(guide[:, 3:4]) * 2.0
        x = residual_base + residual * residual_gain
        return x[..., :h, :w]

    @torch.no_grad()
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


REALFAST_PRESETS = {
    "realfast-v0": dict(
        width=48,
        enc_blk_nums=(2, 3, 4),
        middle_blk_num=6,
        dec_blk_nums=(2, 2, 1),
        high_expand=1.25,
        low_expand=2.0,
        ffn_expand=2.0,
    ),
    "realfast-v0-lite": dict(
        width=44,
        enc_blk_nums=(2, 2, 4),
        middle_blk_num=5,
        dec_blk_nums=(2, 1, 1),
        high_expand=1.25,
        low_expand=2.0,
        ffn_expand=2.0,
    ),
    "realfast-v1": dict(
        width=48,
        enc_blk_nums=(2, 3, 4),
        middle_blk_num=5,
        dec_blk_nums=(2, 2, 1),
        high_expand=1.25,
        high_ffn_expand=1.5,
        high_ffn_enc_stages=(0,),
        high_ffn_dec_stages=(2,),
        low_expand=2.0,
        ffn_expand=2.0,
    ),
    "realfast-v1-enc": dict(
        width=48,
        enc_blk_nums=(2, 3, 4),
        middle_blk_num=5,
        dec_blk_nums=(2, 2, 1),
        high_expand=1.25,
        high_ffn_expand=1.5,
        high_ffn_enc_stages=(0,),
        high_ffn_dec_stages=(),
        low_expand=2.0,
        ffn_expand=2.0,
    ),
    "realfast-v1-dec": dict(
        width=48,
        enc_blk_nums=(2, 3, 4),
        middle_blk_num=5,
        dec_blk_nums=(2, 2, 1),
        high_expand=1.25,
        high_ffn_expand=1.5,
        high_ffn_enc_stages=(),
        high_ffn_dec_stages=(2,),
        low_expand=2.0,
        ffn_expand=2.0,
    ),
}


def build_realfast_preset(name: str) -> NagiRealFast:
    try:
        cfg = REALFAST_PRESETS[name]
    except KeyError as exc:
        names = ", ".join(sorted(REALFAST_PRESETS))
        raise ValueError(f"unknown NagiRealFast preset {name!r}; choose one of: {names}") from exc
    return NagiRealFast(**cfg)
