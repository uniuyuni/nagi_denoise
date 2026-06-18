"""GAMA-IR style sRGB restoration network.

This is a from-paper implementation of the GAMA-IR design described in
"Global Additive Multidimensional Averaging for Fast Image Restoration".
The paper does not ship public code, so this module keeps the architecture
faithful to the text while staying compatible with the local Nagi training and
Core ML export paths.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LayerNorm2d, SimpleGate


class GAMABlock(nn.Module):
    """Global Additive Multidimensional Averaging block.

    For a feature map BxCxHxW, average along C, H, and W separately. Each
    averaged 2D map is filtered with a tiny 7x7 single-channel convolution, then
    broadcast back and added to the input.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.hw = nn.Conv2d(1, 1, kernel_size, padding=pad, bias=True)
        self.cw = nn.Conv2d(1, 1, kernel_size, padding=pad, bias=True)
        self.hc = nn.Conv2d(1, 1, kernel_size, padding=pad, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        hw = x.mean(dim=1, keepdim=True)
        hw = self.hw(hw).expand(b, c, h, w)

        cw = x.mean(dim=2, keepdim=True).transpose(1, 2)
        cw = self.cw(cw).transpose(1, 2).expand(b, c, h, w)

        hc = x.mean(dim=3, keepdim=True).permute(0, 3, 2, 1)
        hc = self.hc(hc).permute(0, 3, 2, 1).expand(b, c, h, w)

        return x + hw + cw + hc


class GAMAIRBlock(nn.Module):
    """NAF-style local processing plus GAMA global mixing."""

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        gama_kernel: int = 7,
        residual_init: float = 0.0,
    ):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        if dw_channels % 2 != 0 or ffn_channels % 2 != 0:
            raise ValueError("expanded channels must be even for SimpleGate")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels, bias=True)
        self.sg = SimpleGate()
        self.gama = GAMABlock(kernel_size=gama_kernel)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1, bias=True)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1, bias=True)

        self.beta = nn.Parameter(torch.full((1, channels, 1, 1), float(residual_init)))
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(residual_init)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.gama(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


def _stack(channels: int, count: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(*[GAMAIRBlock(channels, **kwargs) for _ in range(int(count))])


class PixelUnshuffleDown(nn.Module):
    """GAMA-IR downsample: 1x1 channel reduction followed by pixel unshuffle."""

    def __init__(self, channels: int):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be even for GAMA-IR downsample")
        self.conv = nn.Conv2d(channels, channels // 2, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(self.conv(x), 2)


class PixelShuffleUp(nn.Module):
    """GAMA-IR upsample: 1x1 channel expansion followed by pixel shuffle."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.conv(x), 2)


def depth_to_blocks(depth: int, levels: int = 4) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    """Distribute total block depth over a 4-level encoder/decoder.

    The paper specifies depth as the total number of building blocks. Its figure
    uses a U-shaped architecture, so we use a symmetric 4-level allocation and
    place any extra block in the bottleneck.
    """

    side = [2] * levels
    middle = int(depth) - 2 * sum(side)
    if middle < 1:
        raise ValueError(f"depth {depth} is too small for {levels} GAMA-IR levels")
    return tuple(side), middle, tuple(side)


class GAMAIR(nn.Module):
    """GAMA-IR style sRGB residual network."""

    def __init__(
        self,
        img_channels: int = 3,
        width: int = 42,
        depth: int = 18,
        levels: int = 4,
        enc_blk_nums: tuple[int, ...] | None = None,
        middle_blk_num: int | None = None,
        dec_blk_nums: tuple[int, ...] | None = None,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        gama_kernel: int = 7,
        residual_init: float = 0.0,
        ending_init_std: float = 0.0,
    ):
        super().__init__()
        if enc_blk_nums is None or middle_blk_num is None or dec_blk_nums is None:
            enc, mid, dec = depth_to_blocks(depth, levels=levels)
            enc_blk_nums = enc if enc_blk_nums is None else enc_blk_nums
            middle_blk_num = mid if middle_blk_num is None else middle_blk_num
            dec_blk_nums = dec if dec_blk_nums is None else dec_blk_nums
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc_blk_nums and dec_blk_nums must have the same length")

        self.img_channels = int(img_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.levels = len(enc_blk_nums)
        self.enc_blk_nums = tuple(int(x) for x in enc_blk_nums)
        self.middle_blk_num = int(middle_blk_num)
        self.dec_blk_nums = tuple(int(x) for x in dec_blk_nums)
        self.dw_expand = int(dw_expand)
        self.ffn_expand = int(ffn_expand)
        self.gama_kernel = int(gama_kernel)
        self.residual_init = float(residual_init)
        self.ending_init_std = float(ending_init_std)
        self.size_multiple = 2 ** self.levels

        kwargs = dict(
            dw_expand=dw_expand,
            ffn_expand=ffn_expand,
            gama_kernel=gama_kernel,
            residual_init=residual_init,
        )
        self.intro = nn.Conv2d(self.img_channels, self.width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(self.width, self.img_channels, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = self.width
        for count in self.enc_blk_nums:
            self.encoders.append(_stack(channels, count, **kwargs))
            self.downs.append(PixelUnshuffleDown(channels))
            channels *= 2

        self.middle_blks = _stack(channels, self.middle_blk_num, **kwargs)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for count in self.dec_blk_nums:
            self.ups.append(PixelShuffleUp(channels))
            channels //= 2
            self.decoders.append(_stack(channels, count, **kwargs))

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
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = decoder(x + skip)

        x = self.ending(x)
        x = x + residual_base
        return x[..., :h, :w]

    @torch.no_grad()
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


GAMAIR_PRESETS = {
    "gamair-s": dict(width=42, depth=18),
    "gamair-m56": dict(width=56, depth=19),
    "gamair-l": dict(width=80, depth=19),
}


def build_gamair_preset(name: str) -> GAMAIR:
    try:
        cfg = GAMAIR_PRESETS[name]
    except KeyError as exc:
        names = ", ".join(sorted(GAMAIR_PRESETS))
        raise ValueError(f"unknown GAMAIR preset {name!r}; choose one of: {names}") from exc
    return GAMAIR(**cfg)
