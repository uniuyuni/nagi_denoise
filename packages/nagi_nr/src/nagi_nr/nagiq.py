"""NagiQ: sRGB-first NAFNet-style student for 40 dB SIDD experiments."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LayerNorm2d, SimpleGate


class NAFBlock(nn.Module):
    """Full NAFNet block with token-mixing and FFN sub-blocks.

    This is intentionally close to NAFNet's block, but kept in the core package
    so the future student model does not depend on benchmark-only vendored code.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        dw_channels = int(channels * dw_expand)
        if dw_channels % 2 != 0:
            raise ValueError("channels * dw_expand must be even for SimpleGate")
        ffn_channels = int(channels * ffn_expand)
        if ffn_channels % 2 != 0:
            raise ValueError("channels * ffn_expand must be even for SimpleGate")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels, bias=True)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1, bias=True),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1, bias=True)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1, bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


def _stack(channels: int, n_blocks: int, **block_kwargs) -> nn.Sequential:
    return nn.Sequential(*[NAFBlock(channels, **block_kwargs) for _ in range(int(n_blocks))])


class NagiQ(nn.Module):
    """sRGB residual NAFNet-style student.

    The model accepts and returns sRGB tensors in [0, 1]-like float space. It is
    designed for SIDD PSNR, not HDR/linear-light processing.
    """

    def __init__(
        self,
        img_channels: int = 3,
        width: int = 48,
        enc_blk_nums: tuple[int, ...] = (2, 2, 4, 6),
        middle_blk_num: int = 8,
        dec_blk_nums: tuple[int, ...] = (2, 2, 2, 2),
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc_blk_nums and dec_blk_nums must have the same length")

        self.img_channels = int(img_channels)
        self.width = int(width)
        self.enc_blk_nums = tuple(int(x) for x in enc_blk_nums)
        self.middle_blk_num = int(middle_blk_num)
        self.dec_blk_nums = tuple(int(x) for x in dec_blk_nums)
        self.dw_expand = int(dw_expand)
        self.ffn_expand = int(ffn_expand)
        self.drop_out_rate = float(drop_out_rate)
        self.size_multiple = 2 ** len(self.enc_blk_nums)

        block_kwargs = dict(
            dw_expand=self.dw_expand,
            ffn_expand=self.ffn_expand,
            drop_out_rate=self.drop_out_rate,
        )

        self.intro = nn.Conv2d(self.img_channels, self.width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(self.width, self.img_channels, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = self.width
        for n_blocks in self.enc_blk_nums:
            self.encoders.append(_stack(channels, n_blocks, **block_kwargs))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2, bias=True))
            channels *= 2

        self.middle_blks = _stack(channels, self.middle_blk_num, **block_kwargs)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n_blocks in self.dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(_stack(channels, n_blocks, **block_kwargs))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Residual head starts at zero, so the whole network starts as identity.
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


NAGIQ_PRESETS = {
    "q48-trim": dict(width=48, enc_blk_nums=(2, 2, 4, 6), middle_blk_num=8, dec_blk_nums=(2, 2, 2, 2)),
    "q56-trim": dict(width=56, enc_blk_nums=(2, 2, 4, 6), middle_blk_num=8, dec_blk_nums=(2, 2, 2, 2)),
    "q48-fast": dict(width=48, enc_blk_nums=(1, 2, 3, 5), middle_blk_num=6, dec_blk_nums=(1, 1, 2, 2)),
    "q40-trim": dict(width=40, enc_blk_nums=(2, 2, 4, 6), middle_blk_num=8, dec_blk_nums=(2, 2, 2, 2)),
}


def build_nagiq_preset(name: str) -> NagiQ:
    try:
        cfg = NAGIQ_PRESETS[name]
    except KeyError as exc:
        names = ", ".join(sorted(NAGIQ_PRESETS))
        raise ValueError(f"unknown NagiQ preset {name!r}; choose one of: {names}") from exc
    return NagiQ(**cfg)
