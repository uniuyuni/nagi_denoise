"""NagiPerfect: HDR-first denoiser with a separate luma-detail path.

The first design goal is not novelty for its own sake. It encodes the strongest
lesson from the Perfect NR diagnostics: denoising, chroma stability, and
highlight detail should not all be owned by a single RGB residual head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LayerNorm2d, SimpleGate
from .transforms import linear_to_srgb, srgb_to_linear


class NagiPerfectBlock(nn.Module):
    """NAF-style local block with identity-start residual scales."""

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channels = int(channels * dw_expand)
        ffn_channels = int(channels * ffn_expand)
        if dw_channels % 2 or ffn_channels % 2:
            raise ValueError("expanded channels must be even for SimpleGate")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.dw = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, 1)

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, ffn_channels, 1)
        self.ffn2 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dw(y)
        y = self.gate(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.gate(y)
        y = self.ffn2(y)
        return x + y * self.gamma


def _stack(channels: int, n_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[NagiPerfectBlock(channels) for _ in range(int(n_blocks))])


def linear_luma(x: torch.Tensor) -> torch.Tensor:
    weights = x.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (x[:, :3] * weights).sum(dim=1, keepdim=True)


def apply_luma_detail(base: torch.Tensor, detail: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    """Apply a luma-only residual while preserving base chroma."""

    base_nonneg = base.clamp_min(0.0)
    base_luma = linear_luma(base_nonneg)
    applied = detail * confidence
    target_luma = (base_luma + applied).clamp_min(0.0)
    chroma = base_nonneg / base_luma.clamp_min(1.0e-6)
    out = chroma * target_luma
    return torch.where(base_luma > 1.0e-6, out, base_nonneg)


def local_lowpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    pad = kernel_size // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=kernel_size, stride=1)


def srgb_luma(x: torch.Tensor) -> torch.Tensor:
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x[:, :3] * weights).sum(dim=1, keepdim=True)


def input_highlight_guard(
    denoised: torch.Tensor,
    inp: torch.Tensor,
    threshold: float,
    transition: float,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend high-input-luma pixels back toward the input.

    This is a structural HDR safety valve: above the configured input luma
    threshold the network progressively loses permission to alter RGB. It is
    deliberately input-based, not target-based, so it also works at inference
    time on real HDR EXR frames where the training distribution is least
    trustworthy.
    """

    strength = float(strength)
    if strength <= 0.0:
        mask = denoised.new_zeros((denoised.shape[0], 1, denoised.shape[-2], denoised.shape[-1]))
        return denoised, mask

    inp_nonneg = inp.clamp_min(0.0)
    y = linear_luma(inp_nonneg)
    if transition <= 0.0:
        mask = (y > threshold).to(dtype=denoised.dtype)
    else:
        mask = torch.sigmoid((y - threshold) / max(float(transition), 1.0e-6))
    mask = (mask * min(max(strength, 0.0), 1.0)).to(dtype=denoised.dtype)
    return denoised * (1.0 - mask) + inp_nonneg * mask, mask


class NagiPerfect(nn.Module):
    """HDR denoiser with explicit base RGB, luma detail, and confidence heads.

    Forward input/output is linear RGB float. Internally the shared trunk works
    in HDR-white-aware asinh space. The base head predicts an RGB residual in
    compressed space. The detail head predicts a linear-light luma residual that
    is gated by a confidence head and then applied without changing chroma.
    """

    def __init__(
        self,
        img_channels: int = 3,
        width: int = 32,
        enc_blk_nums: tuple[int, ...] = (1, 2, 3),
        middle_blk_num: int = 4,
        dec_blk_nums: tuple[int, ...] = (1, 1, 2),
        asinh_k: float = 8.0,
        detail_scale: float = 0.25,
        compressed_output_clamp: float = 2.0,
        confidence_bias: float = -2.0,
        highlight_protect_threshold: float = 0.0,
        highlight_protect_transition: float = 0.5,
        highlight_protect_strength: float = 0.0,
        chroma_branch: bool = False,
        chroma_branch_scale: float = 0.05,
        chroma_gate_threshold: float = 0.012,
        chroma_gate_transition: float = 0.004,
        chroma_gate_kernel_size: int = 9,
        chroma_gate_highlight_threshold: float = 1.0,
        chroma_gate_highlight_transition: float = 0.2,
        chroma_gate_strength: float = 1.0,
        chroma_branch_width: int = 0,
        chroma_branch_blocks: int = 0,
        chroma_branch_use_input: bool = False,
        chroma_smooth_branch: bool = False,
        chroma_smooth_strength: float = 0.5,
        chroma_smooth_kernel_size: int = 9,
        chroma_smooth_gate_bias: float = -4.0,
        chroma_smooth_gate_scale: float = 1.0,
        luma_smooth_branch: bool = False,
        luma_smooth_strength: float = 0.4,
        luma_smooth_kernel_size: int = 5,
        luma_smooth_gate_bias: float = -4.0,
        luma_smooth_gate_scale: float = 1.0,
    ):
        super().__init__()
        if img_channels != 3:
            raise ValueError("NagiPerfect currently expects RGB input")
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc_blk_nums and dec_blk_nums must have the same length")

        self.img_channels = int(img_channels)
        self.width = int(width)
        self.enc_blk_nums = tuple(int(x) for x in enc_blk_nums)
        self.middle_blk_num = int(middle_blk_num)
        self.dec_blk_nums = tuple(int(x) for x in dec_blk_nums)
        self.asinh_k = float(asinh_k)
        self.detail_scale = float(detail_scale)
        self.compressed_output_clamp = float(compressed_output_clamp)
        self.highlight_protect_threshold = float(highlight_protect_threshold)
        self.highlight_protect_transition = float(highlight_protect_transition)
        self.highlight_protect_strength = float(highlight_protect_strength)
        self.chroma_branch = bool(chroma_branch)
        self.chroma_branch_scale = float(chroma_branch_scale)
        self.chroma_gate_threshold = float(chroma_gate_threshold)
        self.chroma_gate_transition = float(chroma_gate_transition)
        self.chroma_gate_kernel_size = int(chroma_gate_kernel_size)
        self.chroma_gate_highlight_threshold = float(chroma_gate_highlight_threshold)
        self.chroma_gate_highlight_transition = float(chroma_gate_highlight_transition)
        self.chroma_gate_strength = float(chroma_gate_strength)
        self.chroma_branch_width = int(chroma_branch_width)
        self.chroma_branch_blocks = int(chroma_branch_blocks)
        self.chroma_branch_use_input = bool(chroma_branch_use_input)
        self.chroma_smooth_branch = bool(chroma_smooth_branch)
        self.chroma_smooth_strength = float(chroma_smooth_strength)
        self.chroma_smooth_kernel_size = int(chroma_smooth_kernel_size)
        self.chroma_smooth_gate_bias = float(chroma_smooth_gate_bias)
        self.chroma_smooth_gate_scale = float(chroma_smooth_gate_scale)
        self.luma_smooth_branch = bool(luma_smooth_branch)
        self.luma_smooth_strength = float(luma_smooth_strength)
        self.luma_smooth_kernel_size = int(luma_smooth_kernel_size)
        self.luma_smooth_gate_bias = float(luma_smooth_gate_bias)
        self.luma_smooth_gate_scale = float(luma_smooth_gate_scale)
        self.size_multiple = 2 ** len(self.enc_blk_nums)
        self.register_buffer("_asinh_norm", torch.tensor(math.asinh(self.asinh_k), dtype=torch.float32))

        self.intro = nn.Conv2d(self.img_channels, self.width, 3, padding=1)
        channels = self.width
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for n_blocks in self.enc_blk_nums:
            self.encoders.append(_stack(channels, n_blocks))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle = _stack(channels, self.middle_blk_num)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n_blocks in self.dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(_stack(channels, n_blocks))

        self.base_head = nn.Conv2d(channels, self.img_channels, 3, padding=1)
        self.detail_head = nn.Conv2d(channels, 1, 3, padding=1)
        self.confidence_head = nn.Conv2d(channels, 1, 3, padding=1)
        self.chroma_head = self._build_chroma_head(channels) if self.chroma_branch else None
        self.chroma_smooth_head = nn.Conv2d(channels, 1, 3, padding=1) if self.chroma_smooth_branch else None
        self.luma_smooth_head = nn.Conv2d(channels, 1, 3, padding=1) if self.luma_smooth_branch else None
        self._confidence_bias = float(confidence_bias)
        self._init_weights()

    def _build_chroma_head(self, channels: int) -> nn.Module:
        in_channels = int(channels) + (self.img_channels if self.chroma_branch_use_input else 0)
        branch_width = self.chroma_branch_width
        branch_blocks = self.chroma_branch_blocks
        if branch_width <= 0 and branch_blocks <= 0 and not self.chroma_branch_use_input:
            return nn.Conv2d(in_channels, self.img_channels, 3, padding=1)

        if branch_width <= 0:
            branch_width = max(8, min(int(channels), 24))
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, branch_width, 3, padding=1),
        ]
        layers.extend(NagiPerfectBlock(branch_width) for _ in range(max(0, branch_blocks)))
        layers.append(nn.Conv2d(branch_width, self.img_channels, 3, padding=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.base_head.weight)
        nn.init.zeros_(self.detail_head.weight)
        nn.init.zeros_(self.confidence_head.weight)
        if self.chroma_head is not None:
            final_chroma = self.chroma_head
            if isinstance(final_chroma, nn.Sequential):
                final_chroma = final_chroma[-1]
            if isinstance(final_chroma, nn.Conv2d):
                nn.init.zeros_(final_chroma.weight)
        if self.chroma_smooth_head is not None:
            nn.init.zeros_(self.chroma_smooth_head.weight)
        if self.luma_smooth_head is not None:
            nn.init.zeros_(self.luma_smooth_head.weight)
        if self.base_head.bias is not None:
            nn.init.zeros_(self.base_head.bias)
        if self.detail_head.bias is not None:
            nn.init.zeros_(self.detail_head.bias)
        if self.confidence_head.bias is not None:
            nn.init.constant_(self.confidence_head.bias, self._confidence_bias)
        if self.chroma_head is not None:
            final_chroma = self.chroma_head
            if isinstance(final_chroma, nn.Sequential):
                final_chroma = final_chroma[-1]
            if isinstance(final_chroma, nn.Conv2d) and final_chroma.bias is not None:
                nn.init.zeros_(final_chroma.bias)
        if self.chroma_smooth_head is not None and self.chroma_smooth_head.bias is not None:
            nn.init.constant_(self.chroma_smooth_head.bias, self.chroma_smooth_gate_bias)
        if self.luma_smooth_head is not None and self.luma_smooth_head.bias is not None:
            nn.init.constant_(self.luma_smooth_head.bias, self.luma_smooth_gate_bias)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(x * self.asinh_k) / self._asinh_norm

    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x * self._asinh_norm) / self.asinh_k

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        pad_h = (self.size_multiple - h % self.size_multiple) % self.size_multiple
        pad_w = (self.size_multiple - w % self.size_multiple) % self.size_multiple
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def _input_flat_chroma_gate(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not (self.chroma_branch or self.chroma_smooth_branch or self.luma_smooth_branch)
            or self.chroma_gate_strength <= 0.0
        ):
            return x.new_zeros((x.shape[0], 1, x.shape[-2], x.shape[-1]))

        display = linear_to_srgb(x.clamp_min(0.0))
        y_display = srgb_luma(display)
        detail = (y_display - local_lowpass(y_display, self.chroma_gate_kernel_size)).abs()
        flat = torch.sigmoid(
            (self.chroma_gate_threshold - detail) / max(self.chroma_gate_transition, 1.0e-6)
        )

        y_linear = linear_luma(x.clamp_min(0.0))
        highlight = torch.sigmoid(
            (y_linear - self.chroma_gate_highlight_threshold)
            / max(self.chroma_gate_highlight_transition, 1.0e-6)
        )
        gate = flat * (1.0 - highlight)
        return gate * min(max(self.chroma_gate_strength, 0.0), 1.0)

    def _apply_chroma_residual(
        self,
        output: torch.Tensor,
        chroma_raw: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        residual = chroma_raw - linear_luma(chroma_raw)
        residual = residual * (self.chroma_branch_scale * gate)
        return (output + residual).clamp_min(0.0)

    def _apply_chroma_smoothing(
        self,
        output: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.chroma_smooth_strength <= 0.0:
            return output, output.new_zeros(output.shape)

        output_nonneg = output.clamp_min(0.0)
        display = linear_to_srgb(output_nonneg)
        y = srgb_luma(display)
        chroma = display - y
        smooth_chroma = local_lowpass(chroma, self.chroma_smooth_kernel_size)
        blend = gate * min(max(self.chroma_smooth_strength, 0.0), 1.0)
        smoothed_display = y + chroma * (1.0 - blend) + smooth_chroma * blend
        smoothed_linear = srgb_to_linear(smoothed_display.clamp_min(0.0)).clamp_min(0.0)
        return smoothed_linear, smoothed_linear - output_nonneg

    def _apply_luma_smoothing(
        self,
        output: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.luma_smooth_strength <= 0.0:
            return output, output.new_zeros(output.shape)

        output_nonneg = output.clamp_min(0.0)
        display = linear_to_srgb(output_nonneg)
        y = srgb_luma(display)
        chroma = display - y
        smooth_y = local_lowpass(y, self.luma_smooth_kernel_size)
        blend = gate * min(max(self.luma_smooth_strength, 0.0), 1.0)
        smoothed_display = y * (1.0 - blend) + smooth_y * blend + chroma
        smoothed_linear = srgb_to_linear(smoothed_display.clamp_min(0.0)).clamp_min(0.0)
        return smoothed_linear, smoothed_linear - output_nonneg

    def forward(self, inp: torch.Tensor, return_aux: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        h, w = inp.shape[-2:]
        x, _, _ = self._pad(inp)
        x_c = self.compress(x)

        feat = self.intro(x_c)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            feat = up(feat)
            feat = decoder(feat + skip)

        base_c = x_c + self.base_head(feat)
        if self.compressed_output_clamp > 0:
            base_c = base_c.clamp(-self.compressed_output_clamp, self.compressed_output_clamp)
        base = self.decompress(base_c)
        detail = self.detail_head(feat) * self.detail_scale
        confidence = torch.sigmoid(self.confidence_head(feat))
        output_pre_guard = apply_luma_detail(base, detail, confidence)
        output_pre_chroma = output_pre_guard
        chroma_residual = None
        chroma_gate = None
        chroma_smooth_gate = None
        chroma_smooth_delta = None
        luma_smooth_gate = None
        luma_smooth_delta = None
        if self.chroma_head is not None:
            chroma_feat = feat
            if self.chroma_branch_use_input:
                chroma_feat = torch.cat([feat, x_c], dim=1)
            chroma_raw = self.chroma_head(chroma_feat)
            chroma_gate = self._input_flat_chroma_gate(x)
            chroma_residual = (chroma_raw - linear_luma(chroma_raw)) * (self.chroma_branch_scale * chroma_gate)
            output_pre_guard = (output_pre_guard + chroma_residual).clamp_min(0.0)
        if self.chroma_smooth_head is not None:
            smooth_gate_raw = self.chroma_smooth_head(feat)
            chroma_smooth_gate = torch.sigmoid(smooth_gate_raw * self.chroma_smooth_gate_scale)
            chroma_smooth_gate = chroma_smooth_gate * self._input_flat_chroma_gate(x)
            output_pre_guard, chroma_smooth_delta = self._apply_chroma_smoothing(
                output_pre_guard,
                chroma_smooth_gate,
            )
            chroma_residual = chroma_smooth_delta if chroma_residual is None else chroma_residual + chroma_smooth_delta
            chroma_gate = chroma_smooth_gate if chroma_gate is None else torch.maximum(chroma_gate, chroma_smooth_gate)
        if self.luma_smooth_head is not None:
            luma_smooth_raw = self.luma_smooth_head(feat)
            luma_smooth_gate = torch.sigmoid(luma_smooth_raw * self.luma_smooth_gate_scale)
            luma_smooth_gate = luma_smooth_gate * self._input_flat_chroma_gate(x)
            output_pre_guard, luma_smooth_delta = self._apply_luma_smoothing(
                output_pre_guard,
                luma_smooth_gate,
            )
        output, highlight_guard = input_highlight_guard(
            output_pre_guard,
            x,
            threshold=self.highlight_protect_threshold,
            transition=self.highlight_protect_transition,
            strength=self.highlight_protect_strength,
        )

        output = output[..., :h, :w]
        if not return_aux:
            return output
        aux = {
            "output": output,
            "output_pre_guard": output_pre_guard[..., :h, :w],
            "output_pre_chroma": output_pre_chroma[..., :h, :w],
            "base": base[..., :h, :w],
            "detail": detail[..., :h, :w],
            "detail_confidence": confidence[..., :h, :w],
            "detail_applied": (detail * confidence)[..., :h, :w],
            "highlight_guard": highlight_guard[..., :h, :w],
        }
        if chroma_residual is not None and chroma_gate is not None:
            aux.update(
                {
                    "chroma_residual": chroma_residual[..., :h, :w],
                    "chroma_gate": chroma_gate[..., :h, :w],
                }
            )
        if chroma_smooth_gate is not None and chroma_smooth_delta is not None:
            aux.update(
                {
                    "chroma_smooth_gate": chroma_smooth_gate[..., :h, :w],
                    "chroma_smooth_delta": chroma_smooth_delta[..., :h, :w],
                }
            )
        if luma_smooth_gate is not None and luma_smooth_delta is not None:
            aux.update(
                {
                    "luma_smooth_gate": luma_smooth_gate[..., :h, :w],
                    "luma_smooth_delta": luma_smooth_delta[..., :h, :w],
                }
            )
        return aux

    @torch.no_grad()
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


NAGIPERFECT_PRESETS = {
    "perfect-s": dict(width=32, enc_blk_nums=(1, 2, 3), middle_blk_num=4, dec_blk_nums=(1, 1, 2)),
    "perfect-m": dict(width=48, enc_blk_nums=(2, 2, 4), middle_blk_num=6, dec_blk_nums=(1, 2, 2)),
}


def build_nagiperfect_preset(name: str) -> NagiPerfect:
    try:
        cfg = NAGIPERFECT_PRESETS[name]
    except KeyError as exc:
        names = ", ".join(sorted(NAGIPERFECT_PRESETS))
        raise ValueError(f"unknown NagiPerfect preset {name!r}; choose one of: {names}") from exc
    return NagiPerfect(**cfg)
