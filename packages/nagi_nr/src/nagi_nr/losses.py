"""Loss functions for Nagi NR.

All losses operate in float32 and are safe with HDR-range tensors.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import linear_to_srgb, srgb_to_linear


class Charbonnier(nn.Module):
    """sqrt((y - x)^2 + eps^2) — smooth L1 variant. Robust to HDR outliers."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = float(eps) ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps2).mean()


class FFTL1(nn.Module):
    """Magnitude-spectrum L1 loss (HINet / NAFNet style).

    Penalizes mismatches in frequency content, which helps preserve fine
    detail and avoid over-smoothing. Operates on rfft2, so the FFT is real-valued.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        P = torch.fft.rfft2(pred, norm="ortho")
        T = torch.fft.rfft2(target, norm="ortho")
        return (P.abs() - T.abs()).abs().mean()


class NagiDistillLoss(nn.Module):
    """Phase 3 distillation loss (Charbonnier in compressed space).

    For each sample, supervises the prediction against the GT and, when a
    teacher target is available, additionally against the NAFNet teacher output:

        has_teacher == 1:  loss = gt_weight * C(pred, gt) + distill_weight * C(pred, teacher)
        has_teacher == 0:  loss = 1.0       * C(pred, gt)   (teacher term disabled)

    Charbonnier is computed per-sample so the teacher term can be masked without
    contaminating GT-only (synthetic-noise) samples. NaN-safe when a batch has no
    teacher samples.

    Args:
        compress_fn: model.compress (tracks the exact asinh curve).
        gt_weight / distill_weight: blend used when a teacher is present.
    """

    def __init__(
        self,
        compress_fn,
        gt_weight: float = 0.5,
        distill_weight: float = 0.5,
        charbonnier_eps: float = 1e-3,
    ):
        super().__init__()
        self.compress_fn = compress_fn
        self.gt_weight = float(gt_weight)
        self.distill_weight = float(distill_weight)
        self.eps2 = float(charbonnier_eps) ** 2

    def _charb_per_sample(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a - b
        per = torch.sqrt(diff * diff + self.eps2)
        return per.flatten(1).mean(dim=1)  # (B,)

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        teacher: torch.Tensor,
        has_teacher: torch.Tensor,
    ) -> dict:
        pred_c = self.compress_fn(pred)
        gt_c = self.compress_fn(gt)
        teacher_c = self.compress_fn(teacher)

        ht = has_teacher.to(pred.dtype).view(-1)  # (B,) in {0,1}
        gt_ps = self._charb_per_sample(pred_c, gt_c)
        dt_ps = self._charb_per_sample(pred_c, teacher_c)

        # Per-sample weights: GT-only when no teacher, blended when present.
        w_gt = torch.where(ht > 0, self.gt_weight, 1.0)
        w_dt = torch.where(ht > 0, self.distill_weight, 0.0)
        total = (w_gt * gt_ps + w_dt * dt_ps).mean()

        n_teach = ht.sum().clamp_min(1.0)
        return {
            "total": total,
            "gt": gt_ps.mean().detach(),
            "distill": ((dt_ps * ht).sum() / n_teach).detach(),
            "teacher_frac": (ht.mean()).detach(),
        }


class NagiLoss(nn.Module):
    """Composite loss for Nagi NR training.

    The primary Charbonnier + FFT terms are computed in a chosen reconstruction
    space, with a smaller weight on linear-space reconstruction.

    ``recon_space`` selects where the primary terms live:
      * ``"compressed"`` (default): asinh-compressed space (``compress_fn``).
        HDR-friendly, but mismatched to an sRGB-PSNR metric.
      * ``"srgb"``: sRGB-encoded space (``linear_to_srgb``). Aligns the training
        objective with sRGB PSNR, the metric used by the SIDD-Val benchmark.

    Args:
        compress_fn: callable taking a linear tensor and returning compressed tensor.
            Pass model.compress so the loss tracks the model's exact compression curve.
        fft_weight: weight for FFT magnitude loss (in the recon space).
        linear_weight: weight for linear-space Charbonnier (clamped to avoid HDR blowup).
        linear_clamp: clamp value for the linear loss (acts only on extreme HDR).
        recon_space: "compressed" or "srgb" (see above).
        luma_ms_weight: optional multi-scale sRGB-luma low-frequency loss weight.
        luma_ms_scales: average-pooling scales for the luma loss.
    """

    def __init__(
        self,
        compress_fn,
        fft_weight: float = 0.05,
        linear_weight: float = 0.1,
        linear_clamp: float = 16.0,
        charbonnier_eps: float = 1e-3,
        recon_space: str = "compressed",
        luma_ms_weight: float = 0.0,
        luma_ms_scales: tuple[int, ...] = (4, 8),
    ):
        super().__init__()
        if recon_space not in ("compressed", "srgb"):
            raise ValueError(f"recon_space must be 'compressed' or 'srgb', got {recon_space!r}")
        self.recon_space = recon_space
        self.compress_fn = compress_fn
        self.fft_weight = float(fft_weight)
        self.linear_weight = float(linear_weight)
        self.linear_clamp = float(linear_clamp)
        self.luma_ms_weight = float(luma_ms_weight)
        self.luma_ms_scales = tuple(int(s) for s in luma_ms_scales)
        self.charb = Charbonnier(eps=charbonnier_eps)
        self.fft = FFTL1()
        self.register_buffer(
            "_srgb_luma",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _to_recon(self, x: torch.Tensor) -> torch.Tensor:
        # Map a linear-light tensor into the chosen reconstruction space.
        if self.recon_space == "srgb":
            return linear_to_srgb(x)
        return self.compress_fn(x)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        pred_c = self._to_recon(pred)
        target_c = self._to_recon(target)

        loss_c = self.charb(pred_c, target_c)
        total = loss_c
        out = {"recon": loss_c.detach()}

        if self.fft_weight > 0:
            loss_fft = self.fft(pred_c, target_c)
            total = total + self.fft_weight * loss_fft
            out["fft"] = loss_fft.detach()

        if self.linear_weight > 0:
            c = self.linear_clamp
            pl = pred.clamp(-c, c)
            tl = target.clamp(-c, c)
            loss_lin = self.charb(pl, tl)
            total = total + self.linear_weight * loss_lin
            out["linear"] = loss_lin.detach()

        if self.luma_ms_weight > 0:
            pred_y = (linear_to_srgb(pred) * self._srgb_luma.to(dtype=pred.dtype)).sum(dim=1, keepdim=True)
            target_y = (linear_to_srgb(target) * self._srgb_luma.to(dtype=target.dtype)).sum(dim=1, keepdim=True)
            loss_ms = pred_y.new_tensor(0.0)
            n_scales = 0
            for scale in self.luma_ms_scales:
                if scale <= 1:
                    py, ty = pred_y, target_y
                else:
                    if pred_y.shape[-2] < scale or pred_y.shape[-1] < scale:
                        continue
                    py = F.avg_pool2d(pred_y, kernel_size=scale, stride=scale)
                    ty = F.avg_pool2d(target_y, kernel_size=scale, stride=scale)
                loss_ms = loss_ms + self.charb(py, ty)
                n_scales += 1
            if n_scales > 0:
                loss_ms = loss_ms / n_scales
                total = total + self.luma_ms_weight * loss_ms
                out["luma_ms"] = loss_ms.detach()

        out["total"] = total
        return out


def _linear_luma(x: torch.Tensor) -> torch.Tensor:
    weights = x.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (x[:, :3] * weights).sum(dim=1, keepdim=True)


def _chroma_ratio(x: torch.Tensor) -> torch.Tensor:
    rgb = x[:, :3].clamp_min(0.0)
    y = _linear_luma(rgb).clamp_min(1.0e-6)
    return rgb / y


def _local_lowpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    pad = kernel_size // 2
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


class NagiPerfectLoss(nn.Module):
    """Loss for the base/detail/confidence NagiPerfect design.

    The final output is supervised normally, while additional terms keep the
    design honest:

    - the base RGB path must remain a good denoised image;
    - highlight luma must not drift;
    - highlight chroma must stay close to target;
    - luma high-frequency detail is supervised only through luma, not chroma.
    """

    def __init__(
        self,
        compress_fn,
        final_weight: float = 1.0,
        base_weight: float = 0.35,
        srgb_weight: float = 0.0,
        srgb_base_weight: float = 0.0,
        body_srgb_weight: float = 0.0,
        body_srgb_base_weight: float = 0.0,
        flat_srgb_hf_weight: float = 0.0,
        flat_luma_hf_weight: float = 0.0,
        flat_chroma_hf_weight: float = 0.0,
        flat_chroma_damp_weight: float = 0.0,
        flat_chroma_distill_weight: float = 0.0,
        flat_chroma_distill_strength: float = 0.5,
        flat_chroma_distill_kernel_size: int = 9,
        flat_chroma_delta_distill_weight: float = 0.0,
        edge_luma_hf_weight: float = 0.0,
        edge_threshold: float = 0.040,
        edge_transition: float = 0.015,
        flat_threshold: float = 0.015,
        flat_transition: float = 0.010,
        flat_kernel_size: int = 9,
        linear_weight: float = 0.08,
        highlight_luma_weight: float = 0.20,
        highlight_chroma_weight: float = 0.08,
        highlight_ratio_weight: float = 0.0,
        highlight_luma_weight_end: float | None = None,
        highlight_chroma_weight_end: float | None = None,
        highlight_ratio_weight_end: float | None = None,
        highlight_ramp_start: float = 0.0,
        highlight_ramp_end: float = 1.0,
        detail_weight: float = 0.12,
        confidence_l1_weight: float = 0.002,
        highlight_threshold: float = 1.0,
        highlight_transition: float = 0.5,
        detail_kernel_size: int = 9,
        linear_clamp: float = 16.0,
        charbonnier_eps: float = 1.0e-3,
    ):
        super().__init__()
        self.compress_fn = compress_fn
        self.final_weight = float(final_weight)
        self.base_weight = float(base_weight)
        self.srgb_weight = float(srgb_weight)
        self.srgb_base_weight = float(srgb_base_weight)
        self.body_srgb_weight = float(body_srgb_weight)
        self.body_srgb_base_weight = float(body_srgb_base_weight)
        self.flat_srgb_hf_weight = float(flat_srgb_hf_weight)
        self.flat_luma_hf_weight = float(flat_luma_hf_weight)
        self.flat_chroma_hf_weight = float(flat_chroma_hf_weight)
        self.flat_chroma_damp_weight = float(flat_chroma_damp_weight)
        self.flat_chroma_distill_weight = float(flat_chroma_distill_weight)
        self.flat_chroma_distill_strength = float(flat_chroma_distill_strength)
        self.flat_chroma_distill_kernel_size = int(flat_chroma_distill_kernel_size)
        self.flat_chroma_delta_distill_weight = float(flat_chroma_delta_distill_weight)
        self.edge_luma_hf_weight = float(edge_luma_hf_weight)
        self.edge_threshold = float(edge_threshold)
        self.edge_transition = float(edge_transition)
        self.flat_threshold = float(flat_threshold)
        self.flat_transition = float(flat_transition)
        self.flat_kernel_size = int(flat_kernel_size)
        self.linear_weight = float(linear_weight)
        self.highlight_luma_weight = float(highlight_luma_weight)
        self.highlight_chroma_weight = float(highlight_chroma_weight)
        self.highlight_ratio_weight = float(highlight_ratio_weight)
        self.highlight_luma_weight_end = (
            None if highlight_luma_weight_end is None else float(highlight_luma_weight_end)
        )
        self.highlight_chroma_weight_end = (
            None if highlight_chroma_weight_end is None else float(highlight_chroma_weight_end)
        )
        self.highlight_ratio_weight_end = (
            None if highlight_ratio_weight_end is None else float(highlight_ratio_weight_end)
        )
        self.highlight_ramp_start = float(highlight_ramp_start)
        self.highlight_ramp_end = float(highlight_ramp_end)
        self.detail_weight = float(detail_weight)
        self.confidence_l1_weight = float(confidence_l1_weight)
        self.highlight_threshold = float(highlight_threshold)
        self.highlight_transition = float(highlight_transition)
        self.detail_kernel_size = int(detail_kernel_size)
        self.linear_clamp = float(linear_clamp)
        self.charb = Charbonnier(eps=charbonnier_eps)
        self._step = 0
        self._total_steps = 1

    def set_step(self, step: int, total_steps: int) -> None:
        self._step = int(step)
        self._total_steps = max(int(total_steps), 1)

    def _scheduled_weight(self, start: float, end: float | None) -> float:
        if end is None:
            return float(start)

        ramp_start = self.highlight_ramp_start
        ramp_end = self.highlight_ramp_end
        if ramp_start <= 1.0:
            ramp_start = ramp_start * self._total_steps
        if ramp_end <= 1.0:
            ramp_end = ramp_end * self._total_steps
        if ramp_end <= ramp_start:
            t = 1.0
        else:
            t = (self._step - ramp_start) / max(ramp_end - ramp_start, 1.0)
            t = min(max(t, 0.0), 1.0)
        return float(start) + (float(end) - float(start)) * t

    def _highlight_mask(self, target: torch.Tensor) -> torch.Tensor:
        y = _linear_luma(target.clamp_min(0.0))
        return torch.sigmoid((y - self.highlight_threshold) / max(self.highlight_transition, 1.0e-6))

    def _luma_detail(self, x: torch.Tensor) -> torch.Tensor:
        y = _linear_luma(x.clamp_min(0.0))
        return y - _local_lowpass(y, self.detail_kernel_size)

    def _srgb_luma(self, x: torch.Tensor) -> torch.Tensor:
        weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x[:, :3] * weights).sum(dim=1, keepdim=True)

    def _flat_mask(self, target_srgb: torch.Tensor, body_mask: torch.Tensor) -> torch.Tensor:
        y = self._srgb_luma(target_srgb)
        detail = (y - _local_lowpass(y, self.flat_kernel_size)).abs()
        flat = torch.sigmoid((self.flat_threshold - detail) / max(self.flat_transition, 1.0e-6))
        return flat * body_mask

    def forward(self, pred: torch.Tensor | dict[str, torch.Tensor], target: torch.Tensor) -> dict:
        if isinstance(pred, dict):
            output = pred["output"]
            base = pred.get("base", output)
            confidence = pred.get("detail_confidence")
        else:
            output = pred
            base = output
            confidence = None

        target_c = self.compress_fn(target)
        output_c = self.compress_fn(output)
        base_c = self.compress_fn(base)

        loss_final = self.charb(output_c, target_c)
        loss_base = self.charb(base_c, target_c)
        total = self.final_weight * loss_final + self.base_weight * loss_base
        out = {
            "final": loss_final.detach(),
            "base": loss_base.detach(),
        }

        if self.srgb_weight > 0 or self.srgb_base_weight > 0:
            target_srgb = linear_to_srgb(target)
            if self.srgb_weight > 0:
                loss_srgb = self.charb(linear_to_srgb(output), target_srgb)
                total = total + self.srgb_weight * loss_srgb
                out["srgb"] = loss_srgb.detach()
            if self.srgb_base_weight > 0:
                loss_srgb_base = self.charb(linear_to_srgb(base), target_srgb)
                total = total + self.srgb_base_weight * loss_srgb_base
                out["srgb_base"] = loss_srgb_base.detach()

        if self.linear_weight > 0:
            c = self.linear_clamp
            loss_linear = self.charb(output.clamp(-c, c), target.clamp(-c, c))
            total = total + self.linear_weight * loss_linear
            out["linear"] = loss_linear.detach()

        mask = self._highlight_mask(target)
        mask_mean = mask.mean().clamp_min(1.0e-6)
        body_mask = (1.0 - mask).clamp_min(0.0)
        body_mask_mean = body_mask.mean().clamp_min(1.0e-6)

        if self.body_srgb_weight > 0 or self.body_srgb_base_weight > 0:
            target_srgb = linear_to_srgb(target)
            body_mask_rgb = body_mask.expand(-1, 3, -1, -1)
            if self.body_srgb_weight > 0:
                diff = linear_to_srgb(output) - target_srgb
                loss_body = (torch.sqrt(diff.pow(2) + self.charb.eps2) * body_mask_rgb).sum()
                loss_body = loss_body / (body_mask_mean * body_mask.numel() * 3.0)
                total = total + self.body_srgb_weight * loss_body
                out["body_srgb"] = loss_body.detach()
            if self.body_srgb_base_weight > 0:
                diff = linear_to_srgb(base) - target_srgb
                loss_body_base = (torch.sqrt(diff.pow(2) + self.charb.eps2) * body_mask_rgb).sum()
                loss_body_base = loss_body_base / (body_mask_mean * body_mask.numel() * 3.0)
                total = total + self.body_srgb_base_weight * loss_body_base
                out["body_srgb_base"] = loss_body_base.detach()

        if self.flat_srgb_hf_weight > 0:
            target_srgb = linear_to_srgb(target)
            output_srgb = linear_to_srgb(output)
            flat_mask = self._flat_mask(target_srgb, body_mask)
            flat_mask_mean = flat_mask.mean().clamp_min(1.0e-6)
            residual = output_srgb - target_srgb
            residual_hf = residual - _local_lowpass(residual, self.flat_kernel_size)
            flat_mask_rgb = flat_mask.expand(-1, 3, -1, -1)
            loss_flat_hf = (torch.sqrt(residual_hf.pow(2) + self.charb.eps2) * flat_mask_rgb).sum()
            loss_flat_hf = loss_flat_hf / (flat_mask_mean * flat_mask.numel() * 3.0)
            total = total + self.flat_srgb_hf_weight * loss_flat_hf
            out["flat_srgb_hf"] = loss_flat_hf.detach()
            out["flat_mask_mean"] = flat_mask.mean().detach()

        if self.flat_luma_hf_weight > 0 or self.flat_chroma_hf_weight > 0:
            target_srgb = linear_to_srgb(target)
            output_srgb = linear_to_srgb(output)
            flat_mask = self._flat_mask(target_srgb, body_mask)
            flat_mask_mean = flat_mask.mean().clamp_min(1.0e-6)
            target_y = self._srgb_luma(target_srgb)
            output_y = self._srgb_luma(output_srgb)

            if self.flat_luma_hf_weight > 0:
                residual_y = output_y - target_y
                residual_y_hf = residual_y - _local_lowpass(residual_y, self.flat_kernel_size)
                loss_flat_luma = (torch.sqrt(residual_y_hf.pow(2) + self.charb.eps2) * flat_mask).sum()
                loss_flat_luma = loss_flat_luma / (flat_mask_mean * flat_mask.numel())
                total = total + self.flat_luma_hf_weight * loss_flat_luma
                out["flat_luma_hf"] = loss_flat_luma.detach()

            if self.flat_chroma_hf_weight > 0:
                output_chroma = output_srgb - output_y
                target_chroma = target_srgb - target_y
                residual_chroma = output_chroma - target_chroma
                residual_chroma_hf = residual_chroma - _local_lowpass(residual_chroma, self.flat_kernel_size)
                flat_mask_rgb = flat_mask.expand(-1, 3, -1, -1)
                loss_flat_chroma = (
                    torch.sqrt(residual_chroma_hf.pow(2) + self.charb.eps2) * flat_mask_rgb
                ).sum()
                loss_flat_chroma = loss_flat_chroma / (flat_mask_mean * flat_mask.numel() * 3.0)
                total = total + self.flat_chroma_hf_weight * loss_flat_chroma
                out["flat_chroma_hf"] = loss_flat_chroma.detach()

            out["flat_mask_mean"] = flat_mask.mean().detach()

        if self.flat_chroma_damp_weight > 0:
            target_srgb = linear_to_srgb(target)
            output_srgb = linear_to_srgb(output)
            flat_mask = self._flat_mask(target_srgb, body_mask)
            flat_mask_mean = flat_mask.mean().clamp_min(1.0e-6)
            output_y = self._srgb_luma(output_srgb)
            output_chroma = output_srgb - output_y
            output_chroma_hf = output_chroma - _local_lowpass(output_chroma, self.flat_kernel_size)
            flat_mask_rgb = flat_mask.expand(-1, 3, -1, -1)
            loss_chroma_damp = (
                torch.sqrt(output_chroma_hf.pow(2) + self.charb.eps2) * flat_mask_rgb
            ).sum()
            loss_chroma_damp = loss_chroma_damp / (flat_mask_mean * flat_mask.numel() * 3.0)
            total = total + self.flat_chroma_damp_weight * loss_chroma_damp
            out["flat_chroma_damp"] = loss_chroma_damp.detach()
            out["flat_mask_mean"] = flat_mask.mean().detach()

        if self.edge_luma_hf_weight > 0:
            target_srgb = linear_to_srgb(target)
            output_srgb = linear_to_srgb(output)
            target_y = self._srgb_luma(target_srgb)
            output_y = self._srgb_luma(output_srgb)
            target_y_hf = target_y - _local_lowpass(target_y, self.flat_kernel_size)
            output_y_hf = output_y - _local_lowpass(output_y, self.flat_kernel_size)
            edge_mask = torch.sigmoid(
                (target_y_hf.abs() - self.edge_threshold) / max(self.edge_transition, 1.0e-6)
            )
            edge_mask = edge_mask * body_mask
            edge_mask_mean = edge_mask.mean().clamp_min(1.0e-6)
            loss_edge_luma = (
                torch.sqrt((output_y_hf - target_y_hf).pow(2) + self.charb.eps2) * edge_mask
            ).sum()
            loss_edge_luma = loss_edge_luma / (edge_mask_mean * edge_mask.numel())
            total = total + self.edge_luma_hf_weight * loss_edge_luma
            out["edge_luma_hf"] = loss_edge_luma.detach()
            out["edge_mask_mean"] = edge_mask.mean().detach()

        if self.flat_chroma_distill_weight > 0 and isinstance(pred, dict) and "output_pre_chroma" in pred:
            teacher_base = pred["output_pre_chroma"].detach()
            teacher_srgb = linear_to_srgb(teacher_base)
            output_srgb = linear_to_srgb(output)

            teacher_y = self._srgb_luma(teacher_srgb)
            teacher_chroma = teacher_srgb - teacher_y
            smooth_chroma = _local_lowpass(teacher_chroma, self.flat_chroma_distill_kernel_size)

            teacher_detail = (teacher_y - _local_lowpass(teacher_y, self.flat_kernel_size)).abs()
            teacher_flat = torch.sigmoid(
                (self.flat_threshold - teacher_detail) / max(self.flat_transition, 1.0e-6)
            )
            teacher_highlight = torch.sigmoid(
                (_linear_luma(teacher_base.clamp_min(0.0)) - self.highlight_threshold)
                / max(self.highlight_transition, 1.0e-6)
            )
            distill_mask = (teacher_flat * (1.0 - teacher_highlight)).detach()
            distill_mask_mean = distill_mask.mean().clamp_min(1.0e-6)
            blend = (distill_mask * min(max(self.flat_chroma_distill_strength, 0.0), 1.0)).detach()

            target_chroma = (teacher_chroma * (1.0 - blend) + smooth_chroma * blend).detach()
            output_chroma = output_srgb - self._srgb_luma(output_srgb)
            diff_chroma = output_chroma - target_chroma
            distill_mask_rgb = distill_mask.expand(-1, 3, -1, -1)
            loss_chroma_distill = (
                torch.sqrt(diff_chroma.pow(2) + self.charb.eps2) * distill_mask_rgb
            ).sum()
            loss_chroma_distill = loss_chroma_distill / (distill_mask_mean * distill_mask.numel() * 3.0)
            total = total + self.flat_chroma_distill_weight * loss_chroma_distill
            out["flat_chroma_distill"] = loss_chroma_distill.detach()
            out["flat_chroma_distill_mask_mean"] = distill_mask.mean().detach()

        if (
            self.flat_chroma_delta_distill_weight > 0
            and isinstance(pred, dict)
            and "output_pre_chroma" in pred
            and "chroma_residual" in pred
        ):
            teacher_base = pred["output_pre_chroma"].detach()
            teacher_srgb = linear_to_srgb(teacher_base.clamp_min(0.0))
            teacher_y = self._srgb_luma(teacher_srgb)
            teacher_chroma = teacher_srgb - teacher_y
            smooth_chroma = _local_lowpass(teacher_chroma, self.flat_chroma_distill_kernel_size)

            teacher_detail = (teacher_y - _local_lowpass(teacher_y, self.flat_kernel_size)).abs()
            teacher_flat = torch.sigmoid(
                (self.flat_threshold - teacher_detail) / max(self.flat_transition, 1.0e-6)
            )
            teacher_highlight = torch.sigmoid(
                (_linear_luma(teacher_base.clamp_min(0.0)) - self.highlight_threshold)
                / max(self.highlight_transition, 1.0e-6)
            )
            distill_mask = (teacher_flat * (1.0 - teacher_highlight)).detach()
            distill_mask_mean = distill_mask.mean().clamp_min(1.0e-6)
            blend = (distill_mask * min(max(self.flat_chroma_distill_strength, 0.0), 1.0)).detach()

            teacher_srgb_smooth = teacher_y + teacher_chroma * (1.0 - blend) + smooth_chroma * blend
            teacher_linear_smooth = srgb_to_linear(teacher_srgb_smooth.clamp(0.0, 1.0)).detach()
            target_delta = (teacher_linear_smooth - teacher_base.clamp_min(0.0)).detach()
            chroma_residual = pred["chroma_residual"]
            distill_mask_rgb = distill_mask.expand(-1, 3, -1, -1)
            loss_delta_distill = (
                torch.sqrt((chroma_residual - target_delta).pow(2) + self.charb.eps2) * distill_mask_rgb
            ).sum()
            loss_delta_distill = loss_delta_distill / (distill_mask_mean * distill_mask.numel() * 3.0)
            total = total + self.flat_chroma_delta_distill_weight * loss_delta_distill
            out["flat_chroma_delta_distill"] = loss_delta_distill.detach()

        highlight_luma_weight = self._scheduled_weight(
            self.highlight_luma_weight,
            self.highlight_luma_weight_end,
        )
        highlight_chroma_weight = self._scheduled_weight(
            self.highlight_chroma_weight,
            self.highlight_chroma_weight_end,
        )
        highlight_ratio_weight = self._scheduled_weight(
            self.highlight_ratio_weight,
            self.highlight_ratio_weight_end,
        )

        if highlight_luma_weight > 0:
            loss_hl = (torch.sqrt((_linear_luma(output) - _linear_luma(target)).pow(2) + self.charb.eps2) * mask).sum()
            loss_hl = loss_hl / (mask_mean * mask.numel())
            total = total + highlight_luma_weight * loss_hl
            out["highlight_luma"] = loss_hl.detach()

        if highlight_chroma_weight > 0:
            loss_chroma = ((_chroma_ratio(output) - _chroma_ratio(target)).abs() * mask).sum()
            loss_chroma = loss_chroma / (mask_mean * mask.numel() * 3.0)
            total = total + highlight_chroma_weight * loss_chroma
            out["highlight_chroma"] = loss_chroma.detach()

        if highlight_ratio_weight > 0:
            output_y = _linear_luma(output.clamp_min(0.0)).clamp_min(1.0e-5)
            target_y = _linear_luma(target.clamp_min(0.0)).clamp_min(1.0e-5)
            ratio_diff = torch.log(output_y) - torch.log(target_y)
            loss_ratio = (torch.sqrt(ratio_diff.pow(2) + self.charb.eps2) * mask).sum()
            loss_ratio = loss_ratio / (mask_mean * mask.numel())
            total = total + highlight_ratio_weight * loss_ratio
            out["highlight_ratio"] = loss_ratio.detach()

        if self.detail_weight > 0:
            loss_detail = (torch.sqrt((self._luma_detail(output) - self._luma_detail(target)).pow(2) + self.charb.eps2) * mask).sum()
            loss_detail = loss_detail / (mask_mean * mask.numel())
            total = total + self.detail_weight * loss_detail
            out["detail_luma"] = loss_detail.detach()

        if confidence is not None and self.confidence_l1_weight > 0:
            loss_conf = confidence.mean()
            total = total + self.confidence_l1_weight * loss_conf
            out["confidence_l1"] = loss_conf.detach()

        out["highlight_mask_mean"] = mask.mean().detach()
        out["body_mask_mean"] = body_mask.mean().detach()
        out["highlight_luma_weight"] = output.new_tensor(highlight_luma_weight)
        out["highlight_chroma_weight"] = output.new_tensor(highlight_chroma_weight)
        out["highlight_ratio_weight"] = output.new_tensor(highlight_ratio_weight)
        out["total"] = total
        return out
