"""SIDD Validation evaluator (sRGB).

Computes standard PSNR/SSIM on the 40 x 32 = 1280 patches of size 256x256.
PSNR is computed directly in sRGB space, the convention used by NAFNet,
Restormer, SCUNet, etc.

Entry point: ``nagi-eval-sidd``.

Models supported:
  * nagi   : Nagi NR (loads via nagi_denoise.Denoiser)
  * nagi_v2 : NagiV2 HDR-first denoiser (loads via nagi_denoise.Denoiser)
  * scunet : SCUNet color real_psnr (loads via .third_party.scunet)
  * nafnet : NAFNet-SIDD-width64 (loads via .third_party.nafnet)
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def psnr_srgb(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR on uint8 sRGB arrays, peak=255."""
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    if mse <= 0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def ssim_srgb(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight SSIM on uint8 RGB images (mean over channels)."""
    from scipy.signal import convolve2d

    def _ssim_ch(x, y, K1=0.01, K2=0.03, L=255.0, win=11, sigma=1.5):
        coords = np.arange(win) - win // 2
        gauss = np.exp(-(coords ** 2) / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel = gauss[:, None] * gauss[None, :]
        mu1 = convolve2d(x, kernel, mode="valid")
        mu2 = convolve2d(y, kernel, mode="valid")
        mu1_sq = mu1 * mu1; mu2_sq = mu2 * mu2; mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(x * x, kernel, mode="valid") - mu1_sq
        sigma2_sq = convolve2d(y * y, kernel, mode="valid") - mu2_sq
        sigma12 = convolve2d(x * y, kernel, mode="valid") - mu1_mu2
        C1 = (K1 * L) ** 2; C2 = (K2 * L) ** 2
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        return (num / den).mean()

    a64 = a.astype(np.float64); b64 = b.astype(np.float64)
    return float(np.mean([_ssim_ch(a64[..., c], b64[..., c]) for c in range(a.shape[-1])]))


# ---- Model loaders ----
def load_nagi(weights: str, device: str):
    from ..infer import Denoiser
    from ..transforms import srgb_to_linear, linear_to_srgb

    dn = Denoiser.load(weights, device=device)

    @torch.inference_mode()
    def forward(np_noisy_uint8: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np_noisy_uint8).permute(2, 0, 1).float() / 255.0
        t_lin = srgb_to_linear(t)
        out_lin = dn(t_lin, input_space="linear", tile=256, overlap=0)
        out_srgb = linear_to_srgb(out_lin.clamp(0, 1))
        return (out_srgb.numpy().transpose(1, 2, 0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

    n_params = sum(p.numel() for p in dn.model.parameters())
    return forward, n_params


def load_nagi_v2(weights: str, device: str):
    from ..infer import Denoiser
    from ..transforms import srgb_to_linear, linear_to_srgb

    dn = Denoiser.load(weights, device=device)

    @torch.inference_mode()
    def forward(np_noisy_uint8: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np_noisy_uint8).permute(2, 0, 1).float() / 255.0
        t_lin = srgb_to_linear(t)
        out_lin = dn(t_lin, input_space="linear", tile=256, overlap=0)
        out_srgb = linear_to_srgb(out_lin.clamp(0, 1))
        return (out_srgb.numpy().transpose(1, 2, 0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

    n_params = sum(p.numel() for p in dn.model.parameters())
    return forward, n_params


def load_scunet(weights: str, device: str):
    from ..devices import resolve_device
    from .third_party.scunet import SCUNet

    device = resolve_device(device)
    model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    @torch.inference_mode()
    def forward(np_noisy_uint8: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np_noisy_uint8).permute(2, 0, 1).float() / 255.0
        t = t.unsqueeze(0).to(device, dtype=torch.float32)
        out = model(t)
        out = out.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

    n_params = sum(p.numel() for p in model.parameters())
    return forward, n_params


def load_nafnet(weights: str, device: str):
    from ..devices import resolve_device
    from .third_party.nafnet import NAFNet

    device = resolve_device(device)
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    state = ckpt["params"] if "params" in ckpt else (
        ckpt["params_ema"] if "params_ema" in ckpt else ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    @torch.inference_mode()
    def forward(np_noisy_uint8: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np_noisy_uint8).permute(2, 0, 1).float() / 255.0
        t = t.unsqueeze(0).to(device, dtype=torch.float32)
        out = model(t)
        out = out.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

    n_params = sum(p.numel() for p in model.parameters())
    return forward, n_params


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="nagi-eval-sidd",
        description="Evaluate a denoiser on SIDD Validation (sRGB, 1280 patches).",
    )
    ap.add_argument("--model", choices=["nagi", "nagi_v2", "scunet", "nafnet"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    ap.add_argument("--ssim", action="store_true", help="also compute SSIM (slower)")
    ap.add_argument("--max-patches", type=int, default=0, help="limit number of patches (0 = all)")
    args = ap.parse_args()

    print(f"loading {args.noisy_mat} ...")
    noisy = sio.loadmat(args.noisy_mat)
    nkey = next(k for k in noisy if not k.startswith("__"))
    noisy = noisy[nkey]
    print(f"loading {args.gt_mat} ...")
    gt = sio.loadmat(args.gt_mat)
    gkey = next(k for k in gt if not k.startswith("__"))
    gt = gt[gkey]
    assert noisy.shape == gt.shape, (noisy.shape, gt.shape)
    n_img, n_blk = noisy.shape[:2]
    total = n_img * n_blk
    print(f"validation set: {noisy.shape} -> {total} patches of {noisy.shape[2:4]}")

    loaders = {
        "nagi": load_nagi,
        "nagi_v2": load_nagi_v2,
        "scunet": load_scunet,
        "nafnet": load_nafnet,
    }
    forward, n_params = loaders[args.model](args.weights, args.device)
    print(f"model: {args.model}, params: {n_params/1e6:.2f}M")

    psnr_in_list: list[float] = []
    psnr_out_list: list[float] = []
    ssim_in_list: list[float] = []
    ssim_out_list: list[float] = []

    t0 = time.time()
    done = 0
    limit = total if args.max_patches <= 0 else min(args.max_patches, total)
    for i in range(n_img):
        for j in range(n_blk):
            if done >= limit:
                break
            n_patch = noisy[i, j]
            g_patch = gt[i, j]
            out = forward(n_patch)

            psnr_in_list.append(psnr_srgb(n_patch, g_patch))
            psnr_out_list.append(psnr_srgb(out, g_patch))
            if args.ssim:
                ssim_in_list.append(ssim_srgb(n_patch, g_patch))
                ssim_out_list.append(ssim_srgb(out, g_patch))

            done += 1
            if done % 64 == 0 or done == limit:
                el = time.time() - t0
                eta = el / done * (limit - done)
                print(
                    f"[{done:4d}/{limit}] "
                    f"PSNR in={np.mean(psnr_in_list):.3f} out={np.mean(psnr_out_list):.3f} "
                    f"({el:.1f}s, eta {eta:.0f}s)"
                )
        if done >= limit:
            break

    el = time.time() - t0
    p_in = float(np.mean(psnr_in_list))
    p_out = float(np.mean(psnr_out_list))
    print()
    print(f"== {args.model} ({n_params/1e6:.2f}M params) on SIDD Validation ({done} patches) ==")
    print(f"  PSNR noisy : {p_in:.3f} dB")
    print(f"  PSNR denoised: {p_out:.3f} dB  (delta +{p_out - p_in:.3f})")
    if args.ssim:
        s_in = float(np.mean(ssim_in_list))
        s_out = float(np.mean(ssim_out_list))
        print(f"  SSIM noisy : {s_in:.4f}")
        print(f"  SSIM denoised: {s_out:.4f}")
    print(f"  total time : {el:.1f}s ({el/done*1000:.1f} ms/patch)")


if __name__ == "__main__":
    main()
