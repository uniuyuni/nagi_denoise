"""Residual diagnostics for SIDD Validation.

This script uses the same Nagi inference path and sRGB uint8 PSNR convention as
``nagi-eval-sidd``, then decomposes the residuals to make the next experiment
less guessy.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def _mat_payload(path: str) -> np.ndarray:
    data = sio.loadmat(path)
    key = next(k for k in data if not k.startswith("__"))
    return data[key]


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = a.astype(np.float64) - b.astype(np.float64)
    return _psnr_from_mse(float(np.mean(err * err)))


def _ycc(x: np.ndarray) -> np.ndarray:
    # Residual-space YCbCr projection. No offsets are used for chroma because
    # residuals are centered values, not absolute pixel values.
    mat = np.array(
        [
            [0.299000, 0.587000, 0.114000],
            [-0.168736, -0.331264, 0.500000],
            [0.500000, -0.418688, -0.081312],
        ],
        dtype=np.float64,
    )
    return x @ mat.T


def _freq_masks(h: int, w: int) -> dict[str, np.ndarray]:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    r = np.sqrt(fx * fx + fy * fy)
    return {
        "low": r < 0.08,
        "mid": (r >= 0.08) & (r < 0.22),
        "high": r >= 0.22,
    }


def _patch_freq_power(y_resid: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    spec = np.fft.rfft2(y_resid)
    power = (spec.real * spec.real) + (spec.imag * spec.imag)
    return {name: float(power[mask].sum()) for name, mask in masks.items()}


def _quartile_bins(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    qs = np.quantile(values, [0.25, 0.50, 0.75])
    return [
        (f"<= {qs[0]:.3f}", values <= qs[0]),
        (f"{qs[0]:.3f}..{qs[1]:.3f}", (values > qs[0]) & (values <= qs[1])),
        (f"{qs[1]:.3f}..{qs[2]:.3f}", (values > qs[1]) & (values <= qs[2])),
        (f"> {qs[2]:.3f}", values > qs[2]),
    ]


def _mean_for_mask(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(values[mask].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze SIDD Validation residuals for a Nagi checkpoint.")
    ap.add_argument("--weights", default="runs/nagi_nr_m/nagi_nr_m_final.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    ap.add_argument("--output-json", default="runs/residual_m_final.json")
    ap.add_argument("--output-md", default="runs/residual_m_final.md")
    ap.add_argument("--max-patches", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=64)
    args = ap.parse_args()

    from nagi_nr.infer import Denoiser
    from nagi_nr.transforms import linear_to_srgb, srgb_to_linear

    print(f"loading {args.noisy_mat} ...")
    noisy = _mat_payload(args.noisy_mat)
    print(f"loading {args.gt_mat} ...")
    gt = _mat_payload(args.gt_mat)
    if noisy.shape != gt.shape:
        raise ValueError(f"shape mismatch: noisy={noisy.shape}, gt={gt.shape}")

    n_img, n_blk, h, w = noisy.shape[:4]
    total = n_img * n_blk
    limit = total if args.max_patches <= 0 else min(args.max_patches, total)
    print(f"validation set: {noisy.shape}, limit={limit}")
    print(f"loading model: {args.weights}")
    dn = Denoiser.load(args.weights, device=args.device)
    n_params = sum(p.numel() for p in dn.model.parameters())
    print(f"params: {n_params / 1e6:.2f}M, device={dn.device}")

    ycc_names = ["Y", "Cb", "Cr"]
    ch_names = ["R", "G", "B"]
    freq_masks = _freq_masks(h, w)
    freq_out = {k: 0.0 for k in freq_masks}
    freq_in = {k: 0.0 for k in freq_masks}

    pixel_count = 0
    patch_count = 0
    sse_in = 0.0
    sse_out = 0.0
    sse_ch_in = np.zeros(3, dtype=np.float64)
    sse_ch_out = np.zeros(3, dtype=np.float64)
    sum_res_ch = np.zeros(3, dtype=np.float64)
    sum_abs_res_ch = np.zeros(3, dtype=np.float64)
    sse_ycc_in = np.zeros(3, dtype=np.float64)
    sse_ycc_out = np.zeros(3, dtype=np.float64)

    # Closed-form diagnostics.
    alpha_a2 = 0.0
    alpha_ad = 0.0
    alpha_d2 = 0.0
    affine_n = 0
    affine_sum_x = np.zeros(3, dtype=np.float64)
    affine_sum_y = np.zeros(3, dtype=np.float64)
    affine_sum_x2 = np.zeros(3, dtype=np.float64)
    affine_sum_y2 = np.zeros(3, dtype=np.float64)
    affine_sum_xy = np.zeros(3, dtype=np.float64)

    corr_n = 0
    corr_sum_x = 0.0
    corr_sum_y = 0.0
    corr_sum_x2 = 0.0
    corr_sum_y2 = 0.0
    corr_sum_xy = 0.0

    scene_sse_in = np.zeros(n_img, dtype=np.float64)
    scene_sse_out = np.zeros(n_img, dtype=np.float64)
    scene_pixels = np.zeros(n_img, dtype=np.int64)
    patch_rows: list[dict[str, float | int]] = []
    clip_low = 0
    clip_high = 0

    t0 = time.time()
    for i in range(n_img):
        for j in range(n_blk):
            if patch_count >= limit:
                break

            n_patch_u8 = noisy[i, j]
            g_patch_u8 = gt[i, j]
            t = torch.from_numpy(n_patch_u8).permute(2, 0, 1).float() / 255.0
            t_lin = srgb_to_linear(t)
            with torch.inference_mode():
                out_lin = dn(t_lin, input_space="linear", tile=256, overlap=0)
                out_srgb = linear_to_srgb(out_lin.clamp(0, 1))
            out_u8 = (
                out_srgb.numpy().transpose(1, 2, 0) * 255.0 + 0.5
            ).clip(0, 255).astype(np.uint8)

            n_patch = n_patch_u8.astype(np.float64)
            g_patch = g_patch_u8.astype(np.float64)
            out = out_u8.astype(np.float64)
            err_in = n_patch - g_patch
            err_out = out - g_patch

            p_pixels = int(err_out.shape[0] * err_out.shape[1] * err_out.shape[2])
            pixel_count += p_pixels
            scene_pixels[i] += p_pixels
            patch_count += 1

            patch_sse_in = float(np.sum(err_in * err_in))
            patch_sse_out = float(np.sum(err_out * err_out))
            sse_in += patch_sse_in
            sse_out += patch_sse_out
            scene_sse_in[i] += patch_sse_in
            scene_sse_out[i] += patch_sse_out

            sse_ch_in += np.sum(err_in * err_in, axis=(0, 1))
            sse_ch_out += np.sum(err_out * err_out, axis=(0, 1))
            sum_res_ch += np.sum(err_out, axis=(0, 1))
            sum_abs_res_ch += np.sum(np.abs(err_out), axis=(0, 1))

            ycc_in = _ycc(err_in)
            ycc_out = _ycc(err_out)
            sse_ycc_in += np.sum(ycc_in * ycc_in, axis=(0, 1))
            sse_ycc_out += np.sum(ycc_out * ycc_out, axis=(0, 1))

            y_res_in = ycc_in[..., 0]
            y_res_out = ycc_out[..., 0]
            for name, value in _patch_freq_power(y_res_in, freq_masks).items():
                freq_in[name] += value
            for name, value in _patch_freq_power(y_res_out, freq_masks).items():
                freq_out[name] += value

            a = err_in
            d = out - n_patch
            alpha_a2 += float(np.sum(a * a))
            alpha_ad += float(np.sum(a * d))
            alpha_d2 += float(np.sum(d * d))

            affine_n += int(out.shape[0] * out.shape[1])
            affine_sum_x += np.sum(out, axis=(0, 1))
            affine_sum_y += np.sum(g_patch, axis=(0, 1))
            affine_sum_x2 += np.sum(out * out, axis=(0, 1))
            affine_sum_y2 += np.sum(g_patch * g_patch, axis=(0, 1))
            affine_sum_xy += np.sum(out * g_patch, axis=(0, 1))

            x = err_in.reshape(-1)
            y = err_out.reshape(-1)
            corr_n += int(x.size)
            corr_sum_x += float(x.sum())
            corr_sum_y += float(y.sum())
            corr_sum_x2 += float(np.dot(x, x))
            corr_sum_y2 += float(np.dot(y, y))
            corr_sum_xy += float(np.dot(x, y))

            clip_low += int(np.count_nonzero(out_u8 <= 0))
            clip_high += int(np.count_nonzero(out_u8 >= 255))

            gt_y = _ycc(g_patch)[..., 0]
            patch_rows.append(
                {
                    "scene": i,
                    "block": j,
                    "psnr_in": _psnr(n_patch_u8, g_patch_u8),
                    "psnr_out": _psnr(out_u8, g_patch_u8),
                    "mse_in": patch_sse_in / p_pixels,
                    "mse_out": patch_sse_out / p_pixels,
                    "gt_luma": float(gt_y.mean()),
                    "noisy_luma": float(_ycc(n_patch)[..., 0].mean()),
                }
            )

            if patch_count % args.progress_every == 0 or patch_count == limit:
                elapsed = time.time() - t0
                eta = elapsed / patch_count * (limit - patch_count)
                print(
                    f"[{patch_count:4d}/{limit}] "
                    f"PSNR in={_psnr_from_mse(sse_in / pixel_count):.3f} "
                    f"out={_psnr_from_mse(sse_out / pixel_count):.3f} "
                    f"({elapsed:.1f}s, eta {eta:.0f}s)"
                )
        if patch_count >= limit:
            break

    patch_psnr_in = np.array([float(r["psnr_in"]) for r in patch_rows])
    patch_psnr_out = np.array([float(r["psnr_out"]) for r in patch_rows])
    patch_luma = np.array([float(r["gt_luma"]) for r in patch_rows])
    patch_noisy_psnr = np.array([float(r["psnr_in"]) for r in patch_rows])

    mse_in = sse_in / pixel_count
    mse_out = sse_out / pixel_count
    alpha_opt = -alpha_ad / alpha_d2 if alpha_d2 > 0 else float("nan")
    alpha_sse = alpha_a2 + 2.0 * alpha_opt * alpha_ad + alpha_opt * alpha_opt * alpha_d2
    alpha_mse = alpha_sse / pixel_count

    affine = []
    affine_sse_total = 0.0
    for c in range(3):
        n = float(affine_n)
        denom = n * affine_sum_x2[c] - affine_sum_x[c] * affine_sum_x[c]
        if abs(denom) < 1e-12:
            scale = 1.0
            bias = 0.0
        else:
            scale = (n * affine_sum_xy[c] - affine_sum_x[c] * affine_sum_y[c]) / denom
            bias = (affine_sum_y[c] - scale * affine_sum_x[c]) / n
        sse_c = (
            scale * scale * affine_sum_x2[c]
            + 2.0 * scale * bias * affine_sum_x[c]
            + bias * bias * n
            - 2.0 * scale * affine_sum_xy[c]
            - 2.0 * bias * affine_sum_y[c]
            + affine_sum_y2[c]
        )
        affine_sse_total += float(sse_c)
        affine.append({"channel": ch_names[c], "scale": float(scale), "bias": float(bias), "mse": float(sse_c / n)})
    affine_mse = affine_sse_total / pixel_count

    corr_num = corr_sum_xy - corr_sum_x * corr_sum_y / corr_n
    corr_den_x = corr_sum_x2 - corr_sum_x * corr_sum_x / corr_n
    corr_den_y = corr_sum_y2 - corr_sum_y * corr_sum_y / corr_n
    corr = corr_num / math.sqrt(max(corr_den_x * corr_den_y, 1e-30))

    def bin_table(values: np.ndarray, label: str) -> list[dict[str, float | str | int]]:
        rows = []
        for bin_name, mask in _quartile_bins(values):
            rows.append(
                {
                    "bin": f"{label} {bin_name}",
                    "count": int(mask.sum()),
                    "psnr_in_mean": _mean_for_mask(patch_psnr_in, mask),
                    "psnr_out_mean": _mean_for_mask(patch_psnr_out, mask),
                    "gain_mean": _mean_for_mask(patch_psnr_out - patch_psnr_in, mask),
                }
            )
        return rows

    scene_rows = []
    for i in range(n_img):
        if scene_pixels[i] <= 0:
            continue
        scene_rows.append(
            {
                "scene": i,
                "psnr_in": _psnr_from_mse(float(scene_sse_in[i] / scene_pixels[i])),
                "psnr_out": _psnr_from_mse(float(scene_sse_out[i] / scene_pixels[i])),
                "gain": _psnr_from_mse(float(scene_sse_out[i] / scene_pixels[i]))
                - _psnr_from_mse(float(scene_sse_in[i] / scene_pixels[i])),
            }
        )
    worst_scenes = sorted(scene_rows, key=lambda r: float(r["psnr_out"]))[:8]
    best_scenes = sorted(scene_rows, key=lambda r: float(r["psnr_out"]), reverse=True)[:8]

    total_freq_in = sum(freq_in.values())
    total_freq_out = sum(freq_out.values())
    freq_rows = []
    for name in ["low", "mid", "high"]:
        freq_rows.append(
            {
                "band": name,
                "input_share": freq_in[name] / total_freq_in if total_freq_in > 0 else float("nan"),
                "output_share": freq_out[name] / total_freq_out if total_freq_out > 0 else float("nan"),
                "output_vs_input_power": freq_out[name] / freq_in[name] if freq_in[name] > 0 else float("nan"),
            }
        )

    result = {
        "weights": args.weights,
        "patches": patch_count,
        "pixels": pixel_count,
        "params_m": n_params / 1e6,
        "psnr_in": float(patch_psnr_in.mean()),
        "psnr_out": float(patch_psnr_out.mean()),
        "psnr_in_global_mse": _psnr_from_mse(mse_in),
        "psnr_out_global_mse": _psnr_from_mse(mse_out),
        "mse_in": mse_in,
        "mse_out": mse_out,
        "channel": [
            {
                "channel": ch_names[c],
                "psnr_in": _psnr_from_mse(float(sse_ch_in[c] / (pixel_count / 3))),
                "psnr_out": _psnr_from_mse(float(sse_ch_out[c] / (pixel_count / 3))),
                "bias": float(sum_res_ch[c] / (pixel_count / 3)),
                "mae": float(sum_abs_res_ch[c] / (pixel_count / 3)),
            }
            for c in range(3)
        ],
        "ycbcr": [
            {
                "component": ycc_names[c],
                "psnr_in": _psnr_from_mse(float(sse_ycc_in[c] / (pixel_count / 3))),
                "psnr_out": _psnr_from_mse(float(sse_ycc_out[c] / (pixel_count / 3))),
                "mse_out": float(sse_ycc_out[c] / (pixel_count / 3)),
            }
            for c in range(3)
        ],
        "freq_luma": freq_rows,
        "alpha_noisy_to_pred": {
            "alpha": float(alpha_opt),
            "psnr_unclipped": _psnr_from_mse(alpha_mse),
            "mse_unclipped": float(alpha_mse),
            "interpretation": "alpha > 1 means the model is under-correcting along its own denoise direction; alpha < 1 means over-correcting.",
        },
        "affine_output_to_gt": {
            "psnr_unclipped": _psnr_from_mse(affine_mse),
            "mse_unclipped": float(affine_mse),
            "channels": affine,
        },
        "residual_corr_input_output": float(corr),
        "clip_fraction": {
            "low": clip_low / pixel_count,
            "high": clip_high / pixel_count,
        },
        "bins_by_gt_luma": bin_table(patch_luma, "gt_luma"),
        "bins_by_noisy_psnr": bin_table(patch_noisy_psnr, "noisy_psnr"),
        "worst_scenes": worst_scenes,
        "best_scenes": best_scenes,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def md_table(headers: list[str], rows: list[list[str]]) -> str:
        out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        out.extend("| " + " | ".join(r) + " |" for r in rows)
        return "\n".join(out)

    lines = [
        "# Residual Analysis",
        "",
        f"Weights: `{args.weights}`",
        f"Patches: {patch_count}",
        f"PSNR noisy (mean patch, eval convention): {result['psnr_in']:.3f} dB",
        f"PSNR denoised (mean patch, eval convention): {result['psnr_out']:.3f} dB",
        f"PSNR noisy (global MSE): {result['psnr_in_global_mse']:.3f} dB",
        f"PSNR denoised (global MSE): {result['psnr_out_global_mse']:.3f} dB",
        f"Residual correlation input/output: {corr:.4f}",
        "",
        "## Channel Residuals",
        md_table(
            ["channel", "psnr_in", "psnr_out", "bias", "mae"],
            [
                [
                    r["channel"],
                    f"{r['psnr_in']:.3f}",
                    f"{r['psnr_out']:.3f}",
                    f"{r['bias']:.4f}",
                    f"{r['mae']:.4f}",
                ]
                for r in result["channel"]
            ],
        ),
        "",
        "## YCbCr Residuals",
        md_table(
            ["component", "psnr_in", "psnr_out", "mse_out"],
            [
                [r["component"], f"{r['psnr_in']:.3f}", f"{r['psnr_out']:.3f}", f"{r['mse_out']:.4f}"]
                for r in result["ycbcr"]
            ],
        ),
        "",
        "## Luma Frequency Residuals",
        md_table(
            ["band", "input_share", "output_share", "out/input power"],
            [
                [
                    r["band"],
                    f"{r['input_share']:.4f}",
                    f"{r['output_share']:.4f}",
                    f"{r['output_vs_input_power']:.4f}",
                ]
                for r in result["freq_luma"]
            ],
        ),
        "",
        "## Closed-Form Output Diagnostics",
        f"Denoise strength alpha: {alpha_opt:.4f} -> unclipped PSNR {_psnr_from_mse(alpha_mse):.3f} dB",
        f"Per-channel affine output correction -> unclipped PSNR {_psnr_from_mse(affine_mse):.3f} dB",
        md_table(
            ["channel", "scale", "bias", "mse"],
            [[r["channel"], f"{r['scale']:.6f}", f"{r['bias']:.4f}", f"{r['mse']:.4f}"] for r in affine],
        ),
        "",
        "## Brightness Bins",
        md_table(
            ["bin", "count", "psnr_in", "psnr_out", "gain"],
            [
                [r["bin"], str(r["count"]), f"{r['psnr_in_mean']:.3f}", f"{r['psnr_out_mean']:.3f}", f"{r['gain_mean']:.3f}"]
                for r in result["bins_by_gt_luma"]
            ],
        ),
        "",
        "## Noisy-PSNR Bins",
        md_table(
            ["bin", "count", "psnr_in", "psnr_out", "gain"],
            [
                [r["bin"], str(r["count"]), f"{r['psnr_in_mean']:.3f}", f"{r['psnr_out_mean']:.3f}", f"{r['gain_mean']:.3f}"]
                for r in result["bins_by_noisy_psnr"]
            ],
        ),
        "",
        "## Worst Scenes",
        md_table(
            ["scene", "psnr_in", "psnr_out", "gain"],
            [[str(r["scene"]), f"{r['psnr_in']:.3f}", f"{r['psnr_out']:.3f}", f"{r['gain']:.3f}"] for r in worst_scenes],
        ),
        "",
        "## Best Scenes",
        md_table(
            ["scene", "psnr_in", "psnr_out", "gain"],
            [[str(r["scene"]), f"{r['psnr_in']:.3f}", f"{r['psnr_out']:.3f}", f"{r['gain']:.3f}"] for r in best_scenes],
        ),
        "",
    ]
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
