# Nagi NR

Ultra-lightweight, HDR-aware **blind image denoiser** for PyTorch (Apple Silicon / MPS first).

- **float32 everywhere** (training and inference)
- **HDR-safe**: reversible `asinh` range compression inside the graph
- **Blind**: no noise-level input required at inference
- Trained on **SIDD Medium sRGB** real-noise pairs (sRGB→linear inside the data loader)

## Variants

| Variant | base_channels | num_blocks       | params  | SIDD Val PSNR (sRGB) | ms / 256² patch (MPS) |
|---------|--------------:|------------------|--------:|---------------------:|----------------------:|
| S       | 32            | (2, 2, 4, 2, 2)  | 0.45 M  | 36.803 dB            | 26.2                  |
| **M**   | 48            | (4, 4, 8, 4, 4)  | 1.81 M  | **37.463 dB**        | 58.4                  |
| M2      | 48            | (4, 4, 8, 4, 4)  | 1.81 M  | 37.320 dB            | 57.3                  |
| L       | 64            | (4, 4, 8, 4, 4)  | 3.18 M  | 37.389 dB            | 78.2                  |

Recommended checkpoint: `runs/nagi_nr_m/nagi_nr_m_final.pt`.

`M2` is a phase-2 fine-tune from `M` with synthetic degradations. It may still be
useful for non-SIDD noise experiments, but on real SIDD Validation it is 0.143 dB
behind `M`.

`L` is a distillation experiment from NAFNet teacher outputs. It is larger and
slower than `M`, but did not beat `M` on SIDD Validation.

For reference (same eval harness, same data):

| Comparator | Params | PSNR | ms / patch |
|---|---:|---:|---:|
| SCUNet `color_real_psnr` | 17.95 M | 35.11 dB | 504.1 |
| NAFNet-width64           | 115.98 M | 40.21 dB | 387.1 |

## Repo layout

```
NR/
├── pixi.toml / pixi.lock           # reproducible local environment + tasks
├── packages/
│   ├── nagi_nr/                    # model, training, inference package
│   │   ├── src/nagi_nr/            # model, transforms, infer, data, losses, train
│   │   ├── configs/                # nagi_nr_s.yaml, nagi_nr_m.yaml
│   │   └── pyproject.toml
│   └── nagi_nr_bench/              # SIDD Val evaluator + comparators
│       ├── src/nagi_nr_bench/
│       │   ├── eval_sidd_val.py
│       │   └── third_party/        # vendored SCUNet, NAFNet
│       └── pyproject.toml
├── benchmarks/                     # third-party pretrained .pth files (not bundled)
│   ├── scunet/scunet_color_real_psnr.pth
│   └── nafnet/NAFNet-SIDD-width64.pth
├── data/                           # SIDD Validation .mat files
├── runs/                           # trained checkpoints + logs
├── SIDD_Medium_Srgb/               # training set
├── tests/
└── README.md
```

## Pixi Environment

```bash
pixi install
pixi run check-env
pixi run test
```

`pixi install` installs PyTorch plus both local packages in editable mode:
`nagi-nr` and `nagi-nr-bench`.

In the Codex sandbox, PyTorch can see that MPS support is built but the MPS
device is hidden. `pixi run check-env` should therefore report `auto device:
cpu` inside the sandbox. For real MPS training/evaluation, run the pixi task
outside the sandbox / with escalated execution and verify it first:

```bash
pixi run check-mps
```

The project sets `PYTORCH_ENABLE_MPS_FALLBACK=1` in the pixi activation
environment. CLI entry points accept `--device auto`, `mps`, `cuda`, or `cpu`.
Requesting `--device mps` from a sandboxed process now fails with a direct
message instead of a vague PyTorch backend error.

Pixi registers these commands:

| Command | From | Purpose |
|---|---|---|
| `nagi-train`     | nagi-nr      | Train on SIDD Medium sRGB |
| `nagi-denoise`   | nagi-nr      | Denoise a single image |
| `nagi-eval-sidd` | nagi-nr-bench | Evaluate on SIDD Validation (1280 patches) |

## Smoke test

```bash
pixi run test
```

## Train

Point `--sidd-root` at either `SIDD_Medium_Srgb/` or its `Data/` subdirectory.

```bash
# Small (existing recipe)
pixi run train-s

# Medium (speed-first variant)
pixi run train-m

# Medium phase-2 fine-tune
pixi run train-m2

# Large/distillation variant, resumes latest checkpoint
pixi run train-l
```

Equivalent direct CLI form:

```bash
pixi run nagi-train \
    --config packages/nagi_nr/configs/nagi_nr_s.yaml \
    --sidd-root SIDD_Medium_Srgb \
    --output runs/nagi_nr_s \
    --device mps \
    --ckpt-prefix nagi_nr_s
```

Checkpoints store `state_dict` (EMA weights, recommended for inference) plus
`model_state_dict`, `optimizer`, and the config used. Resume with:

```bash
pixi run nagi-train --config <yaml> --sidd-root ... --resume runs/nagi_nr_s/nagi_nr_s_0100000.pt
```

## Inference

CLI:

```bash
pixi run nagi-denoise \
    --weights runs/nagi_nr_m/nagi_nr_m_final.pt \
    --input  photo.png \
    --output photo_denoised.png \
    --device auto
```

For HDR (`.exr`, `.hdr`) the loader feeds linear data straight through and writes
linear output. `--input-space {auto,srgb,linear}` overrides auto-detection.

Python:

```python
from nagi_nr import Denoiser

dn = Denoiser.load("runs/nagi_nr_m/nagi_nr_m_final.pt", device="auto")
out = dn(linear_img, input_space="linear")   # HDR-safe, float32
```

The checkpoint embeds its config, so `Denoiser.load` auto-detects whether to
instantiate an S or M model — no flag needed.

## Evaluate on SIDD Validation

```bash
pixi run eval-s
pixi run eval-m
pixi run eval-m2
pixi run eval-l
pixi run eval-scunet
pixi run eval-nafnet
```

Defaults expect `data/ValidationNoisyBlocksSrgb.mat` and `data/ValidationGtBlocksSrgb.mat`
at the repo root.

## Architecture

```
x (linear float32, HDR-safe)
  └─ asinh(k·x) / asinh(k)          # k=8, reversible HDR compression
  └─ PixelUnshuffle(2)              # cheap 2x downsample
  └─ 3-level NAFLite U-Net          # residual in compressed space
  └─ PixelShuffle(2) + add(x_c)     # full-res compressed output
  └─ sinh(y_c · asinh(k)) / k       # decompress
y (linear float32)
```

- **NAFLiteBlock**: LayerNorm → 1×1 expand → DWConv3×3 → SimpleGate → SCA → 1×1, learnable beta residual.
- **Loss**: Charbonnier in compressed space + 0.05 · FFT magnitude L1 + 0.1 · linear Charbonnier.
- **Augment**: log-uniform exposure jitter ×0.25 – ×4 (exposes the net to HDR ranges).
- **EMA** with decay 0.999. Schedule: cosine, 2e-4 → 1e-6 over 300K iters with 2K warmup.
