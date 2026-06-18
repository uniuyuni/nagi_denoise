# nagi-nr

Core Nagi NR package — model, transforms, inference, and the training rig.

## Install (editable)

From the repo root:

```bash
pixi install
pixi run check-env
pixi run test
```

The pixi environment installs this package in editable mode. In Codex, MPS is
hidden inside the sandbox, so `check-env` falls back to CPU there. Use
`pixi run check-mps` outside the sandbox / with escalated execution before
running MPS training or full validation.

## Variants

A single `NagiNR` class drives all sizes; pick a YAML config to pin a variant.

| Variant | base_channels | num_blocks      | params | SIDD Val PSNR      |
|---------|---------------|-----------------|--------|--------------------|
| S       | 32            | (2, 2, 4, 2, 2) | 0.45 M | 36.803 dB |
| **M**   | 48            | (4, 4, 8, 4, 4) | 1.81 M | **37.463 dB** |
| M2      | 48            | (4, 4, 8, 4, 4) | 1.81 M | 37.320 dB |
| L       | 64            | (4, 4, 8, 4, 4) | 3.18 M | 37.389 dB |

Recommended checkpoint: `runs/nagi_nr_m/nagi_nr_m_final.pt`.

Configs live in `packages/nagi_nr/configs/`:

- `nagi_nr_s.yaml` — Small (current trained recipe)
- `nagi_nr_m.yaml` — Medium (current recommended checkpoint)
- `nagi_nr_m2.yaml` — M fine-tuned with synthetic degradations
- `nagi_nr_l.yaml` — L/distillation experiment; larger/slower than M, not recommended for SIDD

## CLI

```bash
# Train (Small variant on SIDD Medium sRGB)
pixi run train-s

# Train (Medium variant)
pixi run train-m

# Denoise a single image
pixi run nagi-denoise \
    --weights runs/nagi_nr_m/nagi_nr_m_final.pt \
    --input  photo.png \
    --output photo_denoised.png \
    --device auto
```

## Python API

```python
import torch
from nagi_nr import Denoiser, srgb_to_linear, linear_to_srgb

dn = Denoiser.load("runs/nagi_nr_m/nagi_nr_m_final.pt", device="auto")
out = dn(linear_img, input_space="linear")  # HDR-safe, float32
```

The trained checkpoint embeds its own `config["model"]`, so `Denoiser.load`
automatically instantiates the correct S/M/L variant — no flag needed at inference.

## Architecture overview

See top-level `README.md` for the full design discussion; in short:

```
x (linear float32, HDR-safe)
  └─ asinh(k·x) / asinh(k)          # k=8, reversible HDR compression
  └─ PixelUnshuffle(2)
  └─ 3-level NAFLite U-Net          # residual in compressed space
  └─ PixelShuffle(2) + add(x_c)
  └─ sinh(y_c · asinh(k)) / k       # decompress
y (linear float32)
```
