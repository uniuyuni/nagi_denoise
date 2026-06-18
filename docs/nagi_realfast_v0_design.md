# Nagi-RealFast v0 Design

## Goal

Build a practical real-image denoiser for the local Mac/MPS/Core ML path.

Priority:

1. Image quality
2. Inference speed
3. Training speed
4. Model size

This design is not a paper-faithful clone and not a novelty-first model. It is a
purpose-built architecture for sRGB real-noise denoising under the measured local
constraints.

## Constraints From Prior Runs

- NAFNet width64 reaches the quality target, but PyTorch MPS is about 520 ms per
  256x256 patch. Stage timing shows most cost is in repeated full-resolution and
  near-full-resolution NAF blocks, especially `enc0` and `dec3`.
- NagiQ keeps NAF blocks and remains around 208-317 ms per 256x256 patch. It is
  faster than full NAFNet but still dominated by 1x1 expanded channel mixing.
- GAMAIR-S has fewer parameters, but measured speed is about 209 ms per 256x256
  patch. Its global averaging/broadcast/transposes do not convert into local MPS
  speed wins.
- Long GAMAIR-S teacher restart reached only 37.033 dB at 20k and flattened.
  That path is not a likely 39-40 dB route.

Therefore v0 must not spend most compute on high-resolution repeated NAF-style
FFN blocks, and must avoid global mixing patterns that are cheap on paper but
memory-unfriendly on MPS.

## Core Hypothesis

Real denoising needs different operations at different frequency bands:

- High-resolution path: preserve edges and remove local chroma/luma noise with
  cheap local filters.
- Low-resolution path: estimate broad noise strength, color cast, and texture
  context with a small number of stronger blocks.
- Output: predict a conservative residual, not a clean image directly.

NAFNet spends expensive learned mixing everywhere. v0 should spend high-capacity
mixing only after downsampling, while keeping full-resolution processing shallow
and local.

## Architecture

Working name: `NagiRealFast`

Input/output space:

- Input: sRGB float `[0, 1]`
- Output: sRGB float `[0, 1]`
- Model predicts residual `r`; final output is `x + scale * r`
- Initial residual scale/head is zero or near-zero to start from identity

Default v0 preset:

```text
width: 48
levels: 3
stem: 3x3 conv, 3 -> 48
enc blocks: [2, 3, 4]
middle blocks: 6
dec blocks: [2, 2, 1]
downsample: stride-2 3x3 conv
upsample: 1x1 conv + PixelShuffle
high-res block expansion: 1.25
low-res block expansion: 2.0
```

### Block Types

#### LocalDenoiseBlock

Used at full and half resolution.

Structure:

```text
identity
 -> LayerNorm2d
 -> 1x1 C -> round_even(1.25C)
 -> 3x3 depthwise
 -> SimpleGate
 -> 1x1 round_even(0.625C) -> C
 -> residual scale beta
```

No second FFN branch at high resolution.

Reason:

NAFBlock has two expensive 1x1 branches per block. The second FFN is useful, but
at full resolution it is the main speed tax. For denoising, the first
depthwise-gated branch already provides local adaptive smoothing. Capacity is
moved to lower resolutions where spatial area is 1/4 or 1/16.

#### ContextBlock

Used at quarter/eighth/middle resolution.

Structure:

```text
LocalDenoiseBlock with expansion 2.0
plus lightweight FFN branch:
  LayerNorm2d
  1x1 C -> 2C
  SimpleGate
  1x1 C -> C
  residual scale gamma
```

Reason:

Low-resolution channels can afford richer channel mixing. This handles texture
classification, noise strength, and broad color consistency without repeatedly
paying full-resolution cost.

#### NoiseGuideHead

Small side head from the stem:

```text
3x3 depthwise/separable convs -> 4 maps
maps: luma_noise, chroma_noise, edge_guard, residual_gain
```

The maps modulate only residual strength, not feature routing:

```text
residual = residual * sigmoid(residual_gain)
```

Reason:

Real images fail visually when flat chroma noise remains, or when textured/edge
areas are over-smoothed. A cheap guide head gives the model an explicit place to
represent that distinction without adding attention.

## Estimated Cost

The measured NagiQ `q48-fast` is 23.83 GMAC and about 208 ms per 256x256 patch.
v0 removes the second NAF FFN from the high-resolution and half-resolution
blocks, keeps fewer blocks near output resolution, and spends capacity deeper.

Target estimates:

```text
params: 18-28M
GMAC 256x256: 12-16
PyTorch MPS random patch: 115-150 ms
Core ML 512 tile: must be benchmarked after export
```

Gate:

- If untrained random-input MPS speed is slower than 160 ms per 256 patch, do not
  train.
- If Core ML export has unsupported or slow ops, revise before training.

## Training Recipe

No long exploratory training before speed/export gates pass.

Data:

- Primary: SIDD Medium sRGB
- Optional fine-tune: PolyU real/mean pairs only after SIDD quality is viable
- Patch size: 256 for main training
- Batch size: 1, grad accumulation 2 on MPS
- `num_workers: 1`, low prefetch to keep system usable

Loss:

```text
loss = gt_mse
     + 0.15 * teacher_mse
     + 0.01 * grad_l1
     + 0.02 * chroma_l1
     + 0.01 * lowfreq_l1
```

Details:

- `gt_mse`: direct SIDD PSNR alignment.
- `teacher_mse`: use NAFNet teacher softly; do not let it dominate.
- `grad_l1`: protects edges from mush.
- `chroma_l1`: YCbCr Cb/Cr residual penalty, aimed at visible color noise.
- `lowfreq_l1`: downsampled prediction vs GT, preventing color/brightness drift.

Schedule:

```text
total_iters: 80000 first real run
warmup: 500
lr: 2e-4 for v0 from scratch, cosine to 1e-6
ema: 0.998
validation: every 2000, SIDD val 128 patches
save: best only
```

Why higher LR than the recent GAMAIR restart:

The restart was fine-tuning an already-trained path with low LR. A new identity
initialized residual model needs enough LR to learn denoising quickly. The zero
head keeps the early phase stable.

## Decision Gates

### Gate 0: Static/Random Speed

Before training:

- instantiate model
- count params
- estimate GMAC
- random MPS forward, 256x256
- Core ML export smoke test if supported

Pass:

```text
MPS 256 patch <= 160 ms
no unsupported Core ML operation
```

Fail:

- reduce full-res blocks first
- then reduce width to 44
- do not reduce low-res/middle blocks first

### Gate 1: 2k Training

This is the only short screen before committing to a real run.

Pass:

```text
2k val >= 36.2 dB
loss descending normally
no color cast on sample output
```

Fail:

- if PSNR is low but images look clean: adjust loss weights
- if images are smeared: reduce chroma/lowfreq and increase grad
- if both low: architecture lacks capacity

### Gate 2: 10k Training

Pass:

```text
10k val >= 37.3 dB
trend from 8k to 10k >= +0.05 dB
```

Fail:

- if below 37.0: stop; architecture is not worth long training
- if 37.0-37.3 but trend healthy: continue to 20k only

### Gate 3: 80k Production Run

Expected useful target:

```text
SIDD val: 38.5-39.2 dB
real image: less chroma noise than GAMAIR-S, fewer tile artifacts than NAFNet tiling
speed: at least 1.5x faster than local NAFNet baseline
```

If it cannot beat 38 dB, treat it as a speed-specialized model, not the main
quality path.

## Implementation Plan

1. Add `nagi_realfast.py` with the model and preset builder.
2. Extend `train_q.py` and `eval_sidd_val.py` with `kind: realfast`.
3. Add a benchmark script or extend `benchmark_nagiq.py` to include realfast.
4. Add one config: `nagi_realfast_v0_80k.yaml`.
5. Run Gate 0 speed/export checks.
6. If Gate 0 passes, run Gate 1 2k.

No broad search, no many-variant sweep, and no long run until the gates pass.

## Rejection Criteria

Reject v0 if any of these happen:

- Speed is not clearly better than `q48-fast`.
- 10k PSNR is below the recent GAMAIR-S restart trend.
- Color noise is visibly worse than the half/fast practical outputs.
- Core ML path is slower or less stable due to unsupported ops.

The point is not to defend the new model. The point is to quickly prove whether
this theory buys practical quality per millisecond.
