# GAMA-IR Implementation Plan

## Source

Paper:

```text
GAMA-IR: Global Additive Multidimensional Averaging for Fast Image Restoration
arXiv:2404.00807
```

Public code and pretrained weights were not found, so the implementation is
from the paper description.

Relevant paper facts:

| variant | depth | width | SIDD PSNR claim | latency claim |
| --- | ---: | ---: | ---: | ---: |
| GAMA-IR-S | 18 | 42 | 40.02 dB | 15.1 ms on RTX A6000 |
| GAMA-IR-L | 19 | 80 | 40.41 dB | 20.7 ms on RTX A6000 |

Local exploratory variant:

| variant | depth | width | purpose |
| --- | ---: | ---: | --- |
| GAMAIR-M56 | 19 | 56 | middle-size capacity test between S and L |

The paper trains S for 6.4M iterations and L for 25.6M iterations, so local
training will not be a faithful full reproduction at first. We use short
screens to test whether the architecture rises faster than prior NagiQ attempts.

## Local Implementation

Implemented:

```text
packages/nagi_nr/src/nagi_nr/gamair.py
scripts/benchmark_gamair.py
packages/nagi_nr/configs/gamair_s_sidd_mse_2k.yaml
packages/nagi_nr/configs/gamair_l_sidd_mse_2k.yaml
```

Tasks:

```text
pixi run bench-gamair
pixi run train-gamair-s-sidd-mse-2k
pixi run train-gamair-l-sidd-mse-2k
```

Middle-size architecture screen:

```text
pixi run train-gamair-m56-faststart-128-2k
```

PolyU mixed task after the SIDD-only fast-start run:

```text
pixi run train-gamair-s-faststart-128-polyu-5k
```

Final quality fine-tune task after the PolyU 128px run:

```text
pixi run train-gamair-s-polyu-256-ft-10k
```

Teacher-restart diagnostic after the S 256px mixed fine-tune:

```text
pixi run train-gamair-s-teacher-restart-256-10k
pixi run train-gamair-s-teacher-restart-256-30k
```

This initializes from:

```text
runs/nagiq_gamair_s_faststart_128_10k/nagiq_gamair_s_faststart_128_10k_best.pt
```

and mixes SIDD Medium sRGB with PolyU cropped real/mean JPEG pairs:

```text
PolyU-Real-World-Noisy-Images-Dataset-master/CroppedImages/*_real.JPG
PolyU-Real-World-Noisy-Images-Dataset-master/CroppedImages/*_mean.JPG
```

PolyU samples have no NAFNet teacher target, so the training loss falls back to
GT-only supervision for those samples while keeping teacher distillation on SIDD.

The 256px fine-tune keeps the same mixed data but lowers the learning rate and
teacher weight. The intent is not more fast screening; it is the quality pass
that lets the model see larger spatial context after the cheaper 128px stages.

The teacher restart removes PolyU for one phase and raises teacher supervision
again. Its purpose is to test whether GAMAIR-S can still move toward the NAFNet
teacher on SIDD, rather than continuing the near-flat low-LR tail.

Architecture choices:

- U-shaped encoder/decoder.
- Total depth distributed as encoder `[2,2,2,2]`, decoder `[2,2,2,2]`,
  middle `2` for S and `3` for L.
- GAMA block averages along C, H, and W separately, applies 7x7 single-channel
  convolutions, broadcasts back, and adds to the feature map.
- NAF-style SimpleGate and pointwise/depthwise convolutions are used in each
  block.
- Output head starts at zero, so the model starts as exact identity.

## Local Speed

MPS random 256x256 benchmark:

| model | params | ms/patch |
| --- | ---: | ---: |
| GAMAIR-S | 10.42M | 209.4 |
| GAMAIR-L | 47.46M | 422.3 |

Interpretation:

- S is the first genuinely promising speed candidate in this line.
- L is likely too slow on this Mac unless Core ML changes the picture.
- S should be trained first.

## Screen Plan

Start with `GAMAIR-S` 2k:

| step | continue if |
| ---: | --- |
| 500 | clearly above noisy identity and rising |
| 1000 | stronger than W56Q direct at 1000, ideally >30 dB |
| 2000 | high enough to justify 10k extension |

Because this is a scratch model and the paper uses millions of iterations, 2k is
not expected to reach final quality. The useful signal is slope and stability.

If S rises fast:

1. extend S to 10k
2. export/check Core ML speed
3. then consider L only if S quality saturates below target

If S stays near noisy identity:

1. adjust initialization/head training
2. consider teacher-heavy distillation
3. do not launch L blindly

## GAMAIR-S 2k Result

Run:

```text
task: pixi run train-gamair-s-sidd-mse-2k
output: runs/nagiq_gamair_s_sidd_mse_2k
best/final: runs/nagiq_gamair_s_sidd_mse_2k/nagiq_gamair_s_sidd_mse_2k_best.pt
```

Validation curve:

| step | val128 PSNR | noisy | ms/patch |
| ---: | ---: | ---: | ---: |
| 500 | 23.451 dB | 21.530 dB | 301.2 |
| 1000 | 26.767 dB | 21.530 dB | 316.7 |
| 1500 | 28.556 dB | 21.530 dB | 308.5 |
| 2000 | 29.613 dB | 21.530 dB | 381.0 |

Judgment:

```text
GAMAIR-S is not near target at 2k, but it is not a failed/noisy-identity path.
It rises much more clearly than the W48/W56 channel-slice students.
```

Comparison:

| model | early result | note |
| --- | ---: | --- |
| W48Q head0 | 22.210 dB at 1000 | failed |
| W56Q head0 | 23.381 dB at 1000 | failed |
| W56Q residual-first | 21.577 dB at 500 | failed |
| GAMAIR-S | 26.767 dB at 1000, 29.613 dB at 2000 | viable slope |

Next decision:

- do not start GAMAIR-L yet; it is slower and more expensive
- extend GAMAIR-S to 10k only if we accept an overnight run
- a useful 10k gate would be `>=34 dB`; below that, the local recipe is too
  slow for this hardware
- if extending, reduce checkpoint churn and keep only best/final

## Active GAMAIR-S Extension

Started an additional 8k run from the 2k best checkpoint:

```text
task: pixi run train-gamair-s-sidd-mse-extend10k
output: runs/nagiq_gamair_s_sidd_mse_extend10k
init: runs/nagiq_gamair_s_sidd_mse_2k/nagiq_gamair_s_sidd_mse_2k_best.pt
pid: 90014
```

Recipe:

```text
lr: 8e-5 -> 1.5e-5
teacher_weight: 0.45 -> 0.25
ema: 0.997
save_every: 0
val128: every 1000
```

Judgment gates:

| local step | 10k-equivalent step | continue if |
| ---: | ---: | --- |
| 1000 | 3000 | clearly above 29.6 dB |
| 2000 | 4000 | preferably >=31.5 dB |
| 4000 | 6000 | preferably >=33 dB |
| 8000 | 10000 | >=34 dB to justify longer training |

## GAMAIR-S 10k-Equivalent Result

The additional 8k run completed:

```text
best/final: runs/nagiq_gamair_s_sidd_mse_extend10k/nagiq_gamair_s_sidd_mse_extend10k_best.pt
```

Full local curve:

| total-equivalent step | run | val128 PSNR | noisy | ms/patch |
| ---: | --- | ---: | ---: | ---: |
| 500 | 2k screen | 23.451 dB | 21.530 | 301.2 |
| 1000 | 2k screen | 26.767 dB | 21.530 | 316.7 |
| 1500 | 2k screen | 28.556 dB | 21.530 | 308.5 |
| 2000 | 2k screen | 29.613 dB | 21.530 | 381.0 |
| 3000 | extend10k | 30.588 dB | 21.530 | 216.1 |
| 4000 | extend10k | 31.789 dB | 21.530 | 215.9 |
| 5000 | extend10k | 32.791 dB | 21.530 | 216.0 |
| 6000 | extend10k | 33.416 dB | 21.530 | 216.0 |
| 7000 | extend10k | 33.901 dB | 21.530 | 215.9 |
| 8000 | extend10k | 34.295 dB | 21.530 | 220.9 |
| 9000 | extend10k | 34.623 dB | 21.530 | 220.4 |
| 10000 | extend10k | 34.810 dB | 21.530 | 218.0 |

Judgment:

```text
GAMAIR-S passes the 10k viability gate, but is not on a practical path to 39 dB
with the current local recipe.
```

Reason:

- the model rises cleanly and is much better than the failed W48/W56 students
- speed remains attractive at about 216-220 ms/patch on val128
- however the curve is flattening: +0.394, +0.328, +0.187 dB over the last
  three 1k intervals
- extrapolating this recipe toward 39 dB would likely require an impractically
  long run on this Mac

Next options:

1. Try a recipe change before longer training:
   - higher GT weight / lower teacher weight
   - stronger random crop diversity
   - possibly non-zero ending initialization so internal features learn earlier
2. Run GAMAIR-L only as a quality probe, not a speed candidate.
3. Treat GAMAIR-S as a fast architecture candidate but not yet a NAFNet
   replacement.
