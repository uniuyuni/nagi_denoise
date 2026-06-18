# NAFNet-Fast Next Design

## Current Facts

The last W56Q residual-curriculum screen failed:

| method | checkpoint | val128 PSNR | noisy |
| --- | ---: | ---: | ---: |
| W48Q head0 direct | 1000 | 22.210 dB | 21.530 dB |
| W56Q head0 direct | 1000 | 23.381 dB | 21.530 dB |
| W56Q residual-first | 250 | 21.564 dB | 21.530 dB |
| W56Q residual-first | 500 | 21.577 dB | 21.530 dB |

This rejects the current width-reduction recipe. The problem is not just that
W48/W56 need more iterations. The raw channel-sliced head collapses to about
5 dB, and the safe zero head cannot reconstruct a useful teacher residual from
the sliced internal features. That means the denoising function is not preserved.

The good path we still have:

| path | quality | real EXR speed | note |
| --- | ---: | ---: | --- |
| full NAFNet Core ML fp16 512 batch4 | teacher quality | 97.7s | too slow |
| P3 fine-tuned | 39.760 dB val128 | not enough MAC reduction | quality candidate only |
| half-res residual upsample | visually close, some noise left | 27.7s | fast preview |
| 2/3-res residual upsample | worse visually than half-res | 43.8s | rejected |

The target remains quality first:

```text
primary: visual quality and SIDD PSNR
secondary: real EXR speed, preferably <=60s for 7728x5152
avoid: another long low-signal training run
```

## Design Decision

Move the main experiment from single-model width reduction to a cascade:

```text
NAFNet-Fast-C

input sRGB
  -> half-resolution full NAFNet residual pass
  -> full-resolution cleanup network
  -> output sRGB
```

The half-resolution pass gives the large-scale denoising and color stability.
The cleanup network is trained only to remove the remaining full-resolution
noise, texture error, and chroma speckle.

This is no longer an exact NAFNet prune. It is a production-speed approximation
designed around the evidence we have:

- half-res residual is already fast enough
- its visual failure mode is residual noise, not total collapse
- a small full-res network is more likely to learn the missing local correction
  than a sliced W56 model is to relearn the whole denoising function
- the speed budget allows a small cleanup pass after the 27.7s broad pass

## Why This Is More Logical Than W48/W56

W48/W56 tries to learn:

```text
noisy image -> full denoised image
```

from a damaged initialization. It must recover the whole mapping.

The cascade cleanup learns:

```text
noisy image
half-res denoised image
half-res residual
    -> full NAFNet correction delta or GT correction delta
```

That is a smaller problem. The broad denoise has already happened. If the
remaining error is mostly local high-frequency noise, a small full-resolution
network can plausibly fix it.

The first gate must prove that assumption before any long training:

```text
measure:
  half-res residual PSNR
  full NAFNet PSNR
  delta = full_NAFNet - half_res_output
  delta energy by luma/chroma and frequency band

continue only if:
  delta is small compared with full output residual
  delta is mostly local/high-frequency
  a small cleanup model can overfit a tiny subset quickly
```

If the delta is global or exposure/color-shift dominated, the cleanup design is
wrong and should be stopped early.

## Architecture

Name:

```text
NAFNet-Fast-C1
```

Inputs:

```text
noisy_srgb:       3 channels
coarse_srgb:      3 channels  # half-res NAFNet residual upsample result
coarse_residual:  3 channels  # coarse_srgb - noisy_srgb
```

Total input channels: 9.

Output:

```text
delta_srgb: 3 channels
final = coarse_srgb + residual_scale * delta_srgb
```

Initialize the final conv to zero, so the whole cascade starts exactly as the
half-resolution output. This avoids the W48/W56 raw-head collapse failure.

Recommended cleanup model:

```text
width: 32 or 40
levels: 3
enc blocks: [1, 2, 2]
middle blocks: 4
dec blocks: [1, 1, 1]
block type: NAF-lite / depthwise separable residual block
params target: 1M to 4M
```

The model must be Core ML friendly:

- Conv2d
- depthwise Conv2d
- pointwise Conv2d
- SimpleGate-like multiply
- bilinear-free inside model
- no attention windows
- no dynamic control flow

Do not use SCUNet-style window attention. It is too slow and too complex for the
current hardware target.

## Training Targets

Use precomputed teacher/coarse pairs so training does not run the large model
online.

For every SIDD crop:

```text
noisy
gt
teacher_full = full NAFNet output
coarse = half-res NAFNet residual-upsample output
target_delta_teacher = teacher_full - coarse
target_delta_gt = gt - coarse
```

Loss:

```text
pred = coarse + delta

loss =
  1.00 * MSE(pred, gt)
  0.50 * MSE(pred, teacher_full)
  0.10 * MSE(delta, target_delta_teacher)
  0.01 * gradient_loss(pred, gt)
  0.001 * soft_range_penalty(pred)
```

Teacher is a stabilizer, not the only target. GT MSE remains primary because
SIDD PSNR is the metric.

## Validation Gates

Do not train this for days before it earns the right.

Stage A: residual audit, no training.

| metric | continue if |
| --- | --- |
| half-res val128 PSNR | within a plausible cleanup gap, preferably >=37.5 dB |
| delta spectrum | mostly high-frequency/local |
| chroma delta | not dominant global color shift |

Stage B: tiny overfit.

Train on 16 to 32 fixed crops for 300 to 500 steps.

| metric | continue if |
| --- | --- |
| train PSNR | climbs quickly above coarse baseline |
| train teacher-delta MSE | falls clearly |
| output | no color shift, no ringing |

Stage C: 2k screen.

| checkpoint | continue if |
| --- | --- |
| 500 | val128 improves over coarse by clear margin |
| 1000 | val128 is on a trajectory toward 39 dB |
| 2000 | val128 >=38.5 dB, otherwise redesign |

Stage D: speed gate.

Core ML export the cleanup model and measure on the real EXR.

```text
half-res broad pass: 27.7s known baseline
cleanup budget: <=25s preferred, <=32s maximum
total target: <=60s
```

If cleanup alone exceeds 32s, shrink the cleanup model before training longer.

## Implementation Plan

1. Add a script to generate a small residual-audit dataset.
   - Run or reuse full NAFNet teacher output.
   - Run half-res residual output.
   - Save `noisy`, `gt`, `teacher`, `coarse`, and `delta` as tensors or npz.
   - task: `pixi run audit-nafnet-fast-c`

2. Add a residual audit report.
   - PSNR: noisy, coarse, teacher.
   - Delta statistics: Y/CbCr, low/mid/high frequency energy.
   - Example crops for visual inspection.

3. Implement `FastCleanupNet`.
   - 9-channel input, 3-channel delta output.
   - Zero final conv.
   - Width 32 first, width 40 only if speed budget permits.

4. Add tiny-overfit training config.
   - 16 to 32 fixed crops.
   - 300 to 500 steps.
   - This is a capability test, not a quality run.

5. Benchmark untrained cleanup Core ML.
   - Reject slow architecture before spending training time.

6. If Stage A/B/C pass, launch the 2k screen.

## Backup: Teacher-Compatible Lite Blocks

If the cascade residual audit fails, the next single-model route is not
W48/W56 slicing. It should be teacher-compatible block replacement:

```text
start from full width64 teacher
replace selected late NAFBlocks with LiteNAFBlocks
initialize replacement as identity
distill only the replaced block groups first
then fine-tune whole model
```

This preserves the teacher's channel width and skip topology. It avoids the
feature-space break caused by slicing channels. The cost reduction is smaller
than W48, but the quality risk is much lower.

Use this only after the cascade audit, because it is more implementation work
and likely gives less speed for the real EXR target.

## Stage A Result

Implemented:

```text
scripts/audit_nafnet_fast_cascade.py
task: pixi run audit-nafnet-fast-c
output: runs/nafnet_fast_cascade_audit/audit.md
```

Result on the first 128 SIDD validation patches:

| metric | value |
| --- | ---: |
| noisy PSNR | 21.530 dB |
| half-res coarse PSNR | 24.582 dB |
| full teacher PSNR | 39.553 dB |
| cleanup gap | 14.970 dB |
| coarse vs teacher PSNR | 24.692 dB |

The residual was mostly high-frequency, but the magnitude was far too large:

| metric | value |
| --- | ---: |
| delta RMS RGB | 0.065359 |
| delta energy vs teacher residual | 0.482957 |
| Y high-band share | 0.830205 |
| chroma/luma energy | 0.279541 |

Judgment:

```text
cascade cleanup as the SIDD-quality path: reject
```

Reason: the full-resolution cleanup model would not be learning a small
correction. It would need to recover a large part of the denoising function.
That brings us back to the failed small-student problem.

The real EXR half-res preview can remain a fast preview mode, but it should not
be the main 39 dB-class model path.

Next step: teacher-compatible branch pruning. Instead of slicing width or
removing whole blocks, measure whether individual NAFBlock branches can be
removed or replaced:

```text
scripts/audit_nafnet_branch_prune.py
task: pixi run audit-nafnet-branch-prune
```

## Branch-Prune Result

Implemented:

```text
scripts/audit_nafnet_branch_prune.py
scripts/search_nafnet_branch_prune.py
scripts/eval_nafnet_branch_prune.py

tasks:
  pixi run audit-nafnet-branch-prune
  pixi run search-nafnet-branch-prune
  pixi run eval-nafnet-branch-prune-p6
```

Single-branch audit on 8 validation patches showed many safe-looking deep
branches:

| rank | branch | drop | saved GMAC |
| ---: | --- | ---: | ---: |
| 1 | enc3.3.attn | +0.055 dB | 0.815 |
| 2 | enc3.2.attn | +0.054 dB | 0.815 |
| 3 | middle.9.ffn | +0.039 dB | 0.805 |
| 4 | enc3.7.attn | +0.034 dB | 0.815 |
| 5 | dec0.1.ffn | +0.032 dB | 0.805 |
| 6 | middle.10.ffn | +0.032 dB | 0.805 |

Dangerous branches:

| branch | drop |
| --- | ---: |
| dec0.0.attn | -28.521 dB |
| enc3.0.attn | -0.439 dB |

Greedy search on 8 patches selected:

```text
dec0.1.ffn
enc3.1.attn
enc3.2.attn
enc3.3.attn
enc3.7.attn
middle.2.ffn
```

The 8-patch search looked excellent, but 128-patch validation showed the greedy
P6 mask was too aggressive:

| mask | val128 PSNR | drop | saved GMAC | ideal speed |
| --- | ---: | ---: | ---: | ---: |
| full teacher | 39.553 dB | 0 | 0 | 1.000x |
| branch P4: enc3 attn x4 | 39.317 dB | -0.235 dB | 3.260 | 1.054x |
| branch P5: P4 + middle.2.ffn | 39.272 dB | -0.281 dB | 4.065 | 1.069x |
| branch P5: P4 + dec0.1.ffn | 39.191 dB | -0.361 dB | 4.065 | 1.069x |
| branch P6 greedy | 39.133 dB | -0.419 dB | 4.871 | 1.083x |

Judgment:

```text
branch pruning is quality-controllable, but not enough for the 1.5x speed target
by itself.
```

Compared with P3:

| model | val128 initial | saved GMAC | ideal speed |
| --- | ---: | ---: | ---: |
| P3-middle whole-block prune | 39.141 dB | 4.86 | 1.083x |
| branch P6 | 39.133 dB | 4.871 | 1.083x |
| branch P5-middle | 39.272 dB | 4.065 | 1.069x |

The branch route is still useful because it identifies safer, more granular
removals. But the first branch masks only reproduce P3-scale speed. They do not
solve the real EXR 60s target.

Next engineering implication:

- P5-middle is the safest branch-prune fine-tune candidate.
- P6 is too aggressive without fine-tune, but could recover similarly to P3.
- Neither is likely to reach 1.5x unless followed by a second, more aggressive
  round after fine-tuning.
- Before training, physical branch-skipping export is required; the current
  audit gates zero branches after computing them, so it measures quality only,
  not real speed.

## Immediate Next Step

Implement physical branch-skipping export for `branch P5-middle`, then benchmark
Core ML/PyTorch speed before training.

The key question is:

```text
Does branch-skipping produce real speedup on MPS/Core ML, or is the runtime
still dominated by unchanged memory traffic and surrounding ops?
```

If speedup is real, fine-tune P5/P6 briefly. If speedup is not real, the
remaining route is backend/scheduler optimization around full/P3.
