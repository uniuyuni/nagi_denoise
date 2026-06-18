# NagiQ Next Training Design

## Confirmed Facts

The old `q48-trim` 20k result is no longer a reliable architecture verdict.
That run mixed three bad conditions:

```text
fixed crop repetition
train-time hard clamp in the old loss path
EMA-only evaluation
```

The recovery experiments established:

| experiment | evaluation | result |
| --- | --- | ---: |
| q40 micro 500 live | train subset | 38.021 dB |
| q40 micro 1000 live | train subset | 41.925 dB |
| q40 micro 1000 EMA | train subset | 33.204 dB |
| q40 fixed final live | SIDD full val | 34.467 dB |
| q40 random final live | SIDD full val | 34.957 dB |
| q40 random final EMA | SIDD val 128 | 30.506 dB |
| q48-fast random 2k live | SIDD val 128 | 32.956 dB |
| q48-fast random 6k live | SIDD val 128 | 36.106 dB |
| q48-fast random 6k EMA | SIDD val 128 | 36.244 dB |
| q48-fast random 12k live | SIDD val 128 | 36.767 dB |
| q48-fast random 12k EMA | SIDD val 128 | 36.809 dB |
| q48-fast random 12k EMA | SIDD full val | 37.206 dB |
| q48-trim corrected 12k EMA | SIDD val 128 | 36.816 dB |
| q48-trim corrected 12k EMA | SIDD full val | 37.212 dB |
| q48-trim corrected 20k live | SIDD val 128 | 37.006 dB |
| q48-trim corrected 20k EMA | SIDD val 128 | 37.075 dB |
| q48-trim corrected 20k EMA | SIDD full val | 37.409 dB |

Interpretation:

```text
model/loss/teacher alignment can learn
random crop per access improves generalization by about +0.49 dB
EMA decay 0.999 is too stale for short screening
live weights must be the primary early evaluation target
```

## Design Constraint

Do not restart a long q48-trim run yet. First isolate the corrected recipe on a
cheaper model with enough capacity to be informative.

The next run should change only the confirmed broken parts:

```text
randomize_each_access: true
train loss clamp: false
evaluate live weights
less-stale EMA for observation only
resume from live weights, not EMA
```

Avoid adding extra architecture ideas, linear auxiliary losses, synthetic data,
or new loss terms in this run. Those would make attribution muddy.

## Next Candidate: q48-fast Random Screen

Use `q48-fast` before returning to `q48-trim`.

Reason:

```text
q40-trim:   32.45M, 21.12 GMAC, 213.6 ms/patch, full val 34.957 after 3k/64 pairs
q48-fast:   35.71M, 23.83 GMAC, 208.0 ms/patch
q48-trim:   46.66M, 30.28 GMAC, 262.0 ms/patch
```

`q48-fast` is close to q40 in measured inference cost but has wider channels.
It is the best low-risk probe for whether the corrected recipe scales before
spending another long q48-trim run.

## q48-fast Random Screen Config

```yaml
model:
  preset: q48-fast

data:
  patch_size: 256
  patches_per_image: 4
  chunk_size: 4
  exposure_jitter: null
  flip_rot: true
  randomize_each_access: true
  num_workers: 1
  prefetch_factor: 1
  return_teacher: true
  output_space: srgb

train:
  batch_size: 1
  grad_accum_steps: 4
  total_iters: 12000
  warmup_iters: 600
  lr: 2.0e-4
  lr_min: 2.0e-5
  weight_decay: 1.0e-3
  grad_clip: 1.0
  ema_decay: 0.995
  log_every: 100
  save_every: 2000
  keep_last_ckpts: 5

loss:
  charbonnier_eps: 1.0e-3
  teacher_weight_start: 0.90
  teacher_weight_end: 0.60
  grad_weight: 0.02
  clamp_pred: false
```

Why these values:

```text
full SIDD pairs instead of max_pairs=64
random crops to avoid repeated patch memorization
teacher remains strong because the teacher is the only local 40 dB reference
GT weight increases enough to avoid pure teacher imitation
EMA 0.995 is less stale, but live remains the primary metric
keep_last_ckpts limits disk growth
```

## Evaluation Gates

Primary metric is live SIDD validation PSNR.

Use 128-patch validation for quick checks:

```text
2k checkpoint: smoke check only
6k checkpoint: trajectory check
12k final: full validation if 128-patch trend is sane
```

Run full validation at 12k regardless if the model is stable, because q40
showed that 128-patch ranking matched full validation closely.

Gates:

| q48-fast 12k full live | decision |
| ---: | --- |
| < 36.0 dB | recipe still insufficient; do not scale up |
| 36.0-37.5 dB | continue q48-fast to 30k only if curve is still rising |
| 37.5-38.5 dB | recipe is working; choose between q48-fast continuation and q48-trim |
| > 38.5 dB | strong signal; run q48-trim corrected recipe |

For each evaluated checkpoint also compare:

```text
live validation
EMA validation
train-subset live
```

Expected diagnostic patterns:

```text
train high, validation low  -> overfit or insufficient crop/data diversity
both low                    -> recipe/optimization still wrong
validation rising           -> continue or scale model
EMA far below live           -> keep evaluating live, reduce/delay EMA later
```

## q48-fast 12k Result

The `q48-fast` corrected-recipe screen passed, but it is not yet close enough
to 40 dB to justify only extending the same run as the main strategy.

```text
q48-fast random 12k EMA full validation: 37.206 dB
q40 random 3k live full validation:      34.957 dB
improvement:                             +2.249 dB
```

The corrected recipe clearly scales beyond q40. The model is still behind:

```text
Nagi M:             37.463 dB
NAFNet teacher:     40.212 dB
gap to Nagi M:      -0.257 dB
gap to teacher:     -3.006 dB
```

This means:

```text
recipe fix is real
q48-fast capacity/training length is not enough for 40 dB
q48-trim corrected run is now justified
```

## After q48-fast

If q48-fast proves the corrected recipe, then run q48-trim with the same data
and loss policy:

```text
model: q48-trim
randomize_each_access: true
teacher_weight: 0.90 -> 0.55 or 0.60
ema_decay: 0.995 for screening
evaluate live at every gate
```

Only consider q56-trim if q48-trim approaches the target but plateaus below
about 39.6 dB.

## q48-trim Corrected Result

The corrected q48-trim run improved over q48-fast, but not enough to become a
40 dB candidate by itself.

```text
q48-fast random 12k EMA full:       37.206 dB
q48-trim corrected 12k EMA full:    37.212 dB
q48-trim corrected 20k EMA full:    37.409 dB
Nagi M:                             37.463 dB
NAFNet teacher:                     40.212 dB
```

Interpretation:

```text
q48-trim capacity helps only modestly: +0.203 dB over q48-fast
20k still remains slightly below Nagi M
gap to teacher is still about 2.8 dB
simply extending the same q48-trim recipe is unlikely to reach 40 dB
```

The next design step should not be a blind longer q48-trim run. It should target
the remaining recipe/model gap: stronger teacher use, better late GT fine-tune,
checkpoint averaging/SWA, or a q56/q64 capacity test with a short gate.

## What Not To Do Yet

Do not:

```text
start a 100k run
judge short runs by EMA only
add linear-space auxiliary losses
change NAFBlock internals
increase batch/worker count aggressively
delete old checkpoints without an explicit cleanup decision
```

The next run is a controlled test of the corrected training recipe, not a final
40 dB attempt.
