# NagiQ Recovery Design After q48-trim 20k Failure

## Situation

`NagiQ q48-trim` 20k screening result:

```text
PSNR: 35.800 dB
speed: 276.2 ms/patch
```

This is below both:

```text
Nagi M: 37.463 dB
NAFNet teacher: 40.212 dB
```

Conclusion: do not extend this run. The current training pipeline failed the 20k quality gate.

## Most Likely Root Cause

The first issue is probably not architecture. It is the training/data recipe.

Current `SIDDPatchDataset.__getitem__` seeds crop/augmentation as:

```text
base_seed + idx * constant
```

The sampler changes index order per epoch, but each index always maps to the same crop and same augmentation. For `nagiq_q48_trim.yaml`:

```text
SIDD pairs: 320
patches_per_image: 4
dataset items: 1280
```

So the 46.66M-param NagiQ model was trained on only 1280 fixed 256x256 crops, repeated for 20k optimizer steps. That is far too little diversity for a large sRGB student. Nagi M survived a similar limitation because it is much smaller, uses 128x128 patches, and has a very different inductive bias.

Fixing this is higher priority than trying another architecture.

## Secondary Suspects

### 1. Loss clamps prediction

`NagiQSrgbDistillLoss` currently does:

```python
pred = pred.clamp(0.0, 1.0)
```

This can zero gradients when the model outputs outside range. For training, use unclipped output in the loss. Clamp only for evaluation/image export.

### 2. Identity start may be too slow for a large model

`NagiQ.ending` is initialized to zero to start as exact identity. This is stable, but initially blocks gradient into most of the network until the ending layer moves. For a large scratch model, this may slow early learning too much.

Candidate fix:

```text
ending weight: small normal/trunc_normal, not zero
ending bias: zero
residual scale: optional learnable or fixed small scale
```

Keep NAFBlock beta/gamma zero as in NAFNet-style safe residual blocks.

### 3. EMA-only evaluation may hide live weights

Checkpoints save EMA as `state_dict`; evaluation uses EMA. With scratch training and short runs, live weights may be better early. Always evaluate both:

```text
EMA weights
live model_state_dict weights
```

Then choose EMA decay from evidence.

### 4. No proof of train-set overfit

Before a long run, the model must prove it can overfit a small set toward GT/teacher. If it cannot reach high PSNR on a tiny training subset, the recipe is broken.

## Recovery Plan

### Phase A: Instrumentation and Fixes

Make the training pipeline measurable before spending more MPS days.

Required changes:

1. Add random crop per access.
2. Remove train-time output clamp from NagiQ loss.
3. Add live-weight checkpoint/eval option.
4. Add small-set train PSNR evaluator.
5. Add validation subset evaluator for quick feedback.

#### A1. Random crop per access

Add a dataset option:

```yaml
data:
  randomize_each_access: true
```

Implementation idea:

```text
seed = base_seed + idx * constant + access_counter * another_constant
```

For worker safety, maintain a per-dataset-instance counter. Perfect reproducibility is less important than crop diversity here. If reproducibility is needed later, pass epoch/sample id explicitly.

Expected effect:

```text
20k steps x effective batch 4 = 80k distinct-ish crops
instead of 1280 fixed crops
```

#### A2. Remove loss clamp

For NagiQ training:

```text
loss(pred, target), no clamp
eval/export: clamp to [0, 1]
```

If values explode, use a soft penalty:

```text
range_loss = relu(-pred).mean() + relu(pred - 1).mean()
```

but do not hard-clamp inside the main loss.

#### A3. Evaluate live vs EMA

Add a helper to export live weights:

```text
state_dict = model_state_dict
```

Evaluate both at the same checkpoint:

```text
q48_002000_ema.pt
q48_002000_live.pt
```

If live is consistently better before 50k, reduce EMA decay:

```text
0.999 -> 0.995 or delayed EMA start
```

#### A4. Train-subset evaluator

Create a diagnostic evaluator for fixed training crops:

```text
eval target:
  - GT PSNR
  - teacher PSNR
  - student vs teacher PSNR
```

If train-subset PSNR is low, do not run full validation; the model has not even learned the supervised target.

### Phase B: Micro Overfit Test

Purpose: prove the model and loss can learn before running overnight.

Config:

```text
model: q48-fast or q40-trim first
images: 8-16 SIDD pairs
patch_size: 256
randomize_each_access: false for this test
batch_size: 1
grad_accum_steps: 4
steps: 2000-5000
teacher_weight: 0.7
gt_weight: 0.3
```

Gate:

```text
train fixed-crop PSNR should climb rapidly
student-vs-teacher PSNR should exceed 38-40 dB on the tiny subset
```

If it cannot overfit:

```text
bug in model/loss/data/teacher alignment
```

Do not proceed.

### Phase C: Small Random-Crop Screening

Purpose: verify that dynamic crops improve generalization.

Candidate:

```text
model: q40-trim or q48-fast
patch_size: 256
randomize_each_access: true
steps: 10k
eval: max-patches 128 at 2k/5k/10k, full only if promising
```

Why smaller first:

```text
q48-trim is 276 ms/patch and slow to train.
q48-fast/q40-trim can validate the recipe faster.
```

Gate:

```text
10k 128-patch validation > NagiQ old curve by a large margin
10k full validation should be at least near Nagi M trajectory
```

If still around 35-36 dB:

```text
distillation/training recipe is still wrong
```

### Phase D: Real 40 dB Candidate

Only after Phase B/C pass.

Main candidate:

```text
model: q48-trim
patch_size: 256
randomize_each_access: true
batch_size: 1
grad_accum_steps: 8 if tolerable, otherwise 4
steps: 50k first gate
save_every: 2k
eval: 10k, 20k, 50k
```

Loss schedule:

```text
0-10k:   teacher 0.9, GT 0.1
10-30k:  teacher 0.7, GT 0.3
30-50k:  teacher 0.5, GT 0.5
50k+:    teacher 0.3, GT 0.7
```

Reason:

The previous run shifted toward GT too early while the model had not yet learned the teacher mapping. For a 40 dB student, first imitate the 40 dB teacher, then move toward GT.

Gate:

```text
10k full val < 37.0: stop / recipe still bad
20k full val < 38.0: stop q48-trim
50k full val < 39.0: switch design
50k full val >= 39.3: continue to 100k
```

### Phase E: If q48-trim Works But Is Too Slow

If q48-trim reaches quality but speed is not acceptable:

1. Train q48-fast with q48-trim as teacher.
2. Distill q48-trim -> q48-fast, not NAFNet -> q48-fast.
3. Compare quality loss vs speed gain.

This two-stage distillation is more realistic than asking q48-fast to learn NAFNet directly.

## Immediate Next Actions

Do these before any new long training:

1. Patch dataset with `randomize_each_access`.
2. Remove hard clamp from NagiQ loss.
3. Add live-weight export/eval helper.
4. Run micro-overfit.
5. Run q40/q48-fast 10k recipe screening.
6. Only then restart q48-trim.

## Do Not Do

Do not:

- continue current q48-trim checkpoint
- start q56-trim immediately
- run another 20k with fixed crops
- interpret 35.8 dB as architecture-only failure

The current result is most likely a training pipeline failure.

