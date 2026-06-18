# Perfect NR Experiment Log

## 2026-06-18 Kickoff

### Starting Point

We are starting a fresh research track after the NAFNet / SCUNet / Core ML line
exposed several failure modes:

- Core ML `ALL` was not automatically faster or safer.
- `neuralnetwork + ALL` converted but was too slow.
- `mlprogram + cpu_and_gpu` was the practical Core ML path.
- SCUNet and PyTorch also failed when HDR input exceeded the assumed working range.
- HDR-white-aware log normalization fixed the major purple/cyan highlight failure.
- Low-frequency transfer parameters are easy to misread and can remove highlight detail.

### Known Diagnostic Crop

Primary crop:

```text
/Users/uniuyuni/PythonProjects/platypus/SCUNet_CoreML/test_inputs/X-T5 Room.EXR
```

Measured facts:

```text
RGB max:        about 5.19
luma p99:       about 3.81
peak > 1.0:     about 9.17% of pixels
luma > 1.0:     about 7.29% of pixels
channels:       B/G/R float
```

This is a real HDR test crop, not merely an EXR container with SDR values.

### Important Outputs From The Previous Investigation

Useful references in Platypus `SCUNet_CoreML/test_outputs`:

```text
xt5_current_log_preview.png
xt5_hdrwhite_log_preview.png
xt5_coreml_mlprogram_cpu_gpu_hdr_log_preview.png
xt5_coreml_helper_cpu_gpu_hdr_scaled_transfer_preview.png
xt5_coreml_cpu_gpu_hdrscaled070_040_d010_preview.png
```

Current interpretation:

- `current_log` shows why fixed SDR log normalization is invalid for HDR.
- `hdrwhite_log` shows the main color failure can be removed before postprocessing.
- low-frequency transfer can improve or harm detail depending on highlight parameters.

### Open Questions

1. What should replace overloaded low-frequency transfer?
2. Can we preserve original highlight detail without restoring noise?
3. Is a confidence-map model better than a plain RGB residual model?
4. Can a small local model beat SCUNet/NAFNet on real HDR diagnostic crops even if
   SIDD PSNR is lower?
5. What is the minimum tile overlap for each model without seams?

### Next Experiment

Build a diagnostic runner for EXR crops that outputs:

- processed EXR;
- preview PNG with fixed tone mapping;
- metadata JSON;
- high-frequency / chroma drift metrics;
- one markdown summary row.

The first runner should support external commands or simple Python callables so
SCUNet, Core ML, and future NagiPerfect models can be compared with the same
measurement code.

## 2026-06-18 Phase 0 Harness

Added:

```text
scripts/perfect_nr_probe.py
scripts/perfect_nr_compare.py
```

`perfect_nr_probe.py` reads EXR/TIFF/PNG/JPEG, writes stable preview PNG and
linear/HDR JSON stats.

`perfect_nr_compare.py` compares candidate outputs against a reference crop using
the same preview tone curve and a chosen diagnostic mask.

First comparison:

```text
runs/perfect_nr/compare/xt5_room/comparison.md
```

Top-luma 1% mask results:

| candidate | chroma drift | luma delta | HF retention | interpretation |
| --- | ---: | ---: | ---: | --- |
| no_transfer | 0.01546 | +0.02453 | 0.494 | color/luma stable, but highlight detail weak |
| scaled_d020 | 0.01393 | -1.78287 | 1.052 | detail returns, but highlight luma is pulled down |
| scaled_d010 | 0.01685 | -2.00880 | 1.205 | stronger detail, more luma loss |
| fixed095_d010 | 0.02818 | -2.47142 | 0.061 | bad: luma and detail both collapse |

Conclusion:

The current low-frequency transfer is not the right abstraction for Perfect NR.
It can recover visible highlight texture only by pulling down highlight luma.
The next postprocess should explicitly separate:

```text
1. low-frequency chroma stabilization
2. highlight luma preservation
3. high-frequency detail reconstruction / retention
```

Next hypothesis:

Start from `no_transfer` because it preserves highlight luma, then add a
highlight-only detail guard that borrows high-frequency luma from the reference
without restoring chroma noise.

## 2026-06-18 Phase 1A Highlight Luma Detail Guard

Added:

```text
scripts/perfect_nr_detail_guard.py
```

Purpose:

Separate highlight detail reconstruction from low-frequency color transfer.
The new guard starts from the stable `no_transfer` output, preserves its chroma,
and adds only coherent luma detail from the reference/input inside a smooth
highlight mask.

Key rule:

```text
output chroma = base/no_transfer chroma
output luma   = base/no_transfer luma + coherent_reference_luma_detail
```

This is intentionally not a final denoiser. It is a design probe for the model:
Perfect NR should treat noise removal, chroma stability, highlight luma, and
fine detail as separate responsibilities.

Comparison on the X-T5 Room top-luma 1% mask:

```text
runs/perfect_nr/compare/xt5_room_luma_guard_wide/comparison.md
```

| candidate | chroma drift | luma delta | HF retention | interpretation |
| --- | ---: | ---: | ---: | --- |
| no_transfer | 0.01546 | +0.02453 | 0.494 | safe color/luma, too smooth |
| scaled_d020 | 0.01393 | -1.78287 | 1.052 | detail returns by badly darkening highlights |
| luma_guard_s110 | 0.01546 | +0.05400 | 0.778 | stable, useful but conservative |
| luma_guard_s160 | 0.01546 | +0.06739 | 0.928 | best natural candidate so far |
| sigma22_s120 | 0.01546 | +0.07699 | 1.010 | strongest detail, possible over-texture |
| sigma22_s160 | 0.01546 | +0.09448 | 1.219 | too strong / upper-bound probe |

Visual crop sheet:

```text
runs/perfect_nr/compare/xt5_room_luma_guard_wide/curtain_crop_sheet.png
```

Conclusion:

The separated luma-detail guard is a better abstraction than the old overloaded
low-frequency transfer. It can restore highlight texture without the large
negative luma shift that made the previous transfer variants visually suspect.

Current best operating range:

```text
detail_sigma:      1.2 to 2.2
strength:          1.1 to 1.6
max_detail_frac:   0.18 to 0.24
zero_mean_sigma:   8 to 10
```

Design implication for `NagiPerfect`:

- predict denoised base RGB in bounded HDR-log space;
- predict or infer a confidence/detail gate;
- reconstruct chroma from the base path;
- reconstruct highlight luma detail through a separate luma-detail path;
- train with explicit penalties for highlight luma drift and chroma drift.

Next experiment:

Turn this postprocess into a model-side target. Build a tiny training/evaluation
prototype that predicts:

```text
1. denoised base
2. residual luma detail
3. detail confidence mask
```

The first loss should use SIDD/PolyU for denoise sanity and the HDR diagnostic
crops only as acceptance gates, not as a benchmark to overfit.

## 2026-06-18 Phase 1B NagiPerfect Skeleton

Added:

```text
packages/nagi_nr/src/nagi_nr/nagiperfect.py
NagiPerfectLoss in packages/nagi_nr/src/nagi_nr/losses.py
tests/test_model.py smoke coverage
```

The first `NagiPerfect` prototype encodes the Phase 1A finding directly:

```text
shared HDR-log trunk
  -> base RGB residual head
  -> luma detail residual head
  -> detail confidence head
  -> chroma-preserving luma-detail reconstruction
```

The detail head cannot change chroma. It can only modify luma through a gated
residual. This is the structural guardrail that the old single RGB residual and
low-frequency transfer path did not have.

Initial presets:

| preset | params | CPU 256x256 sanity time |
| --- | ---: | ---: |
| perfect-s | 2.809M | 433.5 ms |
| perfect-m | 8.712M | 1043.9 ms |

These CPU timings are only scale checks, not production speed claims.

Verification:

```text
pixi run python tests/test_model.py
```

Result:

```text
all tests passed
```

Next experiment:

Build the first short training recipe for `perfect-s`:

- SIDD/PolyU paired crops for ordinary denoise behavior;
- HDR diagnostic crops as non-training gates;
- `NagiPerfectLoss` with explicit highlight luma/chroma/detail terms;
- identity-start check before training;
- 500 to 1000 step pilot only, then evaluate on the fixed diagnostic harness.

## 2026-06-18 Phase 1C Perfect-S Pilot Training Path

Added:

```text
packages/nagi_nr/src/nagi_nr/train_perfect.py
packages/nagi_nr/configs/nagiperfect_s_pilot_500.yaml
scripts/launch_perfect_training.py
scripts/denoise_exr_nagiperfect.py
pixi task: train-nagiperfect-s-pilot-500
```

`train_perfect.py` is intentionally separate from `train_q.py`. The existing
`train_q.py` path is sRGB teacher-distillation oriented, while `NagiPerfect`
needs linear/HDR supervision and the split-head `NagiPerfectLoss`.

Smoke check:

```text
pixi run python -m nagi_nr.train_perfect \
  --config packages/nagi_nr/configs/nagiperfect_s_pilot_500.yaml \
  --sidd-root SIDD_Medium_Srgb \
  --output /private/tmp/nagiperfect_smoke \
  --device cpu \
  --ckpt-prefix smoke \
  --max-iters 2
```

Result:

```text
[0/2] loss=0.1560 ... highlight_chroma=0.3432 ...
saved /private/tmp/nagiperfect_smoke/smoke_final.pt
```

Sandbox note:

`num_workers: 1` failed in the managed sandbox because PyTorch worker shared
memory startup was denied. The pilot config now uses `num_workers: 0`, which is
slower but more predictable and lower impact.

Current run:

```text
runs/nagiperfect_perfect_s_pilot_500
pid: 65462
device: mps
target: 500 steps
```

Early trend:

```text
step 0:   loss=0.1560
step 25:  loss=0.0306
step 100: loss=0.0268
```

Next evaluation when the pilot finishes:

1. Run `scripts/denoise_exr_nagiperfect.py` on the X-T5 Room EXR crop.
2. Compare against `no_transfer`, `luma_guard_s160`, and the reference with
   `scripts/perfect_nr_compare.py`.
3. Inspect `detail_confidence.png`; if confidence is flat, the detail head has
   not learned meaningful routing yet.

Pilot result:

```text
runs/nagiperfect_perfect_s_pilot_500/nagiperfect_perfect_s_pilot_500_final.pt
```

Training finished normally on MPS.

```text
final checkpoint: 43 MB
throughput:        about 1.5 it/s after warmup
SIDD val64 input:  19.681 dB
SIDD val64 output: 20.293 dB
delta:             +0.612 dB
```

HDR X-T5 Room diagnostic:

```text
runs/perfect_nr/compare/xt5_room_nagiperfect_500/comparison.md
runs/perfect_nr/compare/xt5_room_nagiperfect_500/curtain_crop_sheet.png
```

Top-luma 1% mask:

| candidate | chroma drift | luma delta | HF retention | interpretation |
| --- | ---: | ---: | ---: | --- |
| no_transfer | 0.01546 | +0.02453 | 0.494 | safe but smooth |
| luma_guard_s160 | 0.01546 | +0.06739 | 0.928 | good postprocess candidate |
| sigma22_s120 | 0.01546 | +0.07699 | 1.010 | strong detail, possible over-texture |
| nagiperfect_500 | 0.00455 | -0.01733 | 0.989 | no HDR breakage; likely still input-preserving |

Important interpretation:

This is encouraging but not a victory yet. The HDR metrics are excellent partly
because the 500-step pilot is still conservative and close to the input. SIDD
val64 confirms it is doing real denoising (`+0.612 dB`), but it is nowhere near
production strength.

Next recipe:

- keep the split-head architecture;
- increase denoise pressure through longer training and/or stronger base loss;
- reduce `confidence_l1_weight` or bias so the detail head can become more
  selective instead of uniformly low;
- add a validation row that tracks both SIDD PSNR and X-T5 highlight metrics
  after every short run;
- do not change architecture again until a 2k to 5k run proves whether the
  current design saturates.

## 2026-06-18 Phase 1D Perfect-S Base-Push 2k

Added:

```text
packages/nagi_nr/configs/nagiperfect_s_basepush_2k.yaml
pixi task: train-nagiperfect-s-basepush-2k
```

Recipe change from the 500-step pilot:

- initialize from the 500-step final checkpoint;
- fresh optimizer, not resumed optimizer;
- total 2000 steps;
- stronger base denoise supervision (`base_weight: 0.55`);
- slightly stronger final/linear supervision;
- lower confidence sparsity penalty (`0.002 -> 0.0005`);
- still no architecture change.

Current run:

```text
runs/nagiperfect_perfect_s_basepush_2k
pid: 75262
device: mps
```

Early trend:

```text
step 0:   loss=0.1408
step 50:  loss=0.0523
step 100: loss=0.0294
```

Initial status:

Training is stable so far. The real decision point is not loss; it is whether
the 500/1000/2000 checkpoints increase SIDD PSNR without damaging the X-T5 HDR
highlight gate.

Outcome:

`basepush_2k` failed. At step 500, loss became NaN and the saved 500-step
checkpoint contained NaNs in almost all tensors:

```text
runs/nagiperfect_perfect_s_basepush_2k/nagiperfect_perfect_s_basepush_2k_0000500.pt
bad tensors: 272 / 273
```

The process was stopped. This checkpoint is invalid and must not be used for
evaluation or initialization.

Likely cause:

The aggressive recipe combined higher LR, stronger base/final pressure, wider
exposure jitter, and no bound on the compressed reconstruction before `sinh`
decompression. One bad update can push compressed values into a numerically
unsafe range on MPS.

Fixes added:

- `NagiPerfect.compressed_output_clamp` defaults to `2.0`;
- `train_perfect.py` now aborts before optimizer/checkpoint when loss is non-finite;
- new safe recipe: `nagiperfect_s_stable_2k.yaml`.

## 2026-06-18 Phase 1E Perfect-S Stable 2k

Added:

```text
packages/nagi_nr/configs/nagiperfect_s_stable_2k.yaml
pixi task: train-nagiperfect-s-stable-2k
```

Recipe:

- initialize from the valid 500-step pilot final checkpoint;
- lower LR (`5e-5 -> 2e-5`);
- narrower exposure jitter (`max 3.0 -> 2.0`);
- weaker synthetic noise pressure;
- gradient clip tightened (`1.0 -> 0.5`);
- compressed output clamp enabled;
- no architecture change besides the numerical safety clamp.

Current run:

```text
runs/nagiperfect_perfect_s_stable_2k
pid: 80700
device: mps
```

Initial status:

```text
step 0: loss=0.1134, finite
```

500-step checkpoint:

```text
runs/nagiperfect_perfect_s_stable_2k/nagiperfect_perfect_s_stable_2k_0000500.pt
bad tensors: 0 / 273
```

SIDD val64:

```text
input:  19.681 dB
output: 20.445 dB
delta:  +0.764 dB
```

X-T5 Room top-luma 1%:

| candidate | chroma drift | luma delta | HF retention |
| --- | ---: | ---: | ---: |
| pilot_500 | 0.00455 | -0.01733 | 0.989 |
| stable_0500 | 0.01134 | -0.03221 | 0.983 |
| luma_guard_s160 | 0.01546 | +0.06739 | 0.928 |

Interpretation:

Stable 500 has stronger denoise behavior than pilot 500 and remains HDR-safe so
far. It slightly increases highlight chroma drift and darkens top highlights a
little more than the pilot, so 1000-step evaluation is important before letting
this recipe run much longer.

1000-step checkpoint:

```text
runs/nagiperfect_perfect_s_stable_2k/nagiperfect_perfect_s_stable_2k_0001000.pt
bad tensors: 0 / 273
```

SIDD val64:

```text
input:  19.681 dB
output: 20.630 dB
delta:  +0.949 dB
```

X-T5 Room top-luma 1%:

| candidate | chroma drift | luma delta | HF retention |
| --- | ---: | ---: | ---: |
| pilot_500 | 0.00455 | -0.01733 | 0.989 |
| stable_0500 | 0.01134 | -0.03221 | 0.983 |
| stable_1000 | 0.01569 | -0.05561 | 0.972 |

Interpretation:

Denoising strength is increasing, but the HDR highlight gate is slowly eroding:
more chroma drift and a larger negative luma delta. This is not a hard failure
yet, but 1500-step evaluation should decide whether to continue to final or stop
and redesign the loss.

Final result:

```text
runs/nagiperfect_perfect_s_stable_2k/nagiperfect_perfect_s_stable_2k_final.pt
bad tensors: 0 / 273
```

SIDD val64 trend:

| checkpoint | output PSNR | delta vs noisy |
| --- | ---: | ---: |
| pilot_500 | 20.293 | +0.612 |
| stable_0500 | 20.445 | +0.764 |
| stable_1000 | 20.630 | +0.949 |
| stable_1500 | 20.798 | +1.117 |
| stable_final | 20.966 | +1.285 |

X-T5 Room top-luma 1% trend:

| checkpoint | chroma drift | luma delta | HF retention |
| --- | ---: | ---: | ---: |
| pilot_500 | 0.00455 | -0.01733 | 0.989 |
| stable_0500 | 0.01134 | -0.03221 | 0.983 |
| stable_1000 | 0.01569 | -0.05561 | 0.972 |
| stable_1500 | 0.01545 | -0.06443 | 0.965 |
| stable_final | 0.01765 | -0.06009 | 0.963 |

Artifacts:

```text
runs/perfect_nr/compare/xt5_room_nagiperfect_stable_final/comparison.md
runs/perfect_nr/compare/xt5_room_nagiperfect_stable_final/curtain_crop_sheet.png
```

Conclusion:

The architecture is viable: stable training improved denoise behavior without
NaNs and without catastrophic HDR failure. However, the loss recipe is still not
Perfect NR quality. As SIDD denoising improves, HDR highlight luma and chroma
slowly drift. This confirms the model needs a stronger highlight preservation
term or a two-phase schedule that freezes/regularizes highlight reconstruction
after ordinary denoise starts improving.

Current best checkpoint depends on priority:

- `stable_final`: strongest SIDD denoise in this run;
- `pilot_500` or `stable_0500`: safest HDR highlight behavior;
- `stable_1000`: reasonable middle point.

Next design change:

Keep the architecture. Change the recipe:

1. add an explicit input/output highlight ratio preservation loss;
2. raise highlight luma/chroma weight after warmup instead of keeping it static;
3. evaluate X-T5 at every checkpoint and stop on luma drift threshold;
4. consider training base RGB and luma-detail heads in phases so base denoise
   cannot steal highlight brightness.

## NagiPerfect-S hlguard 1k and input highlight guard

Checkpoint:

```text
runs/nagiperfect_perfect_s_hlguard_1k/nagiperfect_perfect_s_hlguard_1k_final.pt
bad tensors: 0 / 273
```

SIDD val64:

| checkpoint / mode | output PSNR | delta vs noisy |
| --- | ---: | ---: |
| stable_final | 20.966 | +1.285 |
| hlguard_0500 | 21.081 | +1.400 |
| hlguard_final | 21.210 | +1.529 |

X-T5 Room top-luma 1%:

| candidate | chroma drift | luma delta | HF retention | rgb max |
| --- | ---: | ---: | ---: | ---: |
| stable_final | 0.01765 | -0.06009 | 0.963 | 4.991 |
| hlguard_0500 | 0.01774 | -0.05935 | 0.962 | 4.990 |
| hlguard_final | 0.01841 | -0.06860 | 0.957 | 4.972 |
| pilot_500 | 0.00455 | -0.01733 | 0.989 | 5.115 |

Interpretation:

The scheduled highlight luma/chroma/ratio loss improved ordinary denoise
metrics, but did not solve real HDR highlight preservation. It slightly worsened
top-luma drift by the final checkpoint. This falsifies the "just raise highlight
loss weights" path.

New structural guard:

```text
highlight_protect_threshold = 1.0
highlight_protect_transition = 0.15
highlight_protect_strength = 0.85
```

This guard is input-luma based. Above the threshold the final RGB output is
progressively blended back toward the input, reducing the model's permission to
alter out-of-distribution HDR highlights. It is not a target/loss hint; it is an
inference-time safety invariant.

X-T5 Room top-luma 1% with `hlguard_final`:

| candidate | chroma drift | luma delta | HF retention | rgb max |
| --- | ---: | ---: | ---: | ---: |
| no guard | 0.01841 | -0.06860 | 0.957 | 4.972 |
| guard s=0.85, t=0.15 | 0.00272 | -0.01029 | 0.993 | 5.161 |
| guard s=1.00, t=0.15 | 0.00000 | 0.00000 | 0.999 | 5.195 |

SIDD val64 side effect:

| mode | output PSNR | delta vs noisy |
| --- | ---: | ---: |
| no guard | 21.210 | +1.529 |
| guard s=0.85, t=0.15 | 21.204 | +1.523 |
| guard s=1.00, t=0.15 | 21.203 | +1.522 |

Conclusion:

The current best practical mode is `hlguard_final` with input highlight guard
`threshold=1.0, transition=0.15, strength=0.85`. It gives nearly the same SIDD
denoise score as the strongest checkpoint while restoring real HDR highlight
luma/chroma/detail beyond the early pilot checkpoint. Strength `1.0` is the
strict safety mode; `0.85` is the better quality/default candidate because it
still allows small denoise changes around highlights.

Artifacts:

```text
runs/perfect_nr/compare/xt5_room_nagiperfect_inputguard_transition/comparison.md
runs/perfect_nr/nagiperfect_exr/xt5_room_hlguard_final_inputguard_s085_t015/
runs/perfect_nr/nagiperfect_exr/xt5_room_hlguard_final_inputguard_s100_t015/
```

Implementation notes:

- `NagiPerfect` now has an optional input-luma highlight guard.
- `scripts/denoise_exr_nagiperfect.py` accepts
  `--highlight-protect-threshold`, `--highlight-protect-transition`, and
  `--highlight-protect-strength`.
- `scripts/eval_nagiperfect_sidd_val.py` accepts the same guard options.

## Tiled full-resolution EXR inference

Full-frame CPU inference on `samples/coreml_exr_input/sample_cat_noisy.EXR`
(`7728x5152`) was too slow as a single tensor. Added overlap-blended tiling to
`scripts/denoise_exr_nagiperfect.py`:

```text
--tile-size 512 --tile-overlap 64
```

X-T5 tiled sanity check against the full recommended output:

```text
abs diff mean: 0.0000857
abs diff p95:  0.0002509
abs diff p99:  0.0007289
abs diff max:  0.0840
```

The p99 difference is tiny; the max is a localized boundary/context difference.
This is acceptable for practical large-image inference with overlap blending.

Cat EXR recommended output:

```text
runs/perfect_nr/nagiperfect_exr/sample_cat_hlguard_final_inputguard_recommended_tiled512/
```

Stats:

| metric | input | output |
| --- | ---: | ---: |
| rgb max | 4.6690 | 4.6288 |
| luma max | 3.5730 | 3.5425 |
| luma p99 | 0.7260 | 0.7305 |
| peak_gt_1_fraction | 0.01564 | 0.01484 |
| highlight guard mask mean | - | 0.00645 |

Interpretation:

The cat image is mostly SDR-ish with sparse HDR channel peaks, so the guard only
touches a small fraction of pixels. The preview keeps fine whiskers visible and
does not show obvious tile seams at preview scale.

Convenience tasks:

```text
pixi run denoise-nagiperfect-xt5-recommended
pixi run denoise-nagiperfect-cat-recommended
```

## Current speed profile

Measured on `X-T5 Room.EXR` (`749x690`, about `0.517 MP`) with
`hlguard_final + input highlight guard s=0.85/t=0.15`.

| backend | diagnostics | tiling | seconds | MP/s |
| --- | --- | --- | ---: | ---: |
| PyTorch CPU | on | full | 2.747 | 0.188 |
| PyTorch CPU | on | 768/64 | 2.347 | 0.220 |
| PyTorch CPU | fast | full | 2.214 | 0.233 |
| PyTorch CPU | fast | 768/64 | 2.150 | 0.240 |
| PyTorch MPS | on | full | 1.961 | 0.264 |
| PyTorch MPS | on | 768/64 | 1.360 | 0.380 |
| PyTorch MPS | fast | full | 1.296 | 0.399 |
| PyTorch MPS | fast | 768/64 | 1.259 | 0.410 |

Interpretation:

Fast mode helps, but only modestly on CPU because the convolution trunk is the
main cost. MPS is about `1.7x` faster than CPU fast on this image. This is still
not final production speed; the practical next target is Core ML export and
larger/batched tiles. PyTorch is now mainly the quality reference path.

## Denoise strength push

The previous `hlguard_final` checkpoint was HDR-safe but still left visible
residual noise. Added sRGB-space reconstruction terms and trained
`perfect_s_denoisepush_5k` from:

```text
runs/nagiperfect_perfect_s_hlguard_1k/nagiperfect_perfect_s_hlguard_1k_final.pt
```

SIDD val64 result:

| checkpoint | PSNR in | PSNR out | delta |
| --- | ---: | ---: | ---: |
| hlguard final | 19.6810 | 21.2100 | +1.5290 |
| denoisepush 1000 | 19.6810 | 21.2144 | +1.5334 |
| denoisepush 2000 | 19.6810 | 21.3369 | +1.6559 |
| denoisepush 3000 | 19.6810 | 21.6139 | +1.9329 |
| denoisepush 4000 | 19.6810 | 21.8793 | +2.1984 |
| denoisepush final | 19.6810 | 22.0586 | +2.3776 |

Top-luma X-T5 comparison with recommended inference guard
(`threshold=1.0`, `transition=0.15`, `strength=0.85`):

| checkpoint | chroma drift | luma delta | HF retention | rgb max |
| --- | ---: | ---: | ---: | ---: |
| hlguard final | 0.002719 | -0.010290 | 0.993 | 5.161 |
| denoisepush 3000 | 0.002957 | -0.000546 | 0.993 | 5.168 |
| denoisepush 4000 | 0.003040 | 0.000602 | 0.992 | 5.167 |
| denoisepush final | 0.003002 | 0.001144 | 0.990 | 5.167 |

Interpretation:

The sRGB push worked: denoising strength improved substantially. The tradeoff is
that highlight high-frequency retention began to drop by final, so further
global denoise pressure is risky for thin bright detail.

## Body/midtone denoise push

Added `body_srgb_weight` and `body_srgb_base_weight` to `NagiPerfectLoss`.
These apply sRGB reconstruction only outside the highlight mask. The purpose is
to push dark/midtone residual noise while avoiding extra pressure on HDR peaks
and bright lines.

Trained `perfect_s_bodypush_4k` from:

```text
runs/nagiperfect_perfect_s_denoisepush_5k/nagiperfect_perfect_s_denoisepush_5k_final.pt
```

Stopped at 2000 because SIDD kept improving, but top-luma HF retention continued
to decline. The 2000 checkpoint is the current denoise-strength candidate:

```text
runs/nagiperfect_perfect_s_bodypush_4k/nagiperfect_perfect_s_bodypush_4k_0002000.pt
```

SIDD val64 result:

| checkpoint | PSNR in | PSNR out | delta |
| --- | ---: | ---: | ---: |
| denoisepush final | 19.6810 | 22.0586 | +2.3776 |
| bodypush 1000 | 19.6810 | 22.1824 | +2.5015 |
| bodypush 2000 | 19.6810 | 22.3599 | +2.6790 |

Top-luma X-T5 comparison:

| checkpoint | chroma drift | luma delta | HF retention | rgb max |
| --- | ---: | ---: | ---: | ---: |
| denoisepush final | 0.003002 | 0.001144 | 0.990 | 5.167 |
| bodypush 1000 | 0.003245 | 0.001423 | 0.989 | 5.165 |
| bodypush 2000 | 0.003269 | 0.001219 | 0.987 | 5.163 |

Interpretation:

`bodypush_2000` is the best numerical denoiser so far, improving SIDD val64 by
about `+0.30 dB` over `denoisepush_final`. The cost is a visible-risk signal:
top-luma HF retention drops from `0.990` to `0.987`. Do not continue this exact
recipe blindly; the next quality step should keep the stronger body/midtone
objective but raise non-highlight edge/detail preservation instead of adding
more denoise pressure.

## Flat-region residual noise push

User feedback: zoom-visible residual grain is still unacceptable, even if
highlight protection should not be sacrificed.

Added `flat_srgb_hf_weight` to `NagiPerfectLoss`. It builds a mask from the GT
sRGB luma high-frequency magnitude, then applies high-frequency residual loss
only where the target is flat and outside the highlight mask. The goal is to
attack flat dark/midtone grain while avoiding direct pressure on bright lines,
whiskers, text, and HDR peaks.

Trained `perfect_s_flatpush_2k` from:

```text
runs/nagiperfect_perfect_s_bodypush_4k/nagiperfect_perfect_s_bodypush_4k_0002000.pt
```

Stopped at 1000 because the SIDD metric improved, but top-luma chroma drift kept
creeping upward.

SIDD val64 result:

| checkpoint | PSNR in | PSNR out | delta |
| --- | ---: | ---: | ---: |
| bodypush 2000 | 19.6810 | 22.3599 | +2.6790 |
| flatpush 500 | 19.6810 | 22.3811 | +2.7001 |
| flatpush 1000 | 19.6810 | 22.4305 | +2.7495 |

Top-luma X-T5 comparison:

| checkpoint | chroma drift | luma delta | HF retention | rgb max |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 0.003269 | 0.001219 | 0.987 | 5.163 |
| flatpush 500 | 0.003346 | 0.001464 | 0.987 | 5.163 |
| flatpush 1000 | 0.003437 | 0.001555 | 0.987 | 5.162 |

Interpretation:

The flat-region objective is directionally correct for denoising, but weak:
`flatpush_1000` only improves SIDD by about `+0.07 dB` over `bodypush_2000`,
while chroma drift increases. This does not solve the visible-grain complaint by
itself. The next attempt should explicitly separate luma and chroma grain:
stronger flat-region chroma suppression, weaker global RGB pressure, and a
harder edge/line exclusion mask.

## Flat chroma/luma split attempt

Added `flat_luma_hf_weight` and `flat_chroma_hf_weight` to split flat-region
high-frequency residual loss into luma and chroma components. The recipe
`perfect_s_chromaflat_1k` starts from `bodypush_2000`, uses a harder flat mask,
reduces global/body RGB pressure, and emphasizes flat chroma residuals:

```text
runs/nagiperfect_perfect_s_bodypush_4k/nagiperfect_perfect_s_bodypush_4k_0002000.pt
```

Stopped at 500. It did not beat the tradeoff frontier.

SIDD val64 result:

| checkpoint | PSNR in | PSNR out | delta |
| --- | ---: | ---: | ---: |
| bodypush 2000 | 19.6810 | 22.3599 | +2.6790 |
| flatpush 500 | 19.6810 | 22.3811 | +2.7001 |
| flatpush 1000 | 19.6810 | 22.4305 | +2.7495 |
| chromaflat 500 | 19.6810 | 22.3785 | +2.6976 |

Top-luma X-T5 comparison:

| checkpoint | chroma drift | luma delta | HF retention | rgb max |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 0.003269 | 0.001219 | 0.987 | 5.163 |
| flatpush 500 | 0.003346 | 0.001464 | 0.987 | 5.163 |
| flatpush 1000 | 0.003437 | 0.001555 | 0.987 | 5.162 |
| chromaflat 500 | 0.003375 | 0.001594 | 0.987 | 5.163 |

Interpretation:

Separating flat luma/chroma is sensible but this particular weighting is not
better. `chromaflat_500` slightly improves SIDD over `bodypush_2000`, but less
than `flatpush_500`, and it still increases top-luma chroma/luma drift. Current
best practical checkpoint remains `bodypush_2000` for safer highlights, while
`flatpush_1000` remains the best numerical denoiser but with higher visible-risk
signals.

## Real-photo noise evaluator

Added `scripts/real_photo_noise_eval.py`.

This evaluator treats the original real image as the noisy reference, not as GT.
It builds masks from the input:

- `flat`: non-highlight, midtone, low display-luma high-frequency and low edge
  magnitude;
- `edge`: non-highlight, midtone, high edge magnitude;
- `highlight`: top-luma pixels in linear space.

It then reports:

- flat luma high-frequency ratio vs input;
- flat chroma high-frequency ratio vs input;
- edge luma high-frequency retention;
- highlight chroma drift and luma delta;
- previews and mask overlays.

X-T5 strict-flat run:

```text
runs/perfect_nr/real_noise_eval/xt5_room_key_candidates_strict_flat/
```

| checkpoint | flat luma ratio | flat chroma ratio | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: |
| denoisepush final | 0.800 | 0.921 | 0.864 | 0.003002 |
| bodypush 2000 | 0.759 | 0.916 | 0.850 | 0.003269 |
| flatpush 1000 | 0.754 | 0.915 | 0.847 | 0.003437 |
| chromaflat 500 | 0.758 | 0.916 | 0.849 | 0.003375 |

Interpretation:

The latest recipe improvements mostly reduce luma grain. Flat chroma
high-frequency energy barely changes, even when the training loss targets
flat chroma residuals. This explains why zoom-visible color grain remains.

## Flat chroma energy damping attempt

Added `flat_chroma_damp_weight`, which penalizes output flat-region chroma
high-frequency energy directly instead of matching target chroma residuals.
This tests whether GT residual chroma texture was the reason prior chroma losses
failed.

Recipe:

```text
packages/nagi_nr/configs/nagiperfect_s_chromadamp_1k.yaml
```

Started from:

```text
runs/nagiperfect_perfect_s_bodypush_4k/nagiperfect_perfect_s_bodypush_4k_0002000.pt
```

Stopped at 500 because the real-photo evaluator showed no improvement:

```text
runs/perfect_nr/real_noise_eval/xt5_room_chromadamp_500/
```

| checkpoint | flat luma ratio | flat chroma ratio | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 0.759 | 0.916 | 0.850 | 0.003269 |
| flatpush 1000 | 0.754 | 0.915 | 0.847 | 0.003437 |
| chromadamp 500 | 0.758 | 0.916 | 0.849 | 0.003367 |

Interpretation:

Direct chroma-energy damping still did not reduce measured flat chroma noise.
This strongly suggests the current `NagiPerfect` structure lacks an effective
and isolated chroma-control path. Further loss-only attempts are unlikely to
solve zoom-visible color grain. The next structural experiment should add a
small zero-initialized chroma residual branch, gated by flat/non-highlight masks,
while keeping the existing luma-detail path and highlight guard intact.

## Zero-init chroma branch attempts

First structural test:

```text
packages/nagi_nr/configs/nagiperfect_s_chromabranch_1k.yaml
```

This kept `bodypush_2000` frozen and trained only a zero-initialized
`chroma_head`. Trainable parameters were only 867, so the initial output was
identical to the old model and the test specifically asked whether a tiny RGB
head on frozen decoder features could remove flat chroma grain.

Stopped at 500 for evaluation:

```text
runs/nagiperfect_perfect_s_chromabranch_1k/nagiperfect_perfect_s_chromabranch_1k_0000500.pt
```

Cat 1024 crop real-photo evaluator:

```text
runs/perfect_nr/real_noise_eval/sample_cat_crop_1024_chromabranch_500/
```

| checkpoint | flat luma ratio | flat chroma ratio | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 1.232 | 0.881 | 0.850 | 0.024602 |
| chromabranch 500 | 1.233 | 0.880 | 0.850 | 0.024607 |

Interpretation:

The branch did not meaningfully move flat chroma noise. The measured difference
is too small to matter visually. This falsifies the "small head is enough"
version of the chroma-branch idea.

Second structural test prepared:

```text
packages/nagi_nr/configs/nagiperfect_s_chromabranch2_1k.yaml
```

Changes:

- still starts from `bodypush_2000`;
- still zero-initializes the final chroma residual, so initial output remains
  the old model;
- keeps the main model frozen;
- gives the chroma branch direct compressed-input RGB access;
- adds a tiny dedicated chroma adapter with two `NagiPerfectBlock`s;
- increases trainable parameters from 867 to 10,131.

This is the next logical test: if flat chroma grain still does not move, the
problem is likely not merely branch capacity. It would point toward the real
photo noise/target mismatch and the training objective itself, not just the
model head.

`chromabranch2` was trained on MPS to 500 and stopped for evaluation:

```text
runs/nagiperfect_perfect_s_chromabranch2_1k/nagiperfect_perfect_s_chromabranch2_1k_0000500.pt
```

Cat 1024 crop result:

| checkpoint | flat luma ratio | flat chroma ratio | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 1.232 | 0.881 | 0.850 | 0.024602 |
| chromabranch 500 | 1.233 | 0.880 | 0.850 | 0.024607 |
| chromabranch2 500 | 1.235 | 0.879 | 0.850 | 0.024770 |

Interpretation:

Increasing the frozen chroma branch from 867 to 10,131 trainable parameters
still barely moved real-photo flat chroma noise. The failure is now unlikely to
be "the head is too tiny" by itself. The training objective/data are not
teaching the desired real-photo chroma behavior.

## Direct flat chroma smoothing sanity check

Added:

```text
scripts/apply_flat_chroma_smoother.py
```

This applies an inference-time diagnostic filter:

- convert denoised linear RGB to display sRGB;
- separate display luma and display chroma;
- low-pass only chroma in flat/non-highlight regions;
- keep display luma unchanged before converting back to linear RGB.

This is a proof target, not the final pipeline. It asks whether chroma grain can
be reduced without losing luma edges.

Cat 1024 crop results from `bodypush_2000`:

```text
runs/perfect_nr/real_noise_eval/sample_cat_crop_1024_chromasmooth_grid/
```

| candidate | flat luma ratio | flat chroma ratio | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 1.232 | 0.881 | 0.850 | 0.024602 |
| smooth s050 | 1.226 | 0.657 | 0.850 | 0.028449 |
| smooth s060 t008 | 1.226 | 0.656 | 0.850 | 0.028476 |
| smooth s075 | 1.223 | 0.546 | 0.849 | 0.032224 |
| smooth s090 | 1.221 | 0.464 | 0.849 | 0.036043 |

Interpretation:

Direct chroma smoothing works: flat chroma high-frequency energy drops
substantially while edge luma retention is effectively unchanged. The practical
target is therefore clear. The next model/training step should not keep
optimizing `output` against clean GT alone. It should either:

- add this guarded chroma smoother as a cheap postprocess after `bodypush_2000`,
  with conservative defaults around `s050`; or
- train a branch by distilling this smoother output on real/noisy frames,
  while retaining the existing supervised losses as safety rails.

## Chroma smoother distillation

Added online distillation losses to `NagiPerfectLoss`.

The teacher is generated during training from `output_pre_chroma`, the frozen
`bodypush_2000`-equivalent output before the chroma branch is applied:

1. convert teacher base to display sRGB;
2. split display luma and display chroma;
3. low-pass only chroma in flat/non-highlight regions;
4. train the chroma branch to imitate that chroma-smoothing behavior.

Three recipes were tested to 500:

```text
packages/nagi_nr/configs/nagiperfect_s_chromadistill_1k.yaml
packages/nagi_nr/configs/nagiperfect_s_chromadistill2_1k.yaml
packages/nagi_nr/configs/nagiperfect_s_chromadistill3_1k.yaml
```

Cat 1024 crop result:

| candidate | flat luma ratio | flat chroma ratio | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 1.232 | 0.881 | 0.850 | 0.024602 |
| smooth s050 teacher | 1.226 | 0.657 | 0.850 | 0.028449 |
| chromadistill 500 | 1.236 | 0.877 | 0.850 | 0.024774 |
| chromadistill2 500 | 1.243 | 0.860 | 0.850 | 0.024888 |
| chromadistill3 500 | 1.244 | 0.856 | 0.850 | 0.024971 |

Interpretation:

Distillation does move in the right direction, but the current small branch is
far from matching the direct smoother. Increasing loss weight and adding direct
residual-delta supervision only changed the cat flat chroma ratio from 0.881 to
0.856 at 500, while the deterministic teacher reaches 0.657 immediately.

This suggests the bottleneck is not only the objective. The chroma branch is
probably too small/local to reproduce the guarded low-pass behavior. The
practical next step is either to use the deterministic smoother as a postprocess
or to replace the tiny chroma branch with a slightly wider/deeper local-filter
branch whose receptive field clearly covers the smoother kernel.

## Flatpush completion for normal denoise performance

Postprocess decisions were deferred, so `perfect_s_flatpush_2k` was resumed
from 1000 and trained to final. This is the cleanest continuation of the recipe
that was already improving normal SIDD denoising.

SIDD val64:

| checkpoint | PSNR in | PSNR out | delta |
| --- | ---: | ---: | ---: |
| bodypush 2000 | 19.6810 | 22.3599 | +2.6790 |
| flatpush 1000 | 19.6810 | 22.4306 | +2.7496 |
| flatpush 1500 | 19.6810 | 22.4763 | +2.7954 |
| flatpush final | 19.6810 | 22.5157 | +2.8348 |

Cat 1024 crop real-photo evaluator:

| checkpoint | flat luma ratio | flat chroma ratio | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 1.232 | 0.881 | 0.850 | 0.024602 |
| flatpush 1000 | 1.248 | 0.877 | 0.846 | 0.025518 |
| flatpush 1500 | 1.261 | 0.875 | 0.844 | 0.026083 |
| flatpush final | 1.271 | 0.872 | 0.842 | 0.026435 |

Interpretation:

Normal denoise performance improved clearly: `flatpush_final` is now the best
SIDD-val64 NagiPerfect-S checkpoint measured in this branch. The tradeoff is
also clear: real-photo edge retention slowly drops and highlight chroma drift
creeps up. Further normal-performance training should use `flatpush_final` as
the numeric baseline, but should add a stronger edge/detail safety rail rather
than simply continuing the same flat-HF pressure.

## Flatguard edge-luma safety rail

Added `edge_luma_hf_weight` to `NagiPerfectLoss`. It builds an edge mask from
the target sRGB luma high-frequency energy outside highlights, then matches
output and target luma high-frequency detail under that mask.

Recipe:

```text
packages/nagi_nr/configs/nagiperfect_s_flatguard_500.yaml
```

Started from:

```text
runs/nagiperfect_perfect_s_flatpush_2k/nagiperfect_perfect_s_flatpush_2k_final.pt
```

SIDD val64:

| checkpoint | PSNR in | PSNR out | delta |
| --- | ---: | ---: | ---: |
| flatpush final | 19.6810 | 22.5157 | +2.8348 |
| flatguard final | 19.6810 | 22.5201 | +2.8392 |

Cat 1024 crop real-photo evaluator:

| checkpoint | flat luma ratio | flat chroma ratio | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 1.232 | 0.881 | 0.850 | 0.024602 |
| flatpush final | 1.271 | 0.872 | 0.842 | 0.026435 |
| flatguard final | 1.271 | 0.872 | 0.841 | 0.026587 |

Interpretation:

The edge-luma safety rail did not recover real-photo edge retention. It only
gave a tiny SIDD gain. The likely reason is that this supervised edge term
matches GT edge detail on synthetic/SIDD-like patches, but it does not directly
protect real-photo thin-line high-frequency energy under the evaluator's mask.
Further normal-performance work should not simply increase this edge loss.

## Strict-flat long continuation

User allowed a longer run, so a strict-flat recipe was added:

```text
packages/nagi_nr/configs/nagiperfect_s_strictflat_5k.yaml
```

The intent was to keep denoising pressure high while narrowing the flat mask:

- start from `flatpush_final`;
- lower `flat_threshold` from `0.018` to `0.011`;
- sharpen `flat_transition` from `0.010` to `0.004`;
- keep strong `flat_srgb_hf_weight`;
- keep small `edge_luma_hf_weight` and stronger highlight/detail safety rails.

SIDD val64:

| checkpoint | PSNR in | PSNR out | delta |
| --- | ---: | ---: | ---: |
| flatpush final | 19.6810 | 22.5157 | +2.8348 |
| flatguard final | 19.6810 | 22.5201 | +2.8392 |
| strictflat 1000 | 19.6810 | 22.5477 | +2.8668 |
| strictflat 2000 | 19.6810 | 22.5901 | +2.9091 |
| strictflat 3000 | 19.6810 | 22.6345 | +2.9535 |
| strictflat 4000 | 19.6810 | 22.6643 | +2.9833 |
| strictflat final | 19.6810 | 22.6826 | +3.0016 |

Cat 1024 crop real-photo evaluator:

| checkpoint | flat luma ratio | flat chroma ratio | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 1.232 | 0.881 | 0.850 | 0.024602 |
| flatpush final | 1.271 | 0.872 | 0.842 | 0.026435 |
| strictflat 3000 | 1.291 | 0.865 | 0.835 | 0.027283 |
| strictflat 4000 | 1.298 | 0.863 | 0.834 | 0.027561 |
| strictflat final | 1.302 | 0.862 | 0.833 | 0.027679 |

Interpretation:

This is the strongest normal-SIDD improvement so far, but it still trades away
real-photo line/detail retention. Narrowing the flat mask did not prevent the
erosion visible to the evaluator. It improved flat chroma ratio only modestly
while making luma HF ratio rise and edge retention fall. The current best
numeric checkpoint is `strictflat_final`; the better practical image-quality
candidate is still likely `flatpush_final` or an intermediate checkpoint,
pending visual inspection.

## Local chroma denoise continuation

Strict-flat improved SIDD but kept eroding real-photo detail, so a more local
chroma-focused recipe was added:

```text
packages/nagi_nr/configs/nagiperfect_s_chromalocal_3k.yaml
```

The recipe starts from `flatpush_final`, removes RGB-wide flat HF pressure, and
uses flat-region chroma HF/damping terms instead. The hypothesis was that the
remaining zoom-visible grain is mostly chroma noise, so luma/detail should not
be pushed as hard.

Run:

```text
runs/nagiperfect_perfect_s_chromalocal_3k/
```

The run was stopped after the 1000 checkpoint evaluation because the effect was
too small to justify continuing to 3000.

SIDD val64:

| checkpoint | PSNR in | PSNR out | delta |
| --- | ---: | ---: | ---: |
| flatpush final | 19.6810 | 22.5157 | +2.8348 |
| chromalocal 500 | 19.6810 | 22.5255 | +2.8445 |
| chromalocal 1000 | 19.6810 | 22.5504 | +2.8694 |

Cat 1024 crop real-photo evaluator, strict mask
(`flat_hf_threshold=0.01`, `flat_edge_threshold=0.018`, `edge_threshold=0.04`):

| checkpoint | flat luma ratio | flat chroma ratio | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: |
| bodypush 2000 | 1.232 | 0.881 | 0.850 | 0.024602 |
| flatpush final | 1.271 | 0.872 | 0.842 | 0.026435 |
| strictflat 3000 | 1.291 | 0.865 | 0.835 | 0.027283 |
| chromalocal 500 | 1.271 | 0.872 | 0.841 | 0.026576 |
| chromalocal 1000 | 1.273 | 0.870 | 0.840 | 0.026803 |

Interpretation:

`chromalocal` is safer than strict-flat but barely moves the visible chroma
noise. It gives a small SIDD gain and a tiny flat-chroma improvement, while
edge retention and highlight drift already begin moving in the wrong direction.
Conclusion: continuing the same backbone/loss pressure is not enough. The next
useful step should change the mechanism, not just train longer. Either add a
deterministic/local postprocess path, or give the model an explicit masked
chroma-smoothing output with a stronger teacher target and a preservation gate.

## Masked chroma smooth gate

Added an explicit bounded chroma smoother inside `NagiPerfect`:

```text
packages/nagi_nr/configs/nagiperfect_s_smoothgate_3k.yaml
```

Unlike the earlier free `chroma_branch`, this branch predicts only a one-channel
gate. The operation itself is deterministic: in flat non-highlight regions,
display-space chroma is blended toward its local lowpass, then converted back to
linear RGB. The main denoiser is frozen; only `chroma_smooth_head` trains
(`289` parameters). This directly tests whether model-integrated local chroma
smoothing can remove the visible color grain without eroding luma detail.

Run:

```text
runs/nagiperfect_perfect_s_smoothgate_3k/
```

Stopped after evaluating the 1500 checkpoint because chroma reduction kept
improving, but highlight chroma drift also rose. The useful candidates are 1000
and 1500, not an unbounded continuation.

SIDD val64:

| checkpoint | PSNR in | PSNR out | delta |
| --- | ---: | ---: | ---: |
| flatpush final | 19.6810 | 22.5157 | +2.8348 |
| smoothgate 500 | 19.6810 | 22.5508 | +2.8698 |
| smoothgate 1000 | 19.6810 | 22.6438 | +2.9628 |
| smoothgate 1500 | 19.6810 | 22.7201 | +3.0392 |

Cat 1024 crop real-photo evaluator, strict mask
(`flat_hf_threshold=0.01`, `flat_edge_threshold=0.018`, `edge_threshold=0.04`):

| checkpoint | flat luma ratio | flat chroma ratio | flat chroma reduction | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| bodypush 2000 | 1.232 | 0.881 | 11.9% | 0.850 | 0.024602 |
| flatpush final | 1.271 | 0.872 | 12.8% | 0.842 | 0.026435 |
| smoothgate 500 | 1.270 | 0.840 | 16.0% | 0.841 | 0.026852 |
| smoothgate 1000 | 1.268 | 0.747 | 25.3% | 0.841 | 0.029150 |
| smoothgate 1500 | 1.266 | 0.669 | 33.1% | 0.840 | 0.031135 |

Interpretation:

This is the first model-side experiment that clearly attacks the real-photo
flat chroma noise without the strict-flat edge-retention collapse. Edge HF
stays essentially flat (`0.842 -> 0.840/0.841`) while flat chroma drops far more
than any previous learned recipe. The trade-off is low-frequency/highlight
color drift: 1000 is likely the balanced checkpoint, while 1500 is the stronger
denoise candidate if visual color shift is acceptable. Next useful work is not
more plain continuation; it is a controlled strength sweep or a highlight/color
preservation term specifically for the smooth gate.

### Smoothgate inference strength sweep

Added inference-time `--chroma-smooth-strength` overrides to:

```text
scripts/denoise_exr_nagiperfect.py
scripts/eval_nagiperfect_sidd_val.py
```

This allows the trained gate to be reused with a lower deterministic smoothing
strength, separating "where to smooth" from "how strongly to smooth".

Cat 1024 crop, strict mask, using the 1500 checkpoint with lower strengths:

| candidate | flat luma ratio | flat chroma ratio | flat chroma reduction | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| flatpush final | 1.271 | 0.872 | 12.8% | 0.842 | 0.026435 |
| smoothgate 1000 | 1.268 | 0.747 | 25.3% | 0.841 | 0.029150 |
| smoothgate 1500 strength 0.55 | 1.267 | 0.722 | 27.8% | 0.841 | 0.029498 |
| smoothgate 1500 strength 0.65 | 1.266 | 0.695 | 30.5% | 0.841 | 0.030296 |
| smoothgate 1500 strength 0.75 | 1.266 | 0.669 | 33.1% | 0.840 | 0.031135 |

Interpretation:

Strength override works cleanly. `smoothgate_1500_s055` is a better balanced
candidate than raw 1500: it beats 1000 on chroma reduction while keeping almost
the same edge retention and only a small additional highlight-chroma drift. For
quality-first use, the practical shortlist is now:

1. `smoothgate_1000` for conservative color stability.
2. `smoothgate_1500` with `--chroma-smooth-strength 0.55` for stronger denoise.

The next training recipe should reduce highlight/color drift at the source,
not keep increasing smoothing strength.

### Smoothgate preserve attempt

Added a short preserve recipe:

```text
packages/nagi_nr/configs/nagiperfect_s_smoothgate_preserve_500.yaml
```

It starts from `smoothgate_1500`, sets `chroma_smooth_strength=0.55`, adds
`chroma_smooth_gate_l1_weight`, and increases color-preservation weights. The
intent was to keep the trained gate but shrink unnecessary smoothing.

The run stopped at step 337 due to non-finite loss; checkpoint 250 was still
available and evaluated.

SIDD val64:

| checkpoint | PSNR in | PSNR out | delta |
| --- | ---: | ---: | ---: |
| smoothgate preserve 250 | 19.6810 | 22.6656 | +2.9847 |

Cat 1024 crop, strict mask:

| candidate | flat luma ratio | flat chroma ratio | flat chroma reduction | edge HF retention | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoothgate 1000 | 1.268 | 0.747 | 25.3% | 0.841 | 0.029150 |
| smoothgate 1500 strength 0.55 | 1.267 | 0.722 | 27.8% | 0.841 | 0.029498 |
| smoothgate preserve 250 | 1.267 | 0.727 | 27.3% | 0.841 | 0.029379 |

Interpretation:

The preserve idea is directionally plausible but not yet worth replacing the
simple inference-strength override. It gives a tiny drift improvement compared
with `smoothgate_1500_s055`, but also slightly less chroma reduction and had a
non-finite training event. If revisited, lower LR substantially and add loss
guards around the gate/sRGB conversion path. Current practical best remains
`smoothgate_1500` with `--chroma-smooth-strength 0.55`, with `smoothgate_1000`
as the conservative fallback.
