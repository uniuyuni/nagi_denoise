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

### Chromaaxis final and post overshrink cleanup

The later weak-teacher `chromaaxis_4k` run became the better base than the older
smoothgate branch. It saturated early, but its final checkpoint improved the
real-photo chroma tail without hurting luma/detail:

```text
runs/nagiperfect_perfect_s_weakteacher_chromaaxis_4k/
runs/nagiperfect_perfect_s_weakteacher_chromaaxis_4k/nagiperfect_perfect_s_weakteacher_chromaaxis_4k_final.pt
```

Cat noisy EXR, final checkpoint:

| candidate | flat luma ratio | flat chroma ratio | luma p99 | luma visible | chroma p99 | chroma visible | edge HF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tailquality final | 0.631 | 0.315 | 0.939 | 0.635 | 0.382 | 0.307 | 0.799 |
| chromaaxis final | 0.623 | 0.305 | 0.936 | 0.629 | 0.369 | 0.295 | 0.796 |

Discarded directions:

- `chromaresidual_3k` is worse at the useful checkpoints. At 500/1000 steps it
  regressed chroma p99/visible back to about `0.382/0.306`, so do not resume it.
- `chroma_smooth_kernel_size=13` worsened flat chroma and visible chroma
  compared with kernel 9, so wider uniform chroma smoothing is not the path.

The useful cleanup is a post-model display-space chroma highpass overshrink
driven by the learned `chroma_smooth_gate`. This must happen after the model
output, not by pushing the internal `chroma_smooth_strength` beyond 1.0.

Cat noisy EXR strength sweep:

| post strength | flat chroma | chroma p99 | chroma visible | luma p99 | edge HF | highlight chroma drift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 0.246 | 0.310 | 0.239 | 0.935 | 0.796 | 0.060956 |
| 1.6 | 0.236 | 0.299 | 0.229 | 0.934 | 0.796 | 0.062494 |
| 1.7 | 0.226 | 0.289 | 0.219 | 0.934 | 0.796 | 0.064058 |
| 1.85 | 0.212 | 0.275 | 0.206 | 0.934 | 0.796 | 0.066448 |
| 2.0 | 0.200 | 0.263 | 0.195 | 0.933 | 0.796 | 0.068885 |

Cross-checks with the same `quality` strength 2.0:

| image | flat chroma | chroma p99 | chroma visible | luma p99 | edge HF | highlight chroma drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| X-T5 Room | 0.292 | 0.577 | 0.277 | 0.784 | 0.821 | 0.010809 |
| X-T5 Hydrangea | 0.093 | 0.149 | 0.089 | 0.559 | 0.881 | 0.027974 |

Decision:

Use `--post-chroma-overshrink-preset quality` as the current practical best.
It maps to strength `2.0`; `balanced` maps to `1.6` if a future image shows
visible desaturation. The default preset stays `off` for backwards-compatible
script behavior.

### Post luma HF cleanup after chromaaxis quality

After chroma overshrink, the remaining visible defect is mostly display-luma
grain. The previous learned luma-push directions were risky because they traded
real-photo edge/highlight safety for denoise. A smaller post step is more
controllable: shrink only small display-luma high-frequency residuals under a
flat/non-edge/highlight-safe gate.

Tested `apply_luma_hf_shrink_filter.py` on top of chromaaxis quality. Cat noisy
EXR sweep:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | edge HF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chromaaxis quality | 0.619 | 0.933 | 0.626 | 0.263 | 0.195 | 0.796 |
| luma hf strong | 0.479 | 0.872 | 0.486 | 0.263 | 0.197 | 0.788 |
| luma hf xstrong | 0.405 | 0.837 | 0.411 | 0.263 | 0.197 | 0.783 |
| luma hf ultra | 0.277 | 0.754 | 0.282 | 0.264 | 0.199 | 0.773 |

Cross-checks for the `ultra` luma cleanup:

| image | luma visible before | luma visible after | edge HF before | edge HF after | chroma visible before | chroma visible after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| X-T5 Room | 0.529 | 0.340 | 0.821 | 0.815 | 0.277 | 0.288 |
| X-T5 Hydrangea | 0.491 | 0.205 | 0.881 | 0.877 | 0.089 | 0.094 |

Decision:

Keep this branch. It meaningfully reduces the remaining luma grain while the
edge cost is small on three real photos. The slight chroma visible regression is
expected because luma/chroma decomposition changes a little after luma
reconstruction; it remains far below the original. Integrated this as
`--post-luma-hf-preset quality`, which maps to the existing `ultra` luma-HF
filter. Existing chroma-only `quality` tasks are preserved; new `quality-plus`
tasks enable both post chroma and post luma cleanup.

### Stage11 hybrid finish: line restore plus signed chroma v3 plus luma tail

The strongest current practical route is no longer the older chromaaxis branch.
The base is the stage8 surface/luma teacher output with deterministic coherent
line luma restore, followed by two tightly gated post steps:

```text
line_restore -> signed_chroma_outlier_v3 -> luma_tail_balanced
```

Current reusable entry point:

```text
scripts/apply_hybrid_nr_finish.py
```

Inputs are the already denoised/line-restored EXR and an optional guide image.
The default mode is `quality`; `hdr_safe` is intentionally opt-in for HDR stress
images such as X-T5 Room. An attempted automatic HDR switch was weaker on the
main real-photo set because bright backgrounds and sparse peaks triggered safe
mode unnecessarily.

Example:

```bash
pixi run python scripts/apply_hybrid_nr_finish.py \
  --input runs/refiner_pilot_stage11_hybrid_best/line_restore_all/xt5_cat_hybrid_line_refined.exr \
  --guide-input runs/refiner_pilot_stage11_hybrid_best/line_restore_all/xt5_cat_hybrid_line_refined.exr \
  --output-dir runs/refiner_pilot_stage11_hybrid_best/hybrid_finish_manual \
  --name xt5_cat
```

For HDR stress testing:

```bash
pixi run python scripts/apply_hybrid_nr_finish.py \
  --input <line-restored-or-base.exr> \
  --guide-input <line-restored-or-base.exr> \
  --hdr-reference /Users/uniuyuni/PythonProjects/test_photos/X-T5\ Room.EXR \
  --mode hdr_safe \
  --output-dir runs/refiner_pilot_stage11_hybrid_best/hybrid_finish_hdr_safe \
  --name xt5_room
```

Final four-scene average, measured by `real_photo_noise_eval.py` against the
original noisy EXRs:

| candidate | luma p99 | chroma p99 | chroma visible | magenta p99 | shadow magenta p99 | edge HF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid line only | 0.3795 | 0.1249 | 0.0849 | 0.0914 | 0.0700 | 0.3512 |
| previous balanced | 0.3694 | 0.1143 | 0.0783 | 0.0809 | 0.0592 | 0.3459 |
| signed chroma v2 only | 0.3749 | 0.1119 | 0.0742 | 0.0770 | 0.0544 | 0.3487 |
| stage11 final v2 | 0.3687 | 0.1120 | 0.0742 | 0.0769 | 0.0543 | 0.3465 |
| stage11 final v3 | 0.3679 | 0.1094 | 0.0730 | 0.0730 | 0.0487 | 0.3450 |

Interpretation:

- v2 was a real improvement over previous balanced for visible chroma and
  magenta-dot tails; v3 pushes the same idea slightly harder on the magenta axis.
- v3 is the new default `quality` setting because it further reduces dark
  magenta dots with modest edge cost. The main tradeoff is edge HF
  `0.3465 -> 0.3450` against shadow magenta p99 `0.0543 -> 0.0487`.
- The luma-tail step still matters for the remaining luma p99, with only small
  edge cost.
- `--luma-preset strong` is available, but it is not the default: it improved
  average luma p99 from `0.3687` to `0.3651`, while luma visible stayed flat and
  edge HF fell from `0.3465` to `0.3454`. Visual crops were nearly identical.
- Defaulting to `hdr_safe` or automatic HDR safe mode loses cleanup on Ice,
  Occi, and Dance. Keep `quality` as the production default.
- Use `--mode hdr_safe` only when a crop shows highlight microstructure/chroma
  instability, or when testing dedicated HDR samples.

Key output directories:

```text
runs/refiner_pilot_stage11_hybrid_best/line_restore_all/
runs/refiner_pilot_stage11_hybrid_best/final_v3_balanced/
runs/refiner_pilot_stage11_hybrid_best/compare_final_v3_crops/
runs/refiner_pilot_stage11_hybrid_best/room_hdr_probe/
```

X-T5 Room HDR probe:

- Room source has `rgb_max=5.924`, `peak_gt_1_fraction=0.0663`,
  `peak_gt_2_fraction=0.0371`, `peak_gt_4_fraction=0.0139`.
- `hdr_safe` preserved max/p99 highlight range and reduced the chroma-filter
  blend compared with v2.
- This supports keeping HDR safe as a manual mode, not the default.

### Stage11 v4: add the missing red/cyan opponent axis

Axis evaluation of `stage11 final v3` showed that the visible residual was not
only magenta/blue. The remaining flat-region color tail was strongest on the
red/cyan opponent axis:

| axis | flat p99 ratio | shadow p99 ratio |
| --- | ---: | ---: |
| magenta+ | 0.0730 | 0.0487 |
| blue+ | 0.0876 | 0.0620 |
| red+ | 0.1042 | 0.0815 |
| cyan+ | 0.1069 | 0.0765 |

The signed chroma outlier filter previously corrected only magenta/green and
blue/yellow axes. I added a third signed `red` axis
`[1.0, -0.5, -0.5]` while keeping the v3 thresholds and luma-tail settings
unchanged.

Four-scene final comparison:

| candidate | luma p99 | chroma p99 | chroma visible | magenta p99 | shadow magenta p99 | edge HF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage11 final v3 | 0.3679 | 0.1094 | 0.0730 | 0.0730 | 0.0487 | 0.3450 |
| stage11 final v4 red055 | 0.3682 | 0.1012 | 0.0678 | 0.0691 | 0.0426 | 0.3449 |

Follow-up red-axis strength sweep:

| candidate | luma p99 | luma visible | chroma p99 | chroma visible | magenta p99 | shadow magenta p99 | edge HF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| red055 | 0.3682 | 0.1679 | 0.1012 | 0.0678 | 0.0691 | 0.0426 | 0.3449 |
| red085 | 0.3684 | 0.1680 | 0.0973 | 0.0652 | 0.0676 | 0.0407 | 0.3449 |
| red115 | 0.3686 | 0.1681 | 0.0939 | 0.0626 | 0.0666 | 0.0396 | 0.3449 |
| red150 | 0.3689 | 0.1682 | 0.0905 | 0.0599 | 0.0658 | 0.0395 | 0.3449 |

Blue-axis follow-up with `red_weight=1.15`:

| candidate | luma p99 | luma visible | chroma p99 | chroma visible | magenta p99 | shadow magenta p99 | edge HF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blue090 | 0.3686 | 0.1681 | 0.0939 | 0.0626 | 0.0666 | 0.0396 | 0.3449 |
| blue120 | 0.3678 | 0.1679 | 0.0921 | 0.0610 | 0.0660 | 0.0389 | 0.3446 |
| blue150 | 0.3669 | 0.1678 | 0.0905 | 0.0595 | 0.0656 | 0.0385 | 0.3443 |

Luma-tail follow-up with `red_weight=1.15` and `blue_weight=1.20`:

| candidate | luma p99 | luma visible | chroma p99 | chroma visible | edge HF |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced | 0.3678 | 0.1679 | 0.0921 | 0.0610 | 0.3446 |
| firm | 0.3662 | 0.1680 | 0.0921 | 0.0611 | 0.3441 |
| strong | 0.3642 | 0.1680 | 0.0921 | 0.0611 | 0.3435 |

Detailguard follow-up after the user noted that detail was getting too soft:

| candidate | luma p99 | luma visible | chroma p99 | chroma visible | magenta p99 | shadow magenta p99 | edge HF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current gate | 0.3678 | 0.1679 | 0.0921 | 0.0610 | 0.0660 | 0.0389 | 0.3446 |
| mild detailguard | 0.3680 | 0.1679 | 0.0943 | 0.0615 | 0.0672 | 0.0397 | 0.3453 |
| strong detailguard | 0.3684 | 0.1679 | 0.0968 | 0.0623 | 0.0687 | 0.0410 | 0.3458 |

Axis check, red055 minus v3:

| region | red p99 | cyan p99 | magenta p99 | blue p99 | luma p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat | -0.0120 | -0.0115 | -0.0039 | -0.0036 | +0.0003 |
| shadow flat | -0.0148 | -0.0131 | -0.0060 | -0.0040 | +0.0008 |

Decision:

- Promote v4 to the `quality` default by setting `red_weight=1.15`.
- `red_weight=1.50` gives better aggregate chroma metrics, but visual crops
  show the first hints of over-smoothing on colored flat surfaces and bokeh
  texture. Use `1.15` as the safer quality default.
- Raise `blue_weight` from `0.90` to `1.20`. `blue_weight=1.50` is also
  numerically better, but the Ice/Cat crops start to show slightly thinner
  blue-toned texture and a larger edge-HF drop. Keep `1.20` as the default and
  reserve `1.50` as an aggressive experiment.
- Keep luma-tail at `balanced`. `firm` and `strong` reduce luma p99, but luma
  visible does not improve and edge retention falls. This is not worth making
  the production default unless a future visual crop specifically needs it.
- Adopt mild detailguard for the production `quality` gate:
  `detail_threshold=0.018`, `detail_transition=0.009`,
  `edge_threshold=0.027`, `edge_transition=0.013`. This gives back about
  `+0.0007` edge retention and visibly preserves fine texture better on the
  Ice/Cat/Occi crops. The cost is a small chroma-tail regression, accepted
  because the user prioritizes detail/quality over maximum smoothing.
- Keep `hdr_safe` red correction disabled for now because HDR stress behavior
  was not re-tested for this new axis.
- Keep the v3 and edgeguard outputs in `runs/` as recent rollback candidates.

### Stage11 v5: perceptual luma detail restore

The user pointed out the core visual issue: noisy photos still contain
human-readable detail under the noise. Pure smoothing removes noise and detail
together, so edges and fine texture become soft. The next direction is not more
smoothing, but restoring plausible luma structure where the original noisy
image has coherent high-frequency signal.

Added:

```text
scripts/apply_perceptual_luma_detail_restore.py
```

The filter works in display-luma space:

1. extract signed high-frequency luma detail from the original/noisy reference;
2. reject random noise with a local signed-coherence gate and local-energy gate;
3. subtract the detail already present in the denoised base;
4. add a small clipped luma-only correction back to the base;
5. preserve base chroma, so color noise is not restored.

Integrated into `scripts/apply_hybrid_nr_finish.py` as an optional final step:

```bash
pixi run python scripts/apply_hybrid_nr_finish.py \
  --input <line-restored-base.exr> \
  --guide-input <line-restored-base.exr> \
  --detail-reference <original-noisy.exr> \
  --restore-detail \
  --output-dir <out> \
  --name <name>
```

Strength sweep on top of v4 mild detailguard:

| candidate | luma p99 | luma visible | chroma p99 | chroma visible | edge HF | global luma MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.3680 | 0.1679 | 0.0943 | 0.0615 | 0.3453 | 0.0197 |
| detail s018 | 0.3714 | 0.1624 | 0.0942 | 0.0614 | 0.3575 | 0.0194 |
| detail s024 | 0.3731 | 0.1613 | 0.0942 | 0.0614 | 0.3617 | 0.0193 |

Interpretation:

- `s024` gives a stronger perceptual detail return, especially on Ice and
  Occi, but it starts to bring back dark grain on Cat/Dance.
- `s018` is the safer default experiment: edge retention improves by about
  `+0.012`, luma visible improves, and chroma metrics are effectively unchanged.
- luma p99 rises because restored coherent detail is still high-frequency
  energy. In this branch, luma p99 alone should not be treated as a failure.

Decision:

- Keep detail restore optional, not always-on, because it needs the original
  noisy reference and can reintroduce some dark luma grain.
- Use `--restore-detail` with default strength `0.18` when the priority is
  perceptual detail/edge crispness over the absolute smoothest flat fields.
- This is the first step that matches the desired direction of "reconstruct
  visually acceptable detail" rather than only suppressing noise.

### Stage11 v6: structure luma graft for over-soft areas

The Occi tree/root crop exposed a different failure mode from normal residual
grain: the v4/v5 outputs were not only missing fine texture, but had lost
low/mid-frequency luma shape. In that case, high-frequency detail restore is
too weak because the base image no longer contains enough structure to sharpen.

Added:

```text
scripts/apply_structure_luma_graft.py
```

The filter works in display-luma space:

1. build a denoised luma structure image from the original/noisy reference with
   a guided filter;
2. detect coherent structure with edge and local-contrast gates;
3. blend a clipped luma correction into the denoised base;
4. preserve the denoised base chroma, so chroma noise is not restored.

Occi root crop probe:

| candidate | result |
| --- | --- |
| `base_v4` | visibly too soft; root/rock shape is flattened |
| `detail_s018` | restores fine crispness, but does not recover the missing mid-frequency shape |
| `graft_mid` | restores root/rock luma shape with moderate noise return |
| `graft_strong` | stronger structure return, but starts to risk bringing back noisy luma texture |

Full Occi outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/final_v6_structure_luma_graft/
```

`mid` stats:

- `gate_mean=0.2090`
- `correction_abs_p99=0.0093`
- `correction_abs_max=0.0465`

`strong` stats:

- `gate_mean=0.2066`
- `correction_abs_p99=0.0136`
- `correction_abs_max=0.0735`

Decision:

- Use `graft_mid` as the current candidate for over-soft structure recovery.
- Keep `graft_strong` as a visual reference only until face/hair crops are
  checked; it may be too aggressive for delicate areas.
- This is not a replacement for the chroma/luma NR pipeline. It is a final
  luma-only structure recovery stage for cases where denoising made the result
  look worse than the noisy original.

### Stage11 v7: aggressive luma rebuild

The user judged the v6 Occi root output still far too soft. A stronger probe
showed that the issue is not just missing high-frequency detail; the root and
hair need a much higher replacement ratio from the original/reference luma.

Extended:

```text
scripts/apply_structure_luma_graft.py
```

New presets:

- `rebuild`: low-radius guided reference luma, broad structure gate, effectively
  a strong luma skeleton replacement in textured regions.
- `rebuild_clarity`: a slightly smoother reference luma plus explicit mid/fine
  band restoration for a local-contrast look.

Full Occi outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/final_v7_luma_rebuild/
```

`rebuild` stats:

- `gate_mean=0.6059`
- `correction_abs_p99=0.0599`
- `correction_abs_max=0.2800`

`rebuild_clarity` stats:

- `gate_mean=0.4848`
- `correction_abs_p99=0.0610`
- `correction_abs_max=0.2694`

Visual interpretation:

- Root/tree structure is much less sleepy than v6.
- Hair strands and clumps recover substantially more shape than v4/v6.
- Skin also gets some original-luma grain back, especially in face crops.
- PL still has stronger local contrast and cleaner reconstruction, so this is
  an improvement but not the final "perfect" direction.

Decision:

- Keep v7 as an aggressive recovery candidate, not the new default.
- Next design step should be region-aware strength: strong rebuild for hair,
  roots, wood, rock, and other coherent texture; weak or disabled rebuild for
  skin/flat surfaces.
- A global luma rebuild cannot be the final answer because it trades softness
  for visible luma grain on skin.

### Stage11 v8: region-aware cleanup after rebuild

The user preferred `rebuild` despite remaining noise, because it removes the
most objectionable softness. The next probe tried to keep `rebuild` in textured
areas while reducing returned luma grain on skin-like flat regions.

Added:

```text
scripts/apply_region_aware_luma_cleanup.py
```

The filter:

1. computes a texture mask from original/reference luma;
2. computes a broad skin-color mask from the denoised base;
3. limits cleanup to skin-like flat regions;
4. blends rebuild luma toward a clean target from the v4 base;
5. preserves rebuild chroma.

Full Occi output:

```text
runs/refiner_pilot_stage11_hybrid_best/final_v8_region_rebuild/
```

`hybrid_clean` stats:

- `texture_mean=0.5266`
- `skin_mean=0.2844`
- `skin_flat_mean=0.1359`
- `skin_blend_p99=0.9098`
- `delta_vs_rebuild_abs_p99=0.0056`

Visual interpretation:

- Hair and root/tree structure are effectively preserved from `rebuild`.
- Skin grain is reduced slightly, but the visual change is modest.
- This is not enough to close the gap to PL, but it is a safer cleanup layer
  than global luma smoothing.

Decision:

- Keep `rebuild` as the main candidate.
- Keep v8 `hybrid_clean` as an optional skin/flat cleanup after `rebuild`.
- Bigger PL-like gains likely require either a better semantic/texture mask or
  a learned reconstruction stage; pure hand-authored luma cleanup is now giving
  diminishing returns.

### Stage11 v9: Dance flat/sky protection

The Dance sample is a counterexample for global `rebuild`: it improves perceived
structure, but the dark sky is mostly random noise and gets restored as if it
were texture.

Dance `rebuild` stats:

- `gate_mean=0.8400`
- `correction_abs_p99=0.0621`
- visual: dancer/snow structure improves, but dark-sky grain is too visible

Tested a safer Dance rebuild:

```text
runs/refiner_pilot_stage11_hybrid_best/final_v7_luma_rebuild/k5_dance_luma_rebuild_safe.exr
```

Stats:

- `gate_mean=0.2310`
- `correction_abs_p99=0.0276`

Interpretation: safer, but still not enough for the sky because the core issue
is not just rebuild strength; the cleanup mask needs to treat sky/flat darkness
as non-texture.

Extended `scripts/apply_region_aware_luma_cleanup.py` with:

```text
--texture-source base
preset: flat_protect
```

Dance flat-protect output:

```text
runs/refiner_pilot_stage11_hybrid_best/final_v8_region_rebuild/
k5_dance_rebuild_region_base_texture_flatclean_strong.exr
```

Stats:

- `texture_source=base`
- `texture_mean=0.0160`
- `non_skin_flat_mean=0.5803`
- `delta_vs_rebuild_abs_p99=0.0608`

Visual interpretation:

- Dark sky is much better protected than plain `rebuild`.
- Dancer and snow retain some rebuild benefit, but the result moves back toward
  the softer v4/base look in broad flat regions.
- This is the right trade for Dance-like images, but too strong for Occi-like
  detailed portraits.

Decision:

- Use `rebuild` for detail-heavy images like Occi.
- Use `rebuild + flat_protect` for images with large dark sky/flat areas like
  Dance.
- A future learned or semantic mask should decide this per region rather than
  per image.

### Stage11 v10: adaptive region mask

The user pointed out that per-photo preset switching is less desirable than
per-region selection. The next step was to combine reference and base texture
signals:

- reference texture catches lost real detail, but also treats noise as texture;
- base texture is safer for flat/noisy areas, but misses detail that v4 already
  made too soft;
- agreement texture keeps rebuild where both signals support structure;
- dark/low-saturation flat cleanup protects sky-like areas without flattening
  all bright texture.

Extended `scripts/apply_region_aware_luma_cleanup.py` with:

```text
texture_source=agreement
preset: adaptive_flat
preset: adaptive_sky
```

Dance outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/final_v9_adaptive_region/
k5_dance_rebuild_adaptive_flat.exr
k5_dance_rebuild_adaptive_sky.exr
```

Dance `adaptive_sky` stats:

- `texture_mean=0.0477`
- `non_skin_flat_mean=0.2750`
- `delta_vs_rebuild_abs_p99=0.0462`

Occi output:

```text
runs/refiner_pilot_stage11_hybrid_best/final_v9_adaptive_region/
xt5_occi_rebuild_adaptive_sky.exr
```

Occi `adaptive_sky` stats:

- `texture_mean=0.1254`
- `non_skin_flat_mean=0.1966`
- `delta_vs_rebuild_abs_p99=0.0204`

Visual interpretation:

- Dance: `adaptive_sky` is better balanced than plain `rebuild`; it protects
  dark sky more while keeping more structure than the strongest flat-protect
  variant.
- Occi: `adaptive_sky` is less destructive than `adaptive_flat`, but still a
  little softer than `v8_skin` on hair/root detail.

Decision:

- `adaptive_sky` is the best current step toward per-region automation.
- It is not yet a universal default: Occi still prefers `rebuild` or
  `rebuild + v8_skin`, while Dance prefers `adaptive_sky`/flat protection.
- Next improvement should make the agreement mask more semantic or structure
  aware: dark sky/noise should be flat, but hair/root/wood detail should remain
  rebuild even if the v4 base texture is weak.

### Stage11 v11: coherent structure protection

Tried to improve `adaptive_sky` by protecting directional/coherent structure
from non-skin flat cleanup. The goal was to keep dark hair/root/wood detail
while still cleaning random dark-sky noise.

Extended `scripts/apply_region_aware_luma_cleanup.py` with:

```text
preset: adaptive_coherent
coherent_structure mask from a luma structure tensor
```

Dance output:

```text
runs/refiner_pilot_stage11_hybrid_best/final_v9_adaptive_region/
k5_dance_rebuild_adaptive_coherent.exr
```

Stats:

- `coherent_structure_mean=0.1242`
- `non_skin_flat_mean=0.2491`
- `delta_vs_rebuild_abs_p99=0.0438`

Occi output:

```text
runs/refiner_pilot_stage11_hybrid_best/final_v9_adaptive_region/
xt5_occi_rebuild_adaptive_coherent.exr
```

Stats:

- `coherent_structure_mean=0.1706`
- `non_skin_flat_mean=0.1732`
- `delta_vs_rebuild_abs_p99=0.0190`

Visual interpretation:

- Dance: coherent protection is safe, but only modestly different from
  `adaptive_sky`; sky remains controlled, and dancer detail is similar.
- Occi: coherent protection slightly reduces over-cleaning, but hair/root
  detail still looks closer to `rebuild`/`v8_skin` than to a universal adaptive
  result.

Decision:

- `adaptive_coherent` is a small improvement, not a breakthrough.
- The hand-authored masks are now close to their practical limit. Further
  progress likely needs either a stronger semantic/structure classifier or a
  learned mask/reconstruction stage.

### Stage12 v1: learned candidate blend selector

Built `scripts/train_blend_selector.py` as a practical first learned selector.
It does not denoise from scratch; it predicts per-pixel weights for three
existing candidates:

```text
base    = final_v4_red115_blue120_detailguard_mild
rebuild = final_v7_luma_rebuild
cleanup = region/adaptive cleanup candidate
```

Rationale:

- Occi hair/root prefers `rebuild` because hand cleanup makes detail sleepy.
- Dance dark sky prefers cleanup/base because `rebuild` restores noise.
- A learned selector is the smallest useful AI step: it replaces fragile
  threshold masks without treating PhotoLab as an absolute teacher.

Smoke run:

```text
runs/refiner_pilot_stage11_hybrid_best/blend_selector_pilot_smoke/
steps=20, cpu
```

Result:

- Pipeline worked end to end, but the pseudo-teacher was too cleanup-heavy.
- Occi teacher rebuild mean was only `0.1616`, so hair/root remained too soft.

v2 run after teacher redesign:

```text
runs/refiner_pilot_stage11_hybrid_best/blend_selector_pilot_v2/
steps=300, batch=3, patch=192, width=28, cpu
```

Teacher means:

- Occi: `base=0.2651`, `rebuild=0.2607`, `cleanup=0.4742`
- Dance: `base=0.2696`, `rebuild=0.0914`, `cleanup=0.6390`

Applied outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/blend_selector_pilot_v2_outputs/
k5_dance_blend_selector_v2.exr
xt5_occi_blend_selector_v2.exr
```

Predicted weight stats:

- Dance: mean `base=0.2734`, `rebuild=0.1002`, `cleanup=0.6264`,
  rebuild p95 `0.6497`
- Occi: mean `base=0.2573`, `rebuild=0.2641`, `cleanup=0.4786`,
  rebuild p95 `0.9983`

Visual interpretation:

- Dance: selector keeps sky safe by mostly avoiding rebuild in flat dark sky,
  while still allowing rebuild on structured regions.
- Occi: selector now strongly switches to rebuild on selected structure, but
  final appearance is still bounded by the quality of the three candidates.
- This is promising as a region controller, not yet sufficient as the final
  "perfect NR" mechanism.

Next:

- Add crop/feature caching so training no longer spends minutes on full-image
  SciPy preprocessing every run.
- Add stronger structure labels for hair/root/wood and stricter flat labels for
  sky/background.
- If candidate blending still cannot reach PL-level detail/noise balance, the
  next stage needs a learned reconstruction/detail branch rather than only a
  selector.

### Stage12 v2: crop-mode selector and ROI-biased labels

Extended `scripts/train_blend_selector.py` with crop-mode feature generation and
ROI label bias. The aim was to avoid full-image feature preprocessing and make
the selector learn a clearer policy:

```text
sky/noise_dark -> cleanup/base
hair/root/person/house -> rebuild
skin/face -> cleanup
```

Training:

```text
runs/refiner_pilot_stage11_hybrid_best/blend_selector_pilot_v3/
feature-mode=crop
steps requested=600
checkpoint used=blend_selector_step_000300.pt
```

The run was interrupted after seeing buffered logs, but the step-300 checkpoint
was saved and evaluated.

Applied outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/blend_selector_pilot_v3_outputs/
k5_dance_blend_selector_v3_step300.exr
xt5_occi_blend_selector_v3_step300.exr
```

Predicted weight stats:

- Dance: mean `base=0.2047`, `rebuild=0.0918`, `cleanup=0.7035`,
  rebuild p95 `0.4653`
- Occi: mean `base=0.1982`, `rebuild=0.2420`, `cleanup=0.5597`,
  rebuild p95 `0.9914`

Reference-free metric comparison:

```text
runs/refiner_pilot_stage11_hybrid_best/blend_selector_eval/dance_v2_v3/
runs/refiner_pilot_stage11_hybrid_best/blend_selector_eval/occi_v2_v3/
```

Dance:

- `selector_v2`: luma visible ratio `0.181`, luma p99 ratio `0.501`,
  edge HF retention `0.873`
- `selector_v3`: luma visible ratio `0.188`, luma p99 ratio `0.485`,
  edge HF retention `0.871`

Occi:

- `selector_v2`: luma visible ratio `0.271`, luma p99 ratio `0.634`,
  edge HF retention `0.972`
- `selector_v3`: luma visible ratio `0.288`, luma p99 ratio `0.646`,
  edge HF retention `0.970`

Visual/technical interpretation:

- Crop-mode works and makes future selector training easier to iterate.
- ROI bias makes Occi structure switch strongly to rebuild, but the final image
  still looks bounded by candidate quality.
- v3 is not a clear improvement over v2. Dance remains cleanup-heavy; Occi is
  slightly worse on noise metrics than v2.

Decision:

- Keep `blend_selector_pilot_v2` as the better current selector baseline.
- Keep crop-mode/ROI-bias infrastructure, but do not spend much more time
  tuning selector-only training.
- The next serious improvement must change candidate generation or add a
  learned reconstruction/detail branch. Candidate blending alone is unlikely to
  become "perfect NR".

### Stage13 v1: learned luma reconstruction branch

Added `scripts/train_luma_rebuilder.py`.

Design:

- Input is the current selector output plus noisy/base/rebuild/cleanup
  candidates.
- Output is display-luma residual only. Chroma remains from the selector output.
- Pseudo target:
  - flat dark/skin regions stay near cleanup/current,
  - structure regions pull toward rebuild,
  - a small high-pass synthesis term from rebuild is allowed only under the
    structure gate.

The first smoke target was too weak (`target_delta_abs_mean` around `0.0013`),
so the target was changed to include rebuild high-pass directly. The useful
pilot is:

```text
runs/refiner_pilot_stage11_hybrid_best/luma_rebuilder_pilot_v3/
steps=300, width=36, blocks=5
```

Applied outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/luma_rebuilder_pilot_v3_outputs/
xt5_occi_luma_rebuilder_v3_s1.exr
k5_dance_luma_rebuilder_v3_s1.exr
xt5_occi_luma_rebuilder_v3_s055.exr
k5_dance_luma_rebuilder_v3_s055.exr
```

Full-strength delta stats:

- Occi: delta abs mean `0.00210`, p95 `0.00722`, p99 `0.01509`
- Dance: delta abs mean `0.00240`, p95 `0.00685`, p99 `0.01070`

Metric comparison versus selector v2:

Occi:

- `selector_v2`: edge HF retention `0.972`, luma visible ratio `0.271`
- `luma_rebuilder_s1`: edge HF retention `1.139`, luma visible ratio `0.336`
- `luma_rebuilder_s055`: edge HF retention `1.063`, luma visible ratio `0.305`

Dance:

- `selector_v2`: edge HF retention `0.873`, luma visible ratio `0.181`
- `luma_rebuilder_s1`: edge HF retention `1.015`, luma visible ratio `0.262`
- `luma_rebuilder_s055`: edge HF retention `0.950`, luma visible ratio `0.223`

Interpretation:

- This is the first branch that clearly moves beyond candidate blending:
  measured edge/detail energy increases.
- It also increases luma grain/noise, especially at full strength.
- The 0.55 scale is the better balance, but still not a breakthrough by eye.

Decision:

- Keep `luma_rebuilder_v3_s055` as a useful diagnostic output, not as the
  current best final image.
- The next version needs a stronger distinction between coherent detail and
  stochastic luma grain. A plain high-pass synthesis target is too blunt.
- The right next step is to teach a separate structure/noise discriminator for
  luma detail, then use the rebuilder only where that discriminator is confident.

### Stage13 v2: deterministic luma detail discriminator

Added `scripts/apply_luma_detail_discriminator.py` as a deterministic probe
before training a learned discriminator. It gates the full-strength rebuilder
delta using:

- coherent luma structure tensor evidence,
- base/rebuild texture agreement,
- dark low-saturation flat suppression,
- skin suppression,
- stochastic grain suppression from noisy high-pass energy.

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/luma_detail_discriminator_v1/
xt5_occi_luma_disc_v1.exr
k5_dance_luma_disc_v1.exr
```

Gate stats:

- Occi: gate mean `0.2803`, gate p95 `0.8291`,
  raw delta abs mean `0.00207`, gated `0.00111`
- Dance: gate mean `0.1838`, gate p95 `0.4634`,
  raw delta abs mean `0.00239`, gated `0.00084`

Metric comparison:

Occi:

- `selector_v2`: edge HF retention `0.972`, luma visible `0.271`
- `luma_rebuilder_s055`: edge HF retention `1.063`, luma visible `0.305`
- `luma_disc_v1`: edge HF retention `1.121`, luma visible `0.294`

Dance:

- `selector_v2`: edge HF retention `0.873`, luma visible `0.181`
- `luma_rebuilder_s055`: edge HF retention `0.950`, luma visible `0.223`
- `luma_disc_v1`: edge HF retention `0.992`, luma visible `0.204`

Interpretation:

- This is a real improvement over simple strength scaling: detail retention
  rises while visible luma noise is lower than the scaled rebuilder.
- The visual change is still subtle, but the tradeoff moved in the correct
  direction.

Decision:

- Keep `luma_disc_v1` as the best current diagnostic for the reconstruction
  branch.
- Next step should train this discriminator-like gate, or use its masks as
  pseudo-labels, instead of continuing to hand tune thresholds.

### Stage13 v3: strict luma detail discriminator scan

Scanned deterministic gate presets on ROI crops with
`scripts/scan_luma_detail_discriminator.py`. The goal was to see whether the
hand-written gate should be more open for detail or stricter against residual
luma grain.

Top combined ROI scores:

- `strict_noise`: `0.1154`
- `v1`: `0.1128`
- `balanced`: `0.1098`

Full-frame strict output:

```text
runs/refiner_pilot_stage11_hybrid_best/luma_detail_discriminator_strict_noise/
xt5_occi_luma_disc_strict_noise.exr
k5_dance_luma_disc_strict_noise.exr
```

Full-frame metric comparison:

Occi:

- `selector_v2`: edge HF retention `0.972`, luma visible `0.271`
- `luma_disc_v1`: edge HF retention `1.121`, luma visible `0.294`
- `strict_noise`: edge HF retention `1.116`, luma visible `0.290`

Dance:

- `selector_v2`: edge HF retention `0.873`, luma visible `0.181`
- `luma_disc_v1`: edge HF retention `0.992`, luma visible `0.204`
- `strict_noise`: edge HF retention `0.987`, luma visible `0.197`

Interpretation:

- `strict_noise` is slightly safer than `v1`.
- It gives up a tiny amount of edge retention but lowers visible luma residue.
- Use `strict_noise` as the deterministic baseline for learned gate distillation.

### Stage13 v4: learned luma detail gate

Added `scripts/train_luma_detail_gate.py`. This learns a small CNN gate for the
full-strength luma-rebuilder residual instead of hand-applying a fixed
structure/noise mask. It keeps chroma from `selector_v2` and only gates display
luma residuals.

Pilot v1:

```text
runs/refiner_pilot_stage11_hybrid_best/luma_detail_gate_pilot_v1/
runs/refiner_pilot_stage11_hybrid_best/luma_detail_gate_pilot_v1_outputs/
```

Result:

- v1 learned too open a gate.
- Occi: edge retention `1.079`, luma visible `0.296`
- Dance: edge retention `0.952`, luma visible `0.206`
- This was worse than `strict_noise` on both scenes, so v1 is rejected.

Pilot v2 strict:

```text
runs/refiner_pilot_stage11_hybrid_best/luma_detail_gate_pilot_v2_strict/
runs/refiner_pilot_stage11_hybrid_best/luma_detail_gate_pilot_v2_strict_outputs/
```

Training recipe:

- `420` CPU steps, batch `3`, patch `192`, context `64`
- `detail_mix=0.30`
- `roi_bias_strength=0.55`
- `smooth_weight=0.035`

Full-frame application stats:

- Occi: gate mean `0.3053`, gate p95 `0.9370`,
  gated delta abs mean `0.00105`
- Dance: gate mean `0.1372`, gate p95 `0.4659`,
  gated delta abs mean `0.00062`

Metric comparison:

Occi:

- `selector_v2`: edge HF retention `0.972`, luma visible `0.271`
- `strict_noise`: edge HF retention `1.116`, luma visible `0.290`
- `learned_gate_v2`: edge HF retention `1.126`, luma visible `0.291`

Dance:

- `selector_v2`: edge HF retention `0.873`, luma visible `0.181`
- `strict_noise`: edge HF retention `0.987`, luma visible `0.197`
- `learned_gate_v2`: edge HF retention `0.991`, luma visible `0.197`

Interpretation:

- v2 is the first learned luma gate that slightly beats the deterministic
  `strict_noise` gate on edge retention without materially increasing measured
  flat luma residue.
- The gain is small, but directionally correct and visually safe on the checked
  crops.
- Current best experimental luma reconstruction branch:
  `luma_detail_gate_pilot_v2_strict_outputs`.

Comparison sheets:

```text
runs/refiner_pilot_stage11_hybrid_best/luma_detail_gate_pilot_v2_strict_compare/
xt5_occi_gate_v2_compare_hair_detail_2x.png
xt5_occi_gate_v2_compare_root_2x.png
xt5_occi_gate_v2_compare_face_center_2x.png
k5_dance_gate_v2_compare_sky_existing_2x.png
k5_dance_gate_v2_compare_dancer_center_2x.png
k5_dance_gate_v2_compare_house_detail_2x.png
```

Next:

- Stop hand-tuning the deterministic gate unless a clear visual failure appears.
- Use learned gate v2 as the baseline and improve the luma-rebuilder teacher
  itself; the gate now preserves safety, but the generated detail is still not
  PL-level.

### Stage13 v5: detail-protected flat cleanup

Question:

- Can the PL-like split be used here: first reconstruct coherent hair/detail,
  then clean flat areas like sky more aggressively?

Added `scripts/apply_detail_protected_flat_cleanup.py`.

Design:

- Input is `learned_gate_v2`.
- Flat cleanup uses display-space luma/chroma smoothing.
- Protection uses:
  - coherent structure from the noisy reference,
  - texture masks from reference/current,
  - learned detail gate PNG,
  - skin mask,
  - highlight protection.

This makes the processing explicitly two-layer:

1. detail/microcontrast branch for coherent structures,
2. flat cleanup branch for sky, skin-like flats, and dark low-detail regions.

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/detail_protected_flat_cleanup_v1/
runs/refiner_pilot_stage11_hybrid_best/detail_protected_flat_cleanup_v2_sky/
```

v1 metrics:

Dance:

- `gate_v2`: flat luma `0.201`, luma visible `0.197`,
  edge retention `0.991`, chroma visible `0.058`
- `flat_cleanup_v1`: flat luma `0.194`, luma visible `0.190`,
  edge retention `0.991`, chroma visible `0.055`

Occi:

- `gate_v2`: flat luma `0.306`, luma visible `0.291`,
  edge retention `1.126`, chroma visible `0.070`
- `flat_cleanup_v1`: flat luma `0.293`, luma visible `0.276`,
  edge retention `1.126`, chroma visible `0.064`

v2_sky metrics:

Dance:

- `gate_v2`: flat luma `0.201`, luma visible `0.197`,
  edge retention `0.991`, chroma visible `0.058`,
  magenta visible `0.058`
- `flat_cleanup_v2_sky`: flat luma `0.189`, luma visible `0.183`,
  edge retention `0.991`, chroma visible `0.053`,
  magenta visible `0.053`

Occi:

- `gate_v2`: flat luma `0.306`, luma visible `0.291`,
  edge retention `1.126`, chroma visible `0.070`,
  magenta visible `0.063`
- `flat_cleanup_v2_sky`: flat luma `0.289`, luma visible `0.271`,
  edge retention `1.126`, chroma visible `0.063`,
  magenta visible `0.056`

Interpretation:

- The split works: flat noise can be reduced further after detail reconstruction
  without reducing the current edge-retention metric.
- The improvement is visible but still far from PL-level flat smoothness.
- v2_sky is the better current finish candidate.
- Next improvement should be a learned flat/sky cleanup mask or cleanup branch,
  not simply stronger global smoothing. The current mask is still conservative
  because it must avoid erasing real hair and texture.

Comparison sheets:

```text
runs/refiner_pilot_stage11_hybrid_best/detail_protected_flat_cleanup_v2_sky_compare/
xt5_occi_flat_cleanup_compare_hair_detail_2x.png
xt5_occi_flat_cleanup_compare_root_2x.png
xt5_occi_flat_cleanup_compare_face_center_2x.png
k5_dance_flat_cleanup_compare_sky_existing_2x.png
k5_dance_flat_cleanup_compare_dancer_center_2x.png
k5_dance_flat_cleanup_compare_house_detail_2x.png
```

### Stage13 v6: flat cleanup ROI scan and v3_more_flat

Added `scripts/scan_detail_protected_flat_cleanup.py` to scan cleanup presets
on diagnostic ROI crops before writing full-frame EXRs.

Scan output:

```text
runs/refiner_pilot_stage11_hybrid_best/detail_protected_flat_cleanup_scan_v1/
scan.md
scan.json
```

Combined crop scores:

- `v3_more_flat`: `0.02388`
- `v3_soft_flat`: `0.02181`
- `v2_sky`: `0.02029`
- `v3_aggressive_sky`: `0.01914`
- `v3_skin_safe`: `0.01705`
- `v3_texture_safe`: `0.01390`

Interpretation:

- More smoothing helps, but the fully aggressive sky preset is not best.
- The best tradeoff is a modestly more open flat gate with stronger smoothing,
  while keeping coherent/detail-gate protection.
- `v3_more_flat` wins every ROI in the scan, including detail and skin ROIs.

Full-frame outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/detail_protected_flat_cleanup_v3_more_flat/
xt5_occi_gate_v2_flat_cleanup_v3_more_flat.exr
k5_dance_gate_v2_flat_cleanup_v3_more_flat.exr
```

Full-frame metrics:

Dance:

- `gate_v2`: flat luma `0.201`, luma visible `0.197`,
  edge retention `0.991`, chroma visible `0.058`,
  magenta visible `0.058`
- `flat_cleanup_v2_sky`: flat luma `0.189`, luma visible `0.183`,
  edge retention `0.991`, chroma visible `0.053`,
  magenta visible `0.053`
- `flat_cleanup_v3_more_flat`: flat luma `0.185`, luma visible `0.178`,
  edge retention `0.991`, chroma visible `0.052`,
  magenta visible `0.051`

Occi:

- `gate_v2`: flat luma `0.306`, luma visible `0.291`,
  edge retention `1.126`, chroma visible `0.070`,
  magenta visible `0.063`
- `flat_cleanup_v2_sky`: flat luma `0.289`, luma visible `0.271`,
  edge retention `1.126`, chroma visible `0.063`,
  magenta visible `0.056`
- `flat_cleanup_v3_more_flat`: flat luma `0.286`, luma visible `0.268`,
  edge retention `1.126`, chroma visible `0.062`,
  magenta visible `0.056`

Decision:

- Current best finish candidate:
  `detail_protected_flat_cleanup_v3_more_flat`.
- Hand-tuned flat cleanup is now showing diminishing returns. The next large
  step should be a learned flat/sky cleanup branch or mask trained from the
  PL-like split, not another global smoothing increase.

Comparison sheets:

```text
runs/refiner_pilot_stage11_hybrid_best/detail_protected_flat_cleanup_v3_more_flat_compare/
xt5_occi_flat_cleanup_v3_compare_hair_detail_2x.png
xt5_occi_flat_cleanup_v3_compare_root_2x.png
xt5_occi_flat_cleanup_v3_compare_face_center_2x.png
k5_dance_flat_cleanup_v3_compare_sky_existing_2x.png
k5_dance_flat_cleanup_v3_compare_sky_center_2x.png
k5_dance_flat_cleanup_v3_compare_dancer_center_2x.png
k5_dance_flat_cleanup_v3_compare_house_detail_2x.png
```

### Stage13 v7: PL-aware scan, v4_flat_open, adaptive blend

Updated `scripts/scan_detail_protected_flat_cleanup.py` so PL outputs are used
only as a flat-noise floor reference, not as an absolute image teacher.

PL-aware scan output:

```text
runs/refiner_pilot_stage11_hybrid_best/detail_protected_flat_cleanup_scan_v2_pl/
scan.md
scan.json
```

Combined scores:

- `v3_more_flat`: `0.02457`
- `v4_flat_open`: `0.02335`
- `v3_soft_flat`: `0.02241`
- `v2_sky`: `0.02082`
- `v3_aggressive_sky`: `0.01964`
- `v4_pl_soft`: `0.01893`
- `v4_pl_flat`: `0.01806`

Interpretation:

- PL is not directly usable as a luma teacher on these crops because exposure
  and tone differences make some PL flat-luma ratios larger than our outputs.
- Chroma/magenta flat floors are still useful as directional hints.
- The scan still prefers `v3_more_flat`. Stronger PL-like variants overprotect
  or lose the local score.

Full-frame `v4_flat_open` outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/detail_protected_flat_cleanup_v4_flat_open/
xt5_occi_gate_v2_flat_cleanup_v4_flat_open.exr
k5_dance_gate_v2_flat_cleanup_v4_flat_open.exr
```

Metrics:

Dance:

- `v3_more_flat`: flat luma `0.185`, luma visible `0.178`,
  edge retention `0.991`, chroma visible `0.052`
- `v4_flat_open`: flat luma `0.184`, luma visible `0.176`,
  edge retention `0.990`, chroma visible `0.052`

Occi:

- `v3_more_flat`: flat luma `0.286`, luma visible `0.268`,
  edge retention `1.126`, chroma visible `0.062`
- `v4_flat_open`: flat luma `0.287`, luma visible `0.268`,
  edge retention `1.126`, chroma visible `0.063`

Added `scripts/apply_adaptive_flat_finish_blend.py` to blend `v4_flat_open`
into `v3_more_flat` only in strong flat regions.

Adaptive outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/adaptive_flat_finish_v1/
xt5_occi_adaptive_flat_finish_v1.exr
k5_dance_adaptive_flat_finish_v1.exr
```

Result:

- Adaptive blend did not materially improve over `v3_more_flat`.
- Dance adaptive stayed at luma visible `0.178`, edge retention `0.990`.
- Occi adaptive matched `v3_more_flat`.

Decision:

- Keep `v3_more_flat` as the robust current best.
- Keep `v4_flat_open` as a diagnostic/optional sky-push variant.
- Reject `adaptive_flat_finish_v1` as not worth using.
- The next meaningful step is a learned flat/sky cleanup branch, not more
  deterministic candidate blending. Hand-tuned finishing is now in diminishing
  returns.

### Stage13 v8: learned flat cleanup residual pilots

Added `scripts/train_flat_cleanup_branch.py`.

Goal:

- Train a tiny post-finish branch after `v3_more_flat`.
- Predict only a bounded display-RGB residual in flat/noisy regions.
- Use a stronger protected-flat cleanup as pseudo target.
- Do not use PL as an absolute image teacher.

Pilot v1:

```text
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_branch_pilot_v1/
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_branch_pilot_v1_outputs/
```

Result:

- Target delta was too small.
- Dance was essentially unchanged.
- Occi worsened chroma/magenta metrics.

Dance:

- `v3_more_flat`: flat luma `0.185`, luma visible `0.178`,
  edge retention `0.991`, chroma visible `0.052`
- `learned_flat_v1`: flat luma `0.185`, luma visible `0.178`,
  edge retention `0.991`, chroma visible `0.052`

Occi:

- `v3_more_flat`: flat luma `0.286`, luma visible `0.268`,
  chroma visible `0.062`, magenta visible `0.056`
- `learned_flat_v1`: flat luma `0.286`, luma visible `0.268`,
  chroma visible `0.064`, magenta visible `0.058`

Pilot v2:

```text
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_branch_pilot_v2_gain/
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_branch_pilot_v2_gain_outputs/
```

Changes:

- Added `target_gain`.
- Reduced over-suppression from the target cleanup weight by using a softer
  cleanup-weight exponent.

Dance result:

- `v3_more_flat`: flat luma `0.185`, luma visible `0.178`,
  edge retention `0.991`, chroma visible `0.052`
- `learned_flat_v2`: flat luma `0.184`, luma visible `0.178`,
  edge retention `0.990`, chroma visible `0.052`

Decision:

- Reject learned residual pilots v1/v2.
- The residual branch is currently learning tiny color/luma nudges, not a
  meaningful PL-like flat reconstruction.
- The failure is useful: a scalar/low-capacity residual target after
  deterministic cleanup is not enough. The next learned attempt should predict
  either:
  - a flat-region cleanup gate for an explicit smoother, or
  - a multi-scale low-frequency/chroma field, not direct per-pixel RGB residual.
- Current best remains `detail_protected_flat_cleanup_v3_more_flat`.

### Stage13 v9: learned flat cleanup gate pilot

Added `scripts/train_flat_cleanup_gate.py`.

Design:

- Model predicts a scalar cleanup gate, not RGB residual.
- A deterministic luma/chroma smoother applies the actual cleanup.
- Target gate is a stronger `v4_flat_open`-like protected flat gate.

Pilot:

```text
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_pilot_v1/
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_pilot_v1_outputs/
```

Training:

- `240` CPU steps
- target gate converged plausibly:
  final `pred_mean=0.1703`, `target_mean=0.1777`

Dance full-frame metrics:

- `v3_more_flat`: flat luma `0.185`, luma visible `0.178`,
  edge retention `0.991`, chroma visible `0.052`
- `v4_flat_open`: flat luma `0.184`, luma visible `0.176`,
  edge retention `0.990`, chroma visible `0.052`
- `learned_gate_s100`: flat luma `0.179`, luma visible `0.172`,
  edge retention `0.949`, chroma visible `0.051`

Added inference-time detail suppression:

```text
k5_dance_flat_cleanup_gate_v1_s100_ds085.exr
```

Dance metrics with detail suppression:

- `learned_gate_ds085`: flat luma `0.181`, luma visible `0.174`,
  edge retention `0.974`, chroma visible `0.051`

Interpretation:

- This is the first learned flat gate that clearly reduces flat luma/chroma
  beyond deterministic v3/v4.
- However, it leaks into structure too much. Even with detail suppression,
  edge retention drops from `0.991` to `0.974`.
- As a sky-only diagnostic it is promising. As a general finish it is not yet
  acceptable because quality priority puts structure preservation first.

Decision:

- Do not replace `v3_more_flat`.
- Keep `flat_cleanup_gate_pilot_v1` as the first useful learned flat-gate
  diagnostic.
- Next learned gate attempt should train with explicit edge/detail negative
  examples or an edge-retention penalty, rather than relying on post-hoc
  detail-gate suppression.

### Stage13 v10: edge-safe learned flat gate and selector rollback

Extended `scripts/train_flat_cleanup_gate.py` with target-time detail/edge
suppression:

```text
--target-detail-suppress 0.85
--target-edge-suppress 0.55
```

Pilot:

```text
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_pilot_v2_edge_safe/
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_pilot_v2_edge_safe_outputs/
```

Dance metrics:

- `v3_more_flat`: flat luma `0.185`, luma visible `0.178`,
  edge retention `0.991`, chroma visible `0.052`
- `v4_flat_open`: flat luma `0.184`, luma visible `0.176`,
  edge retention `0.990`, chroma visible `0.052`
- `learned_gate_v2_edge_safe`: flat luma `0.183`, luma visible `0.176`,
  edge retention `0.955`, chroma visible `0.052`
- `learned_gate_v2_edge_safe_ds085`: flat luma `0.184`, luma visible `0.178`,
  edge retention `0.977`, chroma visible `0.052`

Decision:

- Reject v2. Target suppression alone did not stop structural leakage.
- Inference detail suppression restored some edge retention, but also removed
  the noise-reduction gain. It is not a better trade than `v3_more_flat`.
- Current best remains `detail_protected_flat_cleanup_v3_more_flat`.

### Stage13 v11: learned candidate blend selector probes

Extended `scripts/train_blend_selector.py` so it can use two candidate sets:

- `legacy`: older `v4/v7/v9` candidates.
- `current`: current luma/detail baseline plus `v3_more_flat` cleanup.

Also added an optional structure-lock postprocess for the selector weights:

```text
--candidate-set current
--structure-lock-strength <value>
--structure-cleanup-floor <value>
```

Legacy selector probe:

```text
runs/refiner_pilot_stage11_hybrid_best/blend_selector_pilot_v2_quality/
runs/refiner_pilot_stage11_hybrid_best/blend_selector_pilot_v2_quality_outputs/
```

Dance result:

- `blend_selector_v2_step200`: flat luma `0.190`, luma visible `0.188`,
  edge retention `0.845`, chroma visible `0.058`

Current-candidate selector probe:

```text
runs/refiner_pilot_stage11_hybrid_best/blend_selector_current_pilot_v1_dance/
runs/refiner_pilot_stage11_hybrid_best/blend_selector_current_pilot_v1_dance_outputs/
```

Dance result:

- `selector_current_v1`: flat luma `0.187`, luma visible `0.182`,
  edge retention `0.960`, chroma visible `0.054`
- `selector_current_v1_lock090`: flat luma `0.184`, luma visible `0.180`,
  edge retention `0.831`, chroma visible `0.053`

Interpretation:

- The selector repeatedly collapses toward the cleanup candidate and gives up
  too much structure.
- The structure-lock postprocess was worse because the current rebuild candidate
  is not a universally safe detail fallback on Dance; redirecting cleanup weight
  to rebuild can reduce the measured edge-retention metric.
- This validates the earlier concern: candidate blending is not enough unless
  the candidate pool itself contains a truly clean/detail-preserving branch.

Decision:

- Reject both selector probes for the current finish.
- Keep the new `--candidate-set current` and structure-lock options as
  diagnostics only.
- The next promising direction is not longer training of these selectors. It is
  improving the candidate/teacher itself: a cleaner luma-detail reconstruction
  target or a learned multi-scale low-frequency/chroma field with explicit
  structure/noise separation.

### Stage13 v12: PL-informed frequency-split pseudo teacher

Added:

```text
scripts/build_frequency_split_pseudo_teacher.py
```

Purpose:

- Test whether PL can be used as a weak smoothness/flat-color reference without
  making PL an absolute teacher.
- Preserve Nagi's low-frequency tone/color.
- Borrow PL local residual only in flat regions.
- Split luma and chroma so PL can mostly inform chroma/colored speckles while
  Nagi keeps luma structure.

Initial RGB blend was rejected:

```text
runs/refiner_pilot_stage11_hybrid_best/frequency_split_pseudo_teacher_v1/
```

Dance metrics:

- `base_v3`: flat luma `0.185`, luma visible `0.178`,
  edge retention `0.991`, chroma visible `0.052`,
  magenta visible `0.051`
- `freqsplit_v1`: flat luma `0.267`, luma visible `0.245`,
  edge retention `0.991`, chroma visible `0.025`,
  magenta visible `0.023`

Interpretation:

- RGB PL residual improves chroma strongly but injects/keeps bad luma speckles.
- Do not use RGB PL residual as a teacher.

Luma/chroma split:

```text
runs/refiner_pilot_stage11_hybrid_best/frequency_split_pseudo_teacher_v2_chroma_luma_split/
```

Settings:

```text
--luma-strength 0.12
--chroma-strength 1.0
```

Dance metrics:

- `freqsplit_v2`: flat luma `0.177`, luma visible `0.171`,
  edge retention `0.990`, chroma visible `0.025`,
  magenta visible `0.022`

This is a real improvement on Dance: chroma/magenta dots are reduced
substantially while luma and edge retention remain safe.

Occi failed with the same v2 recipe:

- `base_v3`: luma visible `0.268`, chroma visible `0.062`,
  magenta visible `0.056`
- `freqsplit_v2`: luma visible `0.277`, chroma visible `0.119`,
  magenta visible `0.119`

The face crop showed blue/pink local color residuals from PL. This confirms the
teacher-diversity concern: PL cannot be borrowed broadly across image types.

Added PL chroma safety gating:

```text
--chroma-safety-strength 1.0
--chroma-color-threshold 0.035
--skin-flat-bonus -0.55
```

The gate allows PL chroma only when:

- PL local chroma high-frequency is lower than Nagi's, and
- low-frequency chroma agrees with Nagi enough to avoid PL tone/color drift.

Safe v3 outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/frequency_split_pseudo_teacher_v3_chroma_safe/
```

Dance:

- `base_v3`: luma visible `0.178`, chroma visible `0.052`,
  magenta visible `0.051`, edge retention `0.991`
- `freqsplit_v3`: luma visible `0.172`, chroma visible `0.043`,
  magenta visible `0.043`, edge retention `0.990`

Occi:

- `base_v3`: luma visible `0.268`, chroma visible `0.062`,
  magenta visible `0.056`, edge retention `1.126`
- `freqsplit_v3`: luma visible `0.271`, chroma visible `0.064`,
  magenta visible `0.058`, edge retention `1.127`

Interpretation:

- The safe gate removes the obvious Occi face/color failure.
- It also makes Occi mostly a no-op, which is preferable to a bad teacher.
- Dance still improves, though less than the aggressive v2.
- This supports a region-conditioned PL teacher: use PL chroma only where it is
  demonstrably cleaner and color-compatible.

Decision:

- Keep `freqsplit_v2_chroma_luma_split` as an aggressive Dance/flat-sky teacher
  diagnostic.
- Keep `freqsplit_v3_chroma_safe` as the safer general pseudo-teacher baseline.
- Do not train from PL broadly yet. First expand the pseudo-teacher test to Ice
  and Cat, then use only regions where v3 improves or is neutral.
- The teacher-variation concern is real, but manageable if PL is treated as a
  gated regional hint rather than a full-image target.

### Stage13 v13: frequency-split pseudo teacher across Ice/Cat

Applied `freqsplit_v3_chroma_safe` to Ice and Cat. Cat needed a PL crop offset
because the Nagi Cat sample is a crop of the full PhotoLab export:

```text
pl_offset_xy=(1808, 556)
```

Added this alignment support to
`scripts/build_frequency_split_pseudo_teacher.py`.

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/frequency_split_pseudo_teacher_v3_chroma_safe/
k5_ice_freqsplit_teacher_v3_chroma_safe.exr
xt5_cat_freqsplit_teacher_v3_chroma_safe.exr
```

Four-scene metric summary:

| scene | candidate | flat luma | luma visible | flat chroma | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dance | base | 0.185 | 0.178 | 0.056 | 0.052 | 0.051 | 0.040 | 0.991 |
| Dance | v3 | 0.179 | 0.172 | 0.048 | 0.043 | 0.043 | 0.032 | 0.990 |
| Occi | base | 0.286 | 0.268 | 0.068 | 0.062 | 0.056 | 0.032 | 1.126 |
| Occi | v3 | 0.289 | 0.271 | 0.070 | 0.064 | 0.058 | 0.038 | 1.127 |
| Ice | base | 0.209 | 0.205 | 0.090 | 0.080 | 0.069 | 0.048 | 0.275 |
| Ice | v3 | 0.202 | 0.198 | 0.077 | 0.068 | 0.058 | 0.039 | 0.275 |
| Cat | base | 0.132 | 0.133 | 0.039 | 0.038 | 0.039 | 0.035 | 0.322 |
| Cat | v3 | 0.134 | 0.136 | 0.031 | 0.031 | 0.031 | 0.028 | 0.320 |
| Average | base | 0.203 | 0.196 | 0.064 | 0.058 | 0.054 | 0.039 | 0.678 |
| Average | v3 | 0.201 | 0.194 | 0.056 | 0.051 | 0.048 | 0.034 | 0.678 |

Interpretation:

- Safe PL-chroma teacher improves Dance, Ice, and Cat.
- Occi becomes mostly a no-op with slight metric regression, which is still much
  better than the unsafe v2 face/color failure.
- Average chroma visible improves `0.058 -> 0.051`; magenta visible improves
  `0.054 -> 0.048`; shadow magenta visible improves `0.039 -> 0.034`; edge
  retention is unchanged on average.
- This is a valid weak teacher for chroma correction, but not a full-image
  denoising teacher.

Decision:

- Use `freqsplit_v3_chroma_safe` as the first distillation target for a small
  PL-free chroma branch.
- The branch should learn only a bounded display-chroma residual from noisy/base
  features. It should not learn luma reconstruction yet.
- Weight training regions by the pseudo-teacher safety/improvement signal so
  Occi-like no-op areas do not teach the model to make unnecessary color shifts.

### Stage13 v14: PL-free chroma distillation pilots

Added:

```text
scripts/train_frequency_chroma_distiller.py
```

Goal:

- Distill `freqsplit_v3_chroma_safe` into a model that does not need PL at
  inference.
- Inputs: noisy image and current Nagi base.
- Output: bounded display-chroma residual only; no luma prediction.

Pilot v1:

```text
runs/refiner_pilot_stage11_hybrid_best/frequency_chroma_distiller_pilot_v1/
```

Rejected because the feature tensor accidentally included a teacher-derived
`improve_hint`. At inference this feature became zero, so train/test conditions
did not match.

Pilot v2:

```text
runs/refiner_pilot_stage11_hybrid_best/frequency_chroma_distiller_pilot_v2_no_teacher_feature/
```

Fix:

- Removed teacher-derived features from the model input.
- Kept teacher-derived `improve_hint` only in the loss weight, which is valid
  because it is training-only supervision.

Training:

- `360` CPU steps
- prediction magnitude stayed conservative:
  final `pred_abs=0.00067`, `target_abs=0.00124`

Full-frame v2 results:

| scene | candidate | luma visible | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | base | 0.178 | 0.052 | 0.051 | 0.040 | 0.991 |
| Dance | teacher | 0.172 | 0.043 | 0.043 | 0.032 | 0.990 |
| Dance | distilled | 0.179 | 0.048 | 0.048 | 0.036 | 0.991 |
| Occi | base | 0.268 | 0.062 | 0.056 | 0.032 | 1.126 |
| Occi | teacher | 0.271 | 0.064 | 0.058 | 0.038 | 1.127 |
| Occi | distilled | 0.268 | 0.064 | 0.057 | 0.035 | 1.127 |
| Ice | base | 0.205 | 0.080 | 0.069 | 0.048 | 0.275 |
| Ice | teacher | 0.198 | 0.068 | 0.058 | 0.039 | 0.275 |
| Ice | distilled | 0.206 | 0.082 | 0.071 | 0.052 | 0.275 |
| Cat | base | 0.133 | 0.038 | 0.039 | 0.035 | 0.322 |
| Cat | teacher | 0.136 | 0.031 | 0.031 | 0.028 | 0.320 |
| Cat | distilled | 0.134 | 0.041 | 0.042 | 0.040 | 0.322 |
| Average | base | 0.196 | 0.058 | 0.054 | 0.039 | 0.678 |
| Average | teacher | 0.194 | 0.051 | 0.048 | 0.034 | 0.678 |
| Average | distilled | 0.196 | 0.059 | 0.055 | 0.041 | 0.679 |

The prediction direction had positive correlation with the teacher delta, but
the output was too broad and weak. It helped Dance slightly and worsened Ice/Cat.

Tried post-hoc inference gating:

```text
runs/refiner_pilot_stage11_hybrid_best/frequency_chroma_distiller_pilot_v2_no_teacher_feature_gated_outputs/
```

Middle gate strength `g100`:

| scene | candidate | luma visible | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | base | 0.178 | 0.052 | 0.051 | 0.040 | 0.991 |
| Dance | teacher | 0.172 | 0.043 | 0.043 | 0.032 | 0.990 |
| Dance | distilled-gated | 0.178 | 0.050 | 0.050 | 0.038 | 0.990 |
| Ice | base | 0.205 | 0.080 | 0.069 | 0.048 | 0.275 |
| Ice | teacher | 0.198 | 0.068 | 0.058 | 0.039 | 0.275 |
| Ice | distilled-gated | 0.205 | 0.082 | 0.071 | 0.051 | 0.275 |
| Cat | base | 0.133 | 0.038 | 0.039 | 0.035 | 0.322 |
| Cat | teacher | 0.136 | 0.031 | 0.031 | 0.028 | 0.320 |
| Cat | distilled-gated | 0.133 | 0.040 | 0.042 | 0.039 | 0.322 |

Decision:

- Reject distiller pilots v1/v2.
- Keep `freqsplit_v3_chroma_safe` as a good pseudo-teacher image generator.
- Do not spend long training on the current direct RGB/chroma residual branch.
- Next design should predict a lower-dimensional signed chroma correction or a
  correction gate per opponent axis, not unconstrained RGB chroma residual. The
  target signal is too small for the current direct regression setup and becomes
  broad/ambiguous.

### Stage13 v15: opponent-axis chroma distillation

Added:

```text
scripts/train_opponent_chroma_distiller.py
```

Design:

- Predict three signed opponent coefficients instead of free RGB/chroma
  residuals:
  - magenta / green
  - blue / yellow
  - red / cyan
- Reconstruct the RGB chroma residual from these zero-sum axes.
- This should reduce ambiguity versus direct RGB residual regression.

Pilot:

```text
runs/refiner_pilot_stage11_hybrid_best/opponent_chroma_distiller_pilot_v1/
runs/refiner_pilot_stage11_hybrid_best/opponent_chroma_distiller_pilot_v1_outputs/
```

Training:

- `360` CPU steps on Dance/Ice/Cat.
- final `pred_abs=0.00091`, `target_abs=0.00143`.
- The target scale is better than the free-RGB distiller, but still tiny.

Full-frame result:

| scene | candidate | luma visible | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | base | 0.178 | 0.052 | 0.051 | 0.040 | 0.991 |
| Dance | teacher | 0.172 | 0.043 | 0.043 | 0.032 | 0.990 |
| Dance | opponent | 0.179 | 0.046 | 0.046 | 0.035 | 0.991 |
| Ice | base | 0.205 | 0.080 | 0.069 | 0.048 | 0.275 |
| Ice | teacher | 0.198 | 0.068 | 0.058 | 0.039 | 0.275 |
| Ice | opponent | 0.206 | 0.084 | 0.074 | 0.055 | 0.275 |
| Cat | base | 0.133 | 0.038 | 0.039 | 0.035 | 0.322 |
| Cat | teacher | 0.136 | 0.031 | 0.031 | 0.028 | 0.320 |
| Cat | opponent | 0.134 | 0.043 | 0.045 | 0.043 | 0.321 |
| Average | base | 0.172 | 0.057 | 0.053 | 0.041 | 0.529 |
| Average | teacher | 0.169 | 0.047 | 0.044 | 0.033 | 0.529 |
| Average | opponent | 0.173 | 0.058 | 0.055 | 0.044 | 0.529 |

Interpretation:

- Opponent-axis regression improves Dance more than the free RGB/chroma branch.
- It still worsens Ice/Cat because the model emits broad low-amplitude
  corrections instead of sparse point/outlier corrections.
- The failure is now clearly about spatial selectivity, not color-axis
  representation.

Decision:

- Reject `opponent_chroma_distiller_pilot_v1` as a final branch.
- Do not spend long training on signed coefficient regression yet.
- Next design should predict a sparse gate/strength for the existing signed
  chroma outlier filter, or a per-axis outlier-gate correction, rather than
  directly regressing the correction value.
- Keep the PL frequency-split teacher as supervision for where the outlier
  filter should open/close.

### Stage13 v16: signed chroma outlier scan against PL-safe teacher

Added:

```text
scripts/scan_signed_chroma_outlier_filter.py
```

Reasoning:

- Direct chroma residual regression kept producing broad, weak corrections.
- The hand-built signed outlier filter is more stable because it preserves
  display luma and only pulls opponent-axis chroma outliers toward a robust
  local surface.
- Therefore the next practical step is not another long distillation run, but
  a constrained scan of the filter's axis/threshold/gate settings against the
  PL-safe frequency-split teacher.

Run:

```text
pixi run python scripts/scan_signed_chroma_outlier_filter.py --apply-full
```

Output:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_scan_v5_plsafe/
```

Best coarse setting:

```text
strength=0.88
median_size=7
low_sigma=2.4
outlier_threshold=0.0028
outlier_transition=0.00196
magenta_weight=1.20
red_weight=0.65
blue_weight=1.15
detail_threshold=0.020
edge_threshold=0.030
shadow_threshold=0.58
```

ROI scan summary:

| rank | chroma | magenta | shadow magenta | luma | edge loss | teacher chroma MAE | worse |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.817 | 0.798 | 0.681 | 0.999 | 0.00704 | 0.00107 | 2 |

Full-frame evaluation:

| scene | candidate | luma visible | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | base | 0.178 | 0.052 | 0.051 | 0.040 | 0.991 |
| Dance | teacher | 0.172 | 0.043 | 0.043 | 0.032 | 0.990 |
| Dance | v5 | 0.177 | 0.043 | 0.042 | 0.032 | 0.990 |
| Occi | base | 0.268 | 0.062 | 0.056 | 0.032 | 1.126 |
| Occi | teacher | 0.271 | 0.064 | 0.058 | 0.038 | 1.127 |
| Occi | v5 | 0.269 | 0.055 | 0.048 | 0.025 | 1.127 |
| Ice | base | 0.205 | 0.080 | 0.069 | 0.048 | 0.275 |
| Ice | teacher | 0.198 | 0.068 | 0.058 | 0.039 | 0.275 |
| Ice | v5 | 0.206 | 0.063 | 0.054 | 0.035 | 0.273 |
| Cat | base | 0.133 | 0.038 | 0.039 | 0.035 | 0.322 |
| Cat | teacher | 0.136 | 0.031 | 0.031 | 0.028 | 0.320 |
| Cat | v5 | 0.134 | 0.030 | 0.029 | 0.025 | 0.318 |

Interpretation:

- v5 reduces chroma/magenta on all four full-frame samples, including Occi
  where the PL-safe teacher intentionally does almost nothing.
- The result even beats the PL-safe teacher on Ice/Cat chroma metrics because
  this filter targets high-frequency signed dots without importing PL tone.
- Luma visible noise is essentially unchanged, which is expected because the
  filter operates in display-chroma space.
- Remaining risk: Cat edge retention falls from `0.3216` to `0.3177`, about
  `1.2%` relative. This is not a catastrophic visual break in the inspected
  crops, but it is enough to block final adoption.

Decision:

- Keep `signed_chroma_outlier_scan_v5_plsafe` as the current best chroma-dot
  suppression candidate.
- Do not train another direct chroma correction model yet.
- Next step: add a stricter local line/detail protection gate to the signed
  outlier filter, then rerun a smaller scan. The target is to keep v5's chroma
  gains while restoring Cat edge retention to within `0.5%` of base.

### Stage13 v17: failed restore-after-filter protection scans

Tried two protection variants inside `scripts/scan_signed_chroma_outlier_filter.py`:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_scan_v6_lineprotect/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_scan_v7_coherentprotect/
```

v6:

- After the signed chroma outlier filter, blend base back where fine luma
  high-frequency energy is high.
- Best candidate still chose `line_restore_strength=0.0`.
- The restore mask catches noise-like fine energy as well as real lines, so it
  weakens chroma-dot removal without improving the edge metric.

v7:

- Replaced fine luma restore with a coherent-structure restore mask.
- Best candidate still chose `coherent_restore_strength=0.0`.
- Top protected candidate had worse chroma/magenta ratios and slightly worse
  edge loss than unprotected v5:

| rank | protection | chroma | magenta | shadow magenta | edge loss | score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | none | 0.817 | 0.798 | 0.681 | 0.00704 | 3.0817 |
| 3 | coherent restore | 0.829 | 0.811 | 0.693 | 0.00734 | 3.1260 |

Decision:

- Reject restore-after-filter protection. It is the wrong place in the graph.
- The next protection attempt should inhibit the signed outlier gate before
  correction, not restore base after correction.
- Keep v5 as current best output. Next implementation should add structure
  inhibition into the outlier blend itself, preferably per-axis, so chroma dots
  in flat areas are still removed while coherent hair/whisker lines do not
  open the correction gate.

### Stage13 v18: gate inhibition and density inhibition scans

Added optional internal gate controls to:

```text
scripts/apply_signed_chroma_outlier_filter.py
```

New controls:

- `coherent_inhibit_strength`: reduce the correction gate before applying the
  chroma outlier correction in coherent luma structures.
- `outlier_density_inhibit_strength`: reduce the per-axis outlier gate when the
  local 5x5 outlier gate density is high. The hypothesis was that true random
  chroma dots are sparse, while hair/whisker/texture edges form denser support.

Scans:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_scan_v8_gateinhibit/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_scan_v9_densityinhibit/
```

v8 result:

- Best candidate still selected `coherent_inhibit_strength=0.0`.
- Coherent inhibition protected too broadly and gave up more chroma cleanup
  than it recovered in edge retention.

v9 result:

- Best scan candidate still selected `density_inhibit_strength=0.0`.
- The first density-inhibited candidate landed at rank 7:

| candidate | chroma | magenta | shadow magenta | edge loss | score |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5-equivalent | 0.817 | 0.798 | 0.681 | 0.00704 | 3.0817 |
| density035 | 0.853 | 0.835 | 0.715 | 0.00685 | 3.1961 |

Full density035 output:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_scan_v9_density035_full/
```

Full-frame comparison:

| scene | candidate | luma visible | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | base | 0.178 | 0.052 | 0.051 | 0.040 | 0.991 |
| Dance | v5 | 0.177 | 0.043 | 0.042 | 0.032 | 0.990 |
| Dance | density035 | 0.177 | 0.045 | 0.044 | 0.034 | 0.990 |
| Occi | base | 0.268 | 0.062 | 0.056 | 0.032 | 1.126 |
| Occi | v5 | 0.269 | 0.055 | 0.048 | 0.025 | 1.127 |
| Occi | density035 | 0.268 | 0.056 | 0.049 | 0.026 | 1.126 |
| Ice | base | 0.205 | 0.080 | 0.069 | 0.048 | 0.275 |
| Ice | v5 | 0.206 | 0.063 | 0.054 | 0.035 | 0.273 |
| Ice | density035 | 0.206 | 0.067 | 0.057 | 0.038 | 0.274 |
| Cat | base | 0.133 | 0.038 | 0.039 | 0.035 | 0.322 |
| Cat | v5 | 0.134 | 0.030 | 0.029 | 0.025 | 0.318 |
| Cat | density035 | 0.134 | 0.031 | 0.031 | 0.026 | 0.319 |

Interpretation:

- Density inhibition is a real but small tradeoff knob.
- It recovers some Ice/Cat edge retention, but not enough to fix the underlying
  concern.
- It gives up a visible portion of the chroma-dot cleanup that made v5 useful.
- Therefore this is useful as a conservative/detail-safe preset, but not the
  main quality breakthrough.

Decision:

- Current strong chroma-dot candidate: `signed_chroma_outlier_scan_v5_plsafe`.
- Conservative/detail-safe candidate: `signed_chroma_outlier_scan_v9_density035_full`.
- Do not keep pushing hand-designed protection gates as the main route. The
  remaining gap likely needs a learned region/texture classifier or a stronger
  luma-detail reconstruction stage, because the hand gates cannot separate
  dense real texture from dense chroma noise reliably enough.

### Stage13 v19: adaptive strong/detail-safe chroma blend

Added:

```text
scripts/apply_adaptive_chroma_detail_blend.py
```

Design:

- Use v5 strong chroma-dot cleanup in flat regions.
- Blend toward density035 only where the current base has texture/coherent
  structure.
- This is different from restoring the original base: the fallback is still a
  denoised output, just a slightly more detail-safe one.

Output:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v10_adaptive_detail_blend/
```

Full-frame comparison:

| scene | candidate | luma visible | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | base | 0.178 | 0.052 | 0.051 | 0.040 | 0.991 |
| Dance | v5 | 0.177 | 0.043 | 0.042 | 0.032 | 0.990 |
| Dance | density035 | 0.177 | 0.045 | 0.044 | 0.034 | 0.990 |
| Dance | adaptive | 0.177 | 0.043 | 0.043 | 0.032 | 0.990 |
| Occi | base | 0.268 | 0.062 | 0.056 | 0.032 | 1.126 |
| Occi | v5 | 0.269 | 0.055 | 0.048 | 0.025 | 1.127 |
| Occi | density035 | 0.268 | 0.056 | 0.049 | 0.026 | 1.126 |
| Occi | adaptive | 0.268 | 0.055 | 0.049 | 0.026 | 1.126 |
| Ice | base | 0.205 | 0.080 | 0.069 | 0.048 | 0.275 |
| Ice | v5 | 0.206 | 0.063 | 0.054 | 0.035 | 0.273 |
| Ice | density035 | 0.206 | 0.067 | 0.057 | 0.038 | 0.274 |
| Ice | adaptive | 0.206 | 0.065 | 0.055 | 0.036 | 0.274 |
| Cat | base | 0.133 | 0.038 | 0.039 | 0.035 | 0.322 |
| Cat | v5 | 0.134 | 0.030 | 0.029 | 0.025 | 0.318 |
| Cat | density035 | 0.134 | 0.031 | 0.031 | 0.026 | 0.319 |
| Cat | adaptive | 0.134 | 0.030 | 0.029 | 0.025 | 0.318 |

Interpretation:

- Adaptive is the best practical compromise so far.
- It keeps almost all of v5's chroma/magenta cleanup.
- It recovers part of the Ice/Cat edge loss, especially compared with pure v5.
- It does not solve the full detail-retention problem; Cat edge retention is
  still below base. But it is a better default candidate than pure density035.

Decision:

- Preferred current output for user inspection: v10 adaptive detail blend.
- Keep v5 strong as the aggressive chroma-dot baseline.
- Next real improvement should likely be learned selection between strong,
  detail-safe, and base/rebuild branches, trained on patch-level targets rather
  than more hand-tuned global masks.

### Stage13 v20: teacher-oracle strong/detail-safe selector upper bound

Added:

```text
scripts/build_chroma_selector_oracle_blend.py
```

Purpose:

- Before training a selector, test whether a PL-safe pseudo-teacher can provide
  a useful local selection signal between v5 strong and density035.
- If an oracle that directly sees the teacher cannot beat the hand adaptive
  blend, a learned selector trained from the same signal is unlikely to help.

Output:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v11_teacher_oracle_blend/
```

Teacher-distance result:

| scene | detail-safe weight mean | p90 | p99 | strong loss | detail-safe loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Occi | 0.451 | 0.455 | 0.484 | 0.00195 | 0.00195 |
| Dance | 0.440 | 0.460 | 0.494 | 0.00155 | 0.00160 |
| Ice | 0.453 | 0.475 | 0.541 | 0.00195 | 0.00194 |
| Cat | 0.447 | 0.451 | 0.460 | 0.00093 | 0.00095 |

Cat evaluation:

| candidate | luma visible | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.133 | 0.038 | 0.039 | 0.035 | 0.322 |
| adaptive | 0.134 | 0.030 | 0.029 | 0.025 | 0.318 |
| oracle | 0.134 | 0.030 | 0.030 | 0.025 | 0.318 |

Interpretation:

- The PL-safe teacher sees v5 and density035 as almost equivalent.
- The oracle mask collapses to an almost constant blend around `0.44-0.45`.
- On Cat, the oracle is slightly worse than v10 adaptive: less chroma cleanup
  and no edge advantage.

Decision:

- Do not train a strong-vs-density035 selector from the current PL-safe teacher.
- The selection target is too weak and too ambiguous.
- Keep v10 adaptive as the current chroma-dot best.
- Next improvement should move away from this narrow chroma selector and back
  toward luma/detail reconstruction or a richer teacher with clearer patch
  preferences.

### Stage13 v21: conservative luma detail restore on v10

Added:

```text
scripts/apply_v10_perceptual_luma_detail_restore.py
```

v12:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v12_luma_detail_restore/
```

Parameters:

```text
strength=0.18
energy_threshold=0.013
coherence_threshold=0.42
correction_limit=0.012
max_detail_frac=0.035
```

v12 result:

- Cat and Ice improved strongly: luma visible went down and edge retention went
  up.
- Dance and Occi got slightly worse, probably because noisy high-frequency sky
  or already-detailed regions were treated as recoverable luma detail.

| scene | candidate | luma visible | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | v10 | 0.177 | 0.043 | 0.043 | 0.032 | 0.990 |
| Dance | v12 | 0.181 | 0.043 | 0.043 | 0.032 | 0.990 |
| Occi | v10 | 0.268 | 0.055 | 0.049 | 0.026 | 1.126 |
| Occi | v12 | 0.270 | 0.055 | 0.049 | 0.026 | 1.125 |
| Ice | v10 | 0.206 | 0.065 | 0.055 | 0.036 | 0.274 |
| Ice | v12 | 0.203 | 0.065 | 0.055 | 0.036 | 0.279 |
| Cat | v10 | 0.134 | 0.030 | 0.029 | 0.025 | 0.318 |
| Cat | v12 | 0.131 | 0.030 | 0.029 | 0.025 | 0.323 |

v13:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v13_luma_detail_basegated/
```

Change:

- Require base/v10 to already contain local luma detail before restoring detail
  from the noisy reference.
- This suppresses false detail restoration in flat/noisy regions.

Command:

```text
pixi run python scripts/apply_v10_perceptual_luma_detail_restore.py \
  --output-dir runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v13_luma_detail_basegated \
  --tag v13_luma_detail_basegated \
  --base-energy-threshold 0.006 \
  --base-energy-transition 0.0045
```

v13 result:

| scene | candidate | luma visible | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | v10 | 0.177 | 0.043 | 0.043 | 0.032 | 0.990 |
| Dance | v13 | 0.178 | 0.043 | 0.043 | 0.032 | 0.990 |
| Occi | v10 | 0.268 | 0.055 | 0.049 | 0.026 | 1.126 |
| Occi | v13 | 0.269 | 0.055 | 0.049 | 0.026 | 1.125 |
| Ice | v10 | 0.206 | 0.065 | 0.055 | 0.036 | 0.274 |
| Ice | v13 | 0.204 | 0.065 | 0.055 | 0.036 | 0.277 |
| Cat | v10 | 0.134 | 0.030 | 0.029 | 0.025 | 0.318 |
| Cat | v13 | 0.133 | 0.030 | 0.029 | 0.025 | 0.321 |

Interpretation:

- v13 is the best luma/detail compromise so far.
- It keeps the chroma-dot cleanup from v10.
- It recovers much of the Cat/Ice edge retention lost by denoising.
- It still slightly worsens Dance/Occi versus v10, so it is not yet a universal
  default without a better region selector.

Decision:

- Keep v10 as the safest current default.
- Keep v13 as the current detail-recovery candidate for inspection.
- Next step should be a region selector for applying v13 only where it improves
  luma/edge, rather than applying luma detail restoration globally.

### Stage13 v22: Occi hair detail recovery probes

The user flagged that the detail-recovery candidate looked broadly good on
Occi, but crushed hair texture. I tested several targeted recovery strategies
on top of the current safe v10 chroma cleanup.

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v14_dark_hair_detail_rescue/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v14b_dark_hair_detail_rescue_stronger/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v15_occi_structure_rebuild/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v16_dark_line_structure_rebuild/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v17_dark_line_band_rebuild/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v17b_dark_line_band_rebuild_open/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v18_dark_line_microcontrast/
```

Scripts added:

```text
scripts/apply_dark_coherent_hair_detail_rescue.py
scripts/apply_dark_line_structure_rebuild.py
scripts/apply_dark_line_microcontrast_boost.py
```

Occi evaluation:

| candidate | luma p99 | chroma p99 | magenta p99 | shadow magenta p99 | edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 safe base | 0.011932 | 0.005126 | 0.002675 | 0.001335 | 1.126494 |
| v15 structure rebuild | 0.010414 | 0.005139 | 0.002670 | 0.001278 | 0.854876 |
| v16 dark-line structure | 0.011856 | 0.005131 | 0.002672 | 0.001325 | 1.068421 |
| v17 dark-line band | 0.011871 | 0.005126 | 0.002675 | 0.001333 | 1.119038 |
| v18 microcontrast | 0.012874 | 0.005103 | 0.002676 | 0.001360 | 1.283231 |

Interpretation:

- v14/v14b were too conservative. They barely changed the Occi hair and did not
  solve the crushed-texture complaint.
- v15 restored some visible hair structure, but it behaved like a broad luma
  replacement and reduced the edge metric too much. This confirms that low/mid
  structure grafting is not safe as a universal final stage.
- v16 limited structure reconstruction to dark coherent lines. It was safer
  than v15, but still softened too much.
- v17 switched to band-only restoration. This kept edge retention close to v10,
  but was visually subtle.
- v18 does not borrow from the noisy input. It boosts existing v10 dark-line
  microcontrast, so it gives the clearest Occi hair improvement with no chroma
  regression. The tradeoff is higher luma HF, so it should be an optional
  detail/rebuild mode, not the smoothest default.

Decision:

- Keep v10 as the safe default.
- Do not promote v14, v14b, v15, or v16.
- Keep v17 as the conservative detail recovery reference.
- Keep v18 as the current best Occi hair/detail candidate, but apply it only
  through a region selector or a detail-priority preset. Dance sky and flat fur
  still need guarding before v18 can be always-on.

### Stage13 v23: OkLab edge gate probe

The user suggested OkLab as a better edge-detection color space. I tested it as
a gate feature, not as a denoising color space. The implementation converts the
display-referred preview RGB to OkLab, computes a perceptual edge magnitude in
`L/a/b`, and mixes that gate with the existing dark-line density gate.

Updated:

```text
scripts/apply_dark_line_texture_floor.py
```

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v21_oklab_hair_texture_floor/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v21b_oklab_hair_texture_floor_mild/
```

Occi evaluation:

| candidate | luma p99 | chroma p99 | magenta p99 | shadow magenta p99 | edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| v18 microcontrast | 0.012874 | 0.005103 | 0.002676 | 0.001360 | 1.283231 |
| v21 OkLab texture floor | 0.013257 | 0.005097 | 0.002678 | 0.001409 | 1.309600 |
| v21b OkLab mild | 0.013047 | 0.005100 | 0.002677 | 0.001382 | 1.295726 |

Interpretation:

- OkLab is useful as a structure gate: it increases edge retention without
  hurting chroma or magenta-dot metrics.
- The strong v21 setting visibly fills some flat hair texture, but it starts to
  read as fine grain because the texture source is still derived from the noisy
  reference.
- v21b is the better practical setting so far. It keeps most of the edge gain
  while reducing the added grain.
- OkLab should not be treated as the NR space yet. Its best immediate use is
  gating and region selection for dark hair/detail restoration.

Decision:

- Keep v18 as the safer detail-priority baseline.
- Keep v21b as the current OkLab-gated candidate.
- Next refinement should denoise or synthesize the texture-floor source before
  adding it, rather than increasing the v21/v21b strength directly.

### Stage13 v24: denoised texture-floor and OkLab microcontrast follow-up

The user noted that the hair remained difficult and still contained
over-denoised flat patches. I tested two follow-ups:

1. `v22`: keep the OkLab-gated texture floor, but soft-threshold the reference
   luma texture before adding it. This suppresses weaker random grain.
2. `v23`: do not borrow reference texture at all. Use OkLab only to widen the
   v18-style microcontrast gate, so the source remains the denoised v10/v18
   structure.

Updated:

```text
scripts/apply_dark_line_texture_floor.py
scripts/apply_dark_line_microcontrast_boost.py
```

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v22_oklab_soft_texture_floor/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v23_oklab_microcontrast/
```

Occi evaluation:

| candidate | luma p99 | chroma p99 | magenta p99 | shadow magenta p99 | edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| v18 microcontrast | 0.012874 | 0.005103 | 0.002676 | 0.001360 | 1.283231 |
| v21b OkLab mild | 0.013047 | 0.005100 | 0.002677 | 0.001382 | 1.295726 |
| v22 OkLab soft texture | 0.013184 | 0.005098 | 0.002677 | 0.001400 | 1.304839 |
| v23 OkLab microcontrast | 0.013100 | 0.005103 | 0.002678 | 0.001369 | 1.284652 |

Interpretation:

- Soft-thresholding the reference texture helps avoid the worst noisy speckles,
  but v22 still reads as added fine grain if pushed too far.
- v23 avoids reference-derived grain, but it also cannot fill hair areas that
  v10 already flattened. It is therefore not a meaningful improvement over v18.
- OkLab is still useful for region/gate detection. The current bottleneck is
  the texture source: hand-crafted post-processing can sharpen remaining hair
  lines, but it cannot convincingly reconstruct missing hair surface texture.

Decision:

- Keep v21b as the best practical OkLab candidate.
- Keep v22 as the stronger/detail-priority reference.
- Do not promote v23.
- Next serious step should be a small learned hair/detail residual module or a
  better synthesized texture source, not more direct reference texture grafting.

### Stage13 v25: oriented OkLab texture floor

The next attempt kept the useful OkLab gate, but changed the texture source so
it is not added as isotropic fine grain. The reference luma band is
soft-thresholded, then filtered through a 4-direction line filter chosen from
the local hair/structure orientation. This makes the added texture more
strand-like and less speckle-like.

Updated:

```text
scripts/apply_dark_line_texture_floor.py
```

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v24_oklab_oriented_texture_floor/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v24b_oklab_oriented_texture_floor_mild/
```

Occi evaluation:

| candidate | luma p99 | chroma p99 | magenta p99 | shadow magenta p99 | edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| v18 microcontrast | 0.012874 | 0.005103 | 0.002676 | 0.001360 | 1.283231 |
| v21b OkLab mild | 0.013047 | 0.005100 | 0.002677 | 0.001382 | 1.295726 |
| v24 oriented | 0.013202 | 0.005096 | 0.002677 | 0.001380 | 1.305245 |
| v24b oriented mild | 0.013088 | 0.005099 | 0.002677 | 0.001373 | 1.297341 |

Interpretation:

- Directional texture shaping is useful. v24 is similar to v22 numerically, but
  the added texture reads slightly more like hair strands and less like
  isotropic grain.
- v24b is the better practical setting: it improves edge retention over v21b
  while keeping luma p99 close and slightly reducing shadow-magenta impact.
- The improvement is still subtle. This confirms the hand-crafted ceiling:
  remaining lines can be enhanced, but convincingly reconstructing flattened
  hair surface needs a learned or better-synthesized texture source.

Decision:

- Promote v24b to the current best hand-crafted hair/detail candidate.
- Keep v24 as the stronger detail-priority reference.
- The next direction should be a small learned residual module trained to
  produce strand-like luma detail under the OkLab/orientation gates, rather than
  more direct texture graft strength increases.

### Stage13 v26/v27: HDR restore and shadow-lift detail detection

The Occi crop showed two separate failures:

1. Bokeh and the top hair highlights had lost HDR detail.
2. Dark hair strands were still too sleepy; lifting the shadows visually
   reveals more strand structure in the noisy source.

The HDR issue was a real pipeline bug in the display-space experiments. Several
detail stages converted to display RGB, clipped to 0..1, and then converted back
to EXR. Occi's source reaches about 5.96 linear, while v24b/v18-derived outputs
were capped at exactly 1.0.

Updated:

```text
scripts/apply_hdr_highlight_restore.py
scripts/apply_dark_line_texture_floor.py
```

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v25_hdr_restore/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v26b_shadow_lift_hdr_restore/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v27a_shadow_lift_detail_hdr_occi/
runs/refiner_pilot_stage11_hybrid_best/signed_chroma_outlier_v27b_shadow_lift_detail_hdr_occi/
```

HDR restoration stats:

| scene | restored max | restored >1 frac | source max |
| --- | ---: | ---: | ---: |
| xt5_occi | 5.8821 | 0.102199 | 5.9575 |
| xt5_cat | 2.9536 | 0.000054 | 3.4323 |
| k5_ice | 5.1504 | 0.053664 | 5.4795 |
| k5_dance | 10.0180 | 0.012897 | 10.0180 |

Occi evaluation:

| candidate | luma p99 | chroma p99 | magenta p99 | shadow magenta p99 | edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| v24b | 0.013088 | 0.005099 | 0.002677 | 0.001373 | 1.297341 |
| v25 HDR restore | 0.013200 | 0.006600 | 0.004034 | 0.001406 | 1.196200 |
| v26b shadow lift + HDR | 0.013183 | 0.006600 | 0.004034 | 0.001406 | 1.195637 |
| v27a stronger shadow lift + HDR | 0.013310 | 0.006599 | 0.004034 | 0.001421 | 1.200795 |
| v27b medium shadow lift + HDR | 0.013249 | 0.006600 | 0.004033 | 0.001413 | 1.198921 |

Interpretation:

- HDR restore is necessary and should stay in the pipeline. It restores bokeh
  and top-hair highlight intensity from the original EXR instead of allowing
  display-space detail stages to clip it away.
- Shadow lifting is logically correct for feature detection: it exposes dark
  hair lines that are visible to the eye after lifting. However, applying it to
  the current texture-floor source only gives a small improvement. Stronger
  settings begin to trade detail for visible luma grain.
- This supports the previous ceiling diagnosis: direct reference texture grafts
  can recover some strand-like lines, but they cannot reliably reconstruct
  clean hair detail once the denoised base has flattened it.

Decision:

- Promote HDR restoration as a required post-step for display-space experiments.
- Keep v25/v26b as useful quality references for HDR-safe output.
- Do not promote v27a/v27b as the new default; they are detail-priority probes.
- Next serious improvement should use shadow-lifted/OkLab/orientation masks as
  conditioning for a learned residual or a better synthesized detail source,
  not simply stronger texture grafting.

### Stage13 v28: hair-detail acceptance probe

The user asked to continue until Occi hair detail is acceptable. The prior
oriented texture floor and shadow-lift detail probes were too weak: they kept
noise controlled, but did not recover the readable hair bundles. The next probe
therefore returned to stronger luma-structure grafting, then cleaned only the
areas where the graft visibly over-restores texture.

Added:

```text
scripts/apply_oriented_hair_luma_rebuild.py
scripts/apply_hair_region_luma_blend.py
scripts/apply_pl_luma_detail_probe.py
```

Key candidates:

```text
runs/refiner_pilot_stage11_hybrid_best/structure_graft_hdr_hair_v2_occi/
runs/refiner_pilot_stage11_hybrid_best/graft_hair_region_cleanup_v1_hdr_occi/
runs/refiner_pilot_stage11_hybrid_best/graft_hair_region_cleanup_v3_detail_hdr_occi/
```

Visual acceptance crops:

```text
runs/refiner_pilot_stage11_hybrid_best/hair_detail_acceptance_final_probe/
runs/refiner_pilot_stage11_hybrid_best/hair_detail_acceptance_final_probe_v2/
```

Occi whole-image evaluation:

| candidate | luma p99 | chroma p99 | magenta p99 | shadow magenta p99 | edge | display luma MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v25 HDR | 0.013200 | 0.006600 | 0.004034 | 0.001406 | 1.196200 | 0.006007 |
| clean1 HDR | 0.013491 | 0.007927 | 0.005172 | 0.001983 | 0.768219 | 0.004178 |
| clean3 detail HDR | 0.017026 | 0.007940 | 0.005179 | 0.002655 | 0.774179 | 0.003212 |

Interpretation:

- The direction-filtered/shadow-lifted hair rebuild was too conservative. It
  improved hair only slightly and never reached the visual threshold.
- Strong reference luma grafting recovers the hair bundles much better. The
  cost is that it can also restore background and skin luma texture.
- `clean1_hdr` is the best current balance: it keeps most of the visible hair
  bundle recovery from `graft_v2`, restores HDR highlights afterward, and avoids
  the large luma-tail regression of the stronger `clean3` candidate.
- `clean3_detail_hdr` is useful as a detail-priority reference, but not a safe
  default because luma p99 almost returns to the noisy input level.
- PL-derived detail transfer was informative but not a good default: fine-only
  transfer was too weak, while mid-frequency transfer introduced artificial
  large-scale streaks. PL remains a visual reference, not a teacher.

Decision:

- Promote `graft_hair_region_cleanup_v1_hdr_occi` as the current Occi
  hair-detail acceptance candidate.
- Keep `graft_hair_region_cleanup_v3_detail_hdr_occi` only as a
  detail-priority stress reference.
- The next production step is to make the strong luma-graft + selective cleanup
  region selector less sample-specific, because the current balance depends on
  real-photo texture masks rather than a learned semantic hair/detail mask.

### Stage13 v29: synthetic hair rejection and selective PL luma detail

The user pointed out that fully flattened hair cannot be truly recovered from
the source. Two probes tested the practical alternatives:

1. Generate plausible orientation-aligned luma strands from random sparse noise.
2. Borrow only local PL luma detail under a conservative hair/detail gate.

Added:

```text
scripts/apply_synthetic_hair_texture.py
scripts/apply_selective_pl_hair_detail.py
```

Synthetic probes:

```text
runs/refiner_pilot_stage11_hybrid_best/synthetic_hair_texture_v6_sparse_occi/
runs/refiner_pilot_stage11_hybrid_best/synthetic_hair_texture_v7_sparse_strong_occi/
```

Synthetic result:

| candidate | flat luma ratio | luma p99 ratio | magenta p99 ratio | shadow magenta p99 ratio |
| --- | ---: | ---: | ---: | ---: |
| clean1 HDR | 0.544 | 0.783 | 0.169 | 0.053 |
| synth v6 sparse | 0.574 | 0.862 | 0.170 | 0.079 |
| synth v7 sparse strong | 0.614 | 1.057 | 0.173 | 0.116 |

Interpretation:

- Random strand synthesis is not safe. Weak settings are barely visible, while
  strong settings create fake background/hair texture and make the luma tail
  worse than the input in the strongest probe.
- The failure mode is logical: direction can be estimated, but random line
  texture has no evidence that it belongs to actual hair.

Selective PL luma-detail probes:

```text
runs/refiner_pilot_stage11_hybrid_best/selective_pl_hair_detail_v1_occi/
runs/refiner_pilot_stage11_hybrid_best/selective_pl_hair_detail_v2_occi/
runs/refiner_pilot_stage11_hybrid_best/selective_pl_hair_detail_v3_prox_occi/
runs/refiner_pilot_stage11_hybrid_best/selective_pl_hair_detail_v4_tight_prox_occi/
```

The final v4 setting locally matches PL low-frequency luma to the HDR-safe base,
then transfers only signed mid/fine luma detail where four gates agree:

- current hair/detail mask
- PL coherent or textured structure
- PL energy exceeds base energy
- base currently has low local detail

Occi v4 evaluation:

| candidate | flat luma ratio | luma p99 ratio | luma visible ratio | chroma p99 ratio | magenta p99 ratio | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean1 HDR | 0.544 | 0.783 | 0.535 | 0.206 | 0.169 | 0.768 |
| selective PL v4 | 0.539 | 0.780 | 0.530 | 0.206 | 0.169 | 0.766 |

Visual result:

- v4 restores some readable hair bundles in the bangs and face-hair crops.
- It is far less artificial than random synthesis.
- It is safer than the stronger v2 probe, which introduced more PL-derived
  mottled structure near bright top-hair/background areas.

Decision:

- Reject synthetic random hair texture as a default direction.
- Keep `selective_pl_hair_detail_v4_tight_prox_occi` as the best current
  detail-recovery probe after `clean1_hdr`.
- PL must remain a bounded detail reference, not an absolute teacher. This
  route is promising only when luma detail is local, signed, chroma-free, and
  gated by both base weakness and PL structure evidence.
- Next step: make the hair/detail gate semantic enough to avoid background
  leakage, then convert the expensive full-frame Gaussian implementation into a
  tiled/ROI implementation if the visual direction survives more samples.

### Stage13 v30: SCUNet70 as an RGB reconstruction candidate

The user provided a manually generated Occi result:

```text
/Users/uniuyuni/PythonProjects/test_photos/X-T5 Occi SCUNet.EXR
```

It is SCUNet blended 70% with the original image. 100% SCUNet damaged
highlights, but the 70% blend visually recovered hair and local tone much more
convincingly than the current hand-built pipeline.

Added:

```text
scripts/make_crop_compare.py
scripts/apply_region_rgb_blend.py
```

Comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet70_occi_compare/
runs/refiner_pilot_stage11_hybrid_best/scunet_rgb_region_blend_compare/
runs/refiner_pilot_stage11_hybrid_best/scunet_rgb_global_blend_compare/
```

Key outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_luma_hair_blend_v1_occi/
runs/refiner_pilot_stage11_hybrid_best/scunet_rgb_region_blend_v1_occi/
runs/refiner_pilot_stage11_hybrid_best/scunet_rgb_region_blend_v2_occi/
runs/refiner_pilot_stage11_hybrid_best/scunet_rgb_global_blend_v1_hdr_occi/
```

SCUNet statistics before HDR-safe blending:

| image | rgb max | peak > 4 frac | luma p99 | peak > 1 frac |
| --- | ---: | ---: | ---: | ---: |
| noisy input | 5.9575 | 0.031309 | 3.9698 | 0.106875 |
| SCUNet70 | 4.7973 | 0.005514 | 2.6902 | 0.081155 |
| current v4 | 5.8824 | 0.031180 | 3.9697 | 0.103560 |

SCUNet70 conclusion:

- Visually, SCUNet70 recovers hair bundle flow much better than current v4.
- It also improves flat luma noise and edge-like structure.
- However, it reduces HDR peaks heavily and leaves more chroma/magenta residual
  than current v4.

The best probe was therefore a global display-RGB blend toward SCUNet70, with
HDR peaks restored from current v4:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_rgb_global_blend_v1_hdr_occi/xt5_occi_scunet_rgb_global_blend_v1_hdr.exr
```

Occi whole-image evaluation:

| candidate | flat luma ratio | luma p99 ratio | luma visible ratio | chroma p99 ratio | magenta p99 ratio | edge | highlight luma delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current v4 | 0.539 | 0.780 | 0.530 | 0.206 | 0.169 | 0.766 | -0.000191 |
| SCUNet70 | 0.441 | 0.801 | 0.429 | 0.317 | 0.313 | 1.076 | -1.313147 |
| SCUNet global HDR | 0.460 | 0.735 | 0.449 | 0.252 | 0.245 | 0.896 | -0.000191 |

Interpretation:

- This is the first probe that clearly improves hair readability while also
  improving luma tail metrics relative to current v4.
- HDR-safe global blending fixes SCUNet70's highlight damage.
- Chroma/magenta remains worse than current v4. This suggests the next best
  design is not "use SCUNet output", but "use SCUNet's RGB/luma reconstruction
  for structure and local tone, then reapply the stronger current chroma guard".

Decision:

- Promote `scunet_rgb_global_blend_v1_hdr_occi` as the current visual/detail
  breakthrough candidate.
- Do not adopt raw SCUNet70 as-is because of HDR peak loss and chroma residuals.
- Next step: split SCUNet guidance into luma/local-tone and chroma components,
  keep current v4 chroma suppression, and evaluate whether this preserves the
  SCUNet hair gain while recovering magenta/chroma metrics.

### Stage13 v31: added SCUNet samples and generic HDR restore

The user added more SCUNet outputs:

```text
/Users/uniuyuni/PythonProjects/test_photos/K-5 Dance SCUNet.EXR
/Users/uniuyuni/PythonProjects/test_photos/K-5 Ice SCUNet.EXR
/Users/uniuyuni/PythonProjects/test_photos/X-T5 Cat2 SCUNet.EXR
/Users/uniuyuni/PythonProjects/test_photos/X-T5 Cat SCUNet.EXR
/Users/uniuyuni/PythonProjects/test_photos/X-T5 Room SCUNet.EXR
```

`X-T5 Cat SCUNet.EXR` is full resolution, while the existing `X-T5 Cat noisy.EXR`
and current v25 base are the older crop. It was therefore not mixed into the
current Cat crop. The full-resolution aligned samples are Cat2, Room, Dance,
Ice, and Occi.

Added a generic input mode to `scripts/apply_hdr_highlight_restore.py`:

```text
--reference-file ORIGINAL_HDR_EXR
--candidate-file CANDIDATE_EXR_OR_TIFF
--name OUTPUT_STEM
```

This keeps the existing registered-scene behavior, but allows arbitrary SCUNet
candidate images to receive HDR peaks and highlight luma from the original EXR.

Dance outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_rgb_global_blend_v1_hdr_k5_dance/
runs/refiner_pilot_stage11_hybrid_best/scunet_global_hdr_chroma_cleanup_v1_k5_dance/
runs/refiner_pilot_stage11_hybrid_best/scunet_global_hdr_chroma_cleanup_compare/k5_dance/
runs/refiner_pilot_stage11_hybrid_best/eval_scunet_global_hdr_k5_dance/
```

Dance evaluation:

| candidate | flat luma ratio | luma p99 ratio | chroma p99 ratio | magenta p99 ratio | edge | highlight luma delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.260 | 0.616 | 0.078 | 0.064 | 1.113 | -0.009408 |
| SCUNet | 0.328 | 0.473 | 0.270 | 0.307 | 0.658 | -0.577916 |
| global HDR | 0.301 | 0.487 | 0.195 | 0.223 | 0.733 | -0.011483 |
| chroma clean | 0.305 | 0.485 | 0.145 | 0.149 | 0.732 | -0.013230 |

Dance interpretation:

- SCUNet is visually much smoother in sky/flat areas and reduces luma tail
  noise, but it softens edge HF and weakens HDR.
- Global HDR blending restores highlights.
- The signed chroma cleanup recovers much of the chroma/magenta regression
  without visibly breaking the SCUNet smoothness.

Ice outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_rgb_global_blend_v1_hdr_k5_ice/
runs/refiner_pilot_stage11_hybrid_best/scunet_global_hdr_chroma_cleanup_v1_k5_ice/
runs/refiner_pilot_stage11_hybrid_best/scunet_global_hdr_chroma_cleanup_compare/k5_ice/
runs/refiner_pilot_stage11_hybrid_best/eval_scunet_global_hdr_k5_ice/
```

Ice evaluation:

| candidate | flat luma ratio | luma p99 ratio | chroma p99 ratio | magenta p99 ratio | edge | highlight luma delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.261 | 0.568 | 0.165 | 0.120 | 0.393 | -0.010734 |
| SCUNet | 0.446 | 0.840 | 0.364 | 0.328 | 0.695 | -0.698448 |
| global HDR | 0.374 | 0.699 | 0.272 | 0.241 | 0.580 | -0.010734 |
| chroma clean | 0.373 | 0.696 | 0.233 | 0.188 | 0.579 | -0.010734 |

Ice interpretation:

- Ice is harder than Dance. SCUNet restores structure and improves visual
  coherence, but retains obvious blue/purple speckles.
- Chroma cleanup helps, but does not reach the current v25 chroma/magenta
  suppression.
- This sample argues for a learned or gated SCUNet selector, not unconditional
  adoption.

Ice v2 chroma cleanup:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_global_hdr_chroma_cleanup_v2_k5_ice/
runs/refiner_pilot_stage11_hybrid_best/scunet_global_hdr_chroma_cleanup_compare/k5_ice_v2/
runs/refiner_pilot_stage11_hybrid_best/eval_scunet_chroma_cleanup_v2_k5_ice/
```

The v2 probe lowered the signed outlier threshold and weighted blue/magenta more
strongly after SCUNet global HDR blending:

| candidate | flat chroma ratio | chroma p99 ratio | chroma visible ratio | magenta p99 ratio | shadow magenta visible ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 0.089 | 0.165 | 0.074 | 0.120 | 0.035 |
| global HDR | 0.240 | 0.272 | 0.234 | 0.241 | 0.225 |
| v1 | 0.205 | 0.233 | 0.196 | 0.188 | 0.173 |
| v2 | 0.177 | 0.213 | 0.168 | 0.168 | 0.143 |

v2 is visually safe in the inspected Ice crops and is better than v1, but still
does not match current v25's color-noise suppression. Keep it as the stronger
minimum guard when SCUNet reconstruction is used in blue-shadow scenes.

Room generic HDR restore:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_hdr_restore_v1_xt5_room/
runs/refiner_pilot_stage11_hybrid_best/scunet_hdr_restore_compare/xt5_room/
runs/refiner_pilot_stage11_hybrid_best/eval_scunet_hdr_restore_xt5_room/
```

Room evaluation:

| candidate | flat luma ratio | chroma p99 ratio | magenta p99 ratio | edge | highlight luma delta | rgb max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SCUNet | 0.494 | 0.367 | 0.343 | 0.960 | -1.202024 | 4.154 |
| HDR restore | 0.527 | 0.379 | 0.358 | 0.940 | -0.007281 | 5.523 |

Room interpretation:

- Raw SCUNet loses the curtain/window highlight structure badly.
- Generic HDR restore recovers the curtain weave and highlight luma while
  retaining most of SCUNet's flat/chroma cleanup.

Cat2 generic HDR restore:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_hdr_restore_v1_xt5_cat2/
runs/refiner_pilot_stage11_hybrid_best/scunet_hdr_restore_compare/xt5_cat2/
runs/refiner_pilot_stage11_hybrid_best/eval_scunet_hdr_restore_xt5_cat2/
```

Cat2 evaluation:

| candidate | flat luma ratio | chroma p99 ratio | magenta p99 ratio | edge | highlight luma delta | rgb max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SCUNet | 0.367 | 0.274 | 0.299 | 0.764 | -0.088128 | 2.248 |
| HDR restore | 0.439 | 0.283 | 0.316 | 0.772 | -0.004176 | 3.678 |

Cat2 interpretation:

- SCUNet is already a strong RGB reconstruction candidate on fur/whiskers.
- HDR restore slightly weakens luma cleanup but protects bright fur/whisker
  peaks and makes highlight drift acceptable.

Current decision:

- The best new direction is **SCUNet-derived RGB/luma reconstruction plus
  original-HDR restoration plus Nagi chroma cleanup**.
- SCUNet should be treated as a reconstruction teacher/candidate, not as the
  final output. Its luma/structure is often better than the current hand-built
  result, but its chroma tails and HDR behavior are not production-safe.
- Dance and Room strongly support this route. Ice is the main counterexample:
  it needs a stronger blue/magenta speckle guard or a selector that avoids
  trusting SCUNet in problematic blue-shadow regions.

### Stage13 v32 - Learned SCUNet selector pilot

Goal:

- Stop hand-picking a fixed SCUNet blend ratio.
- Learn a local selector that chooses where SCUNet-derived reconstruction is
  useful, while keeping the current Nagi-safe output in risky chroma/HDR
  regions.
- Treat SCUNet as a reconstruction candidate, not as an absolute teacher.

New script:

```text
scripts/train_scunet_selector.py
```

The selector predicts one gate per pixel from local features built from:

- noisy input
- current safe output
- SCUNet reconstruction candidate
- luma/chroma residuals
- local high-frequency cleanup benefit
- structure/coherence cues
- chroma outlier risk
- HDR risk

The output is a display-space blend:

```text
out = current * (1 - gate) + scunet * gate
```

then HDR peaks are restored from the current safe output. The pseudo-target is a
conditional gate rather than a final image target. It increases where SCUNet
reduces luma/chroma HF noise with coherent structure and decreases where SCUNet
has color mismatch, HDR risk, or blue/magenta outlier risk.

Training scenes:

| scene | current | SCUNet candidate |
| --- | --- | --- |
| xt5_occi | selective PL hair detail v4 tight prox | SCUNet global HDR chroma cleanup v1 |
| k5_dance | signed chroma outlier v25 HDR restore | SCUNet global HDR chroma cleanup v1 |
| k5_ice | signed chroma outlier v25 HDR restore | SCUNet global HDR chroma cleanup v2 |

Cat2 and Room were intentionally kept out of the training set because no aligned
current-safe base exists yet; keep them for validation/extrapolation after the
selector recipe is stable.

Training command:

```bash
pixi run python scripts/train_scunet_selector.py train \
  --output-dir runs/refiner_pilot_stage11_hybrid_best/scunet_selector_pilot_v1 \
  --steps 600 \
  --batch-size 3 \
  --patch-size 192 \
  --context 64 \
  --stats-samples 12 \
  --device cpu \
  --log-every 50 \
  --save-every 300
```

The run reached about step 450 before interruption; the usable saved checkpoint
is:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_selector_pilot_v1/scunet_selector_step_000300.pt
```

Training trace:

| step | loss | pred mean | target mean |
| ---: | ---: | ---: | ---: |
| 1 | 0.303693 | 0.5261 | 0.2372 |
| 50 | 0.102238 | 0.3209 | 0.2868 |
| 100 | 0.088745 | 0.1684 | 0.2027 |
| 150 | 0.067679 | 0.2643 | 0.2131 |
| 200 | 0.058438 | 0.2485 | 0.2268 |
| 250 | 0.050815 | 0.2425 | 0.1923 |
| 300 | 0.028111 | 0.2285 | 0.2156 |
| 350 | 0.060064 | - | - |
| 400 | 0.060169 | - | - |
| 450 | 0.017631 | 0.1848 | 0.1686 |

Scene target statistics:

| scene | target mean | target p95 |
| --- | ---: | ---: |
| xt5_occi | 0.2953 | 0.4091 |
| k5_dance | 0.1909 | 0.2718 |
| k5_ice | 0.2067 | 0.2906 |

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_selector_pilot_v1_outputs/
runs/refiner_pilot_stage11_hybrid_best/scunet_selector_pilot_v1_compare/
```

Gate statistics at strength 1.0:

| scene | gate mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| xt5_occi | 0.2388 | 0.2327 | 0.2860 | 0.3145 |
| k5_dance | 0.2231 | 0.2197 | 0.2463 | 0.2919 |
| k5_ice | 0.2363 | 0.2313 | 0.2726 | 0.3093 |

Key metrics, strength 1.0:

| scene | candidate | flat luma | chroma p99 | luma p99 | chroma visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | current | 0.260 | 0.078 | 0.616 | 0.052 | 1.113 |
| Dance | SCUNet | 0.305 | 0.145 | 0.485 | 0.167 | 0.732 |
| Dance | selector | 0.265 | 0.080 | 0.572 | 0.071 | 1.003 |
| Ice | current | 0.261 | 0.165 | 0.568 | 0.074 | 0.393 |
| Ice | SCUNet | 0.374 | 0.213 | 0.695 | 0.168 | 0.578 |
| Ice | selector | 0.269 | 0.164 | 0.574 | 0.080 | 0.435 |
| Occi | current | 0.539 | 0.206 | - | - | 0.766 |
| Occi | SCUNet | 0.456 | 0.224 | - | - | 0.896 |
| Occi | selector | 0.514 | 0.203 | - | - | 0.790 |

Occi additional metrics:

| candidate | flat chroma | magenta p99 | luma visible |
| --- | ---: | ---: | ---: |
| current | 0.098 | 0.169 | 0.530 |
| SCUNet | 0.217 | 0.195 | 0.445 |
| selector | 0.115 | 0.167 | 0.505 |

Strength 1.5 probe:

| scene | candidate | flat luma | chroma p99 | luma p99 | chroma visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | selector 1.5 | 0.269 | 0.083 | 0.552 | 0.083 | 0.948 |
| Ice | selector 1.5 | 0.278 | 0.166 | 0.583 | 0.089 | 0.459 |

Interpretation:

- The learned selector is real: gate maps are structured and respond to edges,
  coherent detail, hair, bokeh borders, and risky regions. It is not a uniform
  fixed blend.
- v1 is production-safe but conservative. It moves toward SCUNet enough to
  improve luma/detail tails, while keeping chroma/magenta close to the current
  safe output.
- Occi benefits in the intended direction: luma visible noise drops and edge
  score rises without inheriting SCUNet's chroma penalty.
- Dance benefits modestly in luma p99 while preserving most of the current
  chroma behavior.
- Ice confirms the risk model is needed: strength 1.5 already starts to take
  too much of the SCUNet/blue-shadow failure mode.

Decision:

- Keep strength 1.0 as the safe default.
- Use strength 1.5 only as a scene-specific aggressive sky/Dance probe, not as
  the default.
- Do not just train v1 longer. The next useful step is v2 target redesign:
  widen the target gate in coherent luma/detail areas, especially Occi hair and
  Dance sky, while explicitly tightening the blue/magenta outlier penalty on
  Ice-like shadow regions.
- The current pilot validates the architecture: learned SCUNet selection is
  better grounded than manual blend ratios and safer than unconditional SCUNet.

### Stage13 v33 - Luma-only SCUNet selector blend

Problem after v32:

- The learned RGB selector is safe, but visually too conservative.
- Increasing RGB blend strength risks importing SCUNet's chroma/magenta/blue
  failure modes.
- The useful SCUNet signal is mostly luma/structure, while the current Nagi
  output has safer chroma.

Changed `scripts/train_scunet_selector.py`:

```text
apply --blend-mode luma
apply --gate-gamma
apply --chroma-source-mix
apply --edge-inhibit
apply --edge-inhibit-mode {current,detail_loss}
```

Luma mode keeps the same learned gate, but blends only display luma toward the
SCUNet candidate and keeps chroma from the current safe output:

```text
out_y = current_y * (1 - blend) + scunet_y * blend
out_chroma = current_chroma
```

HDR is restored afterward from the current safe output.

Occi probes:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_selector_pilot_v2_luma_outputs/
xt5_occi_scunet_selector_v2_luma_s180_g085.exr
xt5_occi_scunet_selector_v2_luma_s240_g075.exr
xt5_occi_scunet_selector_v2_luma_s240_g075_cmix006.exr
```

Occi evaluation:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | magenta p99 | shadow magenta visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.539 | 0.780 | 0.530 | 0.206 | 0.084 | 0.169 | 0.045 | 0.766 |
| RGB v2 | 0.515 | 0.754 | 0.507 | 0.202 | 0.098 | 0.167 | 0.049 | 0.795 |
| luma180 | 0.491 | 0.738 | 0.483 | 0.206 | 0.083 | 0.169 | 0.040 | 0.849 |
| luma240 | 0.470 | 0.736 | 0.462 | 0.207 | 0.082 | 0.169 | 0.037 | 0.889 |
| cmix006 | 0.470 | 0.736 | 0.462 | 0.206 | 0.085 | 0.169 | 0.038 | 0.889 |
| SCUNet | 0.456 | 0.735 | 0.445 | 0.224 | 0.207 | 0.195 | 0.172 | 0.896 |

Occi interpretation:

- `luma240` is the best current Occi/portrait-detail candidate.
- It nearly reaches SCUNet edge/detail score while preserving current-safe
  chroma/magenta behavior.
- Tiny SCUNet chroma mix (`cmix006`) does not help enough; it slightly worsens
  chroma visible. Keep chroma mix at `0.0`.

Dance probes:

```text
k5_dance_scunet_selector_v2_luma_s240_g075.exr
k5_dance_scunet_selector_v2_luma_s240_g075_edge070.exr
k5_dance_scunet_selector_v2_luma_s240_g075_edgeloss080.exr
```

Dance evaluation:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | magenta p99 | shadow magenta visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.260 | 0.616 | 0.247 | 0.078 | 0.052 | 0.064 | 0.035 | 1.113 |
| RGB v2 | 0.263 | 0.564 | 0.250 | 0.079 | 0.066 | 0.070 | 0.046 | 0.998 |
| luma240 | 0.281 | 0.489 | 0.271 | 0.079 | 0.052 | 0.064 | 0.035 | 0.765 |
| edge070 | 0.279 | 0.518 | 0.268 | 0.079 | 0.052 | 0.064 | 0.035 | 1.011 |
| edgeloss080 | 0.281 | 0.506 | 0.270 | 0.079 | 0.052 | 0.064 | 0.035 | 0.998 |
| SCUNet | 0.305 | 0.485 | 0.299 | 0.145 | 0.167 | 0.149 | 0.153 | 0.732 |

Dance interpretation:

- Plain luma240 smooths the sky well, but makes the subject/fine lines too
  sleepy.
- Edge inhibition fixes that. `edgeloss080` is the best current Dance balance:
  much lower luma p99 than RGB v2, current-safe chroma, and acceptable edge
  retention.
- The `detail_loss` mode is more logical than generic edge protection because
  it protects only where SCUNet weakens current detail.

Ice probe:

```text
k5_ice_scunet_selector_v2_luma_s140_g090_edgeloss080.exr
```

Ice evaluation:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | magenta p99 | shadow magenta visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.261 | 0.568 | 0.245 | 0.165 | 0.074 | 0.120 | 0.035 | 0.393 |
| RGB v2 | 0.267 | 0.575 | 0.250 | 0.163 | 0.076 | 0.119 | 0.036 | 0.433 |
| luma140 | 0.274 | 0.586 | 0.256 | 0.165 | 0.074 | 0.120 | 0.034 | 0.452 |
| luma240 | 0.318 | 0.647 | 0.298 | 0.166 | 0.074 | 0.120 | 0.034 | 0.543 |
| SCUNet | 0.374 | 0.695 | 0.361 | 0.213 | 0.168 | 0.168 | 0.143 | 0.578 |

Ice interpretation:

- Ice is still the counterexample. Luma blending restores structure, but also
  moves too far toward SCUNet's residual luma noise.
- Keep RGB v2 as the current safe Ice candidate.
- Do not make luma240 a global default.

Current best per scene:

| scene type | recommended candidate |
| --- | --- |
| Occi / hair / portrait detail | `luma_s240_g075`, chroma mix `0.0`, edge inhibit off |
| Dance / sky plus subject detail | `luma_s240_g075_edgeloss080` |
| Ice / blue-shadow risk | RGB selector v2 |

Decision:

- This is a real improvement over v32. The key breakthrough is separating the
  learned SCUNet gate from the component being borrowed.
- Luma-only blending gives most of SCUNet's structure gain without importing its
  chroma failures.
- The next production design should turn these scene-specific settings into a
  learned or rule-based per-region policy:
  `portrait/detail -> luma240`, `flat sky with detail loss -> luma240 + detail-loss edge inhibit`,
  `blue-shadow risk -> RGB v2 or lower luma strength`.

### Stage13 v34 - Luma-specific selector target

After v33, I tested whether the RGB selector target was holding back luma-only
blending. Added:

```text
--target-preset v3_luma
```

The v3 luma target:

- opens on flat luma cleanup benefit;
- opens on coherent SCUNet structure where the current output is sleepy;
- penalizes SCUNet luma HF returning in flat non-structure regions;
- keeps blue/magenta risk as a weaker suppressor than in RGB mode, because
  chroma is not borrowed in luma mode.

Training:

```bash
pixi run python scripts/train_scunet_selector.py train \
  --output-dir runs/refiner_pilot_stage11_hybrid_best/scunet_selector_pilot_v3_luma \
  --target-preset v3_luma \
  --steps 500 \
  --batch-size 3 \
  --patch-size 192 \
  --context 64 \
  --stats-samples 12 \
  --device cpu \
  --log-every 50 \
  --save-every 250
```

Training trace:

| step | loss | pred mean | target mean |
| ---: | ---: | ---: | ---: |
| 1 | 0.322929 | 0.5261 | 0.2319 |
| 50 | 0.151923 | 0.3272 | 0.2938 |
| 100 | 0.117441 | 0.1334 | 0.1681 |
| 150 | 0.097533 | 0.2174 | 0.1504 |
| 200 | 0.083538 | 0.2262 | 0.2118 |
| 250 | 0.092228 | 0.2280 | 0.1431 |
| 300 | 0.059519 | 0.2042 | 0.1520 |
| 350 | 0.121314 | 0.1950 | 0.3177 |
| 400 | 0.073599 | 0.2474 | 0.1656 |
| 450 | 0.027766 | 0.1286 | 0.0970 |
| 500 | 0.065822 | 0.1886 | 0.1807 |

Checkpoint:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_selector_pilot_v3_luma/scunet_selector_final.pt
```

Evaluation:

Dance:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | magenta p99 | shadow magenta visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.260 | 0.616 | 0.247 | 0.078 | 0.052 | 0.064 | 0.035 | 1.113 |
| v2 best | 0.281 | 0.506 | 0.270 | 0.079 | 0.052 | 0.064 | 0.035 | 0.998 |
| v3 luma | 0.275 | 0.505 | 0.264 | 0.079 | 0.052 | 0.064 | 0.035 | 0.992 |
| SCUNet | 0.305 | 0.485 | 0.299 | 0.145 | 0.167 | 0.149 | 0.153 | 0.732 |

Ice:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | magenta p99 | shadow magenta visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.261 | 0.568 | 0.245 | 0.165 | 0.074 | 0.120 | 0.035 | 0.393 |
| v2 RGB | 0.267 | 0.575 | 0.250 | 0.163 | 0.076 | 0.119 | 0.036 | 0.433 |
| v2 luma140 | 0.274 | 0.586 | 0.256 | 0.165 | 0.074 | 0.120 | 0.034 | 0.452 |
| v3 luma | 0.272 | 0.588 | 0.254 | 0.165 | 0.074 | 0.120 | 0.034 | 0.461 |
| SCUNet | 0.374 | 0.695 | 0.361 | 0.213 | 0.168 | 0.168 | 0.143 | 0.578 |

Occi:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | magenta p99 | shadow magenta visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.539 | 0.780 | 0.530 | 0.206 | 0.084 | 0.169 | 0.045 | 0.766 |
| v2 luma240 | 0.470 | 0.736 | 0.462 | 0.207 | 0.082 | 0.169 | 0.037 | 0.889 |
| v3 luma | 0.478 | 0.737 | 0.469 | 0.207 | 0.082 | 0.169 | 0.037 | 0.890 |
| SCUNet | 0.456 | 0.735 | 0.445 | 0.224 | 0.207 | 0.195 | 0.172 | 0.896 |

Decision:

- v3_luma is not a universal replacement.
- Use v3_luma for Dance-like sky/subject scenes, where it gives slightly better
  luma metrics than v2 with nearly the same edge retention.
- Keep v2_luma240 for Occi/hair because it removes more luma noise at the same
  edge level.
- Keep RGB v2 for Ice-like blue-shadow scenes when prioritizing noise safety;
  v3_luma gives more structure but still returns too much luma noise.
- The next real design step is a multi-output policy, not another scalar gate:
  one luma reconstruction gate, one edge-preservation/sleepiness gate, and one
  blue-shadow risk gate.

### Stage13 v35: multi-output SCUNet policy v1/v3 and hybrid rejection

Goal:

- Replace the scalar SCUNet selector with a three-output policy:
  `luma_gate`, `edge_keep`, and `risk_gate`.
- Let the model decide where SCUNet luma reconstruction is useful, while
  independently suppressing edge sleepiness and blue-shadow/purple-shadow
  failures.

Implementation:

- Added `scripts/train_scunet_policy.py`.
- `policy_v1` target used a direct learned luma gate plus edge/risk inhibitors.
- `policy_v3_balanced` made the target safer:
  - stronger edge protection,
  - stronger blue/purple shadow risk,
  - lower base luma in risky regions,
  - width 28 / blocks 5 / 800 CPU steps.
- Also tested a hybrid mode:
  scalar selector v2 provides the base luma gate, policy v3 provides only
  edge/risk inhibition.

Training:

```bash
pixi run python scripts/train_scunet_policy.py train \
  --output-dir runs/refiner_pilot_stage11_hybrid_best/scunet_policy_pilot_v3_balanced \
  --target-preset v3_balanced \
  --steps 800 \
  --batch-size 3 \
  --patch-size 192 \
  --context 64 \
  --width 28 \
  --blocks 5 \
  --luma-weight 1.10 \
  --edge-weight 1.05 \
  --risk-weight 1.05 \
  --mean-weight 0.10 \
  --stats-samples 12 \
  --device cpu \
  --log-every 50 \
  --save-every 400
```

Final trace:

| step | loss | pred mean | target mean |
| ---: | ---: | ---: | ---: |
| 400 | 0.042496 | `(0.0746, 0.1926, 0.2478)` | `(0.1165, 0.2059, 0.2293)` |
| 600 | 0.034260 | `(0.0650, 0.1862, 0.2506)` | `(0.0722, 0.1830, 0.2631)` |
| 800 | 0.030477 | `(0.1093, 0.2257, 0.2299)` | `(0.1012, 0.2250, 0.2461)` |

Evaluation:

Occi:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | shadow magenta visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.539 | 0.780 | 0.530 | 0.206 | 0.084 | 0.045 | 0.766 |
| v2_luma240 | 0.470 | 0.736 | 0.462 | 0.207 | 0.082 | 0.037 | 0.889 |
| policy_v1 | 0.506 | 0.745 | 0.497 | 0.206 | 0.083 | 0.041 | 0.844 |
| policy_v3 | 0.517 | 0.760 | 0.510 | 0.206 | 0.083 | 0.043 | 0.815 |
| SCUNet | 0.456 | 0.735 | 0.445 | 0.224 | 0.207 | 0.172 | 0.896 |

Dance:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.260 | 0.616 | 0.247 | 0.078 | 0.052 | 1.113 |
| v3_luma | 0.275 | 0.505 | 0.264 | 0.079 | 0.052 | 0.992 |
| policy_v1 | 0.262 | 0.525 | 0.251 | 0.079 | 0.052 | 0.869 |
| policy_v3 | 0.258 | 0.548 | 0.246 | 0.078 | 0.052 | 0.954 |
| hybrid | 0.273 | 0.520 | 0.261 | 0.078 | 0.052 | 0.905 |

Ice:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.261 | 0.568 | 0.245 | 0.165 | 0.074 | 0.393 |
| v2_rgb | 0.267 | 0.575 | 0.250 | 0.163 | 0.076 | 0.433 |
| policy_v1 | 0.280 | 0.607 | 0.261 | 0.166 | 0.074 | 0.494 |
| policy_v3 | 0.267 | 0.584 | 0.250 | 0.165 | 0.074 | 0.447 |
| hybrid | 0.290 | 0.601 | 0.270 | 0.165 | 0.074 | 0.470 |

Crop comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_scunet_policy_v3_occi/
runs/refiner_pilot_stage11_hybrid_best/compare_scunet_policy_v3_dance_ice/
```

Decision:

- Reject the multi-output policy as the primary luma selector. It becomes safe
  but under-applies SCUNet on Occi, and it still does not beat the previous
  scene-specific scalar selectors.
- Reject the scalar-selector + policy-inhibitor hybrid. It improves Dance p99
  over policy_v3, but drops edge retention and makes Ice luma noise worse.
- Keep the previous per-scene bests as the current quality baseline:
  - Occi: `v2_luma240`
  - Dance: `v3_luma`
  - Ice: `v2_rgb` or `policy_v3` depending on whether edge recovery or luma
    safety is preferred.
- Next direction: do not ask a tiny policy to infer the whole luma decision.
  The selector needs an explicit "SCUNet candidate is trustworthy here" signal
  with a much stronger penalty for luma tail return in blue-shadow scenes, or a
  separate image/region-level preset chooser before local blending.

### Stage13 v36: selector trust target and direct luma-tail guard

Goal:

- Keep the useful scalar selector behavior from v2/v3.
- Suppress the Ice-like failure where SCUNet restores edge/detail but also
  returns visible luma noise in blue-shadow regions.

Experiments:

1. Added `target_preset=v4_trust` to `scripts/train_scunet_selector.py`.
   This target adds a blue/purple shadow luma-tail penalty.
2. Trained:

```bash
pixi run python scripts/train_scunet_selector.py train \
  --output-dir runs/refiner_pilot_stage11_hybrid_best/scunet_selector_pilot_v4_trust \
  --target-preset v4_trust \
  --steps 500 \
  --batch-size 3 \
  --patch-size 192 \
  --context 64 \
  --width 20 \
  --blocks 4 \
  --stats-samples 12 \
  --device cpu \
  --log-every 50 \
  --save-every 250
```

3. Added apply-time `--tail-inhibit`, which directly reduces luma blending when
   SCUNet has more luma HF than current inside signed blue/magenta shadow risk.

Evaluation:

v4_trust:

| scene | candidate | flat luma | luma p99 | luma visible | edge | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Occi | v2_luma240 | 0.470 | 0.736 | 0.462 | 0.889 | baseline |
| Occi | v4_trust | 0.495 | 0.751 | 0.488 | 0.878 | worse |
| Dance | v3_luma | 0.275 | 0.505 | 0.264 | 0.992 | baseline |
| Dance | v4_trust | 0.266 | 0.516 | 0.254 | 1.020 | mixed |
| Ice | policy_v3 | 0.267 | 0.584 | 0.250 | 0.447 | safest tested SCUNet-derived output |
| Ice | v4_trust | 0.287 | 0.621 | 0.267 | 0.492 | worse |

Tail guard on Ice with the same strength as the previous v3_luma Ice output:

| candidate | flat luma | luma p99 | luma visible | edge |
| --- | ---: | ---: | ---: | ---: |
| current | 0.261 | 0.568 | 0.245 | 0.393 |
| v3_luma s140/g090 | 0.272 | 0.588 | 0.254 | 0.461 |
| tailguard060 s140/g090 | 0.272 | 0.586 | 0.254 | 0.450 |
| policy_v3 | 0.267 | 0.584 | 0.250 | 0.447 |

Decision:

- Reject `v4_trust` as a replacement selector. It improves Dance edge retention
  but does not solve Ice and loses to v2_luma240 on Occi.
- Keep `--tail-inhibit` in the tool because it is harmlessly optional and gives
  a small Ice p99 reduction under equal strength, but it is not enough to be the
  main answer.
- Current best practical policy is still per-region/per-scene:
  - Occi/hair: v2_luma240.
  - Dance/sky: v3_luma.
  - Ice/blue shadow: policy_v3 or current/v2_rgb for safety.
- Next serious step should be an explicit preset chooser:
  first classify candidate trust at image/region scale, then use the local
  blend. Local per-pixel gate alone keeps confusing "SCUNet detail recovery" with
  "SCUNet noise return" on Ice.

### Stage13 v37: coarse SCUNet preset chooser v1

Goal:

- Stop asking a per-pixel gate to solve image-level ambiguity.
- First classify the SCUNet candidate into one of a few coarse presets, then run
  the local blend that is already known to work for that class.

Implementation:

- Added `scripts/apply_scunet_preset_chooser.py`.
- It computes downsampled scene metrics:
  - signed blue/magenta risk,
  - shadow-weighted luma-tail risk,
  - blue-shadow structure risk,
  - coherent/detail texture,
  - current vs SCUNet luma HF.
- It chooses:
  - `hair_luma`: selector v2 luma, strength 2.4, gamma 0.75.
  - `sky_luma`: selector v3 luma, strength 2.4, gamma 0.75, detail-loss edge
    inhibit.
  - `blue_shadow_safe`: policy v3 balanced.

Classifier thresholds:

```text
blue_shadow_safe if blue_struct_mean >= 0.13 or blue_tail_p95 >= 0.32
hair_luma        if current_texture_mean >= 0.50 and signed_mean <= 0.26
sky_luma         otherwise
```

Observed metrics:

| scene | preset | signed mean | blue tail p95 | blue struct mean | current texture | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Occi | hair_luma | 0.175 | 0.142 | 0.077 | 0.624 | high texture, low signed risk |
| Dance | sky_luma | 0.362 | 0.251 | 0.072 | 0.265 | default sky cleanup |
| Ice | blue_shadow_safe | 0.385 | 0.362 | 0.192 | 0.616 | blue-shadow structure/tail risk |

Evaluation:

Occi:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.539 | 0.780 | 0.530 | 0.206 | 0.084 | 0.766 |
| v2_luma240 | 0.470 | 0.736 | 0.462 | 0.207 | 0.082 | 0.889 |
| chooser_v1b | 0.470 | 0.736 | 0.462 | 0.207 | 0.082 | 0.889 |

Dance:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.260 | 0.616 | 0.247 | 0.078 | 0.052 | 1.113 |
| v3_luma | 0.275 | 0.505 | 0.264 | 0.079 | 0.052 | 0.992 |
| chooser | 0.275 | 0.503 | 0.264 | 0.079 | 0.052 | 0.992 |

Ice:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.261 | 0.568 | 0.245 | 0.165 | 0.074 | 0.393 |
| v2_rgb | 0.267 | 0.575 | 0.250 | 0.163 | 0.076 | 0.433 |
| policy_v3 | 0.267 | 0.584 | 0.250 | 0.165 | 0.074 | 0.447 |
| chooser | 0.267 | 0.584 | 0.250 | 0.165 | 0.074 | 0.447 |

Decision:

- Keep preset chooser v1. It does not improve the individual best outputs, but
  it correctly selects the best known family for Occi, Dance, and Ice.
- This is a useful architecture pivot: image/region-level trust first, local
  blend second.
- Remaining problem: the blue-shadow-safe preset is still only a compromise.
  It avoids the worst selector failures, but it does not beat current/v2_rgb on
  luma noise. The next improvement should target a better blue-shadow preset,
  not another universal selector.

### Stage13 v38: blue-shadow structure graft preset

Goal:

- Improve the `blue_shadow_safe` preset.
- Preserve Ice-like luma noise close to current/v2_rgb while restoring more
  structure than v2_rgb or policy_v3.

Design:

- Added `scripts/apply_blue_shadow_structure_graft.py`.
- It keeps current chroma and most current luma.
- It borrows only band-limited SCUNet-vs-current luma structure:
  - current/scunet display luma band: Gaussian 0.65 minus Gaussian 2.20,
  - gated by current/SCUNet edge strength,
  - boosted by coherent/texture evidence,
  - suppressed where SCUNet luma tail exceeds current in non-coherent regions,
  - clipped to a small display-luma correction.

Tested Ice variants:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.261 | 0.568 | 0.245 | 0.165 | 0.074 | 0.393 |
| v2_rgb | 0.267 | 0.575 | 0.250 | 0.163 | 0.076 | 0.433 |
| policy_v3 | 0.267 | 0.584 | 0.250 | 0.165 | 0.074 | 0.447 |
| graft_soft | 0.261 | 0.567 | 0.245 | 0.165 | 0.074 | 0.431 |
| graft_mid | 0.265 | 0.571 | 0.248 | 0.165 | 0.074 | 0.471 |
| graft_strong | 0.274 | 0.588 | 0.256 | 0.166 | 0.075 | 0.515 |

Decision:

- Use `graft_mid` for the chooser's `blue_shadow_safe` preset.
- `graft_soft` is extremely safe but barely beats v2_rgb edge restoration.
- `graft_strong` restores more edge but starts to return luma noise.
- `graft_mid` is the best practical tradeoff:
  - better luma p99 than v2_rgb and policy_v3,
  - better edge than v2_rgb and policy_v3,
  - no chroma penalty.

Chooser v2 Ice result:

| candidate | flat luma | luma p99 | luma visible | chroma p99 | chroma visible | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 0.261 | 0.568 | 0.245 | 0.165 | 0.074 | 0.393 |
| v2_rgb | 0.267 | 0.575 | 0.250 | 0.163 | 0.076 | 0.433 |
| policy_v3 | 0.267 | 0.584 | 0.250 | 0.165 | 0.074 | 0.447 |
| chooser_v2 | 0.265 | 0.571 | 0.248 | 0.165 | 0.074 | 0.471 |

Updated `scripts/apply_scunet_preset_chooser.py`:

- `hair_luma`: v2 selector luma without edge inhibit, matching v2_luma240.
- `sky_luma`: v3 selector luma.
- `blue_shadow_safe`: blue-shadow structure graft mid preset.

Crop comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_blue_shadow_structure_graft_v1/
```

Next:

- Re-test the updated chooser on more blue-shadow/noisy samples, especially
  Dance-like scenes with blue shadows and any newly added high-ISO samples.
- If stable, make the preset chooser the main SCUNet-path dispatcher.

### Stage13 v39: custom chooser diagnostics and region preset mixer

Goal:

- Check whether the coarse preset chooser generalizes to newly added Room/Cat2
  diagnostics.
- Test whether a region-aware preset mix can beat the image-level chooser by
  applying hair/sky/blue-shadow behavior in different parts of the same image.

Custom Room/Cat2 diagnostics:

- Extended `scripts/apply_scunet_preset_chooser.py` so it can accept explicit
  `--noisy`, `--current`, and `--scunet` paths.
- Ran it on Room/Cat2 using HDR-restore output as `current` and raw SCUNet as
  `scunet`. This is diagnostic only, not the normal pipeline pairing.

Results:

| scene | chosen preset | flat luma | luma p99 | visible luma | edge | note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Room hdr_restore | n/a | 0.527 | 0.830 | 0.500 | 0.940 | baseline |
| Room chooser | sky_luma | 0.512 | 0.834 | 0.486 | 0.960 | small safe luma cleanup |
| Cat2 hdr_restore | n/a | 0.439 | 0.690 | 0.432 | 0.772 | baseline |
| Cat2 chooser | sky_luma | 0.396 | 0.690 | 0.393 | 0.768 | useful luma cleanup, no chroma change |

Decision:

- The chooser did not break HDR restoration and gives mild luma cleanup on
  Room/Cat2.
- It is still not a breakthrough: chroma/tail behavior is mostly unchanged.

Region preset mixer v1:

- Added `scripts/apply_scunet_region_preset_mixer.py`.
- v1 ran all three presets (`sky_luma`, `hair_luma`, `blue_shadow_safe`) and
  blended them with full-resolution analytic weights.

Evaluation:

| scene | baseline best | candidate | flat luma | luma p99 | visible luma | edge | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Occi | v2_luma | region_mixer_v1 | 0.501 | 0.741 | 0.493 | 0.816 | reject; worse than v2_luma |
| Dance | v3_luma | region_mixer_v1 | 0.262 | 0.545 | 0.250 | 0.967 | mixed; worse p99/edge than v3_luma |
| Ice | chooser_v2 | region_mixer_v1 | 0.272 | 0.581 | 0.254 | 0.463 | reject; returns luma noise |

Region preset mixer v2:

- Changed the mixer to choose an image-level base preset first, then only apply
  high-confidence local overrides.
- Bases selected:
  - Occi: `hair_luma`.
  - Dance: `sky_luma`.
  - Ice: `blue_shadow_safe`.

Evaluation:

| scene | baseline best | candidate | flat luma | luma p99 | visible luma | edge | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Occi | v2_luma 0.470/0.736/0.462/0.889 | mixer_v2 | 0.471 | 0.735 | 0.463 | 0.883 | near-equal |
| Dance | v3_luma 0.275/0.505/0.264/0.992 | mixer_v2 | 0.274 | 0.502 | 0.263 | 0.955 | p99 slightly better, edge worse |
| Ice | chooser_v2 0.265/0.571/0.248/0.471 | mixer_v2 | 0.265 | 0.573 | 0.248 | 0.473 | near-equal |

Decision:

- Reject mixer v1.
- Keep mixer v2 only as a diagnostic architecture. It proves that image-level
  base selection is necessary, but the local override does not yet produce a
  reliable quality gain.
- Do not make region mixing the main path yet. For current best quality, the
  coarse chooser remains better because it is simpler and less likely to leak
  noise/detail mistakes across regions.
- Next work should focus on an evaluator/target for visible residual grain and
  speckle, not more broad preset mixing.

### Stage13 v40: residual speckle evaluator and luma-tail auto postfilter

Goal:

- Target the remaining visible grain/dot noise more directly.
- The previous real-photo evaluator catches average HF and p99 tails, but
  isolated dot-like luma speckles are better measured as median residual p99 /
  p999.

Implementation:

- Added `scripts/residual_speckle_eval.py`.
  - Uses display-space median residuals.
  - Reports flat luma/chroma impulse p99/p999 and visible-weighted impulse.
  - Also reports magenta/blue positive impulse tails in flat/shadow regions.
- Integrated optional luma-tail filtering into
  `scripts/apply_scunet_preset_chooser.py`:

```bash
--luma-tail-preset off|auto|mild|balanced|strong|xstrong
```

- `auto` policy:
  - `hair_luma` and `sky_luma`: `xstrong`
  - `blue_shadow_safe`: `strong`

Reason for the auto split:

- Occi/Dance benefited from `xstrong` with acceptable edge loss.
- Ice had better safety with `strong`: `xstrong` improved p99/p999 but worsened
  visible-luma ratio and reduced edge retention more.

Speckle results:

| scene | base | tail preset | luma p99 | luma p999 | luma visible | note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Occi | v2_luma | none | 0.470 | 0.648 | 0.333 | previous best |
| Occi | v2_luma | strong | 0.410 | 0.592 | 0.297 | safe improvement |
| Occi | v2_luma | xstrong/auto | 0.368 | 0.534 | 0.259 | best tested |
| Dance | v3_luma | none | 0.296 | 0.360 | 0.222 | previous best |
| Dance | v3_luma | strong | 0.256 | 0.322 | 0.201 | safe improvement |
| Dance | v3_luma | xstrong/auto | 0.236 | 0.307 | 0.190 | best tested |
| Ice | chooser_v2 | none | 0.304 | 0.364 | 0.169 | previous best |
| Ice | chooser_v2 | strong/auto | 0.270 | 0.338 | 0.151 | best balanced |
| Ice | chooser_v2 | xstrong | 0.252 | 0.328 | 0.138 | lower speckle, worse real-photo visible/edge |

Real-photo tradeoff:

| scene | base edge | auto edge | base luma visible | auto luma visible | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Occi | 0.889 | 0.885 | 0.462 | 0.406 | accept |
| Dance | 0.992 | 0.972 | 0.264 | 0.252 | likely accept; check visually |
| Ice | 0.471 | 0.468 | 0.248 | 0.251 | use strong, not xstrong |

Additional Room/Cat2 diagnostics:

| scene | chooser luma p99 | tail xstrong p99 | chooser visible | tail xstrong visible | note |
| --- | ---: | ---: | ---: | ---: | --- |
| Room | 0.519 | 0.418 | 0.401 | 0.311 | strong luma speckle reduction |
| Cat2 | 0.368 | 0.283 | 0.331 | 0.264 | strong luma speckle reduction |

Decision:

- Keep `residual_speckle_eval.py`; it better matches the "visible grain remains"
  complaint than average HF alone.
- Promote `apply_scunet_preset_chooser.py --luma-tail-preset auto` as the
  current quality-first candidate.
- This is still not "perfect": chroma speckle tails barely move, and Dance edge
  retention under xstrong needs visual confirmation. But it is the first clean
  improvement in the exact defect the user called out: residual luma grain/tail.

### Stage13 v41: chroma speckle postfilter on top of tail-auto

Goal:

- Reduce the remaining chroma/magenta/blue dot tail after the luma-tail auto
  candidate.
- Keep luma, edge retention, and HDR/highlight behavior unchanged.

Implementation:

- Reused `scripts/apply_chroma_speckle_filter.py`, which fixes display luma and
  only replaces chroma outliers in flat/non-highlight regions.
- Added `--chroma-speckle-preset` to
  `scripts/apply_scunet_preset_chooser.py`.
- `--chroma-speckle-preset auto` now maps to `axisplus`.

Quality preset probe:

| scene | candidate | chroma p999 | chroma visible | magenta p999 | shadow magenta p999 | blue p999 | shadow blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Occi | tail_auto | 0.246 | 0.076 | 0.224 | 0.065 | 0.231 | 0.081 |
| Occi | quality | 0.245 | 0.073 | 0.222 | 0.064 | 0.229 | 0.080 |
| Dance | tail_auto | 0.113 | 0.036 | 0.105 | 0.052 | 0.109 | 0.055 |
| Dance | quality | 0.110 | 0.034 | 0.103 | 0.048 | 0.106 | 0.049 |
| Ice | tail_auto | 0.242 | 0.083 | 0.186 | 0.073 | 0.198 | 0.091 |
| Ice | quality | 0.236 | 0.079 | 0.184 | 0.070 | 0.194 | 0.087 |

Decision:

- `quality` is safe but too small; it proves the filter direction is valid.

Axisplus probe:

| scene | candidate | chroma p999 | chroma visible | magenta p999 | shadow magenta p999 | blue p999 | shadow blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Occi | axisplus | 0.243 | 0.070 | 0.220 | 0.061 | 0.228 | 0.075 |
| Dance | axisplus | 0.107 | 0.030 | 0.100 | 0.029 | 0.102 | 0.031 |
| Ice | axisplus | 0.231 | 0.071 | 0.182 | 0.059 | 0.191 | 0.070 |

Real-photo check:

| scene | candidate | flat chroma | chroma p99 | chroma visible | edge | highlight drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Occi | tail_auto | 0.097 | 0.207 | 0.082 | 0.885 | 0.000007 |
| Occi | axisplus | 0.092 | 0.204 | 0.077 | 0.885 | 0.000007 |
| Ice | tail_auto | 0.090 | 0.165 | 0.074 | 0.468 | 0.005059 |
| Ice | axisplus | 0.083 | 0.158 | 0.068 | 0.468 | 0.005059 |

Visual crops:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_chroma_stage13_axis_probe_occi/
runs/refiner_pilot_stage11_hybrid_best/compare_chroma_stage13_axis_probe_dance/
runs/refiner_pilot_stage11_hybrid_best/compare_chroma_stage13_axis_probe_ice/
```

Decision:

- Promote chroma speckle `axisplus` as the current auto preset.
- The practical quality-first command is now:

```bash
pixi run python scripts/apply_scunet_preset_chooser.py \
  --scene <scene> \
  --luma-tail-preset auto \
  --chroma-speckle-preset auto
```

- This is still not the final "perfect" NR: chroma cleanup is incremental, and
  generated/reconstructed detail is still the hard part. But this is aligned
  with the current defect report: visible luma grain and colored speckle tails
  are both lower, with no measured edge or highlight regression in the checked
  scenes.

### Stage13 v42: dark-dot residual filter and v6 auto

Goal:

- Attack the remaining visually annoying dark purple/blue pin dots without
  globally blurring detail.
- Keep the existing luma-tail and chroma-axis cleanup, then add a signed
  dark-dot pass only where the pixel is darker than its local median.

Implementation:

- Added `scripts/apply_dark_dot_speckle_filter.py`.
- Added `--dark-dot-preset` to
  `scripts/apply_scunet_preset_chooser.py`.
- `--dark-dot-preset auto` now maps by coarse scene preset:
  - `hair_luma` / `sky_luma`: `sky`
  - `blue_shadow_safe`: `strong`

Reason for the split:

- `sky` gives the best luma speckle reduction on Occi/Dance with essentially
  unchanged edge retention in the checked crops.
- Ice/blue-shadow gets slightly better speckle p99 with `sky`, but real-photo
  luma visible is marginally better balanced with `strong`.

Dark-dot strength probe:

| scene | candidate | luma p99 | luma p999 | luma visible | chroma visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Occi | v4 tail+chroma | 0.368 | 0.535 | 0.259 | 0.070 | 0.885 |
| Occi | darkdot strong | 0.339 | 0.502 | 0.232 | 0.068 | 0.884 |
| Occi | darkdot sky | 0.332 | 0.495 | 0.220 | 0.066 | 0.884 |
| Dance | v4 tail+chroma | 0.236 | 0.308 | 0.190 | 0.030 | 0.972 |
| Dance | darkdot strong | 0.215 | 0.298 | 0.172 | 0.029 | 0.969 |
| Dance | darkdot sky | 0.210 | 0.295 | 0.164 | 0.029 | 0.968 |
| Ice | v4 tail+chroma | 0.270 | 0.338 | 0.151 | 0.071 | 0.468 |
| Ice | darkdot strong | 0.254 | 0.329 | 0.138 | 0.069 | 0.467 |
| Ice | darkdot sky | 0.251 | 0.327 | 0.133 | 0.067 | 0.466 |

Real-photo check:

| scene | candidate | flat luma | flat chroma | luma visible | edge | highlight drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Occi | v4 tail+chroma | 0.414 | 0.092 | 0.406 | 0.885 | 0.000007 |
| Occi | darkdot strong | 0.392 | 0.090 | 0.383 | 0.884 | 0.000007 |
| Occi | darkdot sky | 0.382 | 0.089 | 0.372 | 0.884 | 0.000007 |
| Dance | v4 tail+chroma | 0.258 | 0.055 | 0.252 | 0.972 | 0.035575 |
| Dance | darkdot strong | 0.245 | 0.054 | 0.239 | 0.969 | 0.038756 |
| Dance | darkdot sky | 0.239 | 0.054 | 0.233 | 0.968 | 0.038820 |
| Ice | v4 tail+chroma | 0.266 | 0.083 | 0.251 | 0.468 | 0.005059 |
| Ice | darkdot strong | 0.266 | 0.082 | 0.252 | 0.467 | 0.005059 |
| Ice | darkdot sky | 0.266 | 0.080 | 0.253 | 0.466 | 0.005059 |

Visual crops:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_dark_dot_v1_strength_occi/
runs/refiner_pilot_stage11_hybrid_best/compare_dark_dot_v1_strength_dance/
runs/refiner_pilot_stage11_hybrid_best/compare_dark_dot_v1_strength_ice/
runs/refiner_pilot_stage11_hybrid_best/compare_dark_dot_v1_sky_occi/
runs/refiner_pilot_stage11_hybrid_best/compare_dark_dot_v1_sky_dance/
runs/refiner_pilot_stage11_hybrid_best/compare_dark_dot_v1_sky_ice/
```

Current quality-first command:

```bash
pixi run python scripts/apply_scunet_preset_chooser.py \
  --scene <scene> \
  --luma-tail-preset auto \
  --chroma-speckle-preset auto \
  --dark-dot-preset auto
```

Integrated v6 outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v6_tail_chroma_darkdot_auto_outputs/
```

Decision:

- Promote dark-dot auto into the current quality-first pipeline.
- This is the cleanest improvement in the user's "dark/magenta dots remain"
  complaint so far: it reduces isolated luma speckle tails without a meaningful
  measured edge or highlight regression.
- It is still not perfect. The remaining gap is no longer just isolated dots;
  broad residual grain and believable detail reconstruction remain the next hard
  problem.

### Stage13 v43: luma HF grain shrink after v6

Goal:

- Reduce the remaining broad luminance grain that survives v6 dark-dot cleanup.
- Avoid the previous failure mode where stronger cleanup makes hair/branches/ice
  lines look plasticky or sleepy.

Implementation:

- Added `--no-tiff` to `scripts/apply_luma_hf_shrink_filter.py` so probes can
  be run without generating large TIFF sidecars.
- Added `--luma-hf-preset` to
  `scripts/apply_scunet_preset_chooser.py`.
- `--luma-hf-preset auto` currently maps to `grain`.
- The luma HF pass runs after luma-tail, chroma-speckle, and dark-dot cleanup.

Probe decision:

- `grain` is the current best balance. It removes a large fraction of flat-field
  luma grain while keeping edge retention nearly unchanged.
- `ultra` improves the numeric luma ratios further, but visual crops become too
  polished/flat on Dance and Ice, so it is rejected as the default.

Integrated v7 real-photo check:

| scene | candidate | flat luma | flat chroma | luma visible | chroma visible | edge | highlight drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Occi | v6 | 0.382 | 0.089 | 0.372 | 0.074 | 0.884 | 0.000007 |
| Occi | v7 grain | 0.274 | 0.088 | 0.262 | 0.073 | 0.883 | 0.000007 |
| Dance | v6 | 0.239 | 0.054 | 0.233 | 0.047 | 0.968 | 0.038820 |
| Dance | v7 grain | 0.149 | 0.054 | 0.141 | 0.047 | 0.965 | 0.041281 |
| Ice | v6 | 0.266 | 0.082 | 0.252 | 0.067 | 0.467 | 0.005059 |
| Ice | v7 grain | 0.224 | 0.082 | 0.210 | 0.067 | 0.463 | 0.005059 |

Visual crops:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_scunet_preset_chooser_v7_occi/
runs/refiner_pilot_stage11_hybrid_best/compare_scunet_preset_chooser_v7_dance/
runs/refiner_pilot_stage11_hybrid_best/compare_scunet_preset_chooser_v7_ice/
```

Integrated v7 outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v7_tail_chroma_darkdot_lumahf_auto_outputs/
```

Current quality-first command:

```bash
pixi run python scripts/apply_scunet_preset_chooser.py \
  --scene <scene> \
  --luma-tail-preset auto \
  --chroma-speckle-preset auto \
  --dark-dot-preset auto \
  --luma-hf-preset auto
```

Decision:

- Promote luma HF `grain` into the current quality-first pipeline as v7.
- This improves the "shrunken preview looks similar, but equal-size crop still
  has sand-like noise" complaint without a large measured edge penalty.
- The remaining visible problem is more chroma/signed-color than luma: blue or
  dark magenta dots still survive in some shadow/sky regions. That should be the
  next target rather than making luma smoothing stronger.

### Stage13 v44: signed blue/magenta chroma dots after v7

Goal:

- Attack the remaining blue/dark-magenta pin dots that v7 leaves behind.
- Do not add more luma smoothing; preserve v7's detail/edge balance.

Diagnosis:

- Residual speckle evaluation showed v7 mainly improves luma impulse visibility,
  while chroma/magenta/blue impulse metrics are almost unchanged from v6.
- Therefore the next pass should be chroma-only and axis-specific.

Implementation:

- Added `--no-tiff` to `scripts/apply_signed_chroma_outlier_filter.py`.
- Added `--signed-chroma-preset` to
  `scripts/apply_scunet_preset_chooser.py`.
- `--signed-chroma-preset auto` currently maps to `bm_strong`.
- `bm_strong` reuses the signed chroma outlier model, but only opens
  blue/magenta axes:

```text
strength=0.72
median_size=7
low_sigma=2.4
outlier_threshold=0.0032
outlier_transition=0.0022
magenta_weight=1.0
red_weight=0.0
blue_weight=1.05
shadow_threshold=0.58
```

Real-photo check:

| scene | candidate | flat luma | flat chroma | luma visible | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Occi | v7 | 0.274 | 0.088 | 0.262 | 0.073 | 0.070 | 0.024 | 0.883 |
| Occi | v8 bm | 0.274 | 0.084 | 0.262 | 0.070 | 0.065 | 0.021 | 0.883 |
| Dance | v7 | 0.149 | 0.054 | 0.141 | 0.047 | 0.047 | 0.033 | 0.965 |
| Dance | v8 bm | 0.148 | 0.050 | 0.140 | 0.043 | 0.043 | 0.030 | 0.964 |
| Ice | v7 | 0.224 | 0.082 | 0.210 | 0.067 | 0.058 | 0.030 | 0.463 |
| Ice | v8 bm | 0.223 | 0.078 | 0.210 | 0.063 | 0.054 | 0.027 | 0.463 |

Residual speckle check:

| scene | candidate | chroma visible | magenta p999 | shadow magenta p999 | blue p999 | shadow blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Occi | v7 | 0.066 | 0.215 | 0.060 | 0.229 | 0.074 |
| Occi | v8 bm | 0.061 | 0.209 | 0.057 | 0.225 | 0.070 |
| Dance | v7 | 0.029 | 0.096 | 0.029 | 0.099 | 0.029 |
| Dance | v8 bm | 0.026 | 0.092 | 0.022 | 0.095 | 0.023 |
| Ice | v7 | 0.070 | 0.178 | 0.058 | 0.195 | 0.069 |
| Ice | v8 bm | 0.065 | 0.173 | 0.054 | 0.196 | 0.064 |

Visual crops:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_signed_chroma_after_v7_bm_occi/
runs/refiner_pilot_stage11_hybrid_best/compare_signed_chroma_after_v7_bm_dance/
runs/refiner_pilot_stage11_hybrid_best/compare_signed_chroma_after_v7_bm_ice/
```

Current quality-first command:

```bash
pixi run python scripts/apply_scunet_preset_chooser.py \
  --scene <scene> \
  --luma-tail-preset auto \
  --chroma-speckle-preset auto \
  --dark-dot-preset auto \
  --luma-hf-preset auto \
  --signed-chroma-preset auto
```

Decision:

- Promote signed chroma `bm_strong` as the current v8 post-pass.
- The gain is smaller than v7's luma HF cleanup, but it targets the correct
  residual failure: blue/magenta chroma points fall while luma/detail metrics
  remain essentially stable.
- It is still not the final "perfect" answer. Ice flat blue p999 is almost
  unchanged, so the next step needs a better blue-dot detector or a learned
  region classifier rather than simply increasing the current strength.

### Stage13 v45: neutral-region chroma-dot split and v9

Goal:

- Understand why Ice `flat blue p999` did not improve after v8.
- Avoid chasing real cyan/blue image structure as if it were blue noise.
- Add one more chroma-dot pass only where the local low-frequency chroma is
  neutral enough to be safe.

Diagnosis:

- The top Ice `flat blue p999` coordinates were not dark pin dots. They landed
  on mid-luma cyan/ice structures (`display luma ~= 0.34..0.57`, not
  `shadow_flat`).
- Therefore global `flat blue p999` is contaminated by subject color. It should
  not be used as the sole adoption metric.

Implementation:

- Added `neutral_flat` and `blue_struct_flat` split metrics to
  `scripts/residual_speckle_eval.py`.
- Added `scripts/apply_blue_chroma_dot_filter.py` and rejected it:
  - `pin` / `pin_strong` slightly reduce `shadow blue p999`,
  - but do not improve the real target enough and can worsen blue-structure
    blue p999.
- Added `scripts/apply_neutral_chroma_dot_filter.py`.
- Added `--neutral-chroma-preset` to
  `scripts/apply_scunet_preset_chooser.py`.
- `--neutral-chroma-preset auto` currently maps to `neutral_strong`.

Neutral strong parameters:

```text
strength=0.96
median_size=7
low_sigma=2.4
outlier_threshold=0.0024
outlier_transition=0.0016
magenta_weight=1.15
blue_weight=1.20
neutral_threshold=0.080
neutral_transition=0.028
shadow_threshold=0.76
```

Real-photo check:

| scene | candidate | flat luma | flat chroma | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Occi | v8 | 0.274 | 0.084 | 0.070 | 0.065 | 0.021 | 0.883 |
| Occi | v9 neutral | 0.275 | 0.080 | 0.066 | 0.061 | 0.018 | 0.883 |
| Dance | v8 | 0.148 | 0.050 | 0.043 | 0.043 | 0.030 | 0.964 |
| Dance | v9 neutral | 0.147 | 0.046 | 0.040 | 0.039 | 0.026 | 0.964 |
| Ice | v8 | 0.223 | 0.078 | 0.063 | 0.054 | 0.027 | 0.463 |
| Ice | v9 neutral | 0.223 | 0.074 | 0.060 | 0.050 | 0.025 | 0.463 |

Residual speckle split:

| scene | candidate | neutral chroma visible | neutral magenta p999 | neutral blue p999 | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: |
| Occi | v8 | 0.053 | 0.181 | 0.179 | 0.287 |
| Occi | v9 neutral | 0.048 | 0.176 | 0.177 | 0.298 |
| Dance | v8 | 0.024 | 0.085 | 0.089 | 0.083 |
| Dance | v9 neutral | 0.022 | 0.083 | 0.087 | 0.078 |
| Ice | v8 | 0.054 | 0.142 | 0.142 | 0.263 |
| Ice | v9 neutral | 0.048 | 0.137 | 0.138 | 0.270 |

Visual crops:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_neutral_chroma_dot_occi/
runs/refiner_pilot_stage11_hybrid_best/compare_neutral_chroma_dot_dance/
runs/refiner_pilot_stage11_hybrid_best/compare_neutral_chroma_dot_ice/
```

Integrated v9 smoke output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v9_tail_chroma_darkdot_lumahf_signed_neutral_auto_outputs/
```

Current quality-first command:

```bash
pixi run python scripts/apply_scunet_preset_chooser.py \
  --scene <scene> \
  --luma-tail-preset auto \
  --chroma-speckle-preset auto \
  --dark-dot-preset auto \
  --luma-hf-preset auto \
  --signed-chroma-preset auto \
  --neutral-chroma-preset auto
```

Decision:

- Promote neutral chroma `neutral_strong` as v9.
- Reject the one-sided blue `pin` filter as a default. It was aimed at the
  wrong mixed metric.
- v9 is still a small cleanup pass, not a breakthrough. The next useful step is
  learned gating or region classification for where chroma correction should
  open, because hand thresholds are now mostly making incremental gains.

### Stage13 v46: learned neutral chroma gate v1

Goal:

- Replace the hand-tuned neutral chroma gate with a learned safety gate.
- Keep the useful v9 neutral-dot cleanup, but avoid opening on real blue/cyan
  structure where deterministic v9 can slightly worsen `blue_struct_blue_p999`.

Implementation:

- Added `scripts/train_neutral_chroma_gate.py`.
- Trained a small CPU gate (`width=16`, `blocks=2`, `context=48`) for 120 steps
  using v8 as base and deterministic v9 as the candidate correction.
- Checkpoint:

```text
runs/refiner_pilot_stage11_hybrid_best/neutral_chroma_gate_pilot_v1_120/neutral_chroma_gate_final.pt
```

- Applied the learned gate to Occi / Dance / Ice.
- Also scanned post-application gate strengths by reusing saved gate PNGs:
  `s2 = 2.0x`, `s3 = 3.0x`.

Regular real-photo check:

| scene | candidate | flat chroma | chroma visible | magenta visible | shadow magenta visible | edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Occi | v8 | 0.084 | 0.070 | 0.065 | 0.021 | 0.883 |
| Occi | v9 deterministic | 0.080 | 0.066 | 0.061 | 0.018 | 0.883 |
| Occi | gate s3 | 0.083 | 0.068 | 0.063 | 0.020 | 0.883 |
| Dance | v8 | 0.050 | 0.043 | 0.043 | 0.030 | 0.964 |
| Dance | v9 deterministic | 0.046 | 0.040 | 0.039 | 0.026 | 0.964 |
| Dance | gate s3 | 0.048 | 0.042 | 0.041 | 0.028 | 0.964 |
| Ice | v8 | 0.078 | 0.063 | 0.054 | 0.027 | 0.463 |
| Ice | v9 deterministic | 0.074 | 0.060 | 0.050 | 0.025 | 0.463 |
| Ice | gate s3 | 0.076 | 0.062 | 0.052 | 0.026 | 0.463 |

Residual split check:

| scene | candidate | neutral chroma visible | neutral magenta p999 | neutral blue p999 | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: |
| Occi | v8 | 0.053 | 0.181 | 0.179 | 0.287 |
| Occi | v9 deterministic | 0.048 | 0.176 | 0.177 | 0.298 |
| Occi | gate s3 | 0.051 | 0.180 | 0.179 | 0.288 |
| Dance | v8 | 0.024 | 0.085 | 0.089 | 0.083 |
| Dance | v9 deterministic | 0.022 | 0.083 | 0.087 | 0.078 |
| Dance | gate s3 | 0.023 | 0.084 | 0.089 | 0.080 |
| Ice | v8 | 0.054 | 0.142 | 0.142 | 0.263 |
| Ice | v9 deterministic | 0.048 | 0.137 | 0.138 | 0.270 |
| Ice | gate s3 | 0.051 | 0.141 | 0.141 | 0.264 |

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/neutral_chroma_gate_pilot_v1_120_outputs/
runs/refiner_pilot_stage11_hybrid_best/neutral_chroma_gate_pilot_v1_120_strength_scan/
runs/refiner_pilot_stage11_hybrid_best/eval_neutral_chroma_gate_v1_120_strength_{occi,dance,ice}/
runs/refiner_pilot_stage11_hybrid_best/speckle_eval_neutral_chroma_gate_v1_120_strength_{occi,dance,ice}/
```

Decision:

- Do not promote learned gate v1 as the default.
- The learned direction is useful because it mostly avoids the blue-structure
  regression seen in Occi/Ice deterministic v9.
- However, even at 3x strength it underperforms deterministic v9 on the actual
  visible chroma/magenta residuals. It is too conservative.
- Keep deterministic v9 as the current quality-first default.

Next design:

- Train a v2 gate with stronger target calibration instead of merely multiplying
  the output after inference.
- The target should open more on neutral residual chroma dots while keeping a
  separate penalty for blue/cyan structure.
- A practical v2 recipe is:
  - target gain around `1.6..2.2`,
  - lower mean-gate regularization,
  - explicit blue-structure close penalty,
  - longer 300-600 step pilot before any overnight run.

### Stage13 v47: learned gate v2 and blue-structure protector v10

Goal:

- Test whether a stronger learned neutral-chroma gate can replace v9.
- If not, keep v9's cleanup and only protect the blue/cyan structure failure
  that v9 exposed.

Learned gate v2:

- Extended `scripts/train_neutral_chroma_gate.py` with:
  - `--target-gain`
  - `--target-power`
  - `--blue-close-weight`
- Trained a 300-step CPU pilot:

```bash
pixi run python scripts/train_neutral_chroma_gate.py train \
  --output-dir runs/refiner_pilot_stage11_hybrid_best/neutral_chroma_gate_pilot_v2_300 \
  --steps 300 --batch-size 2 --patch-size 160 --context 48 \
  --width 16 --blocks 2 \
  --target-gain 2.0 --target-power 0.72 \
  --mean-weight 0.012 --blue-close-weight 0.055 --smooth-weight 0.020 \
  --device cpu
```

Gate means:

| scene | gate mean | p90 | p99 | elapsed |
| --- | ---: | ---: | ---: | ---: |
| Occi | 0.322 | 0.512 | 0.601 | 145s |
| Dance | 0.426 | 0.584 | 0.668 | 53s |
| Ice | 0.356 | 0.586 | 0.657 | 61s |

v2 regular check:

| scene | candidate | flat chroma | chroma visible | magenta visible | shadow magenta visible |
| --- | --- | ---: | ---: | ---: | ---: |
| Occi | v9 deterministic | 0.080 | 0.066 | 0.061 | 0.018 |
| Occi | gate v2 | 0.083 | 0.068 | 0.064 | 0.020 |
| Dance | v9 deterministic | 0.046 | 0.040 | 0.039 | 0.026 |
| Dance | gate v2 | 0.048 | 0.042 | 0.041 | 0.027 |
| Ice | v9 deterministic | 0.074 | 0.060 | 0.050 | 0.025 |
| Ice | gate v2 | 0.076 | 0.062 | 0.052 | 0.026 |

v2 residual split:

| scene | candidate | neutral chroma visible | neutral magenta p999 | neutral blue p999 | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: |
| Occi | v9 deterministic | 0.048 | 0.176 | 0.177 | 0.298 |
| Occi | gate v2 | 0.051 | 0.180 | 0.179 | 0.287 |
| Dance | v9 deterministic | 0.022 | 0.083 | 0.087 | 0.078 |
| Dance | gate v2 | 0.023 | 0.084 | 0.089 | 0.079 |
| Ice | v9 deterministic | 0.048 | 0.137 | 0.138 | 0.270 |
| Ice | gate v2 | 0.051 | 0.141 | 0.141 | 0.263 |

Decision on learned gate:

- v2 is better calibrated than v1 and protects blue/cyan structure well.
- It still underperforms deterministic v9 on the visible neutral chroma and
  magenta residuals.
- Do not promote learned gate v2 as default.

Blue-structure protector:

- Added `scripts/apply_blue_structure_protector.py`.
- Added `--blue-structure-protect-preset` to
  `scripts/apply_scunet_preset_chooser.py`.
- `--blue-structure-protect-preset auto` currently maps to `mild`.
- The protector runs after neutral chroma cleanup. It restores the pre-neutral
  image only in low-frequency blue/cyan structure regions.

v10 regular check:

| scene | candidate | flat chroma | chroma visible | magenta visible | shadow magenta visible |
| --- | --- | ---: | ---: | ---: | ---: |
| Occi | v9 deterministic | 0.080 | 0.066 | 0.061 | 0.018 |
| Occi | v10 protect | 0.080 | 0.066 | 0.061 | 0.018 |
| Dance | v9 deterministic | 0.046 | 0.040 | 0.039 | 0.026 |
| Dance | v10 protect | 0.046 | 0.040 | 0.039 | 0.026 |
| Ice | v9 deterministic | 0.074 | 0.060 | 0.050 | 0.025 |
| Ice | v10 protect | 0.074 | 0.060 | 0.051 | 0.025 |

v10 residual split:

| scene | candidate | neutral chroma visible | neutral magenta p999 | neutral blue p999 | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: |
| Occi | v9 deterministic | 0.048 | 0.176 | 0.177 | 0.298 |
| Occi | v10 protect | 0.048 | 0.177 | 0.177 | 0.296 |
| Dance | v9 deterministic | 0.022 | 0.083 | 0.087 | 0.078 |
| Dance | v10 protect | 0.022 | 0.083 | 0.087 | 0.078 |
| Ice | v9 deterministic | 0.048 | 0.137 | 0.138 | 0.270 |
| Ice | v10 protect | 0.048 | 0.137 | 0.138 | 0.265 |

Integrated smoke output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v10_tail_chroma_darkdot_lumahf_signed_neutral_blueprotect_auto_outputs/
```

Current quality-first command:

```bash
pixi run python scripts/apply_scunet_preset_chooser.py \
  --scene <scene> \
  --luma-tail-preset auto \
  --chroma-speckle-preset auto \
  --dark-dot-preset auto \
  --luma-hf-preset auto \
  --signed-chroma-preset auto \
  --neutral-chroma-preset auto \
  --blue-structure-protect-preset auto
```

Decision:

- Promote blue-structure protector as v10 candidate.
- It keeps v9's neutral/magenta cleanup almost exactly, while reducing the
  measured blue-structure regression on Ice and slightly on Occi.
- This is a better quality-first move than replacing v9 with the learned gate.
- Remaining gap: v10 is still incremental. It does not solve broad perceptual
  reconstruction or the remaining residual grain/dots; it just avoids one
  measurable overreach introduced by v9.

### Stage13 v48: post-v10 dark-dot probe

Goal:

- After integrated v10, reduce the remaining isolated black/dark dots visible
  in large flat areas, especially the Dance sky.
- Avoid adding broad smoothing or further softening texture/detail.

Implementation:

- Added `--no-tiff` to `scripts/apply_dark_dot_speckle_filter.py` so small
  probes do not create large TIFF sidecars.
- Added optional blue/cyan structure inhibition to the same filter:
  - `--blue-structure-inhibit`
  - `--blue-structure-threshold`
  - `--blue-structure-transition`
  - `--blue-structure-chroma-threshold`
  - `--blue-structure-chroma-transition`
- Probe parameters:

```text
preset=sky
strength=0.65
dark_threshold=0.0026
dark_transition=0.0018
local_gain=0.08
max_lift=0.026
chroma_strength=0.35
line_preserve_strength=0.92
blue_structure_inhibit=0.92
```

Dance result:

| candidate | flat luma | luma p99 | luma visible | edge | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v10 | 0.147 | 0.445 | 0.139 | 0.964 | 0.066 | 0.022 | 0.078 |
| post darkdot | 0.145 | 0.441 | 0.137 | 0.963 | 0.062 | 0.021 | 0.077 |

Occi result:

| candidate | flat luma | luma p99 | luma visible | edge | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v10 | 0.275 | 0.662 | 0.262 | 0.883 | 0.107 | 0.048 | 0.296 |
| post darkdot | 0.270 | 0.657 | 0.258 | 0.883 | 0.101 | 0.047 | 0.307 |

Ice result:

| candidate | flat luma | luma p99 | luma visible | edge | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v10 | 0.223 | 0.545 | 0.210 | 0.463 | 0.094 | 0.048 | 0.265 |
| post darkdot | 0.224 | 0.542 | 0.211 | 0.463 | 0.091 | 0.046 | 0.273 |

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/post_v10_dark_dot_probe/
runs/refiner_pilot_stage11_hybrid_best/post_v10_dark_dot_blueprotect_probe/
runs/refiner_pilot_stage11_hybrid_best/eval_post_v10_dark_dot_blueprotect_{occi,dance,ice}/
runs/refiner_pilot_stage11_hybrid_best/speckle_eval_post_v10_dark_dot_blueprotect_{occi,dance,ice}/
```

Decision:

- Do not promote post-v10 dark-dot as a global default.
- It helps Dance sky exactly as intended, but it worsens the blue-structure
  split on Occi and Ice even with blue-structure inhibition.
- This confirms the next improvement cannot simply be "one more dark-dot pass".
  It needs either a better region classifier or a pass limited to proven flat
  sky/dark neutral regions.
- Keep the code knobs because they are useful for future per-region experiments,
  but leave the current v10 quality-first command unchanged.

### Stage13 v49: sky-flat gate rejected, scene-gated v11 candidate

Goal:

- Salvage the useful Dance-sky gain from Stage13 v48 without touching Occi/Ice
  blue/cyan structure.
- Test whether a stricter deterministic "flat sky / dark neutral" gate can
  separate the failure cases.

Implementation:

- Extended `scripts/apply_dark_dot_speckle_filter.py` with optional sky-flat
  gates:
  - `--sky-flat-strength`
  - `--sky-luma-min`
  - `--sky-luma-max`
  - `--sky-luma-transition`
  - `--sky-neutral-threshold`
  - `--sky-neutral-transition`
  - `--sky-blue-abs-threshold`
  - `--sky-blue-abs-transition`
  - `--sky-line-max`
  - `--sky-line-transition`
- Tested two variants:
  - `skyflat`: normal sky-flat restriction,
  - `skyflat_strict`: lower blue/line/neutral thresholds.

Skyflat result:

| scene | candidate | luma p99 | luma visible | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Occi | v10 | 0.662 | 0.262 | 0.107 | 0.048 | 0.296 |
| Occi | skyflat | 0.662 | 0.261 | 0.105 | 0.048 | 0.306 |
| Dance | v10 | 0.445 | 0.139 | 0.066 | 0.022 | 0.078 |
| Dance | skyflat | 0.445 | 0.138 | 0.065 | 0.022 | 0.078 |
| Ice | v10 | 0.545 | 0.210 | 0.094 | 0.048 | 0.265 |
| Ice | skyflat | 0.545 | 0.210 | 0.094 | 0.047 | 0.273 |

Strict result:

| scene | candidate | luma p99 | luma visible | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | v10 | 0.445 | 0.139 | 0.066 | 0.022 | 0.078 |
| Dance | strict | 0.445 | 0.139 | 0.065 | 0.022 | 0.078 |
| Ice | v10 | 0.545 | 0.210 | 0.094 | 0.048 | 0.265 |
| Ice | strict | 0.545 | 0.210 | 0.094 | 0.048 | 0.273 |

Decision on sky-flat:

- Reject both sky-flat variants.
- The stricter gate becomes safe-ish by becoming too weak. It gives up most of
  the Dance benefit while still failing to improve the Ice blue-structure split.
- This is another sign that hand thresholds are at their limit.

Scene-gated v11:

- Added `--post-dark-dot-preset` to
  `scripts/apply_scunet_preset_chooser.py`.
- `--post-dark-dot-preset auto` maps to:
  - `sky_luma -> sky_tail`
  - `hair_luma -> off`
  - `blue_shadow_safe -> off`
- This deliberately uses the existing coarse preset chooser as a practical
  safety gate. It is not the final per-region solution, but it captures the
  Dance gain while avoiding the measured Occi/Ice regression path.

Integrated Dance v11 result:

| candidate | flat luma | luma p99 | luma visible | edge | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v10 | 0.147 | 0.445 | 0.139 | 0.964 | 0.066 | 0.022 | 0.078 |
| v11 | 0.145 | 0.441 | 0.137 | 0.963 | 0.062 | 0.021 | 0.077 |

Auto routing check:

```text
sky_luma -> sky_tail
hair_luma -> off
blue_shadow_safe -> off
```

Integrated v11 output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v11_sky_post_darkdot_auto_outputs/
```

Current quality-first command:

```bash
pixi run python scripts/apply_scunet_preset_chooser.py \
  --scene <scene> \
  --luma-tail-preset auto \
  --chroma-speckle-preset auto \
  --dark-dot-preset auto \
  --luma-hf-preset auto \
  --signed-chroma-preset auto \
  --neutral-chroma-preset auto \
  --blue-structure-protect-preset auto \
  --post-dark-dot-preset auto
```

Decision:

- Promote scene-gated post dark-dot as v11 candidate.
- It is a practical improvement for Dance-like `sky_luma` images.
- It is not the final answer. The next design should replace the coarse
  scene-gate with a learned or stronger per-region sky/flat classifier so that
  sky cleanup can happen locally without relying on the whole-photo preset.

### Stage13 v50: learned local post-darkdot gate pilot

Goal:

- Replace the coarse v11 scene gate with a local learned gate.
- Blend between the safe v10 output and the stronger post-v10 darkdot candidate.
- Open on Dance sky/snow flat areas, close on Occi hair/blue shadow and Ice
  blue/cyan structures.

Implementation:

- Added `scripts/train_post_dark_dot_gate.py`.
- Inputs:
  - safe base: `scunet_preset_chooser_v10_*_auto.exr`
  - stronger candidate: `post_v10_dark_dot_probe/*_v10_post_darkdot_sky.exr`
  - references: noisy EXRs from `test_photos`
- The model predicts only a blend gate. It does not invent a new correction.
- Features include base/reference/candidate display RGB, candidate delta,
  luma/saturation, flatness, edge, blue-structure signal, luma impulses,
  dark-dot signal, candidate lift, and chroma/magenta/blue impulses.

Pilot v1:

- Training: `post_dark_dot_gate_pilot_v1_260`
- Result: safe but almost fully closed.
- Gate stats:
  - Dance mean `0.030`, p99 `0.065`
  - Ice mean `0.026`, p99 `0.068`
  - Occi mean `0.024`, p99 `0.081`

v1 result:

| scene | candidate | luma p99 | luma visible | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | v10 | 0.445 | 0.139 | 0.066 | 0.022 | 0.078 |
| Dance | global | 0.441 | 0.137 | 0.062 | 0.021 | 0.077 |
| Dance | v1 learned | 0.445 | 0.139 | 0.066 | 0.022 | 0.078 |
| Ice | v10 | 0.545 | 0.210 | 0.094 | 0.048 | 0.265 |
| Ice | global | 0.541 | 0.211 | 0.089 | 0.046 | 0.274 |
| Ice | v1 learned | 0.545 | 0.210 | 0.094 | 0.048 | 0.265 |
| Occi | v10 | 0.662 | 0.262 | 0.107 | 0.048 | 0.296 |
| Occi | global | 0.656 | 0.257 | 0.101 | 0.047 | 0.307 |
| Occi | v1 learned | 0.662 | 0.262 | 0.107 | 0.048 | 0.296 |

Pilot v2:

- Training: `post_dark_dot_gate_pilot_v2_360`
- Recipe changes:
  - stronger target gain,
  - snow as a soft positive,
  - weaker mean/protection penalties,
  - apply-time strength `4.0`.
- Gate stats before strength:
  - Dance mean `0.055`, p99 `0.111`
  - Ice mean `0.047`, p99 `0.111`
  - Occi mean `0.043`, p99 `0.126`

v2 strength-4 result:

| scene | candidate | luma p99 | luma visible | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dance | v10 | 0.445 | 0.139 | 0.066 | 0.022 | 0.078 |
| Dance | global | 0.441 | 0.137 | 0.062 | 0.021 | 0.077 |
| Dance | v2s4 learned | 0.445 | 0.138 | 0.065 | 0.021 | 0.078 |
| Ice | v10 | 0.545 | 0.210 | 0.094 | 0.048 | 0.265 |
| Ice | global | 0.541 | 0.211 | 0.089 | 0.046 | 0.274 |
| Ice | v2s4 learned | 0.544 | 0.210 | 0.093 | 0.047 | 0.265 |
| Occi | v10 | 0.662 | 0.262 | 0.107 | 0.048 | 0.296 |
| Occi | global | 0.656 | 0.257 | 0.101 | 0.047 | 0.307 |
| Occi | v2s4 learned | 0.661 | 0.261 | 0.105 | 0.048 | 0.296 |

Decision:

- Do not promote learned local post-darkdot gate as the main path.
- v2 is directionally correct: it keeps the blue-structure split at v10 level
  while recovering a small part of the global post-darkdot luma gain.
- The effect is too small because the learned gate is still conservative and
  the candidate itself is a narrow postprocess. This is not a strong enough
  mechanism to move toward "perfect" NR.
- Keep the script and checkpoints as a diagnostic tool. It proves a local
  safety gate can prevent the measured blue-structure regression, but the next
  meaningful step needs a stronger restoration/smoothing candidate, not just a
  safer blend mask over this small dark-dot pass.

### Stage13 v51: HDR-safe strong flat cleanup candidate

Goal:

- Move away from tiny dark-dot-only corrections.
- Create a stronger candidate that visibly lowers flat-region grain and chroma
  residue while protecting coherent detail.

Implementation:

- Extended `scripts/apply_detail_protected_flat_cleanup.py`:
  - added `--no-tiff` for faster probes,
  - fixed HDR clipping by restoring original linear HDR peaks after display
    space cleanup.
- Added `--flat-cleanup-preset` to
  `scripts/apply_scunet_preset_chooser.py`.
- New integrated preset:
  - `auto -> strong_v1`
  - default remains `off`, so previous commands are unchanged unless the new
    option is enabled.

Strong flat cleanup v1 parameters:

```text
luma_strength=0.92
chroma_strength=0.95
luma_sigma=2.20
chroma_sigma=3.00
flat_threshold=0.030
edge_threshold=0.030
coherent_protect=0.95
texture_protect=0.72
skin_protect=0.40
gate_blur=1.20
hdr_restore_threshold=0.92
hdr_restore_transition=0.24
```

Standalone v10 -> strongflat result:

| scene | candidate | flat luma | luma p99 | luma visible | flat chroma | chroma visible | edge | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dance | v10 | 0.147 | 0.445 | 0.139 | 0.046 | 0.040 | 0.964 | 0.066 | 0.022 | 0.078 |
| Dance | strongflat | 0.136 | 0.442 | 0.127 | 0.042 | 0.036 | 0.964 | 0.059 | 0.019 | 0.078 |
| Ice | v10 | 0.223 | 0.545 | 0.210 | 0.074 | 0.060 | 0.463 | 0.094 | 0.048 | 0.265 |
| Ice | strongflat | 0.214 | 0.541 | 0.200 | 0.069 | 0.056 | 0.463 | 0.089 | 0.042 | 0.253 |
| Occi | v10 | 0.275 | 0.662 | 0.262 | 0.080 | 0.066 | 0.883 | 0.107 | 0.048 | 0.296 |
| Occi | strongflat | 0.250 | 0.657 | 0.236 | 0.071 | 0.058 | 0.884 | 0.090 | 0.040 | 0.278 |

Integrated v12 Dance command:

```bash
pixi run python scripts/apply_scunet_preset_chooser.py \
  --scene k5_dance \
  --luma-tail-preset auto \
  --chroma-speckle-preset auto \
  --dark-dot-preset auto \
  --luma-hf-preset auto \
  --signed-chroma-preset auto \
  --neutral-chroma-preset auto \
  --blue-structure-protect-preset auto \
  --post-dark-dot-preset auto \
  --flat-cleanup-preset auto
```

Integrated Dance v12 result:

| candidate | flat luma | luma p99 | luma visible | flat chroma | chroma visible | edge | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v10 | 0.147 | 0.445 | 0.139 | 0.046 | 0.040 | 0.964 | 0.066 | 0.022 | 0.078 |
| v11 | 0.145 | 0.441 | 0.137 | 0.046 | 0.040 | 0.963 | 0.062 | 0.021 | 0.077 |
| v12 | 0.134 | 0.438 | 0.125 | 0.041 | 0.035 | 0.963 | 0.056 | 0.018 | 0.077 |

Visual check:

- Dance sky: smoother than v10/v11 with no obvious new artifacts in the crop.
- Ice blue shadow: background noise falls while thin branch structure remains
  close to v10; blue-structure metric improves.
- Occi selected crops are not perfect for hair diagnosis, but available checks
  did not show a large new break. Occi still needs a better hair/detail ROI
  check before promoting v12 as universal.

Decision:

- Promote strong flat cleanup as the next main direction.
- v12 is the best measured integrated Dance candidate so far.
- Before making it the default quality command for all scenes, run integrated
  v12 on Ice and Occi and inspect better hair/edge crops.

### Stage13 v52: integrated v12 on Ice and Occi

Goal:

- Verify that the v12 flat-cleanup direction generalizes beyond Dance.
- Check the two highest-risk cases:
  - Ice: blue/cyan shadow structure must not break.
  - Occi: hair/face detail must not become obviously smeared.

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v12_flat_cleanup_auto_outputs/
```

Integrated preset routing:

| scene | preset | post dark-dot | flat cleanup |
| --- | --- | --- | --- |
| Dance | `sky_luma` | `sky_tail` | `strong_v1` |
| Ice | `blue_shadow_safe` | off | `strong_v1` |
| Occi | `hair_luma` | off | `strong_v1` |

Integrated v12 all-scene metrics:

| scene | candidate | flat luma | luma p99 | luma visible | flat chroma | chroma visible | edge | magenta p99 | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dance | v10 | 0.147 | 0.445 | 0.139 | 0.046 | 0.040 | 0.964 | 0.050 | 0.066 | 0.022 | 0.078 |
| Dance | v11 | 0.145 | 0.441 | 0.137 | 0.046 | 0.040 | 0.963 | 0.050 | 0.062 | 0.021 | 0.077 |
| Dance | v12 | 0.134 | 0.438 | 0.125 | 0.041 | 0.035 | 0.963 | 0.047 | 0.056 | 0.018 | 0.077 |
| Ice | v10 | 0.223 | 0.545 | 0.210 | 0.074 | 0.060 | 0.463 | 0.104 | 0.094 | 0.048 | 0.265 |
| Ice | v12 | 0.214 | 0.541 | 0.200 | 0.069 | 0.056 | 0.463 | 0.102 | 0.089 | 0.042 | 0.253 |
| Occi | v10 | 0.275 | 0.662 | 0.262 | 0.080 | 0.066 | 0.883 | 0.148 | 0.107 | 0.048 | 0.296 |
| Occi | v12 | 0.250 | 0.657 | 0.236 | 0.071 | 0.058 | 0.884 | 0.143 | 0.090 | 0.040 | 0.278 |

Visual check:

- Dance sky: v12 is the smoothest of v10/v11/v12, though some dark dots remain.
- Ice blue shadow: v12 reduces background grain and keeps thin branch/ice
  structure close to v10. This is the first post-v10 direction that improves
  the blue-structure split instead of worsening it.
- Occi face/hairline: v12 smooths skin/background while keeping major hair and
  eyelash structures. It may still be slightly soft, but the checked crops do
  not show a new obvious failure.

Decision:

- Promote integrated v12 as the current best quality candidate.
- The strong flat cleanup is now the main path, not the learned post-darkdot
  gate.
- Remaining issues:
  - some dark dots remain in Dance sky,
  - Occi hair/detail still needs a stronger detail-aware protection/restoration
    mechanism before calling this "perfect",
  - integrated v12 is slow on large Occi because the pipeline has no progress
    logging and applies several full-frame filters.
- Next design:
  - keep v12 as the baseline,
  - add a detail-gate/protection input to strong flat cleanup instead of using
    an all-zero gate,
  - then tighten flat cleanup further only where the detail gate is closed.

### Stage13 v53: auto-detail protection probes for strong flat cleanup

Goal:

- Test whether the v12 strong flat cleanup can be pushed slightly harder if an
  automatic local-detail protection mask is added.
- Avoid restoring noisy detail after cleanup; this is only a gate/protection
  change inside the cleanup stage.

Implementation:

- Added optional `auto_detail_protect`, `auto_detail_threshold`, and
  `auto_detail_transition` inputs to
  `scripts/apply_detail_protected_flat_cleanup.py`.
- The auto-detail gate is computed from current-image local luma detail and
  edge energy, then added to the existing coherent/texture/detail/skin protect
  term.
- `strong_v1` keeps `auto_detail_protect=0.0`, so integrated v12 is unchanged
  unless an explicit experimental preset is selected.

Probe outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/strong_flat_cleanup_detail_v2_probe/
runs/refiner_pilot_stage11_hybrid_best/strong_flat_cleanup_detail_v3_probe/
```

`detail_v2` was overprotected:

- Dance gate mean dropped from `0.191` to `0.118`.
- Ice gate mean dropped from `0.132` to `0.071`.
- Metrics moved back toward v10 instead of improving v12.

`detail_v3` loosened the protection and was closer:

| scene | candidate | flat luma | luma p99 | luma visible | flat chroma | chroma visible | edge | residual luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dance | strong_v1 | 0.136 | 0.442 | 0.127 | 0.042 | 0.036 | 0.964 | 0.059 | 0.019 | 0.078 |
| Dance | detail_v3 | 0.135 | 0.443 | 0.126 | 0.041 | 0.035 | 0.964 | 0.059 | 0.019 | 0.078 |
| Ice | strong_v1 | 0.214 | 0.541 | 0.200 | 0.069 | 0.056 | 0.463 | 0.089 | 0.042 | 0.253 |
| Ice | detail_v3 | 0.212 | 0.542 | 0.198 | 0.069 | 0.056 | 0.463 | 0.089 | 0.043 | 0.253 |
| Occi | strong_v1 | 0.250 | 0.657 | 0.236 | 0.071 | 0.058 | 0.884 | 0.090 | 0.040 | 0.278 |
| Occi | detail_v3 | 0.247 | 0.659 | 0.233 | 0.071 | 0.058 | 0.884 | 0.090 | 0.040 | 0.278 |

Visual check:

- Occi face/hair, bangs, and cloth crops showed no meaningful hair/detail
  improvement over `strong_v1`.
- The metric movement is mixed: flat means improve by a hair, while p99/tails
  slightly regress on some scenes.

Decision:

- Do not promote `detail_v2` or `detail_v3`.
- Keep v12/`strong_v1` as the current baseline.
- The auto-detail gate is useful infrastructure, but not the missing
  breakthrough. It mostly rebalances cleanup coverage; it does not recover
  convincing hair/detail structure.

### Stage13 v54: structure graft check after v12

Goal:

- Check whether the perceived v12 softness on Occi can be reduced by adding
  back low/mid-frequency luma structure after cleanup.
- Test two references:
  - original noisy EXR,
  - user-provided `X-T5 Occi SCUNet.EXR` as a bounded reconstruction candidate.

Probe outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/occi_v12_structure_graft_probe/
runs/refiner_pilot_stage11_hybrid_best/compare_occi_v12_structure_graft_probe/
runs/refiner_pilot_stage11_hybrid_best/compare_occi_v12_structure_graft_hdr_restore/
```

Initial issue:

- The older `scripts/apply_structure_luma_graft.py` operated in display sRGB and
  clipped output to `0..1`, which crushed HDR bokeh/highlights into gray areas.
- Patched the script to restore HDR peaks from the base image after the luma
  graft, with `--hdr-restore-threshold` and `--hdr-restore-transition`.

Occi metrics before the HDR-safe patch, still useful for tradeoff direction:

| candidate | flat luma | luma p99 | luma visible | flat chroma | chroma visible | edge | residual luma visible | blue-structure blue p999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v12 | 0.250 | 0.657 | 0.236 | 0.071 | 0.058 | 0.884 | 0.090 | 0.278 |
| raw graft | 0.232 | 0.582 | 0.220 | 0.070 | 0.057 | 0.774 | 0.078 | 0.238 |
| SCUNet graft | 0.237 | 0.613 | 0.224 | 0.070 | 0.057 | 0.837 | 0.078 | 0.236 |

Visual check after HDR restore/safe patch:

- HDR highlight crushing is fixed.
- Hair/face improvements are subtle at best.
- The edge-retention drop says the graft is changing low/mid luma shape more
  than it is restoring useful crisp detail.

Decision:

- Do not promote structure graft after v12.
- Keep the HDR-safe script patch because the old behavior was a real
  experimental footgun.
- Next direction should be a v12-native learned branch/gate, not a global
  post-graft. Existing learned scripts still point at older v4/v7/v8/v9
  intermediate outputs, so the recipe needs to be re-based on current v12
  inputs before any longer training.

### Stage13 v55: v12-native learned flat cleanup gate pilot

Goal:

- Re-base the learned flat cleanup gate on the current v12 outputs instead of
  old luma-detail-gate/rebuilder intermediates.
- Learn only "where to smooth", then use deterministic luma/chroma smoothing.
- Keep HDR-safe output by restoring HDR peaks from the current v12 image.

Implementation changes:

- `scripts/train_flat_cleanup_gate.py`
  - now uses `scunet_preset_chooser_v12_flat_cleanup_auto_outputs` as current
    inputs,
  - supports missing detail-gate files by using a zero gate,
  - adds Ice as a training/application scene,
  - restores HDR peaks after the learned-gate smoother.
- `scripts/train_flat_cleanup_branch.py`
  - updated for the new `apply_cleanup` signature,
  - restored HDR peaks after predicted RGB residual output.

Pilot v1:

```text
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_v12_native_pilot_v1/
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_v12_native_pilot_v1_outputs/
```

Training:

- CPU, 360 steps, 3 scenes, width 20, 3 blocks.
- Final step: `loss=0.024131`, `pred_mean=0.2931`,
  `target_mean=0.2519`.

Application speed:

| scene | tiles | elapsed |
| --- | ---: | ---: |
| Dance | 35 | 43.5 sec |
| Ice | 35 | 43.9 sec |
| Occi | 77 | 104.8 sec |

Metrics:

| scene | candidate | flat luma | luma p99 | luma visible | flat chroma | chroma visible | edge | speckle luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dance | v12 | 0.134 | 0.438 | 0.125 | 0.041 | 0.035 | 0.963 | 0.056 | 0.018 | 0.077 |
| Dance | gate_v1 s1.0 | 0.116 | 0.404 | 0.107 | 0.035 | 0.030 | 0.862 | 0.045 | 0.015 | 0.057 |
| Dance | gate_v1 s0.6 | 0.123 | 0.416 | 0.114 | 0.038 | 0.032 | 0.902 | 0.049 | 0.016 | 0.065 |
| Dance | gate_v1 s0.4 | 0.127 | 0.423 | 0.118 | 0.039 | 0.033 | 0.922 | 0.051 | 0.017 | 0.069 |
| Ice | v12 | 0.214 | 0.541 | 0.200 | 0.069 | 0.056 | 0.463 | 0.089 | 0.042 | 0.253 |
| Ice | gate_v1 s1.0 | 0.193 | 0.502 | 0.179 | 0.060 | 0.047 | 0.414 | 0.073 | 0.034 | 0.231 |
| Ice | gate_v1 s0.6 | 0.201 | 0.516 | 0.187 | 0.064 | 0.050 | 0.434 | 0.079 | 0.037 | 0.237 |
| Ice | gate_v1 s0.4 | 0.205 | 0.524 | 0.191 | 0.065 | 0.052 | 0.444 | 0.082 | 0.039 | 0.240 |
| Occi | v12 | 0.250 | 0.657 | 0.236 | 0.071 | 0.058 | 0.884 | 0.090 | 0.040 | 0.278 |
| Occi | gate_v1 s1.0 | 0.220 | 0.590 | 0.205 | 0.063 | 0.051 | 0.825 | 0.072 | 0.033 | 0.263 |
| Occi | gate_v1 s0.6 | 0.231 | 0.616 | 0.217 | 0.066 | 0.054 | 0.849 | 0.079 | 0.036 | 0.265 |
| Occi | gate_v1 s0.4 | 0.237 | 0.629 | 0.223 | 0.068 | 0.055 | 0.861 | 0.083 | 0.037 | 0.266 |

Pilot v2:

```text
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_v12_native_pilot_v2/
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_v12_native_pilot_v2_outputs/
```

- Added stronger auto-detail protection to the target and higher
  `target-edge-suppress`.
- Final step: `loss=0.023558`, `pred_mean=0.2011`,
  `target_mean=0.2048`.
- v2 reduced gate means but did not beat v1 s0.4/s0.6 on the main tradeoff:
  it still lost similar edge retention while cleaning a bit more.

Decision:

- Learned v12-native gate is a valid direction: it reduces luma/chroma/speckle
  metrics more than hand v12.
- Strength `1.0` is too aggressive and visibly/quantitatively risks softness.
- Current best compromise is `gate_v1` reused at strength `0.4`:
  - clear residual noise reduction,
  - edge loss much smaller than full-strength learned gate,
  - crop checks on Occi hair, Dance sky, and Ice branch do not show obvious new
    breakage.
- This is not "perfect" yet. The next improvement should be scene/region-aware
  strength, not another global stronger gate:
  - high strength for flat sky/shadow,
  - low or zero strength on hair/branches/fabric,
  - preserve v12 as fallback where the learned gate is uncertain.

### Stage13 v56: region-aware strength map for learned flat gate

Goal:

- Keep the useful learned flat gate from v55, but avoid global softening.
- Replace one scalar strength with a spatial strength map:
  - stronger on flat low-saturation/dark regions,
  - weaker on coherent structure, texture, skin, and HDR highlights.

Implementation:

```text
scripts/apply_region_aware_flat_gate.py
```

Inputs:

- reference: original noisy EXR,
- input: integrated v12 output,
- gate: `flat_cleanup_gate_v12_native_pilot_v1` learned gate PNG.

The script writes:

- EXR output,
- preview PNG,
- mask PNGs for `strength`, `effective_gate`, `structure_protect`,
  `flat_target`, `skin`, `highlight`, and related diagnostics,
- JSON metadata with all parameters and stats.

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v1_outputs/
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v2_outputs/
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v3_outputs/
```

Main parameter direction:

| version | intent |
| --- | --- |
| region_v1 | conservative structure protection; nearly v12 edge retention, weak cleanup |
| region_v2 | slightly stronger flat opening; still conservative |
| region_v3 | strong flat-only opening with very high structure suppression |

Region v3 parameters:

```text
base_strength=0.16
flat_boost=1.05
skin_strength=0.08
structure_suppress=0.98
highlight_suppress=0.88
flat_threshold=0.044
flat_transition=0.018
edge_threshold=0.034
edge_transition=0.014
min_strength=0.02
max_strength=1.00
blur_sigma=1.05
```

Metrics:

| scene | candidate | flat luma | luma p99 | luma visible | flat chroma | chroma visible | edge | speckle luma visible | neutral chroma visible | blue-structure blue p999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dance | v12 | 0.134 | 0.438 | 0.125 | 0.041 | 0.035 | 0.963 | 0.056 | 0.018 | 0.077 |
| Dance | s040 | 0.127 | 0.423 | 0.118 | 0.039 | 0.033 | 0.922 | 0.051 | 0.017 | 0.069 |
| Dance | region_v2 | 0.128 | 0.432 | 0.119 | 0.039 | 0.033 | 0.959 | 0.053 | 0.017 | 0.076 |
| Dance | region_v3 | 0.126 | 0.431 | 0.117 | 0.038 | 0.033 | 0.960 | 0.052 | 0.017 | 0.075 |
| Ice | v12 | 0.214 | 0.541 | 0.200 | 0.069 | 0.056 | 0.463 | 0.089 | 0.042 | 0.253 |
| Ice | s040 | 0.205 | 0.524 | 0.191 | 0.065 | 0.052 | 0.444 | 0.082 | 0.039 | 0.240 |
| Ice | region_v2 | 0.208 | 0.536 | 0.194 | 0.067 | 0.053 | 0.461 | 0.085 | 0.040 | 0.245 |
| Ice | region_v3 | 0.207 | 0.535 | 0.193 | 0.066 | 0.053 | 0.461 | 0.084 | 0.039 | 0.245 |
| Occi | v12 | 0.250 | 0.657 | 0.236 | 0.071 | 0.058 | 0.884 | 0.090 | 0.040 | 0.278 |
| Occi | s040 | 0.237 | 0.629 | 0.223 | 0.068 | 0.055 | 0.861 | 0.083 | 0.037 | 0.266 |
| Occi | region_v2 | 0.239 | 0.649 | 0.225 | 0.068 | 0.055 | 0.883 | 0.084 | 0.038 | 0.268 |
| Occi | region_v3 | 0.237 | 0.647 | 0.222 | 0.068 | 0.055 | 0.883 | 0.083 | 0.037 | 0.268 |

Visual check:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_region_aware_flat_gate_v3_occi/
runs/refiner_pilot_stage11_hybrid_best/compare_region_aware_flat_gate_v3_dance/
runs/refiner_pilot_stage11_hybrid_best/compare_region_aware_flat_gate_v3_ice/
```

- Occi face/hair: region_v3 is very close to v12 detail, unlike scalar s040,
  while still reducing a little residual grain.
- Dance sky: region_v3 is cleaner than v12 but not as flat as scalar s040.
- Ice branch/blue shadow: branch structure stays close to v12; blue shadow
  noise improves modestly.
- Cloth/fabric: region_v3 avoids the obvious extra softening seen in scalar
  learned-gate outputs.

Decision:

- Promote `region_v3` as the next best candidate after v12.
- It does not give the dramatic noise drop of scalar `s040`, but the edge
  retention is much better and matches the quality priority.
- This is a safer basis for the next round than training another stronger
  global gate.
- Next direction:
  - integrate region-aware strength into the chooser as an optional final
    stage,
  - then tune per-scene presets: stronger for Dance sky, cautious for Ice,
    moderate for Occi.


### Stage13 v57: dark-flat cleanup for shadow sky

After the project/data move, paths were updated for the new layout:

```text
project: /Users/uniuyuni/PythonProjects/nagi_denoise
test photos: /Users/uniuyuni/ProjectData/test_photos
```

Problem focus: denoise quality is good, but dark flat regions such as Dance sky
still show visible grain. We should push those flat/shadow areas cleaner while
not adding more global softness.

Implementation updates:

- Added `shadow_flat_boost` to `scripts/apply_region_aware_flat_gate.py`.
  It only opens extra cleanup where the region is simultaneously flat,
  low-saturation, and dark.
- Added CLI overrides for deterministic smoothing strength/sigma so dark-flat
  presets can widen the smoothing kernel without changing the learned gate.
- Updated `scripts/train_flat_cleanup_gate.py` to use
  `/Users/uniuyuni/ProjectData/test_photos`.

Dance candidates from v12 + learned flat gate:

```text
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v4_darkclean045_outputs/
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v4_darkclean075_outputs/
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v4_darkclean075_wide_outputs/
```

Main parameters:

| candidate | shadow_flat_boost | base | flat_boost | edge_threshold | luma_sigma | chroma_sigma | intent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dark045 | 0.45 | 0.14 | 1.05 | 0.032 | 2.75 | 3.45 | mild dark-flat cleanup |
| dark075 | 0.75 | 0.12 | 1.02 | 0.031 | 2.75 | 3.45 | stronger dark sky cleanup |
| wide | 0.75 | 0.12 | 1.02 | 0.031 | 4.00 | 5.00 | smoother broad flat tone |

Lightweight ROI metrics on Dance:

| ROI | candidate | luma impulse p99 | chroma impulse p99 | luma std | chroma std | mean y |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sky_center | v12 | 0.005071 | 0.001307 | 0.010081 | 0.017663 | 0.181615 |
| sky_center | region_v3 | 0.004713 | 0.001221 | 0.009800 | 0.017491 | 0.181619 |
| sky_center | dark075 | 0.004565 | 0.001188 | 0.009692 | 0.017427 | 0.181621 |
| sky_center | wide | 0.004578 | 0.001199 | 0.009533 | 0.017274 | 0.181615 |
| sky_existing | v12 | 0.006125 | 0.001330 | 0.011126 | 0.015914 | 0.155574 |
| sky_existing | region_v3 | 0.005688 | 0.001240 | 0.010811 | 0.015733 | 0.155579 |
| sky_existing | dark075 | 0.005516 | 0.001204 | 0.010686 | 0.015662 | 0.155582 |
| sky_existing | wide | 0.005517 | 0.001214 | 0.010512 | 0.015498 | 0.155577 |
| dancer_center | v12 | 0.003829 | 0.001284 | 0.027399 | 0.025621 | 0.258598 |
| dancer_center | region_v3 | 0.003589 | 0.001193 | 0.027307 | 0.025498 | 0.258614 |
| dancer_center | dark075 | 0.003503 | 0.001162 | 0.027278 | 0.025460 | 0.258620 |
| dancer_center | wide | 0.003513 | 0.001181 | 0.027199 | 0.025354 | 0.258629 |

Crop outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_region_aware_flat_gate_v4_darkclean075_dance/
runs/refiner_pilot_stage11_hybrid_best/compare_region_aware_flat_gate_v4_darkclean_wide_dance/
```

Decision:

- `dark075` is the current best Dance/dark-sky candidate: it lowers both luma
  and chroma impulses more than region_v3 without changing mean brightness.
- `wide` makes broad flat tone smoother (`luma_std/chroma_std` down), but its
  point impulse metrics are slightly worse than `dark075`; keep it as a broad
  flat-area option, not the default.
- Next integration should be scene/region aware:
  - Dance-like dark sky: use `dark075`.
  - General detail-heavy portraits/hair: stay closer to `region_v3`.
  - Very broad flat backgrounds where slight softness is acceptable: consider
    `wide`.
- Full residual-speckle evaluation was intentionally skipped/interrupted while
  the user's background task was heavy; use ROI metrics and crops for now.


### Stage13 v58: chooser integration for dark-flat region gate

Integrated the Stage13 v57 dark-flat cleanup into the main SCUNet preset chooser
as an optional final stage.

Code changes:

- `scripts/apply_scunet_preset_chooser.py`
  - imports `apply_region_aware_gate` and gate PNG loading from
    `scripts/apply_region_aware_flat_gate.py`,
  - adds `REGION_AWARE_FLAT_GATE_PRESETS`:
    - `quality_v3`: prior region-aware v3 behavior,
    - `dark_sky`: current best Dance/dark-sky cleanup,
    - `dark_sky_wide`: broader smoothing for very flat backgrounds,
  - adds CLI:
    - `--region-aware-flat-gate-preset {off,auto,quality_v3,dark_sky,dark_sky_wide}`,
    - `--region-aware-flat-gate-dir`,
  - `auto` currently applies `dark_sky` only for `sky_luma`; detail-heavy
    portraits remain unchanged by default.
- `scripts/train_scunet_selector.py` now points `TEST_PHOTOS` at
  `/Users/uniuyuni/ProjectData/test_photos`, matching the moved data folder.

Verification:

- CLI help shows the new region-aware options.
- `train_scunet_selector.SCENES["k5_dance"].noisy` resolves to
  `/Users/uniuyuni/ProjectData/test_photos/K-5 Dance noisy.EXR`.
- A 32x32 synthetic smoke test of `apply_region_aware_flat_gate_filter(...,
  "dark_sky")` returned `(32, 32, 3) float32`, read the gate, and reported
  `smooth_luma_sigma=2.75`.

Attempted full integrated Dance output:

```text
scripts/apply_scunet_preset_chooser.py   --scene k5_dance   --force-preset sky_luma   --... existing auto postfilters ...   --region-aware-flat-gate-preset dark_sky
```

This was intentionally interrupted after about two minutes because the user's
background task was heavy. The interruption occurred inside the pre-existing
selector feature `median_filter`, before the new region-aware final stage.
Therefore the integration code is smoke-tested, but the full v13 integrated EXR
should be generated later when the machine is free.

Current recommendation:

- Use `dark_sky` for Dance-like dark sky scenes.
- Keep `quality_v3` or no region-aware final gate for hair/detail-heavy images
  until per-scene visual checks prove the stronger cleanup is safe.
- Avoid making `dark_sky_wide` the default; it lowers broad flat std but does not
  beat `dark_sky` on point impulse metrics.


### Stage13 v59: precomputed-base chooser path and all-scene region-gate verification

The full chooser remained too heavy while the user's background task was active:
it spends significant time in the pre-existing full-frame selector feature
`median_filter`. To keep iteration moving without changing quality behavior,
added a lightweight path to the chooser:

```text
--precomputed-base <existing-base.exr>
```

Behavior:

- still resolves the scene and computes scene metrics from noisy/current/SCUNet,
- skips selector/policy inference,
- uses the supplied EXR as `out`,
- then applies the requested postfilters/final region-aware flat gate,
- records `model_kind: precomputed_base` and the source path in metadata.

This allows rapid validation of final-stage design on top of current v12 outputs.
It is not a replacement for full end-to-end inference; it is a practical cached
base path for evaluation and packaging.

Generated outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v13_region_darkflat_precomputed_outputs/
  k5_dance_scunet_preset_chooser_v13_region_darkflat_precomputed.exr

runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v13_region_quality_precomputed_outputs/
  k5_ice_scunet_preset_chooser_v13_region_quality_precomputed.exr
  xt5_occi_scunet_preset_chooser_v13_region_quality_precomputed.exr
```

Equivalence checks against standalone region-aware outputs:

| scene | integrated preset | standalone reference | max abs diff | mean abs diff | p99 abs diff |
| --- | --- | --- | ---: | ---: | ---: |
| Dance | `dark_sky` | `k5_dance_region_aware_flat_gate_v4_darkclean075.exr` | 0.0 | 0.0 | 0.0 |
| Ice | `quality_v3` | `k5_ice_region_aware_flat_gate_v3.exr` | 0.0 | 0.0 | 0.0 |
| Occi | `quality_v3` | `xt5_occi_region_aware_flat_gate_v3.exr` | 0.0 | 0.0 | 0.0 |

Dance ROI metrics through the integrated precomputed path are identical to the
standalone `dark075` candidate:

| ROI | candidate | luma impulse p99 | chroma impulse p99 | luma std | chroma std | mean y |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sky_center | v12 | 0.005071 | 0.001307 | 0.010081 | 0.017663 | 0.181615 |
| sky_center | integrated dark_sky | 0.004565 | 0.001188 | 0.009692 | 0.017427 | 0.181621 |
| sky_existing | v12 | 0.006125 | 0.001330 | 0.011126 | 0.015914 | 0.155574 |
| sky_existing | integrated dark_sky | 0.005516 | 0.001204 | 0.010686 | 0.015662 | 0.155582 |
| dancer_center | v12 | 0.003829 | 0.001284 | 0.027399 | 0.025621 | 0.258598 |
| dancer_center | integrated dark_sky | 0.003503 | 0.001162 | 0.027278 | 0.025460 | 0.258620 |

Decision:

- The chooser integration is now verified for the cached-base path.
- `auto` should remain conservative: `dark_sky` only for `sky_luma`, while
  detail-heavy scenes use `quality_v3` or skip the final region-aware gate.
- Next practical step when the machine is free: run the full chooser end-to-end
  once to verify that uncached output plus `--region-aware-flat-gate-preset auto`
  reaches the same final stage behavior.
- For research direction, continue improving the flat/structure classifier rather
  than globally increasing smoothing; `dark_sky` works because it is restricted
  to flat, dark, low-saturation regions.


### Stage13 v60: ROI evaluator and stricter dark-sky auto preset

Added a lightweight ROI-only evaluator:

```text
scripts/roi_noise_eval.py
```

Purpose:

- Full residual-speckle evaluation is expensive on large EXRs and can interfere
  with background work.
- ROI evaluation reads only candidate EXRs and computes local impulse/std/contrast
  metrics on the important visual failure regions.
- It outputs both JSON and Markdown.

Generated ROI evaluations:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v13_region_darkflat_dance/
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v13_region_quality_ice/
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v13_region_quality_occi/
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v5_darkclean_strict_dance/
```

Finding from ROI eval:

- `dark_sky` improves Dance sky more than `region_v3`, but it also reduces
  local contrast in `dancer_center` and `house_detail` more than desired.
- This confirmed the user's earlier sharpness concern: the cleanup was still
  slightly too broad for midtone detail regions.

Strict dark-sky redesign:

- Added `shadow_luma_threshold` and `shadow_luma_transition` to
  `scripts/apply_region_aware_flat_gate.py`.
- New strict candidate:

```text
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v5_darkclean_strict_outputs/
  k5_dance_region_aware_flat_gate_v5_darkclean_strict.exr
```

Strict parameters relative to `dark_sky`:

```text
shadow_luma_threshold=0.22
shadow_luma_transition=0.06
```

Dance ROI ratios vs v12:

| ROI | candidate | luma p99 | chroma p99 | magenta p999 | blue p999 | contrast |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sky_existing | region_v3 | 0.929 | 0.932 | 0.938 | 0.956 | 0.917 |
| sky_existing | dark_sky | 0.900 | 0.905 | 0.902 | 0.961 | 0.883 |
| sky_existing | strict | 0.908 | 0.913 | 0.910 | 0.956 | 0.891 |
| sky_center | region_v3 | 0.929 | 0.934 | 0.939 | 0.963 | 0.915 |
| sky_center | dark_sky | 0.900 | 0.909 | 0.927 | 0.951 | 0.882 |
| sky_center | strict | 0.912 | 0.920 | 0.935 | 0.952 | 0.894 |
| dancer_center | region_v3 | 0.937 | 0.929 | 0.969 | 0.937 | 0.916 |
| dancer_center | dark_sky | 0.915 | 0.904 | 0.958 | 0.921 | 0.887 |
| dancer_center | strict | 0.938 | 0.930 | 0.967 | 0.939 | 0.914 |
| house_detail | region_v3 | 0.947 | 0.935 | 0.964 | 0.954 | 0.935 |
| house_detail | dark_sky | 0.932 | 0.912 | 0.951 | 0.952 | 0.912 |
| house_detail | strict | 0.950 | 0.936 | 0.964 | 0.957 | 0.934 |

Decision:

- Promote `dark_sky_strict` as the `auto` region-aware final gate for
  `sky_luma` scenes.
- Keep the older `dark_sky` as a manual aggressive preset for cases where the
  user prefers maximum sky cleanup and accepts a little detail softness.
- `dark_sky_strict` keeps the important Dance sky gain while restoring
  dancer/house detail behavior almost exactly to `region_v3`.

Chooser integration:

- Added `dark_sky_strict` to `REGION_AWARE_FLAT_GATE_PRESETS`.
- `choose_region_aware_flat_gate_preset("sky_luma", "auto")` now returns
  `dark_sky_strict`.
- Generated cached-base auto output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v14_region_darkflat_strict_precomputed_outputs/
  k5_dance_scunet_preset_chooser_v14_region_darkflat_strict_precomputed.exr
```

Equivalence check:

| integrated output | standalone reference | max abs diff | mean abs diff | p99 abs diff |
| --- | --- | ---: | ---: | ---: |
| v14 auto strict | v5 standalone strict | 0.0 | 0.0 | 0.0 |

Next direction:

- Use ROI evaluator as the default quick gate before visual crop checks.
- Continue improving the flat/structure classifier; global smoothing is not the
  right direction because it directly causes the sharpness loss the user sees.
- Once the machine is free, run full chooser end-to-end with
  `--region-aware-flat-gate-preset auto` to validate the uncached path.

## Stage13 v61 - ROI evaluator scoring restored after project move

After the project/data folder move, the executable `scripts/roi_noise_eval.py` in the
new project directory was still the older metric-only variant. Restored the current
ROI scoring form with ROI kinds, per-ROI score, and summary score. This makes the
quick evaluator match the Dance/Ice decision logic again.

Updated runtime data paths in the remaining experiment/training scripts from:

```text
/Users/uniuyuni/PythonProjects/test_photos
```

to:

```text
/Users/uniuyuni/ProjectData/test_photos
```

Light checks:

```text
pixi run python -B -m py_compile scripts/roi_noise_eval.py
pixi run python -B -m py_compile scripts/apply_region_aware_flat_gate.py scripts/apply_scunet_preset_chooser.py scripts/roi_noise_eval.py scripts/train_flat_cleanup_gate.py scripts/train_scunet_selector.py scripts/refiner_pilot.py
```

Occi fixed-score ROI evaluation:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v14_score_occi/
```

Summary score, lower is better:

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| v12 | 1.000 | 4 |
| v13_quality | 1.016 | 4 |

Key Occi ratios vs v12:

| ROI | kind | candidate | score | luma p99 | chroma p99 | magenta p999 | blue p999 | contrast |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hair_detail | detail | v13_quality | 1.017 | 0.990 | 1.000 | 0.981 | 1.004 | 0.986 |
| root | detail | v13_quality | 1.030 | 0.983 | 0.998 | 0.999 | 1.001 | 0.978 |
| face_center | skin | v13_quality | 1.019 | 0.991 | 0.996 | 1.002 | 1.004 | 0.985 |
| noise_dark | flat | v13_quality | 0.997 | 0.989 | 0.996 | 0.997 | 0.997 | 0.982 |

Decision:

- Do not auto-apply `region_quality` to hair/detail/skin scenes. Occi confirms
  the same failure mode as Ice: small noise reduction is outweighed by local
  contrast/detail loss.
- Keep `dark_sky_strict` as the only automatic final region-aware gate for
  `sky_luma` scenes.
- The next design step should improve flat/structure classification rather than
  making the final smoothing broader. Flat and dark-sky cleanup is desirable,
  but it must not leak into hair, branches, people, or other readable detail.

## Stage13 v62 - Lightweight ROI gate leakage diagnostics

Added a lightweight design probe:

```text
scripts/eval_region_gate_roi.py
```

Purpose: evaluate the learned flat-cleanup gate and region-aware strength map on
fixed ROIs without writing denoised EXRs. The first implementation accidentally
built full-frame strength maps and was stopped because it was too heavy while
background tasks were running. The script now computes `build_strength_map()`
only on ROI crops.

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/region_gate_roi_eval_v1/
  k5_dance_region_gate_roi_eval.md
  xt5_occi_region_gate_roi_eval.md
  k5_ice_region_gate_roi_eval.md
```

Dance effective gate means:

| ROI | kind | quality_v3 | dark_sky | dark_sky_strict |
| --- | --- | ---: | ---: | ---: |
| sky_existing | flat | 0.1425 | 0.1997 | 0.1875 |
| sky_center | flat | 0.1483 | 0.2050 | 0.1857 |
| dancer_center | detail | 0.1587 | 0.2120 | 0.1630 |
| house_detail | detail | 0.1384 | 0.1849 | 0.1407 |
| snow_ground | flat | 0.0458 | 0.0504 | 0.0386 |

Interpretation: `dark_sky` leaks strongly into Dance detail ROIs.
`dark_sky_strict` keeps most of the sky gain while bringing dancer/house detail
back near `quality_v3`, which explains the better ROI score balance.

Occi effective gate means:

| ROI | kind | quality_v3 | dark_sky_strict |
| --- | --- | ---: | ---: |
| hair_detail | detail | 0.1120 | 0.1240 |
| root | detail | 0.1235 | 0.1110 |
| face_center | skin | 0.1160 | 0.1256 |
| noise_dark | flat | 0.0561 | 0.0535 |

Occi p95 note: `dark_sky_strict` raises hair/face effective p95
(`hair_detail 0.3090 -> 0.3935`, `face_center 0.3028 -> 0.3855`). This
confirms it must not auto-apply to hair/skin scenes.

Ice effective gate means:

| ROI | kind | quality_v3 | dark_sky_strict |
| --- | --- | ---: | ---: |
| blue_shadow | mixed | 0.1178 | 0.1484 |
| ice_branch | detail | 0.0905 | 0.0979 |
| dark_edge | detail | 0.0877 | 0.0911 |
| flat_blue | flat | 0.1098 | 0.1359 |

Ice confirms the same rule: `dark_sky_strict` is useful for sky-like scenes but
not a general-purpose cleanup preset. It raises branch/detail p95 as well.

Design decision:

- Keep `choose_region_aware_flat_gate_preset(auto)` conservative: only
  `sky_luma -> dark_sky_strict`; detail/hair/skin/blue-shadow scenes remain off.
- The next learning target should not be broader smoothing. It should produce a
  sharper flat-vs-structure gate: open in true dark sky/flat fields, close more
  aggressively on coherent lines, hair, branches, faces, and mixed blue-shadow
  regions.
- Candidate training knobs already exist for that direction:
  `--target-detail-suppress`, `--target-edge-suppress`, and ROI bias. The next
  pilot should use these instead of raising final smoothing strength.

## Stage13 v63 - Detail-strict flat gate pilot v3

Before launching heavier full-frame output, the ROI gate evaluator was extended
with checkpoint inference:

```text
scripts/eval_region_gate_roi.py --checkpoint <flat_cleanup_gate_final.pt>
```

This allows evaluating a learned gate on ROI crops without first generating
full-size gate PNGs or denoised EXRs. A sanity check against the existing v2 gate
PNG matched closely on Dance, so this path is suitable for quick design tests.

Existing v2 training recipe:

```text
target_detail_suppress=0.25
target_edge_suppress=0.35
roi_bias_strength=0.55
steps=420
```

New short v3 pilot:

```text
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_v12_native_pilot_v3_detail_strict/

target_detail_suppress=0.45
target_edge_suppress=0.50
roi_bias_strength=0.95
steps=320
CPU, width=20, blocks=3
```

Training completed in about 0.29s/it after warmup. Final log line:

```text
step 00320/320 loss=0.036898 gate=0.036301 smooth=0.000598 pred_mean=0.2164 target_mean=0.2001
```

ROI checkpoint diagnostics:

```text
runs/refiner_pilot_stage11_hybrid_best/region_gate_roi_eval_v3_detail_strict_ckpt/
```

Dance, `dark_sky_strict`, effective gate mean/p95, v1/v2/v3:

| ROI | kind | mean v1 | mean v2 | mean v3 | p95 v1 | p95 v2 | p95 v3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sky_existing | flat | 0.1875 | 0.1616 | 0.1587 | 0.3247 | 0.2942 | 0.2649 |
| sky_center | flat | 0.1857 | 0.1594 | 0.1544 | 0.3213 | 0.2903 | 0.2570 |
| dancer_center | detail | 0.1630 | 0.1386 | 0.1284 | 0.2712 | 0.2423 | 0.2075 |
| house_detail | detail | 0.1407 | 0.1176 | 0.1112 | 0.2607 | 0.2304 | 0.2032 |
| snow_ground | flat | 0.0386 | 0.0287 | 0.0252 | 0.0914 | 0.0730 | 0.0566 |

Occi, `quality_v3`, effective gate p95, v1/v2/v3:

| ROI | kind | p95 v1 | p95 v2 | p95 v3 |
| --- | --- | ---: | ---: | ---: |
| hair_detail | detail | 0.3090 | 0.2893 | 0.2457 |
| root | detail | 0.2563 | 0.2307 | 0.1842 |
| face_center | skin | 0.3028 | 0.2826 | 0.2413 |
| noise_dark | flat | 0.1882 | 0.1646 | 0.1391 |

Ice, `quality_v3`, effective gate p95, v1/v2/v3:

| ROI | kind | p95 v1 | p95 v2 | p95 v3 |
| --- | --- | ---: | ---: | ---: |
| blue_shadow | mixed | 0.2221 | 0.1985 | 0.1806 |
| ice_branch | detail | 0.1885 | 0.1641 | 0.1494 |
| dark_edge | detail | 0.1846 | 0.1634 | 0.1460 |
| flat_blue | flat | 0.2066 | 0.1825 | 0.1670 |

Interpretation:

- v3 does reduce detail/skin/branch p95 more than v2, so the stronger
  detail/edge suppression is moving in the intended direction.
- It also lowers sky/flat gates, so it is not a clean win yet. It is safer but
  may under-clean flat areas.
- For Dance with `dark_sky_strict`, the separation is better than v2:
  detail ROIs fall more than sky ROIs, while sky still has useful effective
  gate.

Decision:

- Do not promote v3 directly to default yet.
- Use v3 as the current "safe detail gate" candidate for a single Dance visual
  output when the machine is free.
- Next design improvement should explicitly preserve/open true sky-flat ROIs
  while keeping the stronger detail/edge suppression. That likely needs either
  stronger sky-specific ROI target bias than the current fixed `ROI_BIAS`, or a
  sky/flat semantic feature in the gate target rather than only global
  detail/edge suppression.

## Stage13 v64 - Sky-open attempts v4/v5/v6

Goal after v3: recover more sky/true-flat gate while keeping the stronger
detail/edge suppression.

Code changes:

- Added `ROI_SUPPRESS_SCALE` to `scripts/train_flat_cleanup_gate.py` so sky/flat
  ROIs can reduce the global detail/edge suppression while hair/branch/skin ROIs
  strengthen it.
- Added `ROI_WEIGHT_MUL` to emphasize selected ROI targets during training.
- Added a new `sky_flat_hint` feature channel: `flat_hint * low_sat * shadow_hint`.
  This raises `FEATURE_CHANNELS` from 22 to 23.
- Fixed checkpoint compatibility: `FlatCleanupGate` now stores/uses the checkpoint
  feature-channel count, and `align_feature_channels()` drops the new
  `sky_flat_hint` channel when evaluating older 22-channel checkpoints. A v2
  checkpoint compatibility probe succeeded.

Pilots:

| pilot | change | result |
| --- | --- | --- |
| v4 | ROI suppress scale, same strong detail/edge suppression | did not recover sky; safer but too closed |
| v5 | stronger sky/flat ROI training weight | recovered sky but also reopened detail/skin/branch too much |
| v6 | moderate ROI weight + `sky_flat_hint` feature | close to v3, slightly safer in some ROIs, still did not recover sky |

Dance `dark_sky_strict`, effective gate mean/p95:

| ROI | kind | v2 | v3 | v5 | v6 |
| --- | --- | ---: | ---: | ---: | ---: |
| sky_existing | flat | 0.1616 / 0.2942 | 0.1587 / 0.2649 | 0.1722 / 0.2954 | 0.1569 / 0.2672 |
| sky_center | flat | 0.1594 / 0.2903 | 0.1544 / 0.2570 | 0.1710 / 0.2921 | 0.1534 / 0.2603 |
| dancer_center | detail | 0.1386 / 0.2423 | 0.1284 / 0.2075 | 0.1499 / 0.2454 | 0.1286 / 0.2108 |
| house_detail | detail | 0.1176 / 0.2304 | 0.1112 / 0.2032 | 0.1297 / 0.2368 | 0.1110 / 0.2048 |

Occi `dark_sky_strict`, effective gate mean/p95:

| ROI | kind | v2 | v3 | v5 | v6 |
| --- | --- | ---: | ---: | ---: | ---: |
| hair_detail | detail | 0.1092 / 0.3681 | 0.0968 / 0.3141 | 0.1117 / 0.3486 | 0.0965 / 0.3149 |
| root | detail | 0.0938 / 0.2079 | 0.0834 / 0.1672 | 0.1011 / 0.2056 | 0.0836 / 0.1703 |
| face_center | skin | 0.1106 / 0.3594 | 0.0979 / 0.3093 | 0.1123 / 0.3423 | 0.0973 / 0.3096 |

Ice `dark_sky_strict`, effective gate mean/p95:

| ROI | kind | v2 | v3 | v5 | v6 |
| --- | --- | ---: | ---: | ---: | ---: |
| blue_shadow | mixed | 0.1252 / 0.2576 | 0.1255 / 0.2365 | 0.1371 / 0.2625 | 0.1239 / 0.2372 |
| ice_branch | detail | 0.0797 / 0.1913 | 0.0810 / 0.1803 | 0.0905 / 0.2013 | 0.0800 / 0.1799 |
| flat_blue | flat | 0.1130 / 0.2323 | 0.1143 / 0.2133 | 0.1249 / 0.2376 | 0.1127 / 0.2138 |

Decision:

- v5 is rejected: it opens sky but also reopens details, undoing the safety gain.
- v4 is rejected: it is safer but closes sky/flat further.
- v6 is not a breakthrough; it behaves almost like v3. Keep v3/v6 as safe-gate
  candidates, with v3 slightly simpler and v6 preserving compatibility with the
  new sky feature path.
- The current learned gate architecture is struggling to recover sky openness
  without broad reopening. The next useful step is likely not another tiny
  weight tweak, but an explicit two-factor policy: keep the safe learned detail
  gate, then apply a deterministic sky/flat reopen multiplier only where the
  region-aware sky mask is high and structure protection is low. This can be
  tested on ROI gates before generating full images.

## Stage13 v65 - Manual dark-sky reopen preset

Implemented the ROI-proven two-factor policy as a manual preset, not as auto:

```text
dark_sky_strict_reopen
```

Code changes:

- `scripts/apply_region_aware_flat_gate.py` now has `build_reopen_map()` and
  optional `reopen_*` parameters. Defaults keep previous behavior unchanged
  (`reopen_strength=0`).
- `apply_region_aware_gate()` multiplies the effective gate by the reopen map
  and records `reopen_mean` / `reopen_p95` in stats.
- CLI options were added for manual experiments:
  `--reopen-strength`, `--reopen-shadow-weight`,
  `--reopen-structure-suppress`, `--reopen-min`, `--reopen-max`,
  `--reopen-shadow-threshold`, and `--reopen-shadow-transition`.
- `scripts/apply_scunet_preset_chooser.py` now exposes manual preset
  `dark_sky_strict_reopen`. `auto` is intentionally unchanged: only
  `sky_luma -> dark_sky_strict`, not reopen.
- `scripts/eval_region_gate_roi.py` can evaluate reopen-enabled presets
  directly.

Preset parameters:

```text
reopen_strength=0.85
reopen_shadow_weight=1.0
reopen_structure_suppress=1.0
reopen_min=1.0
reopen_max=1.45
reopen_shadow_threshold=0.40
reopen_shadow_transition=0.12
```

ROI validation with v3 safe gate:

```text
runs/refiner_pilot_stage11_hybrid_best/region_gate_roi_eval_v7_reopen_preset_check/
```

Dance `dark_sky_strict_reopen` effective gate:

| ROI | kind | reopen mean | effective mean | effective p95 |
| --- | --- | ---: | ---: | ---: |
| sky_existing | flat | 1.1343 | 0.1842 | 0.3465 |
| sky_center | flat | 1.1094 | 0.1749 | 0.3210 |
| dancer_center | detail | 1.0000 | 0.1284 | 0.2075 |
| house_detail | detail | 1.0033 | 0.1118 | 0.2054 |
| snow_ground | flat | 1.0000 | 0.0252 | 0.0566 |

Interpretation:

- This recovers the dark-sky gate strength lost by v3 while keeping dancer and
  house detail essentially at the safe v3 level.
- Occi/Ice ROI probes showed this reopen can fire outside Dance-like sky scenes,
  so it must remain a manual/sky_luma-only candidate until visual full-frame
  checks prove otherwise.

Next step when the machine is available: generate one Dance output with
`dark_sky_strict_reopen` and compare against v12, strict, and dark_sky crops.

## Stage13 v66 - Dance v15 reopen visual candidate

Generated a real Dance output using the manual reopen preset with the v3 safe
learned gate. To keep runtime lower, this used the existing v12 output as a
precomputed base and generated only the final region-aware pass.

Gate generation:

```text
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_v12_native_pilot_v3_detail_strict_outputs/
  k5_dance_flat_cleanup_gate_v12_native_pilot_v1_gate.png
```

The gate was produced with `train_flat_cleanup_gate.py apply --gate-only`, so it
wrote only the gate PNG/json and skipped denoised EXR/TIFF output.

Final output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v15_reopen_precomputed_outputs/
  k5_dance_scunet_preset_chooser_v15_reopen_precomputed.exr
```

Input path confirmed in metadata:

```text
region_aware_flat_gate_preset=dark_sky_strict_reopen
region_aware_flat_gate_path=.../flat_cleanup_gate_v12_native_pilot_v3_detail_strict_outputs/k5_dance_flat_cleanup_gate_v12_native_pilot_v1_gate.png
```

Region-aware stats:

```text
gate_mean=0.19847
reopen_mean=1.03197
reopen_p95=1.19161
effective_gate_mean=0.10885
effective_gate_p95=0.25911
```

ROI evaluation:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v15_reopen_dance/
```

Summary score, lower is better:

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| reopen | 0.999 | 5 |
| v12 | 1.000 | 5 |
| strict | 1.004 | 5 |
| dark_sky | 1.011 | 5 |

Key ROI ratios vs v12:

| ROI | kind | candidate | score | luma p99 | chroma p99 | contrast |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| sky_existing | flat | strict | 0.940 | 0.908 | 0.913 | 0.891 |
| sky_existing | flat | reopen | 0.947 | 0.915 | 0.923 | 0.897 |
| sky_center | flat | strict | 0.947 | 0.912 | 0.920 | 0.894 |
| sky_center | flat | reopen | 0.951 | 0.922 | 0.928 | 0.903 |
| dancer_center | detail | strict | 1.085 | 0.938 | 0.930 | 0.914 |
| dancer_center | detail | reopen | 1.062 | 0.947 | 0.935 | 0.931 |
| house_detail | detail | strict | 1.060 | 0.950 | 0.936 | 0.934 |
| house_detail | detail | reopen | 1.046 | 0.955 | 0.947 | 0.946 |

Crop comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_v15_reopen_dance/
  k5_dance_v15_reopen_sky_center_compare.png
  k5_dance_v15_reopen_sky_existing_compare.png
  k5_dance_v15_reopen_dancer_center_compare.png
  k5_dance_v15_reopen_house_detail_compare.png
  k5_dance_v15_reopen_snow_ground_compare.png
```

Interpretation:

- Reopen is not the strongest sky cleaner; `dark_sky` remains lower on flat sky
  noise metrics.
- Reopen is the best current balance: it gives back detail contrast on dancer
  and house compared with strict/dark_sky while preserving meaningful sky cleanup.
- This supports keeping `dark_sky_strict_reopen` as a manual quality candidate
  for Dance-like dark sky scenes. It is still not promoted to auto because
  Occi/Ice ROI probes showed possible non-sky activation.

Next direction:

- Visual inspect the generated Dance crops. If they look right, consider an
  explicit `sky_luma_reopen` auto option guarded by scene preset and maybe a
  global dark-sky coverage threshold.
- Do not apply reopen to hair/skin/blue-shadow scenes yet.

## Stage13 v67 - Reopen stays explicit, add auto_reopen mode

The generated Dance v15 reopen output is numerically promising, but the local
image viewer could not open files from the moved project path or `/private/tmp`
in this session. The crop PNGs were generated successfully and remain available
for user-side visual inspection:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_v15_reopen_dance/
```

Decision after v15:

- Do not change existing `auto`; it remains conservative:
  `sky_luma -> dark_sky_strict`.
- Add an explicit opt-in mode `auto_reopen` for further testing:
  `sky_luma -> dark_sky_strict_reopen`, all other scene presets -> off.
- This keeps reopen out of hair/skin/blue-shadow scenes while making the Dance
  path easier to reproduce.

Verification:

```text
sky_luma auto=dark_sky_strict auto_reopen=dark_sky_strict_reopen
hair_luma auto=None auto_reopen=None
blue_shadow_safe auto=None auto_reopen=None
```

Next practical step: visually inspect the v15 Dance crops. If they look better
than strict/dark_sky, run one more full/precomputed Dance pass through
`--region-aware-flat-gate-preset auto_reopen` to validate the public CLI path,
then consider whether a dark-sky coverage threshold is needed before any real
auto promotion.

## Stage13 v68 - auto_reopen CLI path validated

The initial `auto_reopen` run failed because the chooser's argparse choices only
allowed `off`, `auto`, and concrete preset names. Fixed the CLI choices to include
`auto_reopen` as an explicit mode.

Validation command produced:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v15_auto_reopen_precomputed_outputs/
  k5_dance_scunet_preset_chooser_v15_auto_reopen_precomputed.exr
```

Metadata confirms the requested mode and resolved preset:

```text
params.region_aware_flat_gate_preset=auto_reopen
region_aware_flat_gate_preset=dark_sky_strict_reopen
```

Equivalence check against the manually requested v15 reopen output:

| comparison | max abs diff | mean abs diff | p99 abs diff |
| --- | ---: | ---: | ---: |
| manual `dark_sky_strict_reopen` vs `auto_reopen` | 0.0 | 0.0 | 0.0 |

Decision:

- `auto_reopen` is now a valid reproducible test path.
- Existing conservative `auto` remains unchanged.
- Continue treating reopen as opt-in until the generated Dance crops are visually
  accepted and until a broader dark-sky coverage guard is designed.

## Stage13 v69 - Generalization probe for dark flat cleanup guard

User concern: the current dark-sky/flat cleanup guard may be tuned only to a few
samples. Added a lightweight probe script:

```text
scripts/eval_flat_region_generalization.py
```

The probe does not write denoised images. It downsamples each EXR and evaluates
region-aware masks with `reference=input`, so it is only a gate/generalization
risk diagnostic, not a final quality metric.

Existing v12-output full-resolution guard check remains intentionally narrow:

| scene | guard | candidate gt_040 | structure mean | interpretation |
| --- | --- | ---: | ---: | --- |
| K-5 Dance v12 | pass | 0.05395 | 0.56671 | intended dark-sky candidate |
| X-T5 Occi v12 | fail | 0.03599 | 0.45111 | insufficient dark-flat coverage |
| K-5 Ice v12 | fail | 0.03996 | 0.69803 | too structure-heavy / risky |

Broader noisy-EXR probe at `--max-side 1200`:

| image | guard | candidate gt_040 | structure mean |
| --- | --- | ---: | ---: |
| K-5 Dance noisy | fail | 0.01742 | 0.57061 |
| K-5 Ice noisy | fail | 0.00592 | 0.81703 |
| X-T5 Occi noisy | fail | 0.01455 | 0.76724 |
| X-T5 Cat noisy | fail | 0.01571 | 0.38223 |
| X-T5 Cat2 noisy | fail | 0.00093 | 0.65350 |
| X-T5 Room | pass | 0.08823 | 0.58824 |
| Z7 bird noisy | fail | 0.00001 | 0.62865 |
| Z7 fix noisy | fail | 0.00000 | 0.90829 |
| Z7 night noisy | fail | 0.00007 | 0.89689 |

SCUNet/processed-EXR probe at `--max-side 1200`:

| image | guard | candidate gt_040 | structure mean |
| --- | --- | ---: | ---: |
| K-5 Dance SCUNet | pass | 0.15531 | 0.45500 |
| K-5 Ice SCUNet | fail | 0.02831 | 0.77688 |
| X-T5 Occi SCUNet | fail | 0.01890 | 0.75779 |
| X-T5 Cat SCUNet | fail | 0.01792 | 0.24777 |
| X-T5 Cat2 SCUNet | fail | 0.00348 | 0.61980 |
| X-T5 Room SCUNet | pass | 0.05863 | 0.59618 |

Interpretation:

- The guard is not image-class universal yet. It is sensitive to the current
  pipeline output, especially whether the preceding stage has turned broad dark
  areas into genuinely flat residual fields.
- That sensitivity is useful for safety: Cat/Occi/Ice do not pass even when some
  structure means are low, while Dance and Room pass after smoothing produces a
  large flat dark region.
- It also means any automatic use must be a local/late-stage decision, not a
  raw-photo global preset. The guard should inspect the actual candidate image
  about to receive flat cleanup.
- The current threshold pair (`candidate.gt_040 >= 0.05` and
  `structure.mean <= 0.62`) is a candidate safety guard, not a final learned
  rule. It is too early to treat it as broadly validated.

Next design direction:

- Keep existing conservative `auto` unchanged.
- Keep `auto_reopen` opt-in.
- Add a late-stage guard path that can refuse `dark_sky_strict_reopen` unless the
  current candidate image passes broad dark-flat coverage and structure-safety
  checks.
- For real generalization, collect more non-sky dark images and bright flat-sky
  images, then convert this hand guard into training features or a learned gate.

## Stage13 v70 - Guarded auto_reopen implementation

Implemented a guarded path for `auto_reopen` in:

```text
scripts/apply_scunet_preset_chooser.py
```

New behavior:

- Existing conservative `auto` is unchanged.
- `auto_reopen` can still resolve to `dark_sky_strict_reopen` for `sky_luma`.
- When `--region-aware-flat-gate-guard` is enabled and the resolved preset is
  `dark_sky_strict_reopen`, the script measures the actual candidate image before
  applying the region-aware flat gate.
- If the candidate does not pass the guard, it falls back to `dark_sky_strict`
  by default, or to off with `--region-aware-flat-gate-guard-fallback off`.

New CLI options:

```text
--region-aware-flat-gate-guard
--region-aware-flat-gate-guard-min-candidate-gt040 0.05
--region-aware-flat-gate-guard-max-structure-mean 0.62
--region-aware-flat-gate-guard-fallback {off,dark_sky_strict}
```

Lightweight function check on downsampled v12 candidates:

| scene | guard | candidate gt_040 | structure mean |
| --- | --- | ---: | ---: |
| K-5 Dance | pass | 0.14936 | 0.51039 |
| K-5 Ice | fail | 0.03857 | 0.79219 |
| X-T5 Occi | fail | 0.02159 | 0.75387 |

Verification:

```text
pixi run python -B -m py_compile scripts/apply_scunet_preset_chooser.py
pixi run python -B scripts/apply_scunet_preset_chooser.py --help
```

Decision:

- The right abstraction is not a universal photo-level dark-sky preset. It is a
  late-stage permission check on the actual candidate image.
- This keeps the strong clean path available for Dance/Room-like broad dark flat
  areas while refusing structure-heavy images.
- Next quality step is to visually test guarded `auto_reopen` on Dance and Room,
  then gather more negative examples before promoting it beyond opt-in.

## Stage13 v71 - Guarded auto_reopen CLI validation

Validated the guarded `auto_reopen` path through the real CLI with precomputed
base images, avoiding selector/model inference.

Dance pass case:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v16_guarded_auto_reopen_precomputed_outputs/
  k5_dance_scunet_preset_chooser_v16_guarded_auto_reopen_precomputed.exr
```

Metadata:

| field | value |
| --- | --- |
| requested preset | `auto_reopen` |
| resolved preset | `dark_sky_strict_reopen` |
| guard passed | `true` |
| candidate gt_040 | 0.053946 |
| structure mean | 0.566710 |

Equivalence against previous v15 auto-reopen output:

| comparison | max abs diff | mean abs diff | p99 abs diff |
| --- | ---: | ---: | ---: |
| Dance v16 guarded vs v15 auto_reopen | 0.0 | 0.0 | 0.0 |

Ice forced-sky rejection case:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v16_guarded_auto_reopen_precomputed_outputs/
  k5_ice_scunet_preset_chooser_v16_guarded_auto_reopen_forced_sky_precomputed.exr
```

Metadata:

| field | value |
| --- | --- |
| requested preset | `auto_reopen` |
| forced scene preset | `sky_luma` |
| guard passed | `false` |
| final resolved preset | `dark_sky_strict` |
| candidate gt_040 | 0.039959 |
| structure mean | 0.698028 |

Interpretation:

- The guard is transparent when the candidate clearly matches the Dance-like dark
  flat-sky case: output is bit-identical to the previous unguarded v15 path.
- The guard refuses a structure-heavy Ice case and falls back to the safer strict
  cleanup instead of reopening the flat gate.
- This supports keeping `auto` conservative, and using guarded `auto_reopen` as
  the next opt-in quality path for broad dark flat areas.

Next step:

- Generate guarded `auto_reopen` crops for Dance and, when a suitable base/gate
  exists, Room. Visual acceptance should focus on whether the dark sky becomes
  cleaner without further softening people/hair/edge structure.

## Stage13 v72 - Shadow-flat sky-only reopen probe

Motivation after v16 guarded:

- v16 guarded is safe and reproducible, but the improvement is small.
- It cleans sky less than strict while preserving more detail than strict.
- Next hypothesis: reduce always-on base strength and move strength into the
  dark flat/shadow reopen path, so detail regions receive less incidental blur.

Generated a direct region-aware flat gate probe, not yet wired into chooser:

```text
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v17_shadow_reopen_skyonly_outputs/
  k5_dance_region_aware_flat_gate_v17_shadow_reopen_skyonly.exr
```

Main parameter changes vs v16/dark_sky_strict_reopen:

| parameter | v16 | v17 probe |
| --- | ---: | ---: |
| base_strength | 0.12 | 0.04 |
| flat_boost | 1.02 | 0.78 |
| skin_strength | 0.05 | 0.02 |
| shadow_flat_boost | 0.75 | 1.55 |
| structure_suppress | 0.99 | 0.995 |
| reopen_strength | 0.85 | 1.05 |
| reopen_max | 1.45 | 1.65 |

Filter stats:

| stat | v17 |
| --- | ---: |
| strength_mean | 0.43555 |
| strength_p95 | 0.84694 |
| reopen_mean | 1.03949 |
| reopen_p95 | 1.23670 |
| effective_gate_mean | 0.10791 |
| effective_gate_p95 | 0.29078 |

Dance ROI evaluation:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v17_shadow_reopen_skyonly_dance/
```

Summary score, lower is better:

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| v17_skyonly | 0.997 | 5 |
| v16_guarded | 0.999 | 5 |
| v12 | 1.000 | 5 |
| strict | 1.004 | 5 |

Key ratios vs v12:

| ROI | kind | candidate | score | luma p99 | chroma p99 | contrast |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| sky_existing | flat | v16_guarded | 0.947 | 0.915 | 0.923 | 0.897 |
| sky_existing | flat | v17_skyonly | 0.941 | 0.904 | 0.914 | 0.882 |
| sky_center | flat | v16_guarded | 0.951 | 0.922 | 0.928 | 0.903 |
| sky_center | flat | v17_skyonly | 0.948 | 0.914 | 0.923 | 0.892 |
| dancer_center | detail | v16_guarded | 1.062 | 0.947 | 0.935 | 0.931 |
| dancer_center | detail | v17_skyonly | 1.063 | 0.950 | 0.938 | 0.933 |
| house_detail | detail | v16_guarded | 1.046 | 0.955 | 0.947 | 0.946 |
| house_detail | detail | v17_skyonly | 1.044 | 0.958 | 0.951 | 0.948 |

Crop comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_v17_shadow_reopen_skyonly_dance/
```

Interpretation:

- The direction is slightly better than v16 numerically: sky is cleaner and house
  detail is marginally better, while dancer detail is essentially tied.
- However, the gain is still small. This is not a breakthrough; it is a better
  local parameterization of the guarded dark-flat cleanup path.
- Visual inspection should decide whether v17's stronger sky smoothing is worth
  the extra local contrast drop in sky ROIs.

Next design direction:

- If v17 crops look acceptable, make `dark_sky_strict_reopen_skyonly` a named
  preset and let guarded `auto_reopen` resolve to it for sky_luma pass cases.
- If sky looks too plasticky, keep v16 and move effort to a learned flat/noise
  residual branch instead of further hand-tuning.

## Stage13 v73 - v17 sky-only reopen named preset

Promoted the v17 hand-run parameters to a reproducible named preset in:

```text
scripts/apply_scunet_preset_chooser.py
```

New preset:

```text
dark_sky_strict_reopen_skyonly
```

This preset keeps the v17 direction explicit but does not yet change the default
`auto_reopen` behavior. The intent is to make visual/ROI validation repeatable
without silently promoting it before user-side crop inspection.

CLI validation output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v17_named_skyonly_precomputed_outputs/
  k5_dance_scunet_preset_chooser_v17_named_skyonly_precomputed.exr
```

Equivalence against the original hand-run v17 output:

| comparison | max abs diff | mean abs diff | p99 abs diff |
| --- | ---: | ---: | ---: |
| named preset v17 vs hand-run v17 | 0.0 | 0.0 | 0.0 |

Decision:

- Keep conservative `auto` unchanged.
- Keep guarded `auto_reopen` resolving to `dark_sky_strict_reopen` for now.
- Use `dark_sky_strict_reopen_skyonly` as an explicit candidate until the v17
  crops are visually accepted. If accepted, guarded `auto_reopen` can resolve to
  this preset for pass cases.

## Stage13 v74 - Generalized guard for all reopen region presets

Safety issue found after naming v17:

- The `--region-aware-flat-gate-guard` condition was tied only to the exact
  preset name `dark_sky_strict_reopen`.
- New reopen presets such as `dark_sky_strict_reopen_skyonly` would bypass the
  guard if used explicitly or promoted later.

Implemented:

```text
is_reopen_region_aware_preset(preset)
```

The guard now applies to any region-aware flat gate preset whose parameters
include `reopen_strength > 0`.

Function check:

| preset | guarded reopen preset |
| --- | --- |
| `dark_sky_strict` | false |
| `dark_sky_strict_reopen` | true |
| `dark_sky_strict_reopen_skyonly` | true |

Dance pass validation:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v18_guarded_named_skyonly_precomputed_outputs/
  k5_dance_scunet_preset_chooser_v18_guarded_named_skyonly_precomputed.exr
```

Metadata:

| field | value |
| --- | --- |
| requested preset | `dark_sky_strict_reopen_skyonly` |
| guard passed | true |
| final preset | `dark_sky_strict_reopen_skyonly` |
| candidate gt_040 | 0.053946 |
| structure mean | 0.566710 |

Equivalence to unguarded v17 named output:

| comparison | max abs diff | mean abs diff | p99 abs diff |
| --- | ---: | ---: | ---: |
| v18 guarded skyonly vs v17 named skyonly | 0.0 | 0.0 | 0.0 |

Ice forced-sky rejection validation:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v18_guarded_named_skyonly_precomputed_outputs/
  k5_ice_scunet_preset_chooser_v18_guarded_named_skyonly_forced_sky_precomputed.exr
```

Metadata:

| field | value |
| --- | --- |
| requested preset | `dark_sky_strict_reopen_skyonly` |
| forced scene preset | `sky_luma` |
| guard passed | false |
| final preset | `dark_sky_strict` |
| candidate gt_040 | 0.039959 |
| structure mean | 0.698028 |

Decision:

- This makes future promotion of v17 safer: the guard remains active even if
  `auto_reopen` is later changed to resolve to `dark_sky_strict_reopen_skyonly`.
- Keep `auto_reopen` unchanged until visual review of v17 crops.

## Stage13 v75 - Explicit auto_reopen_skyonly mode

Added an opt-in chooser mode for the v17 sky-only reopen candidate:

```text
auto_reopen_skyonly
```

Resolution behavior:

| scene preset | resolved region-aware flat gate preset |
| --- | --- |
| `sky_luma` | `dark_sky_strict_reopen_skyonly` |
| `hair_luma` | off |
| `blue_shadow_safe` | off |

This does not change existing behavior:

- `auto` remains conservative and resolves `sky_luma -> dark_sky_strict`.
- `auto_reopen` still resolves `sky_luma -> dark_sky_strict_reopen`.
- `auto_reopen_skyonly` is an explicit v17 test path.

CLI validation output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v19_auto_reopen_skyonly_precomputed_outputs/
  k5_dance_scunet_preset_chooser_v19_auto_reopen_skyonly_precomputed.exr
```

Metadata confirms:

| field | value |
| --- | --- |
| requested preset | `auto_reopen_skyonly` |
| resolved preset | `dark_sky_strict_reopen_skyonly` |
| guard passed | true |

Equivalence against explicit guarded v18 skyonly output:

| comparison | max abs diff | mean abs diff | p99 abs diff |
| --- | ---: | ---: | ---: |
| v19 auto_reopen_skyonly vs v18 explicit guarded skyonly | 0.0 | 0.0 | 0.0 |

Decision:

- Use `auto_reopen_skyonly --region-aware-flat-gate-guard` for convenient v17
  visual tests.
- Do not promote it to default until v17 crops are visually accepted.

## Stage13 v76 - Arbitrary-pair dark-flat coverage probe for Room/Cat2

Generalized the dark-flat coverage evaluator:

```text
scripts/eval_dark_sky_coverage.py
```

New capabilities:

```text
--pair name,reference,current
--max-side 1200
```

This lets us test the guarded reopen condition on arbitrary real-photo pairs
without requiring the fixed `SCENES` table or running full-resolution output.

Room/Cat2 probe:

```text
runs/refiner_pilot_stage11_hybrid_best/dark_sky_coverage_v3_room_cat2/dark_sky_coverage.json
```

Results at `--max-side 1200`:

| pair | guard | candidate mean | gt_025 | gt_040 | structure mean | strong |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| xt5_room_diag | true | 0.04532 | 0.07236 | 0.06038 | 0.58811 | 0.08545 |
| xt5_room_tail | true | 0.04541 | 0.07247 | 0.06047 | 0.58808 | 0.08555 |
| xt5_cat2_diag | false | 0.00303 | 0.00481 | 0.00267 | 0.65170 | 0.00810 |
| xt5_cat2_tail | false | 0.00302 | 0.00481 | 0.00265 | 0.65224 | 0.00809 |

Interpretation:

- The guard is no longer only a Dance-specific accident. It also passes Room,
  another broad dark-flat candidate.
- Cat2 is rejected strongly: there is almost no dark-flat coverage, and structure
  mean is above the safety threshold.
- This supports the late-stage guard idea: it reacts to the actual candidate
  image content rather than only to scene labels.

Caveat:

- Room currently lacks a learned flat cleanup gate PNG, so this is a guard
  eligibility result, not a completed v17 output.
- Next practical step is to add a custom/reference-current gate generation path
  or otherwise create a Room flat gate before running `auto_reopen_skyonly` on
  the full Room image.

## Stage13 v77 - Room full output with custom flat gate

Added a custom flat-gate application path:

```text
scripts/train_flat_cleanup_gate.py apply-custom
```

This allows arbitrary `--reference` / `--current` pairs and an optional
`--detail-gate`. If no detail gate is supplied, the existing reader returns a
zero gate, which is acceptable for this first Room experiment because the later
region-aware strength/structure guard still protects coherent detail.

Generated Room flat gate with v3 detail-strict checkpoint:

```text
runs/refiner_pilot_stage11_hybrid_best/flat_cleanup_gate_v12_native_pilot_v3_detail_strict_room_outputs/
  xt5_room_flat_cleanup_gate_v12_native_pilot_v1_gate.png
```

Gate stats:

| stat | value |
| --- | ---: |
| gate_mean | 0.22497 |
| gate_p95 | 0.39880 |
| elapsed_sec | 210.32 |
| tiles | 54 |

Generated Room `auto_reopen_skyonly + guard` output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v20_room_auto_reopen_skyonly_outputs/
  xt5_room_scunet_preset_chooser_v20_auto_reopen_skyonly.exr
```

Guard metadata:

| field | value |
| --- | ---: |
| guard passed | true |
| candidate gt_040 | 0.09308 |
| structure mean | 0.35993 |
| effective_gate_mean | 0.14963 |
| effective_gate_p95 | 0.47000 |

Representative mixed ROI evaluation:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v20_room_auto_reopen_skyonly/
```

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| base | 1.000 | 5 |
| v20 | 1.010 | 5 |

Interpretation: arbitrary representative points are not all improved. Some
regions lose local contrast for only small residual-noise gains, so Room-wide
promotion would be premature.

Auto-selected flat-region ROI evaluation:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v20_room_auto_reopen_skyonly_autoflat/
```

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| v20 | 0.925 | 8 |
| base | 1.000 | 8 |

Key auto-flat ratios vs base:

| ROI | score | luma p99 | chroma p99 | magenta p999 | blue p999 | contrast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auto_flat_1 | 0.932 | 0.652 | 0.627 | 0.633 | 0.715 | 0.656 |
| auto_flat_2 | 0.900 | 0.580 | 0.571 | 0.583 | 0.576 | 0.599 |
| auto_flat_3 | 0.929 | 0.723 | 0.732 | 0.790 | 0.774 | 0.775 |
| auto_flat_4 | 0.919 | 0.690 | 0.688 | 0.709 | 0.667 | 0.712 |
| auto_flat_5 | 0.975 | 0.669 | 0.693 | 0.732 | 0.732 | 0.658 |
| auto_flat_6 | 0.919 | 0.757 | 0.738 | 0.737 | 0.721 | 0.778 |
| auto_flat_7 | 0.904 | 0.654 | 0.667 | 0.684 | 0.645 | 0.696 |
| auto_flat_8 | 0.920 | 0.725 | 0.716 | 0.718 | 0.698 | 0.745 |

Crop comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_v20_room_auto_reopen_skyonly/
runs/refiner_pilot_stage11_hybrid_best/compare_v20_room_auto_reopen_skyonly_autoflat/
```

Decision:

- The guarded sky-only reopen path generalizes beyond Dance in the intended
  target regions: Room's auto-flat dark regions are much cleaner.
- It also confirms the remaining weakness: outside target flat regions, the
  deterministic smoother can still reduce local contrast. The next design should
  tighten the final effective gate or use region-aware strength maps more
  selectively before any default promotion.

## Stage13 v78 - Room v21 narrower shadow-flat probe

Motivation:

- Room v20 is clearly beneficial on auto-selected flat dark regions, but
  representative mixed ROIs show slight score regression from local contrast
  reduction.
- Tested a narrower v21 parameterization to reduce broad incidental smoothing.

Generated:

```text
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v21_room_shadow_narrow_outputs/
  xt5_room_region_aware_flat_gate_v21_shadow_narrow.exr
```

Main changes vs v20 skyonly:

| parameter | v20 | v21 |
| --- | ---: | ---: |
| base_strength | 0.04 | 0.015 |
| flat_boost | 0.78 | 0.35 |
| skin_strength | 0.02 | 0.0 |
| shadow_flat_boost | 1.55 | 1.85 |
| structure_suppress | 0.995 | 1.0 |
| flat_threshold | 0.044 | 0.040 |
| edge_threshold | 0.031 | 0.026 |
| min_strength | 0.006 | 0.0 |
| reopen_strength | 1.05 | 1.15 |
| reopen_shadow_threshold | 0.40 | 0.43 |

Filter stats:

| stat | v20 | v21 |
| --- | ---: | ---: |
| strength_mean | 0.49450 | 0.36641 |
| effective_gate_mean | 0.14963 | 0.12040 |
| effective_gate_p95 | 0.47000 | 0.46967 |

Representative ROI evaluation:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v21_room_shadow_narrow_representative/
```

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| base | 1.000 | 5 |
| v21 | 1.008 | 5 |
| v20 | 1.010 | 5 |

Auto-flat ROI evaluation:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v21_room_shadow_narrow_autoflat/
```

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| v20 | 0.925 | 8 |
| v21 | 0.937 | 8 |
| base | 1.000 | 8 |

Interpretation:

- v21 does reduce collateral representative-ROI regression slightly, as intended.
- It also weakens the target flat-region cleanup. The tradeoff is not clearly
  better than v20: v20 remains the stronger quality candidate for visible flat
  noise removal, while v21 is a safer but less effective variant.
- Do not replace the current v17/v20 skyonly candidate with v21 yet.

Crop comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_v21_room_shadow_narrow_autoflat/
```

Next direction:

- A better fix is likely not another global scalar tweak. We need a more local
  effective-gate limiter: preserve v20 strength inside high-confidence flat
  islands, but suppress low-confidence spill into mixed/edge/highlight regions.

## Stage13 v79 - Room v22/v23 local limiter probes

Implemented optional effective-gate limiter in:

```text
scripts/apply_region_aware_flat_gate.py
```

The limiter is off by default and only active when `--limiter-strength > 0`.
It uses a high-confidence flat/shadow/safe mask to keep strong cleanup inside
flat islands and reduce low-confidence spill into mixed regions.

Generated v22 strong limiter:

```text
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v22_room_v20_limited_outputs/
  xt5_room_region_aware_flat_gate_v22_v20_limited.exr
```

Generated v23 softer limiter:

```text
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v23_room_v20_soft_limited_outputs/
  xt5_room_region_aware_flat_gate_v23_v20_soft_limited.exr
```

Effective gate stats:

| candidate | effective_gate_mean | effective_gate_p95 | limiter_mean | limiter_p95 |
| --- | ---: | ---: | ---: | ---: |
| v20 | 0.14963 | 0.47000 | n/a | n/a |
| v21 | 0.12040 | 0.46967 | n/a | n/a |
| v22 | 0.10171 | 0.42858 | 0.59449 | 0.91349 |
| v23 | 0.12512 | 0.44983 | 0.78197 | 0.95920 |

Representative ROI summary:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v23_room_v20_soft_limited_representative/
```

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| base | 1.000 | 5 |
| v22 | 1.007 | 5 |
| v21 | 1.008 | 5 |
| v23 | 1.009 | 5 |
| v20 | 1.010 | 5 |

Auto-flat ROI summary:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v23_room_v20_soft_limited_autoflat/
```

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| v20 | 0.925 | 8 |
| v21 | 0.937 | 8 |
| v23 | 0.940 | 8 |
| v22 | 0.956 | 8 |
| base | 1.000 | 8 |

Interpretation:

- v22 is the safest among processed candidates on representative ROIs, but it
  throws away too much of the target flat-region cleanup.
- v20 is still the strongest target cleaner and remains the best quality-first
  candidate for visible flat noise.
- v23 does not dominate: it is worse than v21 on both representative and auto-flat
  summaries, so it should not be promoted.
- The limiter mechanism is useful as an optional experiment hook, but this first
  confidence formula is too blunt. A future limiter should be edge/spill-aware,
  not just confidence-scaled.

Decision:

- Keep v20 / `auto_reopen_skyonly` as the quality-first visual candidate.
- Keep v22 only as evidence that spill can be reduced, not as a replacement.
- Reject v23 for now.

## Stage13 v80 - Room v24 soft shadow-spill limiter

Added a reusable mask ROI evaluator:

```text
scripts/eval_mask_roi.py
```

Used the saved v22 masks to inspect why representative ROIs regress while
auto-flat ROIs improve. Key observation:

- Representative mixed ROIs have low `shadow_flat` and high structure/edge masks.
- Auto-flat target ROIs have high `shadow_flat` and low structure/edge masks.

This suggests spill control should primarily target low-shadow-flat confidence,
not globally weaken flat cleanup.

Generated v24 soft shadow-spill limiter:

```text
runs/refiner_pilot_stage11_hybrid_best/region_aware_flat_gate_v24_room_shadow_spill_soft_outputs/
  xt5_room_region_aware_flat_gate_v24_shadow_spill_soft.exr
```

Main limiter parameters:

| parameter | value |
| --- | ---: |
| limiter_strength | 0.18 |
| limiter_min | 0.82 |
| limiter_flat_threshold | 0.30 |
| limiter_flat_transition | 0.30 |
| limiter_shadow_threshold | 0.18 |
| limiter_shadow_transition | 0.24 |
| limiter_structure_suppress | 0.35 |

Effective gate stats:

| candidate | effective_gate_mean | effective_gate_p95 | limiter_mean | limiter_p95 |
| --- | ---: | ---: | ---: | ---: |
| v20 | 0.14963 | 0.47000 | n/a | n/a |
| v22 | 0.10171 | 0.42858 | 0.59449 | 0.91349 |
| v23 | 0.12512 | 0.44983 | 0.78197 | 0.95920 |
| v24 | 0.13718 | 0.46488 | 0.86385 | 0.99021 |

Representative ROI summary:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v24_room_shadow_spill_soft_representative/
```

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| base | 1.000 | 5 |
| v22 | 1.007 | 5 |
| v24 | 1.009 | 5 |
| v20 | 1.010 | 5 |

Auto-flat ROI summary:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v24_room_shadow_spill_soft_autoflat/
```

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| v20 | 0.925 | 8 |
| v24 | 0.933 | 8 |
| v22 | 0.956 | 8 |
| base | 1.000 | 8 |

Crop comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_v24_room_shadow_spill_soft_representative/
runs/refiner_pilot_stage11_hybrid_best/compare_v24_room_shadow_spill_soft_autoflat/
```

Interpretation:

- v24 is a better limiter direction than v22/v23: it preserves most auto-flat
  cleanup while modestly reducing representative ROI regression vs v20.
- v20 remains the quality-first candidate because auto-flat cleanup is strongest.
- v24 is a viable safety-biased alternative, not a replacement for v20 yet.

Decision:

- Keep `auto_reopen_skyonly`/v20 as the primary visual candidate.
- Keep the limiter mechanism and v24 parameters for future safety preset work.
- Do not promote v24 until visual crops show the slight representative ROI gain is
  worth the small loss in target flat cleanup.

## Stage13 v81 - Named soft-limiter preset

Promoted v24 to an explicit named region-aware flat-gate preset:

```text
dark_sky_strict_reopen_skyonly_soft_limiter
```

This is a safety-biased test candidate. Existing modes are unchanged:

- `auto` remains conservative.
- `auto_reopen` remains the original reopen path.
- `auto_reopen_skyonly` remains the quality-first v20 path.

Implementation detail:

- Fixed `region_aware_reopen_guard()` so guard evaluation strips both `reopen_*`
  and `limiter_*` keys before calling `build_strength_map()`.
- This was required because limiter parameters are consumed by
  `apply_region_aware_gate()`, not by `build_strength_map()`.

CLI validation output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v25_room_soft_limiter_named_outputs/
  xt5_room_scunet_preset_chooser_v25_soft_limiter_named.exr
```

Metadata confirms:

| field | value |
| --- | --- |
| requested preset | `dark_sky_strict_reopen_skyonly_soft_limiter` |
| guard passed | true |
| candidate gt_040 | 0.09308 |
| structure mean | 0.35993 |
| effective_gate_mean | 0.13718 |
| effective_gate_p95 | 0.46488 |

Equivalence against hand-run v24:

| comparison | max abs diff | mean abs diff | p99 abs diff |
| --- | ---: | ---: | ---: |
| v25 named soft-limiter vs v24 hand-run | 0.0 | 0.0 | 0.0 |

Decision:

- v24/v25 is now reproducible as a named safety-biased candidate.
- It is not promoted to `auto_reopen_skyonly`; v20 remains the quality-first
  candidate until visual review says otherwise.

## Stage13 v82 - Dance validation for soft-limiter preset

Ran the named soft-limiter preset on Dance to check whether it is only a Room
safety candidate or a generally useful guarded alternative.

Output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v26_dance_soft_limiter_named_outputs/
  k5_dance_scunet_preset_chooser_v26_soft_limiter_named.exr
```

Metadata:

| field | value |
| --- | ---: |
| preset | `dark_sky_strict_reopen_skyonly_soft_limiter` |
| guard passed | true |
| candidate gt_040 | 0.053946 |
| structure mean | 0.566710 |
| limiter_mean | 0.87478 |
| effective_gate_mean | 0.09921 |
| effective_gate_p95 | 0.28379 |

ROI evaluation:

```text
runs/refiner_pilot_stage11_hybrid_best/roi_eval_v26_dance_soft_limiter/
```

Summary:

| candidate | mean score | ROI count |
| --- | ---: | ---: |
| v26_soft | 0.996 | 5 |
| v17_skyonly | 0.997 | 5 |
| v12 | 1.000 | 5 |

Key ratios vs v12:

| ROI | kind | candidate | score | luma p99 | chroma p99 | contrast |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| sky_existing | flat | v17_skyonly | 0.941 | 0.904 | 0.914 | 0.882 |
| sky_existing | flat | v26_soft | 0.945 | 0.908 | 0.919 | 0.887 |
| sky_center | flat | v17_skyonly | 0.948 | 0.914 | 0.923 | 0.892 |
| sky_center | flat | v26_soft | 0.951 | 0.919 | 0.927 | 0.897 |
| dancer_center | detail | v17_skyonly | 1.063 | 0.950 | 0.938 | 0.933 |
| dancer_center | detail | v26_soft | 1.057 | 0.959 | 0.945 | 0.941 |
| house_detail | detail | v17_skyonly | 1.044 | 0.958 | 0.951 | 0.948 |
| house_detail | detail | v26_soft | 1.040 | 0.962 | 0.957 | 0.954 |

Crop comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_v26_dance_soft_limiter/
```

Interpretation:

- On Dance, the soft limiter slightly weakens sky cleanup but improves detail
  preservation enough to win the mean ROI score.
- On Room, v20 remains the stronger target-flat cleaner, while v24/v26 is a
  safety-biased alternative.
- This suggests the final system should not have one global answer. It should
  choose between quality-first skyonly and soft-limiter based on the amount of
  mixed/detail spill risk.

Decision:

- Keep `auto_reopen_skyonly` as quality-first.
- Keep `dark_sky_strict_reopen_skyonly_soft_limiter` as safety-biased.
- Next logical work is an automatic selector between these two candidates, using
  mask diagnostics such as mixed-region structure/effective-gate overlap.


## Stage13 v83 - Adaptive sky-only flat gate selector

Added `auto_reopen_skyonly_adaptive` to `scripts/apply_scunet_preset_chooser.py`.
The mode keeps the existing quality-first `auto_reopen_skyonly` untouched, but
adds a scene-statistics selector between:

- `dark_sky_strict_reopen_skyonly` for target flat cleanup.
- `dark_sky_strict_reopen_skyonly_soft_limiter` when structure spill risk is high.

The selector computes a downsampled effective-gate estimate from the current
image, learned flat gate, region masks, and reopen map. It does not branch on
scene names or fixed ROIs. It uses:

```text
spill_ratio = mean(effective_gate over structure-weighted spill) /
              mean(effective_gate over dark-flat target)
```

Default soft-limiter threshold is `spill_ratio >= 0.50`.

Probe results with the same formula:

| scene | chosen | spill_ratio | target_mean | spill_mean |
| --- | --- | ---: | ---: | ---: |
| Dance | soft-limiter | 0.538 | 0.3374 | 0.1816 |
| Occi | quality-first | 0.206 | 0.3159 | 0.0650 |
| Room tail | quality-first | 0.320 | 0.3039 | 0.0971 |

Validation output:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v27_dance_adaptive_skyonly_outputs/
  k5_dance_scunet_preset_chooser_v27_adaptive_skyonly.exr
```

Metadata shows the adaptive selector chose
`dark_sky_strict_reopen_skyonly_soft_limiter` on Dance, with guard pass true.
Diff vs the earlier v26 soft-limiter output:

```text
max 0.0011537, mean 3.46e-05, p99 1.49e-04
```

Decision:

- Use `auto_reopen_skyonly_adaptive` as the next candidate for broader images.
- Keep `auto_reopen_skyonly` available as the explicit quality-first/manual mode.
- Next validation should run adaptive on Room/Occi/Ice and then compare crops,
  especially dark flat sky versus hair/structure preservation.


## Stage13 v84 - Adaptive guard-fail should no-op, not fallback weakly

Validated `auto_reopen_skyonly_adaptive` on broader scenes after v83.
The first v27 behavior still fell back to `dark_sky_strict` when the guarded
sky-only candidate failed. That was too eager for general images:

| scene | final v27 preset | mean score | key observation |
| --- | --- | ---: | --- |
| Room | `dark_sky_strict_reopen_skyonly` | 0.970 | strong improvement on flat dark ROIs |
| Dance | soft-limiter | 1.001 | sky improves, detail ROIs lose contrast |
| Occi | `dark_sky_strict` fallback | 1.017 | tiny flat benefit but hair/skin/root detail worsens |
| Ice | `dark_sky_strict` fallback | 1.019 | flat_blue improves but branch/detail ROIs worsen |

Important examples:

- Room `auto_flat_2`: luma p99 ratio 0.577, chroma p99 ratio 0.571, score 0.894.
- Room `auto_flat_4`: luma p99 ratio 0.735, chroma p99 ratio 0.693, score 0.924.
- Occi `hair_detail`: score worsened to 1.019 from detail contrast loss.
- Ice `ice_branch`: score worsened to 1.046 from detail contrast loss.

Decision:

- For `auto_reopen_skyonly_adaptive` only, guard failure now disables the
  region-aware flat gate instead of falling back to `dark_sky_strict`.
- Older explicit guarded modes keep their existing fallback behavior.
- This makes adaptive conservative for diverse images: apply only when the
  dark-flat target is strong enough, otherwise preserve the base image.

Validation outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/scunet_preset_chooser_v28_adaptive_guardoff_outputs/
  xt5_occi_scunet_preset_chooser_v28_adaptive_guardoff.exr
  k5_ice_scunet_preset_chooser_v28_adaptive_guardoff.exr
```

Both guard-fail outputs are exact no-ops against their base images:

```text
occi max 0.0 mean 0.0 p99 0.0
ice  max 0.0 mean 0.0 p99 0.0
```

Current adaptive policy:

- Room-like strong dark-flat target: apply quality-first sky-only reopen.
- Dance-like mixed spill risk with enough target: apply soft limiter.
- Occi/Ice-like weak target or heavy structure: no-op.

Next direction:

- Keep this conservative adaptive selector as the broad-image gate.
- Do not use the flat cleanup branch to chase small local gains on hair/ice
  scenes; detail preservation losses dominate.
- Further noise removal for Occi/Ice should come from a separate structure-aware
  or learned branch, not this flat-gate smoother.


## Stage13 v85 - Existing structure probes and ultratight PL-hair blend

After the adaptive flat gate became conservative, checked existing structure /
hair candidates for Occi and Ice under the current ROI metric.

Ice existing candidates against current v12 base:

| candidate | mean score | decision |
| --- | ---: | --- |
| base_v12 | 1.000 | keep |
| detail_v3 | 1.012 | reject |
| blueprotect_v10 | 1.012 | reject |
| blue_dot_strong | 1.035 | reject |
| graft_soft | 1.493 | reject |
| graft_mid | 1.517 | reject |

Conclusion for Ice: the old blue/structure graft line is not useful on top of
the current v12 base. It raises residual metrics and should stay rejected.

Occi existing candidates:

| candidate | mean score | note |
| --- | ---: | --- |
| base_v12 | 1.000 | baseline |
| structure_hdr | 1.017 | flat helps, detail/skin lose |
| pl_hair_v4 | 1.024 | hair/root/face improve, noise_dark badly worsens |
| shadow_hdr | 1.044 | reject |
| graft_v3 | 1.056 | reject |

The useful signal is `pl_hair_v4`: it improves the detail/skin ROIs but leaks
into flat dark regions. Built a tighter compositor using `apply_region_rgb_blend.py`
with a much narrower hair/proximity/texture mask.

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/region_rgb_blend_v30_plhair_ultratight_occi/
  xt5_occi_v30_plhair_ultratight_s080.exr
  xt5_occi_v30_plhair_ultratight_s100.exr
```

v30 mask stats for s100:

| stat | value |
| --- | ---: |
| mask_mean | 0.09425 |
| mask_p90 | 0.36268 |
| mask_p99 | 0.73204 |
| texture_mean | 0.23123 |
| coherent_mean | 0.06115 |

ROI results:

| ROI | s080 score | s100 score | key ratios for s100 |
| --- | ---: | ---: | --- |
| hair_detail | 0.994 | 0.992 | magenta 0.970, blue 0.985 |
| root | 0.998 | 0.997 | magenta 0.983, blue 0.996 |
| face_center | 0.992 | 0.991 | magenta 0.980, blue 0.956 |
| noise_dark | 1.011 | 1.015 | luma p99 1.022, chroma p99 1.018 |
| mean | 0.999 | 0.999 | tied, both slightly below base |

Crop comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_v30_plhair_ultratight_occi/
```

Decision:

- v30 is only a tiny numeric win, not a breakthrough.
- s100 is the quality-first variant; s080 is safer. Neither should be promoted
  globally yet.
- The important design result is positive: PL-like hair/detail help can be made
  mostly local by a much tighter mask, without the large flat-region damage from
  raw `pl_hair_v4`.
- Next step should be to turn this into a learned/detail-aware branch or a
  better mask predictor; hand-mask tuning has nearly saturated.


## Stage13 v86 - PL-hair learned gate pilot, ROI-only

Tried a first learned/detail-aware branch for the Occi PL-hair signal. The goal
was to avoid more hand mask tuning by learning a binary blend gate between the
current v12 base and the useful-but-leaky `pl_hair_v4` candidate.

First attempt:

- Full-frame feature generation was too slow; interrupted during Gaussian
  feature construction.
- ROI-only v1 trained, but the sparse target collapsed to all-zero prediction.
  Logs showed `pred=0.0000` by step 150.

Second attempt, v2:

- ROI-only training around hair/root/face/noise_dark/cheek_hair.
- Positive-target sampling and positive-weighted BCE.
- 260 steps, small CNN width 12, patch 96, CPU.

Output:

```text
runs/refiner_pilot_stage11_hybrid_best/plhair_gate_pilot_v2_roi_occi/
  xt5_occi_plhair_gate_pilot_v2_roi.exr
  xt5_occi_plhair_gate_pilot_v2_roi_gate.png
```

Training summary:

```text
step 1/260   loss=0.876473 pred=0.0754 target=0.0739
step 65/260  loss=0.621539 pred=0.0624 target=0.0567
step 130/260 loss=0.788225 pred=0.0899 target=0.0739
step 195/260 loss=0.425279 pred=0.0385 target=0.0399
step 260/260 loss=0.612186 pred=0.0585 target=0.0590
gate_mean=0.00130, gate_p99=0.02873
```

ROI result:

| candidate | mean score | note |
| --- | ---: | --- |
| ultra_s100 analytic | 0.999 | still the best tiny win |
| learned_v2 | 1.000 | safe, but too weak |
| base_v12 | 1.000 | baseline |

Learned v2 details:

| ROI | score | observation |
| --- | ---: | --- |
| hair_detail | 1.002 | no useful improvement |
| root | 1.000 | no-op |
| face_center | 0.997 | small blue/magenta improvement |
| noise_dark | 1.001 | flat damage avoided |

Decision:

- The learned branch is conceptually viable for safety, but this target/cap is
  too conservative to beat the analytic v30 ultratight gate.
- Do not promote learned_v2.
- Next learned attempt should not learn from the weak metric-derived target
  directly. Better recipe: use v30 analytic gate as the positive teacher, then
  learn only the flat/spill suppression correction from negative ROIs. That
  keeps the useful hair strength while teaching the model where to close.


## Stage13 v87 - Close-gate probe on top of v30 PL-hair gate

Changed the learned/detail-aware direction after v86. Instead of learning a weak
open gate from sparse metric targets, used the v30 analytic ultratight hair gate
as the strong positive teacher and tested a deterministic close map that only
suppresses risky flat/spill regions.

Implementation was an inline pilot, not yet promoted to a reusable script.
Inputs:

- base: `scunet_preset_chooser_v12_flat_cleanup_auto_outputs/xt5_occi_scunet_preset_chooser_v12_auto.exr`
- candidate: `selective_pl_hair_detail_v4_tight_prox_occi/xt5_occi_selective_pl_hair_detail_v4_tight_prox.exr`
- base gate: reproduced v30 ultratight hair gate.
- close map: high when candidate luma/chroma impulses rise and local region is flat/non-structure; reduced when candidate color/detail benefit is strong.

Outputs:

```text
runs/refiner_pilot_stage11_hybrid_best/plhair_gate_pilot_v3_close_occi/
  xt5_occi_plhair_gate_pilot_v3_close045.exr
  xt5_occi_plhair_gate_pilot_v3_close065.exr
  xt5_occi_plhair_gate_pilot_v3_close085.exr
  xt5_occi_plhair_gate_pilot_v3_close100.exr
```

Gate stats:

| variant | gate_mean | gate_p99 | limiter_mean |
| --- | ---: | ---: | ---: |
| v30 base gate | 0.09425 | 0.73799 | - |
| close045 | 0.08376 | 0.69558 | 0.81839 |
| close065 | 0.07910 | 0.68012 | 0.73768 |
| close085 | 0.07444 | 0.66660 | 0.65697 |
| close100 | 0.07094 | 0.65749 | 0.59649 |

ROI results:

| candidate | mean score | notes |
| --- | ---: | --- |
| close065 | 0.998 | best/tied |
| close045 | 0.998 | best/tied |
| close085 | 0.998 | best/tied |
| close100 | 0.998 | best/tied, safest flat |
| v30 ultra_s100 | 0.999 | previous best tiny win |
| base_v12 | 1.000 | baseline |

Key per-ROI effects:

- `hair_detail`: close045/065 keep score 0.992, matching v30.
- `face_center`: close045/065 keep score 0.991, matching v30.
- `noise_dark`: v30 was 1.015; close100 improves to 1.007, close065 to 1.010.
- `root`: remains around 0.997-0.998.

Crop comparisons:

```text
runs/refiner_pilot_stage11_hybrid_best/compare_v32_plhair_close_gate_occi/
```

Decision:

- This is a real, though still small, improvement over v30.
- The close-gate design is better than directly learning a sparse open gate.
- For quality-first Occi, close065 is the current preferred point: it preserves
  the hair/face improvement while reducing flat spill. close100 is safer but
  gives up a little hair/face gain.
- Next step: either turn the close-map formula into a reusable script, or train
  a tiny model to distill only the close map while keeping v30 as the positive
  opening prior.
## Stage13 v88 - PL-hair close gate script verification and flat-gate generalization check

Carried the v32 Occi PL-hair close-gate pilot into `scripts/apply_plhair_close_gate.py` and fixed two script-transfer bugs:

- shared hair/structure parameters were accidentally popped before `build_close_gate`; they are now copied for both gates.
- hair-only parameters are no longer forwarded to `build_close_gate`; close-gate kwargs are filtered by an explicit key set.
- added `--no-preview` and `--no-masks` so broad probes can avoid writing diagnostic PNGs.

Occi script validation:

```
inline v32 close065 vs script close065:
max 0.04155707359313965 mean 2.3342063286690973e-05 p99 0.00035250186920166016
```

ROI check (`runs/refiner_pilot_stage11_hybrid_best/roi_eval_v33_plhair_close_gate_script_occi/`):

| ROI | inline close065 score | script close065 score | note |
| --- | ---: | ---: | --- |
| hair_detail | 0.992 | 0.993 | reproduced useful magenta/blue tail reduction |
| root | 0.998 | 0.998 | reproduced |
| face_center | 0.991 | 0.992 | reproduced useful color-tail reduction |
| noise_dark | 1.010 | 1.011 | flat spill remains the limiting risk |

The script is close enough to the inline pilot for continued experiments, but the flat spill result confirms it should not become a global operation without a local close/refusal gate.

Flat-gate generalization probe updates:

- Updated `scripts/eval_flat_region_generalization.py` and `scripts/eval_dark_sky_coverage.py` to strip `reopen_*` / `limiter_*` preset keys before calling `build_strength_map`, matching the production guard path.
- Noisy-image-only probe over 9 EXR files with `dark_sky_strict_reopen_skyonly` was conservative: Room passed; Occi/Cat/Cat2/Ice/Z7 failed; Dance also failed because the raw noisy frame appears too structure-heavy at this stage.
- Reference+current probe on the real v12 stage behaved as intended:

| scene | guard | candidate gt040 | structure mean | interpretation |
| --- | --- | ---: | ---: | --- |
| k5_dance | pass | 0.12103 | 0.56724 | valid dark-sky target after current-stage cleanup |
| xt5_occi | fail | 0.02032 | 0.76680 | avoids hair/skin/root spill |
| k5_ice | fail | 0.02973 | 0.81596 | avoids branch/blue-shadow spill |

Design consequence: broad applicability should not be judged from a few named scenes or from the noisy input alone. The gate must be evaluated on the current processing stage and local region evidence. The promising invariant is still local: open only where flat/shadow evidence is high, structure evidence is low, and candidate risk does not add luma/chroma impulses.

