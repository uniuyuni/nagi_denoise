# Perfect NR Research Project

## Goal

Build a production-grade noise reduction system for Platypus-grade real images.

This project does not start from "make NAFNet faster" or "make SCUNet CoreML work".
It starts from the image formation and failure modes we actually observed:

- real input is linear/HDR, not only SIDD sRGB;
- values above 1.0 are common and meaningful;
- highlight fabric, hair, dark thin lines, and saturated points expose failures first;
- Core ML / ANE can be a backend, but it is not a quality guarantee;
- low-frequency color transfer can fix color drift and still destroy highlight detail.

Success means the model is boringly reliable in real photographs, not just high PSNR
on a benchmark.

Priority:

1. quality
2. inference speed
3. training speed
4. model size

## Non-Negotiable Gates

A candidate is not accepted unless it passes all gates below.

| gate | required behavior |
| --- | --- |
| HDR range | no color cast or texture collapse on linear EXR values above 1.0 |
| highlight detail | white fabric / curtains retain plausible detail without cyan/purple drift |
| dark thin lines | no checkerboard or block corruption in hair, whiskers, wires, text |
| color stability | low-frequency hue and white balance stay close to input unless noise demands otherwise |
| tile stability | overlap/tile boundaries are not visible at production tile sizes |
| benchmark sanity | SIDD/PolyU metrics do not regress below known baselines |
| backend sanity | PyTorch and Core ML outputs are close enough on diagnostic crops before full-image use |

## Lessons From The Previous Line

### NAFNet / Core ML

NAFNet quality is real, but Core ML conversion produced unacceptable artifacts in
real images. The most damaging failures were thin-line corruption and local color
breakage. Using `ALL` did not reliably mean "faster"; it often meant expensive or
unstable backend placement.

### SCUNet / Core ML

SCUNet was not automatically safe either. The important discovery was that the
input preprocessing was wrong for HDR:

```text
old: log1p(k * x) / log1p(k)
```

For HDR EXR crops this sent values above 1.0 into a network trained for roughly
bounded image values. On the X-T5 Room crop, network input reached about 1.7.

The safer transform is HDR-white-aware:

```text
work = log1p(k * x) / log1p(k * white)
```

This fixes the first failure: do not feed out-of-distribution highlight values
to the denoiser.

### Low-Frequency Transfer

The existing `apply_low_frequency_transfer` function does not mean "restore all
original highlight detail".

Its highlight branch computes a mask from the reference image and subtracts
a fraction of the restored high-frequency component:

```text
high_restored = restored - blur(restored)
alpha = mask * (1 - highlight_detail_strength)
output = restored + low_diff - alpha * high_restored
```

Therefore:

- `highlight_detail_strength=1.0` keeps restored high frequency;
- `highlight_detail_strength=0.2` removes most restored high frequency;
- strong visible fabric detail after low values may be the original/low-frequency
  structure becoming visible, not SCUNet detail being preserved.

This function is useful but conceptually overloaded. Perfect NR should split
color stabilization, highlight protection, and detail restoration into separate
stages.

## Research Hypothesis

Perfect NR should be a small pipeline, not a single magic model:

```text
linear/HDR input
  -> deterministic HDR normalization
  -> neural denoise in bounded working space
  -> confidence / residual analysis
  -> color-stable reconstruction
  -> highlight/detail guard
  -> tile-safe output
```

The neural model should remove noise. It should not be trusted as the sole owner
of color, HDR reconstruction, or highlight detail.

## Model Direction

The first model family should be called `NagiPerfect` until it earns a better name.

Initial design constraints:

- input/output: linear RGB float32;
- internal space: HDR-white-aware log or learned bounded transform;
- output: residual in bounded space plus optional confidence map;
- architecture: U-Net-like local model with NAF-style blocks only where they pay
  for themselves;
- avoid attention/softmax in the production path unless it proves useful on
  diagnostic crops;
- optional side outputs:
  - noise residual estimate;
  - detail confidence;
  - chroma confidence;
  - highlight protection mask.

After the first highlight-detail probe, the side outputs should be promoted from
"optional" to the default experiment shape:

- base RGB / residual path for denoising;
- luma detail residual path for highlight and thin-line recovery;
- detail confidence gate so random noise is not restored as texture;
- highlight mask or HDR-range confidence for values above ordinary SDR white.

The important constraint is that the detail path should not own chroma. Chroma
comes from the stable base path unless a later experiment proves otherwise.

The first candidate does not need to beat NAFNet on SIDD. It must first pass the
diagnostic real-image gates.

## Dataset Strategy

Use three dataset types separately, not as one soup:

| dataset | role |
| --- | --- |
| SIDD sRGB | benchmark sanity and PSNR comparability |
| PolyU real/mean | real noise behavior with paired averaged target |
| HDR EXR diagnostic crops | failure reproduction and acceptance gates |

The HDR diagnostic set is small but sacred. It should include the exact crops
that broke NAFNet/SCUNet/CoreML:

- X-T5 Room curtain highlight;
- cat hair / whisker crop;
- dark thin-line artifact crop;
- saturated point / colored highlight crop;
- tile boundary stress crop.

## Evaluation Protocol

Each experiment must report:

- model or pipeline revision;
- preprocessing transform and white-point rule;
- tile size and overlap;
- backend and compute units;
- runtime per tile and full crop;
- linear HDR range stats;
- sRGB preview only as a preview, never as the sole judgment;
- crop-level notes: detail, color cast, seams, hallucination, residual noise.

Metrics:

| metric | purpose |
| --- | --- |
| PSNR/SSIM on SIDD | benchmark sanity |
| PolyU PSNR/LPIPS if available | real paired sanity |
| chroma drift in highlights | detect purple/cyan failure |
| high-frequency retention in selected masks | check detail loss |
| tile-difference heatmap | detect seams |
| manual crop verdict | final gate for real-image failures |

## Phase Plan

### Phase 0: Diagnostic Harness

No training. Build a repeatable crop runner that can compare:

- input;
- current Platypus SCUNet path;
- Core ML SCUNet path;
- transfer off;
- transfer variants;
- future NagiPerfect candidates.

Output EXR plus consistent preview PNG, metadata JSON, and a short markdown row.

### Phase 1: Baseline Pipeline

Lock a "known rational" pipeline:

```text
HDR-white log -> denoise -> inverse -> no low-frequency transfer
```

Then test whether a separated color/detail guard beats low-frequency transfer.

Phase 1A result:

```text
no_transfer + coherent highlight luma detail guard
```

This beat the old transfer abstraction on the X-T5 Room crop because it restored
highlight texture without pulling top-highlight luma down by more than one stop.
The model design should absorb this as a separate luma-detail head rather than
keeping it as an ad hoc postprocess.

### Phase 2: Small Model Overfit

Train a tiny `NagiPerfect-S` on a handful of patches and verify:

- identity-start behavior;
- no highlight color inversion;
- residual learns noise before structure;
- inference speed is in the practical range.

Initial model skeleton:

```text
NagiPerfect
  shared HDR-log trunk
  base RGB residual head
  luma detail residual head
  detail confidence head
  chroma-preserving reconstruction
```

The first training run should use `perfect-s`, not `perfect-m`. The goal is not
final quality yet; it is to prove the split-head objective learns the right
behavior before spending long training time.

### Phase 3: Real Dataset Training

Add SIDD/PolyU only after Phase 0/1 gates are automated. Use short runs first.

### Phase 4: Backend

Only export to Core ML after PyTorch passes diagnostic crops. Test:

- `mlprogram + cpu_and_gpu`;
- no ANE unless it beats cpu_and_gpu on both speed and quality;
- no full-image run before diagnostic crops pass.

## First Decisions

1. Stop using full-image runs as the primary experiment loop.
2. Treat HDR preprocessing as part of the model contract.
3. Split low-frequency color transfer from highlight detail protection.
4. Keep Core ML as an implementation backend, not a design goal.
5. Keep all future claims tied to a diagnostic crop and a reproducible output path.
