# NAFNet Core ML Practical Path

## Position

This is not Nagi.

It is a practical NAFNet Core ML inference path for real EXR denoising on the
local Mac. Keep it separate from Nagi model design and training decisions.

The useful result is:

```text
full NAFNet fp16 Core ML
tile 256
batch 4
compute_units cpu_and_gpu
process_scale 1.0
```

This keeps native-resolution quality and avoids the residual noise introduced by
sub-resolution residual upsampling.

## Current Measurements

Real EXR:

```text
input: samples/coreml_exr_input/sample_cat_noisy.EXR
shape: 5152 x 7728
```

Safe full NAFNet crop measurements:

| path | compute units | batch | ms/tile | correctness |
| --- | --- | ---: | ---: | --- |
| full fp16 b1 | cpu_and_gpu | 1 | 142.8 | reference |
| full fp16 b4 | cpu_and_gpu | 4 | 135.5 | PSNR 93.108 dB vs b1 |
| full fp16 b1 | all | 1 | 66.7 | broken, clipped output |
| full fp16 b4 | all | 4 | 77.3 | broken, clipped output |
| full fp16 b1 | cpu_and_ne | 1 | 78.7 | broken, clipped output |
| full fp16 b1 | cpu_only | 1 | 214.8 | correct but slow |
| full fp32 b1 | all | 1 | 282.0 | correct but slow |

Safe P3 crop measurements:

| path | compute units | batch | ms/tile | correctness |
| --- | --- | ---: | ---: | --- |
| P3 fp16 b1 | cpu_and_gpu | 1 | 145.0 | reference |
| P3 fp16 b4 | cpu_and_gpu | 4 | 128.3 | PSNR 94.888 dB vs b1 |

Full-image safe measurements:

| path | compute units | batch | elapsed | ms/tile | note |
| --- | --- | ---: | ---: | ---: | --- |
| full fp16 b4 native | cpu_and_gpu | 4 | 93.0 s | 142.9 | rejected as final: two bad 256px blocks |
| full fp16 b4 native + 512px context patch | cpu_and_gpu | mixed | - | - | current recommended output |
| full neuralnetwork native | all | 1 | 43.2 s | 66.3 | rejected: deterministic bad tiles |
| P3 fp16 b4 native | cpu_and_gpu | 4 | 88.4 s | 135.7 | slightly faster, small color shift |

The apparent ANE speedup from `compute_units=all` or `cpu_and_ne` is invalid for
the fp16 MLProgram exports. The output clips/saturates randomly, with median
denoised sRGB near 1.0. Do not use MLProgram fp16 with `all` or `cpu_and_ne`
unless correctness is revalidated.

## ANE Investigation: Rejected

The public Core ML Tools documentation says:

- `ComputeUnit.ALL` may use the Neural Engine, CPU, and GPU.
- ML Program conversion defaults to fp16 precision unless overridden.
- ML Program tensors are explicitly typed, and the runtime respects those types
  as minimum precision.
- Float32 typed ML Programs avoid NE execution.
- Selective fp32 preservation is possible with `FP16ComputePrecision`.

That matches the failed MLProgram observation:

```text
fp16 + all/cpu_and_ne: fast, but invalid clipped output
fp32 + all: correct, but slow and effectively not an ANE speed path
fp16 + cpu_and_gpu: correct, current safe route
```

A first selective-fp32 export was tested with these ops preserved:

```text
reduce_mean, sub, square, sqrt, real_div, mul
```

Output package:

```text
runs/nafnet_fast_coreml/nafnet_width64_fp16_b1_256_mixed_norm_mul.mlpackage
```

However, loading/running this package with `compute_units=all` did not reach the
first 512px crop prediction after more than two minutes, so this broad mixed
precision partition is not practical. A narrower op search could still be tried,
but every candidate must pass output comparison against the `cpu_and_gpu`
reference before using any speed number.

The older `neuralnetwork` Core ML format initially looked like a practical
workaround:

```bash
pixi run export-coreml-nafnet-full-nn
```

Measured with:

```text
runs/ane_coreml_experiment/nafnet_width64_neuralnetwork_b1_256.mlmodel
compute_units: all
tile: 256
batch: 1
```

Compute-unit split check on the 1536 crop:

| compute units | ms/tile | interpretation |
| --- | ---: | --- |
| all | 66.3 | fast path |
| cpu_and_ne | 66.5 | same speed as all; NE is the fast path |
| cpu_and_gpu | 1274.7 | much slower; GPU is not the fast path for neuralnetwork export |

Core ML does not expose a pure "ANE only" mode. `cpu_and_ne` still allows CPU
participation for scheduling, I/O, and unsupported pieces, but this result shows
the fast neuralnetwork path is not the GPU path.

Manual CPU+NE and CPU+GPU parallel tiling was also tested on the same 1536 crop:

| GPU tile fraction | elapsed | ms/tile | result |
| ---: | ---: | ---: | --- |
| 0.0 | 2.23 s | 61.9 | CPU+NE only baseline |
| 0.2 | 3.03 s | 84.2 | slower |
| 0.3 | 3.33 s | 92.5 | slower |

Adding GPU work in parallel slows the NE worker, likely due to shared memory
bandwidth / system contention. Manual NE+GPU splitting is not useful for this
model on the tested machine.

Initial results:

| test | elapsed | ms/tile | correctness |
| --- | ---: | ---: | --- |
| 256 crop | 0.1 s | 63.7 | PSNR 71.281 dB vs cpu_and_gpu reference |
| 1536 crop | 2.4 s | 66.3 | PSNR 73.680 dB vs cpu_and_gpu reference |
| full EXR | 43.2 s | 66.3 | warnings none, but later rejected |

Full-image comparison against the safe `cpu_and_gpu` MLProgram output looked
mostly close:

```text
PSNR: 39.860 dB
mean abs: 0.0003299
p99 abs: 0.000580
p999 abs: 0.005707
abs diff > 0.01: 0.0613% of channel values
mean delta RGB: +0.00002459, -0.00000657, +0.00008825
```

However, full-image tile analysis found deterministic bad tiles:

```text
bad tile 1: x=0,    y=4864, mean_abs=0.050075, max_abs=1.0
bad tile 2: x=1280, y=3328, mean_abs=0.042868, max_abs=1.0
```

Running those same tiles in isolation reproduced the error. This is not a random
one-off scheduling failure; it is a deterministic neuralnetwork/ANE correctness
failure on some content. Therefore the neuralnetwork ANE path is rejected for
quality-first use.

## Recommended Commands

Export safe full NAFNet batch-4 package:

```bash
pixi run export-coreml-nafnet-full-fp16-b4
```

Run safe full native EXR denoise:

```bash
pixi run denoise-exr-nafnet-full-safe
```

Export safe P3 batch-4 package:

```bash
pixi run export-coreml-nafnet-p3-fp16-b4
```

Run safe P3 native EXR denoise:

```bash
pixi run denoise-exr-nafnet-p3-safe
```

## Quality Notes

The full native path removes the fine residual grain that remained in the
`process_scale < 1` residual-upsample outputs.

P3 native also removes the visible fine noise in the tested crop, but has a
small color shift versus full NAFNet. On the center crop, P3 is slightly cooler /
more magenta:

```text
full mean: R 0.401222 / G 0.308670 / B 0.309955
P3 mean:   R 0.403414 / G 0.306891 / B 0.312863
```

A simple affine color match improved P3-vs-full crop PSNR from 42.593 dB to
43.156 dB, so the P3 error is partly low-frequency color bias rather than
remaining noise.

On the full safe EXR output, P3 versus full has:

```text
PSNR: 28.715 dB
mean delta RGB: +0.001958, -0.001558, +0.002223
mean abs delta RGB: 0.006312, 0.004599, 0.006983
```

## Decision

For review, use the corrected output:

```text
runs/_recommended_quality_corrected/
```

It starts from the full MLProgram CPU/GPU output and replaces the two known bad
256px blocks with 512px context outputs:

```text
x=0,    y=4864
x=1280, y=3328
```

The unpatched `runs/coreml_exr_outputs_full_fp16_b4_cpu_gpu_native/` output is
not recommended for review.

Bad-tile detection is now automated with:

```bash
pixi run python scripts/detect_coreml_artifact_tiles.py \
  --input runs/coreml_exr_outputs_quality_corrected/sample_cat_noisy_input_srgb16.tiff \
  --output runs/coreml_exr_outputs_quality_corrected/sample_cat_noisy_coreml_nafnet_srgb16.tiff
```

The detector uses tile-level high-frequency chroma growth. It finds the two bad
tiles in the unpatched output and reports zero tiles in the corrected output.

Do not use ANE outputs for quality work unless a future experiment proves exact
tile-level correctness on the full image.

P3 is only a secondary speed candidate that needs color review:

```text
test P3 fp16 b4 cpu_and_gpu native
```

For speed experiments:

```text
do not trust random forward latency alone
always compare output against the cpu_and_gpu reference
```
