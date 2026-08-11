# Nagi Denoise

A blind denoiser for real high-ISO photographs, built to be safe on
linear-light HDR data.

You give it a float32 linear-light RGB image; it gives back the same array,
denoised. Values above 1.0 (specular highlights, light sources, blown skies)
are preserved rather than clipped. Images of any size are handled by tiling
that leaves no visible seams. The pipeline is deterministic: same input, same
output, every time.

It is aimed at the case a general-purpose denoiser handles badly — a 40MP raw
frame developed to linear EXR, where the noise is spatially correlated colour
grain sitting on top of fine structure (hair, fabric, foliage) that must
survive.

**What it is not:** a real-time filter, a general image restorer
(no deblur / super-resolution / JPEG artefact removal), or a detail
*generator*. It removes noise; it does not invent texture.

---

## Install

Requires macOS on Apple Silicon (the environment is pinned to `osx-arm64`) and
[pixi](https://pixi.sh).

```bash
pixi install
pixi run check-env     # torch / MPS sanity check
pixi run test          # unit tests
```

`coremltools` is optional. Without it everything works except
`backend="coreml"`.

## Quick start

```python
import numpy as np
from nagi_denoise import denoise

img = np.load("noisy.npy")            # (H, W, 3) float32, linear-light RGB
out = denoise(img)                    # same shape, same dtype, HDR preserved
out = denoise(img, backend="coreml")  # ~3.6x faster model stage on Apple Silicon
```

That is the whole API for normal use. `denoise()` loads the production
checkpoint by default and caches it, so repeated calls in one process pay the
load cost once.

## CLI

```bash
# Production pipeline. Reads .exr / .tif, writes .exr / .tif.
pixi run denoise --input noisy.exr --output clean.exr

# Same, through the Core ML backend.
pixi run denoise --input noisy.exr --output clean.exr --backend coreml

# Every Python knob has a flag:
pixi run denoise --input a.exr --output b.exr --input-blend 0.0 --highlight-guard off
```

Installed as the console script `nagi-denoise-pipeline`; `pixi run denoise` is
a shorthand for it. `pixi run denoise -- --help` lists all flags.

There is a second, *model-only* CLI (`nagi-denoise`, `pixi run denoise-model`)
that runs just the network on a single file and also accepts sRGB PNG/JPEG. It
skips the chroma pass, the highlight guard and `input_blend`, so it does **not**
produce production output. Use it for quick model checks only.

## Knobs

All are keyword-only arguments to `denoise()`.

| knob | default | change it when |
|---|---|---|
| `weights` | production checkpoint | you trained your own NagiV2 checkpoint |
| `device` | `"auto"` | you need to pin `"cpu"` / `"mps"` (auto prefers MPS, then CUDA, then CPU) |
| `backend` | `"torch"` | you want speed: `"coreml"` is ~3.6x faster on the model stage. `"torch"` is the reference — all quality numbers below were measured on it |
| `coreml_package` | exported production `.mlpackage` | you exported your own graph (`$NAGI_DENOISE_COREML_PACKAGE` also works) |
| `coreml_compute_units` | `"cpu_and_gpu"` | **essentially never.** `"all"` lets Core ML use the Neural Engine, which corrupts this graph — see the warning below |
| `tile` | `768` | you are memory-constrained (smaller tiles, more of them). Must equal the exported tile size when `backend="coreml"` |
| `overlap` | `64` | practically never; 64px of Hann overlap is what makes tiling seamless |
| `chroma_cleanup` | `True` | you want the bare model output, e.g. to measure what the chroma pass contributes |
| `input_blend` | `0.20` | you want a cleaner or grittier result. `0.0` is the cleanest possible output; higher restores apparent detail *and* noise in proportion (measured on the K-5 Ice detail ROI: 0% → HF 0.00242 / chroma-noise 0.00186; 15% → 0.00316 / 0.00294; 30% → 0.00414 / 0.00443) |
| `detail_strength` | `None` | never, on the shipped checkpoint — the confidence gate is closed, so this is inert by design. It is wired up so a future checkpoint that opens the gate activates it automatically |
| `highlight_guard` | `True` | `False` disables it; a float pins an explicit luma threshold. Default is conditional: armed only when the image really contains above-SDR content |
| `highlight_guard_transition` | `0.15` | tuning the guard's blend softness (luma units) |
| `highlight_guard_strength` | `0.85` | tuning how hard the guard blends toward the input above the threshold |
| `batch_size` | `1` | benchmarking non-M1 hardware. On this M1 larger batches are *slower* (b2 273s, b4 1014s, vs b1 237s) |
| `amp_dtype` | `None` | benchmarking. fp16 autocast bought nothing here (247s vs 237s); use `backend="coreml"` for the real speedup |

### Core ML: never use `compute_units="all"`

`ALL` lets Core ML dispatch to the Apple Neural Engine, and the ANE computes
this fp16 graph wrongly. On every one of the 23 validation tiles in
`runs/phase5_speed/coreml/validation_report.json`, peak output values run
1.25x–4.8x higher than the PyTorch reference (worst absolute per-pixel error
8.7 on a scene whose true peak is 5.3) — visually, blown-out garbage. The same
graph on `cpu_and_gpu` agrees with PyTorch to 0.018 max, which is just fp16
rounding. `"all"` remains *selectable* so it can be re-benchmarked, but it is
never a default anywhere in this codebase.

## Measured quality and speed

Model: NagiV2-L, 15.43M parameters
(`runs/nagi_v2_l_ft2/nagi_v2_l_ft2_final.pt`).

| benchmark | result |
|---|---|
| SIDD Validation (sRGB PSNR) | **39.030 dB** |
| — NAFNet-w64 teacher, for reference | 40.21 dB, at 116M parameters (7.5x the size) |
| — the retired in-house NagiNR-M | 37.46 dB |

**Structure-vs-noise selectivity**, X-T5 Occi hair ROI — how much more
high-frequency energy is kept on structure than on flat noise:

| pipeline | selectivity | retention on structure |
|---|---|---|
| NagiV2 (this) | **+11.2 pt** | **90.5%** |
| legacy v12 pipeline | +9.8 pt | 82.1% |

More detail retained *and* cleaner — not a trade. (A second mask definition
used by `scripts/phase5_gates.py` reports 20.6 pt for this same comparison.
The two definitions are not interchangeable; never mix numbers across them.)

**HDR highlight retention** ≥ 0.99 on every scene with true HDR content;
1.0000 on X-T5 Room.

**Seams**: across all 8 diagnostic scenes, the worst tile-boundary row is
*less* discontinuous than the top 0.1% of ordinary image rows — i.e. tile
boundaries are indistinguishable from normal image content.

**Speed**, 39.8MP frame, end to end:

| path | time |
|---|---|
| Core ML fp16 / `cpu_and_gpu` | **~83 s** (model 65 s + chroma 7.7 s + I/O ~10 s) |
| pure PyTorch / MPS | ~261 s |

## Requirements the pipeline satisfies

These are hard requirements, verified on the 8 full-resolution diagnostic
scenes, not aspirations:

1. **HDR-safe.** Linear values above 1.0 pass through. The model works in a
   reversible `asinh`-compressed space, so a 10.0 highlight (K-5 Dance) comes
   out at 10.0, and nothing in the pipeline clips.
2. **Seam-free at any resolution.** Tiles are blended with a 2D Hann window
   and the highlight-guard threshold is computed once for the whole image, so
   no per-tile decision can print a boundary into the output.
3. **It never breaks an image.** Non-finite input is sanitised on the way in;
   the output is asserted finite on the way out. There is no adaptive branch
   that can behave differently on two halves of a picture.
4. **Deterministic.** No randomness, no auto-tuning. Same bytes in, same bytes
   out.

## Repo layout

```
nagi_denoise/                 # the package
├── __init__.py               # public API: denoise(), Denoiser, NagiV2, helpers
├── pipeline/denoise.py       # THE entry point: denoise() + the production CLI
├── coreml.py                 # optional Core ML backend (lazy coremltools import)
├── infer.py                  # Denoiser: checkpoint load + Hann-window tiled inference
├── cli.py                    # legacy model-only CLI (`nagi-denoise`)
├── models/                   # nagi_v2.py (production), nagi_nr.py (retired)
├── transforms.py             # sRGB<->linear, asinh compress/decompress
├── losses.py                 # NagiV2Loss and friends
├── data.py                   # SIDD loader, augmentation
├── train/                    # trainers
├── bench/                    # SIDD Validation evaluator + vendored NAFNet/SCUNet
└── pipeline/…                # the chroma pass, plus research/diagnostic drivers
configs/                      # training configs (incl. abandoned routes, marked)
scripts/                      # Core ML export/validation, quality gates, supervisors
tests/                        # unit tests (`pixi run test`)
docs/                         # architecture + full research history — see docs/README.md
runs/                         # checkpoints and delivered outputs (see below)
benchmarks/                   # third-party teacher weights (NAFNet-w64, SCUNet)
data/, SIDD_Medium_Srgb/, PolyU-…/   # datasets (local, not in git)
```

`runs/` after the v1.0 cleanup:

| path | what |
|---|---|
| `nagi_v2_l_ft2/nagi_v2_l_ft2_final.pt` | **the production checkpoint** |
| `nagi_v2_l/nagi_v2_l_final.pt` | the Phase 1 base it was fine-tuned from (16 days of training) |
| `phase5_speed/coreml/*.mlpackage` | exported Core ML graphs (fp16 and fp32, tile 768) |
| `final_full/` | delivered full-resolution outputs + previews |
| `baseline_v12/` | frozen legacy-pipeline reference used by the audit |
| `audit/` | contact sheets and metrics behind `docs/nagi_v2_parameter_audit.md` |

Full-resolution diagnostic photos (noisy EXR plus PhotoLab / SCUNet
references) live outside the repo at `~/ProjectData/test_photos`.

## Further reading

- `docs/architecture.md` — how the two stages work and why the design is what
  it is.
- `docs/nagi_v2_parameter_audit.md` — every quality-affecting parameter,
  measured across all 8 scenes.
- `docs/README.md` — index of everything in `docs/`, including which files are
  historical.

## License

MIT.
