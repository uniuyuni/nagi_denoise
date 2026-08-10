# NagiV2 Pipeline Parameter Audit

Audit of every quality-affecting parameter in the production path
(`nagi_denoise.pipeline.denoise.denoise()`), run over all 8 full-resolution
diagnostic scenes in `~/ProjectData/test_photos/`.

The audit exists because two threshold defects shipped and were caught only by
accident (see *Defects found* below). Numbers here are measured, not assumed.

## Production configuration as audited

| Parameter | Value | Status |
|---|---|---|
| checkpoint | `runs/nagi_v2_l_ft2/nagi_v2_l_ft2_final.pt` (15.43M) | — |
| tile / overlap | 768 / 64, Hann-window blend | PASS (§3) |
| highlight guard | conditional: threshold 1.0 / transition 0.15 / strength 0.85, armed only when input p99.9 luma > 1.0 | PASS (§1) |
| `input_blend` | 0.20 | user decision, §4 |
| chroma pass `CH` | strength 1.0, chroma_sigma 4.5, detail_sigma 1.2, threshold 0.022, transition 0.010, highlight 1.0/0.2, hdr_restore 0.95/0.85/0.25 | PASS (§2) |
| `detail_scale` | 0.25 | inert — confidence gate closed (~0.0001) |
| `compressed_output_clamp` | 2.0 | PASS — observed compressed range −0.48…1.63, no clipping |

## Defects found and fixed

1. **Highlight guard shipped disabled.** The checkpoint carries
   `highlight_protect_strength=0.0`, so the model crushed isolated specular
   highlights on low-dynamic-range files (Z7 fix: top-1% luma retention 0.62,
   max 1.83 → 0.87). `denoise()` now re-arms it at inference.
2. **The first fix was itself wrong.** An adaptive threshold
   `min(1.0, 0.7·p99.9)` engaged over large areas of low-range scenes; because
   the guard blends the *noisy input* back in, it raised flat-region noise 3.7×
   on Z7 bird and read visually as uneven denoising. Replaced by the
   conditional policy above.

Lesson carried forward: a highlight metric must always be read together with a
noise metric. A variant that improves retention while raising noise is not a fix.

## §1 Highlight guard — all 8 scenes

`highlight retention (top 1% of input luma) / flat-region noise`

| scene | p99.9 | guard off | conditional (production) |
|---|---:|---|---|
| occi | 4.03 | 0.998 / 0.0037 | **1.000 / 0.0037** |
| room | 3.90 | 0.999 / 0.0024 | **1.000 / 0.0024** |
| ice | 3.09 | 0.997 / 0.0022 | **1.000 / 0.0022** |
| dance | 3.02 | 0.994 / 0.0008 | **0.996 / 0.0008** |
| night | 1.13 | 0.740 / 0.0068 | 0.774 / 0.0070 |
| fix | 0.65 | 0.621 / 0.0028 | disarmed (= off) |
| bird | 0.51 | 0.789 / 0.0069 | disarmed (= off) |
| cat | 0.42 | 0.918 / 0.0006 | disarmed (= off) |

On true-HDR scenes the guard reaches ~1.000 retention at **identical** noise —
a free win. On low-range scenes it stays disarmed, so it cannot cause the
unevenness of defect 2.

**Known limitation:** with the guard disarmed, peak retention on Z7 fix/bird
stays 0.62–0.92. That is the model treating small isolated specular highlights
as impulse noise, and it is not fixable post-hoc without re-introducing noise.
A model-level fix (training data containing isolated speculars) is the real
remedy.

## §2 Chroma pass — all 8 scenes, `saturated` / `detail` / `flat` ROIs

Swept `chroma_sigma ∈ {2.0,3.0,4.5,6.5}`, `threshold ∈ {0.012,0.022,0.035}`,
`strength ∈ {0.5,0.75,1.0}`, plus `chroma_cleanup=False` as the reference.

**Colour damage: none found.**

| check | result |
|---|---|
| desaturation vs no-chroma-pass | max 4.2% (bird); most < 1.5%; occi/dance slightly *more* saturated |
| dependence on `chroma_sigma` | sat moves only 1–3% across sigma 2.0→6.5, and *upward* on occi — the wide blur is not eroding colour |
| hue / per-channel deviation | < 2% on 7 of 8 scenes |

**Luma detail cost: zero.** Across all 8 scenes the `detail` ROI HF changes by
−0.96%…+0.15% with the chroma pass on vs off — i.e. within measurement noise.
This confirms on every scene what was first observed on Ice: the chroma pass is
not what makes any scene look flat.

**The pass earns its place**, though unevenly: chroma noise reduction is
2.8–33.6% on detail ROIs and 0.1–22.0% on flat ROIs (marginal on cat and dance).

**Verdict:** a single global `CH` setting is defensible. No scene needs
adaptation, so none is introduced — an adaptive rule here would carry the same
unevenness risk as defect 2 for no measured benefit.

**One flagged anomaly:** on the Z7 night `saturated` ROI (a light source; its
noisy saturation of 55.6 shows extreme outlier pixels) the pass lowers all
channels 6–10% with a slight warm→cool differential (R −10.4%, G −5.8%,
B −7.4%). It is **identical at sigma 2.0, 4.5 and 6.5**, so it originates in the
highlight handling, not the blur width. Localised; not addressed here.

## §3 Tiling — seams are not visible

Two tests were run. The first (differencing tile-768 against tile-512 output)
showed rectangular structure, but that is expected: two grids distribute model
context differently, and it says nothing about either output alone.

The decisive test examines a **single** output for discontinuity at its own tile
boundaries — mean row-to-row luma difference in a ±2px band at each boundary
line versus all other rows:

| scene | boundary | elsewhere | ratio | worst boundary line | all-rows p99.9 | verdict |
|---|---:|---:|---:|---:|---:|---|
| occi | 0.018037 | 0.019018 | 0.948 | 0.024958 | 0.037365 | hidden |
| room | 0.013841 | 0.015052 | 0.920 | 0.016478 | 0.144100 | hidden |
| ice | 0.019157 | 0.016388 | 1.169 | 0.033022 | 0.050513 | hidden |
| dance | 0.003955 | 0.006550 | 0.604 | 0.005666 | 0.065384 | hidden |
| night | 0.032706 | 0.031844 | 1.027 | 0.034649 | 0.040003 | hidden |
| fix | 0.021553 | 0.021235 | 1.015 | 0.023173 | 0.028162 | hidden |
| bird (2048² crop) | 0.013814 | 0.012687 | 1.089 | 0.014545 | 0.016116 | hidden |
| cat | 0.003149 | 0.003206 | 0.982 | 0.003408 | 0.006242 | hidden |

**On every scene the worst tile-boundary row is less discontinuous than the top
0.1% of ordinary image rows** — seams are buried in normal image variation.
Four scenes show boundary rows *smoother* than average (ratio < 1).

## §4 `input_blend = 0.20`

User decision, to restore apparent detail. The trade-off is proportional and
measured — Occi hair ROI, structure-vs-noise selectivity:

| blend | retention on-structure | off-structure | selectivity |
|---:|---:|---:|---:|
| v12 (legacy) | 82.1% | 72.3% | +9.8pt |
| 0.00 | 92.2% | 80.9% | +11.3pt |
| **0.20** | **93.5%** | 84.1% | **+9.4pt** |
| 0.30 | 94.2% | 85.9% | +8.3pt |

At 0.20 the pipeline holds ~11pt more on-structure detail than v12 at
comparable selectivity. Set `input_blend=0.0` for the cleanest output.

## Not verified

- Per-scene visual sign-off on the `night` channel anomaly in §2.
- Chroma sweeps on the `bright` and `evenness` ROIs (only `detail`, `flat`,
  `saturated` were swept).
- Behaviour on images outside the 8 diagnostic scenes.

## Artefacts

- `runs/audit/sheets/*.png` — per-scene contact sheets (evenness / detail / flat / bright).
- `runs/audit/chroma/*.png` — chroma sheets (noisy / no-chroma-pass / production).
- `runs/audit/metrics/*.json` — raw per-scene measurements.
- `runs/audit/tile_check/*.png` — tile-difference heatmaps.
