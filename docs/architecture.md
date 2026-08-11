# NagiV2 Architecture

How the shipped pipeline works, and why each part is shaped the way it is.
Every number here was measured; the raw evidence is in
`docs/nagi_v2_parameter_audit.md` (parameter sweeps over all 8 diagnostic
scenes), `runs/audit/` (contact sheets and metrics) and
`runs/phase5_speed/coreml/` (Core ML validation and timing).

Code: `nagi_denoise/pipeline/denoise.py` (`denoise()`),
`nagi_denoise/infer.py` (tiling), `nagi_denoise/models/nagi_v2.py` (network),
`nagi_denoise/pipeline/flat_chroma_smoother.py` (chroma pass),
`nagi_denoise/coreml.py` (Core ML backend).

---

## 1. Two stages, and nothing else

```
float32 linear HWC RGB
      │
      ├─ sanitise (NaN/Inf → 0)
      ├─ resolve highlight-guard policy   ← once, on the whole image
      │
      ▼
 [1] NagiV2-L, tiled                       torch (reference) or Core ML
      │   768px tiles, 64px overlap, 2D Hann blend
      │   in-graph: asinh compress → U-Net → decompress
      │   in-graph: input-luma highlight guard
      ▼
 [2] deterministic chroma pass             flat_chroma_smoother.smooth_chroma
      │   one fixed parameter set (CH), no adaptation
      ▼
      ├─ input_blend (default 0.20)
      ▼
float32 linear HWC RGB, finite, HDR intact
```

That is the entire production path. Two stages, one fixed parameter set each,
no branching on image content beyond the single highlight-guard decision
described in §5.

### Why the legacy stack was retired

The pre-NagiV2 pipeline (`runs/baseline_v12/`, still reachable through the
research driver `nagi_denoise/pipeline/denoise_exr.py`) was a long chain of
hand-tuned post-filters: chroma speckle removal, luma HF shrink, luma tail
speckle, guided luma smoothing, region-aware luma cleanup, flat-region
cleanup, detail guards, HDR highlight restore — each with its own preset
family, dispatched per-image by a "preset chooser", and driven by 31 CLI
flags. Roughly forty tunable filter stages in total.

It was retired for three reasons, in order of importance:

1. **It could not be reasoned about.** Presets interacted. A change that fixed
   one scene moved three others, and no one could say in advance which. The
   preset chooser made this worse: two similar images could take different
   filter paths, so behaviour was discontinuous in the input.
2. **Per-image and per-region decisions print themselves into the output.**
   Any rule that varies across an image will eventually vary *within* one, and
   uneven denoising is far more objectionable than uniformly imperfect
   denoising. §5 is a concrete case where exactly this happened and had to be
   reverted.
3. **A trained model does the same job better.** NagiV2-L beats the v12
   pipeline on structure-vs-noise selectivity by every measure we have:
   +11.2 pt at 90.5% structure retention, versus +9.8 pt at 82.1% — more
   detail kept *and* less noise, not a trade.

What survived is the chroma pass, and only because it was measured to earn its
place: it cuts chroma noise 2.8–33.6% on detail ROIs at a luma-detail cost of
−0.96%…+0.15% across all 8 scenes, i.e. within measurement noise. Its
parameters (`CH` in `pipeline/denoise.py`) are one global setting. The audit
explicitly considered making them adaptive and rejected it — no scene needed
adaptation, so introducing one would have bought the risk of reason (2) for no
measured benefit.

The retired code stays in the repository. It is research history and it still
reproduces the frozen baseline; it is simply not on the production path.

## 2. HDR handling: asinh, in-graph

Real input is linear light. Highlights routinely exceed 1.0 — K-5 Dance peaks
at 10.0. A network trained on roughly-bounded data will do arbitrary things
with a value of 10, and the previous line proved it: SCUNet's `log1p(k·x)`
preprocessing sent HDR crops far outside its training range and produced
colour breakage.

NagiV2 compresses on the way in and decompresses on the way out, inside the
graph (`NagiV2.compress` / `decompress`):

```
compress(x)   = asinh(k·x) / asinh(k)          k = 8
decompress(y) = sinh(y·asinh(k)) / k
```

Properties that matter:

- **Exactly reversible.** Not a tone map. `decompress(compress(x)) == x`.
- **Near-linear near zero, log-like far from it.** Shadow noise keeps its
  scale; a 10.0 highlight lands around 1.5 instead of 10, so the network sees
  one bounded regime.
- **Defined for negatives.** Linear EXR data from raw development legitimately
  contains small negative values (black-point subtraction); `asinh` is odd and
  handles them, where `log1p` cannot.

Everything downstream — the residual heads, the loss, `compressed_output_clamp`
— works in compressed space. The audit observed a compressed range of
−0.48…1.63 in production, comfortably inside the 2.0 clamp, so the clamp never
fires on real data.

## 3. Seam-free tiling

A 40MP frame cannot be forwarded whole, so `Denoiser._tiled_forward` cuts it
into 768px tiles with 64px overlap, weights each tile by a 2D Hann window, and
normalises by the accumulated weight. The window goes to zero at tile edges, so
each output pixel is a smooth partition-of-unity blend of every tile covering
it. There is no hard boundary anywhere to be visible.

The measured check (`scripts/phase5_gates.py`, gate 3) does not compare two
tilings against each other — that only tells you they differ, which they must.
It examines a **single** output for discontinuity at its own tile boundaries:
mean row-to-row luma difference in a ±2px band at each boundary, against the
99.9th percentile of the same statistic over all rows. On all 8 scenes the
worst boundary row is *below* that percentile — tile boundaries look like
ordinary image content.

The tiling scheme is the reason the highlight-guard threshold must be a
whole-image decision (§5): a per-tile threshold would make adjacent tiles
disagree about how bright "the image" is, and that disagreement would print a
seam.

## 4. The confidence-gated detail head (inert by design)

`NagiV2` has three output heads on a shared trunk:

- a **base** head predicting an RGB residual in compressed space (the denoiser
  proper);
- a **luma detail** head predicting a luma-only additive detail term;
- a **confidence** head, a per-pixel sigmoid gate.

The detail term is applied as `base + detail · confidence · detail_scale`,
touching luma only, so it can never shift chroma. The intent was a route to
restoring texture the denoiser removes without also restoring noise: the gate
would learn *where* invented detail is safe.

**On the production checkpoint the gate is closed.** Confidence sits around
0.0001 everywhere, so the detail term contributes nothing regardless of
`detail_scale` (nominally 0.25). This is deliberate, not an oversight:

- Phase 2C attempted to force the gate open with a confidence-activation loss.
  Negative result — the gate reopened only by degrading base denoising.
- Phase 4B attempted it again via a texture-statistics loss (`nagi_v2_l_ft5`).
  Also a negative result; `scripts/probe_detail_head.py` was written as the
  abort criterion and fired.

Both experiments are in `docs/perfect_nr_experiment_log.md`; their configs
(`configs/nagi_v2_l_ft5.yaml`) and code remain in git, their bulky outputs do
not. The head is kept in the architecture, and `denoise(detail_strength=...)`
is wired through to `model.detail_scale`, so a future checkpoint that opens the
gate needs no pipeline change. Today the knob is a documented no-op.

This is why the README says the pipeline does not generate detail: the only
mechanism that could is switched off, and we have not found an honest way to
switch it on.

## 5. The highlight guard, and the version that was rejected

### The defect

The production checkpoint ships with `highlight_protect_strength = 0.0`, i.e.
the model's in-graph HDR safety valve *disabled*. With it disabled the model
treats small isolated specular highlights as impulse noise and crushes them —
on Z7 fix, top-1% luma retention fell to 0.62 and the peak from 1.83 to 0.87.
`denoise()` therefore re-arms the guard at inference time.

The guard itself is simple and pointwise: above a luma threshold, blend the
network output back toward the (collocated, non-negative) input, with a
sigmoid-soft mask.

### Attempt 1 — adaptive threshold. Rejected.

The first fix scaled the threshold to each image's own peak:
`threshold = min(1.0, 0.7 · p99.9(input_luma))`. This engages on *every* image,
including low-dynamic-range ones whose highlights sit at 0.5–0.7.

But the guard works by blending the **noisy input** back in. Engaging it over
large areas of a low-range scene therefore re-introduces noise — and does so
unevenly, because the mask follows local luma. Measured on Z7 bird: **3.7x
more flat-region noise**. Visually it read as "some parts still noisy, others
over-flattened", which is precisely failure mode (2) from §1.

### Attempt 2 — conditional arming. Shipped.

The guard is armed at a **fixed** threshold of 1.0, and only when the image
actually contains above-SDR content worth protecting — specifically when the
whole image's top-0.1% linear luma exceeds `HIGHLIGHT_GUARD_MIN_P999` (1.0).
Otherwise it stays disarmed.

Audited across all 8 scenes (`highlight retention / flat-region noise`):

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
free. On low-range scenes it never arms, so it cannot cause the unevenness of
attempt 1.

The decision is made **once per `denoise()` call, on the whole image**, never
per tile — see §3.

**Known limitation, not fixed:** with the guard disarmed, peak retention on
Z7 fix/bird stays 0.62–0.92. That is the model treating isolated speculars as
impulse noise, and it is not fixable post-hoc without re-introducing noise. The
real remedy is model-level (training data containing isolated speculars).

**Lesson carried forward, stated in the audit:** a highlight metric must always
be read together with a noise metric. A variant that improves retention while
raising noise is not a fix.

## 6. Core ML backend

`backend="coreml"` swaps the per-tile PyTorch forward for an exported Core ML
`mlprogram` graph. On a 39.8MP frame the model stage drops from ~237 s
(PyTorch/MPS) to ~65 s — 3.6x — putting the end-to-end cost at ~83 s including
the chroma pass and I/O.

Export: `scripts/export_coreml_nagi_v2.py` (`pixi run export-coreml`), a fixed
768×768, batch-1 static graph. Two things had to be handled:

- **`asinh` is missing from coremltools 9's MIL builder.** The torch frontend
  calls `mb.asinh`, which does not exist in that version. The export monkeypatches
  `torch.asinh` to the exact identity `asinh(x) = log(x + sqrt(x²+1))` for the
  duration of tracing only. This is a like-for-like substitution — it changes
  which primitives are traced, not the function computed — and it is
  numerically well-behaved over this model's whole operating range.
- **The in-graph highlight guard cannot be baked in**, because its threshold is
  a per-call, whole-image decision (§5). So it is exported disarmed and applied
  post-hoc by `nagi_denoise.coreml.apply_highlight_guard_np`. This is exact,
  not an approximation: the guard is a pointwise blend, affine in the denoised
  value, whose mask depends only on the collocated input pixel; Hann stitching
  is a per-pixel weighted average; the threshold is identical across every tile
  covering a given pixel. Blending after stitching therefore equals blending
  inside each tile before stitching.

The tiling loop itself (`CoreMLTiledDenoiser.tiled_forward`) imports the
coordinate-grid and window helpers from `nagi_denoise.infer`, so the stitching
geometry cannot drift from the PyTorch path.

`coremltools` is imported lazily, only when a `CoreMLTiledDenoiser` is
constructed. `import nagi_denoise` works without it.

### Numerical agreement, and the ANE warning

Validated over 23 real ROI tiles including the historical failure modes (Occi
hair, Cat whiskers, Ice blue shadows, Dance HDR peak) —
`scripts/validate_coreml_nagi_v2.py`, results in
`runs/phase5_speed/coreml/validation_report.json`:

| precision | compute units | worst abs diff vs PyTorch | peak-value ratio |
|---|---|---:|---|
| fp32 | `cpu_and_gpu` | 2.5e-05 | 1.000 |
| fp32 | `all` | 2.5e-05 | 1.000 |
| fp16 | `cpu_and_gpu` | 0.018 | 1.000–1.002 |
| **fp16** | **`all`** | **8.7** | **1.25–4.83** |

> **Never run the fp16 package with `compute_units="all"`.**
> `ALL` permits dispatch to the Apple Neural Engine, and the ANE computes this
> graph wrongly: peak output values run 1.25x to 4.8x high on **every** one of
> the 23 validation tiles — a worst-case per-pixel error of 8.7 on a scene
> whose true peak is 5.3. That is not rounding; it is corruption, and it looks
> like it. fp32 under `ALL` is unaffected only because fp32 does not dispatch
> to the ANE at all.

The default everywhere in this codebase is `cpu_and_gpu`, and
`nagi_denoise.coreml.resolve_compute_units` raises on unrecognised names rather
than falling back, so a typo can never silently select the ANE. `"all"` remains
selectable purely so the finding can be re-measured.

fp16 on `cpu_and_gpu` is the shipped fast path: 0.018 max per-pixel difference
against the fp32 PyTorch reference is fp16 rounding, and the Core ML quality
gates (`runs/phase5_speed/coreml/gates_fp16_cpu_and_gpu.json`) pass — HDR
highlight retention 1.0000, seam check pass.

`backend="torch"` remains the default and the reference. Every quality number
quoted in the README was measured on it.

## 7. Things deliberately not done

- **No adaptive per-image or per-region parameters.** §1 reason (2), §5
  attempt 1. The only content-dependent decision in the pipeline is the binary
  arm/disarm of the highlight guard, taken once on the whole image.
- **No detail generation.** §4.
- **No larger tile batches.** Measured slower on this M1 (b2 273 s, b4 1014 s
  vs b1 237 s); the knob exists for other hardware.
- **No fp16 autocast on the torch path by default.** Bought ~4% (247 s vs
  237 s) for a 0.012 max difference. Core ML is where the speed actually is.
