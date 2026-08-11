# docs/ index

Two of these documents describe the shipped system. The rest are the research
record: earlier lines that were tried, measured and abandoned. They are kept
because the negative results are the reason the current design looks the way
it does — several of them are the only written evidence for why an obvious
idea is not in the pipeline. **Do not treat historical documents as
instructions; they describe systems that no longer exist.**

## Current — describes what ships

| file | lines | what it is |
|---|---:|---|
| [`architecture.md`](architecture.md) | 303 | **Start here.** The two-stage design, asinh HDR handling, seam-free tiling, the inert detail head, the highlight-guard policy (and the rejected version), and the Core ML backend with the ANE corruption warning. |
| [`nagi_v2_parameter_audit.md`](nagi_v2_parameter_audit.md) | 145 | Every quality-affecting parameter in `denoise()`, swept and measured over all 8 full-resolution diagnostic scenes. The two shipped-then-caught defects and their fixes. Raw artefacts in `runs/audit/`. |

## Reference — still consulted, describes the current project's rules

| file | lines | what it is |
|---|---:|---|
| [`perfect_nr_research_project.md`](perfect_nr_research_project.md) | 266 | The charter for the NagiV2 line: goals, priorities, and the non-negotiable acceptance gates (HDR range, highlight detail, dark thin lines, colour stability, tile stability, benchmark sanity, backend sanity). The gates are still what a candidate must pass. Its "lessons from the previous line" section explains the NAFNet/SCUNet Core ML failures that shaped the design. |
| [`perfect_nr_experiment_log.md`](perfect_nr_experiment_log.md) | 8807 | The full running experiment log for the whole project, in date order. Every training run, sweep, negative result and reversal, including the ones whose outputs were deleted in the v1.0 cleanup. The primary source when you want to know *whether something was already tried*. Append-only — do not rewrite it. |

## Historical — abandoned lines, kept for the record

These predate NagiV2. They describe architectures and pipelines that are no
longer in the repository or no longer used. File paths inside them (e.g.
`packages/nagi_nr/...`) refer to an older repository layout and will not
resolve.

| file | lines | line it belongs to | why it ended |
|---|---:|---|---|
| [`nafnet_coreml_practical_path.md`](nafnet_coreml_practical_path.md) | 270 | NAFNet + Core ML | Core ML conversion of NAFNet produced thin-line corruption and local colour breakage on real images. Superseded, but it is where the "`ALL` is not automatically faster or safer" finding originates. |
| [`nafnet_fast_next_design.md`](nafnet_fast_next_design.md) | 427 | NAFNet-Fast | Pruning a NAFNet-w64 teacher down while keeping quality. Screens failed to beat the direct-training baseline. |
| [`nagiq_40db_design.md`](nagiq_40db_design.md) | 327 | NagiQ | Plan to reach 40 dB on SIDD Validation with a fast student. Superseded by NagiV2. |
| [`nagiq_recovery_design.md`](nagiq_recovery_design.md) | 310 | NagiQ | Recovery plan after the q48-trim 20k run came in at 35.8 dB. |
| [`nagiq_next_training_design.md`](nagiq_next_training_design.md) | 255 | NagiQ | Retraining plan after the q48-trim result was found to be confounded by three bad conditions at once. |
| [`nagiq_redesign_from_first_principles.md`](nagiq_redesign_from_first_principles.md) | 1026 | NagiQ → NAFNet-Fast | The decision to stop extending q48 and instead prune down from the verified NAFNet-w64 teacher. Longest of the historical designs. |
| [`nagi_realfast_v0_design.md`](nagi_realfast_v0_design.md) | 304 | Nagi-RealFast | A practical real-image denoiser for the local Mac / MPS / Core ML path. |
| [`gamair_implementation_plan.md`](gamair_implementation_plan.md) | 273 | GAMA-IR | Implementation plan for arXiv:2404.00807 (no public code or weights, so implemented from the paper). Not adopted. |

## Related material outside `docs/`

- `runs/audit/README.md` — the audit's own artefact index (contact sheets,
  chroma sheets, metrics JSON).
- `configs/*.yaml` — training configs. Several document abandoned routes in
  their header comments; `configs/nagi_v2_l_ft.yaml` is explicitly marked as
  one.
- `runs/phase5_speed/coreml/validation_report.json` — the per-tile Core ML
  numerical validation behind the ANE warning.
