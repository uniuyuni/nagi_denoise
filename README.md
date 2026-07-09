# Nagi Denoise

HDR-aware blind image denoiser for real photographs (Apple Silicon / MPS first).

- **float32 numpy in / float32 numpy out** (linear-light RGB, HDR values above 1.0 preserved)
- **HDR-safe**: reversible in-graph `asinh` compression + input-luma highlight guard
- **Seam-free tiling**: Hann-window overlap blending for 40MP+ images
- Quality target: DxO PhotoLab-class NR on real high-ISO photos

The current model line is **NagiV2** (shared trunk + base RGB residual head +
confidence-gated luma detail head). Design rationale and the full research
history live in `docs/` — start with `docs/perfect_nr_research_project.md` and
the experiment log `docs/perfect_nr_experiment_log.md`.

## Repo layout

```
nagi_denoise/            # unified Python package
├── models/              # nagi_v2.py (NagiV2), nagi_nr.py (NagiNR), chromaguard.py
├── transforms.py        # sRGB<->linear, asinh compress/decompress
├── losses.py            # NagiLoss, NagiV2Loss
├── data.py              # SIDD loader, augmentation
├── infer.py             # Denoiser: checkpoint loading + tiled inference + numpy API
├── cli.py               # nagi-denoise CLI
├── train/               # train_nr.py, train_v2.py, train_chromaguard.py
├── bench/               # SIDD Validation evaluator + vendored SCUNet/NAFNet archs
└── pipeline/            # full-image EXR pipeline, guards, ROI eval, teachers
configs/                 # nagi_v2_s.yaml, nagi_nr_{s,m}.yaml
scripts/                 # check_env.py only (CLIs come from pyproject entry points)
tests/                   # test_model.py
data/                    # SIDD Validation .mat files
SIDD_Medium_Srgb/        # training set (local)
PolyU-…/                 # PolyU real-noise dataset (local)
benchmarks/              # third-party teacher weights (NAFNet-w64, SCUNet)
runs/baseline_v12/       # frozen quality baseline from the previous pipeline
docs/                    # design docs + full experiment log
```

Diagnostic full-resolution test photos (noisy EXR + PhotoLab / SCUNet
references) live outside the repo at `~/ProjectData/test_photos`.

## Environment

```bash
pixi install
pixi run check-env      # torch / MPS sanity
pixi run test           # unit tests
```

## Commands

| Task | Purpose |
|---|---|
| `pixi run train-v2-s`       | Train NagiV2-S on SIDD Medium (MPS) |
| `pixi run eval-sidd-v2`     | SIDD Validation PSNR for a NagiV2 checkpoint |
| `pixi run eval-sidd-nafnet` / `eval-sidd-scunet` | Teacher / comparator baselines |
| `pixi run denoise`          | Single-image CLI (PNG/TIFF/EXR) |
| `pixi run denoise-exr`      | Full-image HDR EXR pipeline (tiled, guards) |
| `pixi run roi-eval`         | ROI noise metrics vs a baseline candidate |
| `pixi run crop-compare`     | Side-by-side crop comparison sheets |
| `pixi run precompute-teacher` | Precompute NAFNet teacher outputs for distillation |

## Python API

```python
import numpy as np
from nagi_denoise import Denoiser

dn = Denoiser.load("runs/nagi_v2_s/nagi_v2_s_final.pt", device="auto")
out = dn.denoise_array(img, input_space="linear", tile=512, overlap=64)
# img/out: float32 HWC RGB numpy, linear light, HDR-safe
```

`Denoiser.load` auto-detects the architecture (NagiV2 or NagiNR) from the
checkpoint config.

## Quality gates

A candidate is only promoted if it passes the gates defined in
`docs/perfect_nr_research_project.md`: HDR range safety, highlight detail,
dark thin lines, color stability, tile seams, and SIDD/PolyU benchmark sanity.
The frozen `runs/baseline_v12/` outputs and the PhotoLab references in
`~/ProjectData/test_photos` are the comparison standard (`pixi run roi-eval`).
