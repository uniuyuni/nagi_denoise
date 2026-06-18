# nagi-nr-bench

Benchmark harness for Nagi NR — third-party comparators and the SIDD Validation evaluator. Pulls in extra deps (`scipy`, `einops`, `timm`, `gdown`) that aren't needed at inference time.

## Install (editable)

```bash
pixi install
pixi run check-env
```

## CLI

```bash
pixi run eval-s
pixi run eval-m
pixi run eval-m2
pixi run eval-l
pixi run eval-scunet
pixi run eval-nafnet
```

Required inputs (defaults assumed to live at the repo root):
- `data/ValidationNoisyBlocksSrgb.mat`
- `data/ValidationGtBlocksSrgb.mat`

Both downloadable from <http://130.63.97.225/share/download_benchmark.html>.

## Vendored third-party code

Located under `src/nagi_nr_bench/third_party/`:

- **`scunet/arch.py`** — from <https://github.com/cszn/SCUNet>, MIT/CC0 license (see upstream). Minor tweak: removed `thop` import for environments without it.
- **`nafnet/arch.py`** — from <https://github.com/megvii-research/NAFNet>, MIT license. Tweak: rewrote `from basicsr...` imports to relative imports so it stands alone.
- **`nafnet/arch_util.py`, `nafnet/local_arch.py`** — same upstream.

## Pretrained weights (not bundled)

Download from upstream and pass to the eval CLI with `--weights`:

| Model | URL | Size |
|---|---|---|
| SCUNet color real_psnr | <https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth> | 69 MB |
| NAFNet-SIDD-width64 | Google Drive (see NAFNet README): file id `14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR` | 464 MB |

Suggested paths kept in this repo:
- `benchmarks/scunet/scunet_color_real_psnr.pth`
- `benchmarks/nafnet/NAFNet-SIDD-width64.pth`

## Last measured results (SIDD Validation, 1280 patches, MPS)

| Model | Params | PSNR | ms/patch |
|---|---:|---:|---:|
| Nagi NR-S | 0.45 M | 36.803 dB | 26.2 |
| Nagi NR-M | 1.81 M | 37.463 dB | 58.4 |
| Nagi NR-M2 | 1.81 M | 37.320 dB | 57.3 |
| Nagi NR-L | 3.18 M | 37.389 dB | 78.2 |
| SCUNet `color_real_psnr` | 17.95 M | 35.11 dB | 504.1 |
| NAFNet-width64 | 115.98 M | 40.21 dB | 387.1 |

(SCUNet is trained on synthetic noise, so it underperforms its own paper number on
heavy-noise SIDD images; NAFNet number is within 0.09 dB of the paper's 40.30 dB.)
