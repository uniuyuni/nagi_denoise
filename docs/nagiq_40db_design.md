# NagiQ 40 dB Design

## Goal

SIDD Validation mean-patch sRGB PSNRで40 dB級を狙う。優先順位は:

1. 画質
2. 推論速度
3. 学習速度
4. 容量

現行Nagi Mの延長ではなく、NAFNetをteacher/上限基準にした高速studentを作る。

## Current Baselines

| model | params | PSNR | speed |
| --- | ---: | ---: | ---: |
| Nagi M | 1.81M | 37.463 dB | 58.4 ms/patch |
| Nagi L | 3.18M | 37.389 dB | 78.2 ms/patch |
| NAFNet-width64 | 115.98M | 40.212 dB | 387.1 ms/patch |

NAFNet論文はSIDDで40.30 dBを報告している。ローカル評価でも40.212 dBなので、40 dBはこの環境/評価で実在する目標。

References:

- NAFNet: https://arxiv.org/abs/2204.04676
- Restormer: https://arxiv.org/abs/2111.09881
- SIDD: https://www.eecs.yorku.ca/~kamel/sidd/files/SIDD_CVPR_2018.pdf

## Why NAFNet Is Slow

NAFNet-width64の遅さは、主に巨大なattentionではなくNAFBlock内のpointwise convとFFNによる。

NAFBlockの主な計算:

- `conv1`: 1x1, C -> 2C
- `conv2`: depthwise 3x3, 2C
- `conv3`: 1x1, C -> C
- `conv4`: 1x1, C -> 2C
- `conv5`: 1x1, C -> C

depthwise 3x3は比較的軽い。支配的なのは4本の1x1 convで、だいたい `6 * C^2 * H * W`。

U-Netでは解像度が半分になるたびにchannelが倍になるため、`C^2 * H * W` は各段でほぼ一定になる。つまり低解像度側でもblock 1個あたりの計算量は大きく減らない。結果として、総block数がほぼそのまま速度に効く。

NAFNet-width64の構成:

```text
enc:    [2, 2, 4, 8]
middle: 12
dec:    [2, 2, 2, 2]
total NAFBlocks: 32
```

概算MAC for 256x256 patch:

| variant | params | GMAC | relative to NAFNet64 |
| --- | ---: | ---: | ---: |
| NAFNet64 | 115.98M | 63.0 | 1.00 |
| Q48 same blocks | 65.36M | 35.6 | 0.56 |
| Q56 trim | 63.43M | 41.0 | 0.65 |
| Q48 trim | 46.66M | 30.1 | 0.48 |
| Q40 trim | 32.45M | 21.0 | 0.33 |
| Q48 fast | 35.71M | 23.7 | 0.38 |

NAFNet64の重い部分:

| stage | blocks | channels | spatial | GMAC |
| --- | ---: | ---: | ---: | ---: |
| middle | 12 | 1024 | 16x16 | 19.4 |
| enc4 | 8 | 512 | 32x32 | 13.0 |
| enc3 | 4 | 256 | 64x64 | 6.5 |
| enc1 | 2 | 64 | 256x256 | 3.4 |
| dec4 | 2 | 64 | 256x256 | 3.4 |

改善すべきもの:

1. widthを64から48/56へ落とす
2. middle blockを12から8以下へ落とす
3. enc4 blockを8から6以下へ落とす
4. NAFBlockは維持するが、必要なら一部をLite/FFN縮小へ置換
5. teacher distillationで削った容量を補う

## Linear vs sRGB

SIDD評価はsRGB PSNRなので、主目的関数はsRGBに置く。

ただしlinear情報は捨てない。設計は以下:

```text
input  : sRGB RGB
aux    : pseudo-linear RGB or luma/noise features
output : sRGB residual
loss   : main sRGB loss + small linear/gradient/luma auxiliary loss
```

現行Nagiのasinh/linear主軸はHDR用途には良いが、SIDD sRGB PSNRには目的関数のズレがある。40 dB狙いではsRGB直結を本線にする。

## Proposed Architecture: NagiQ

### NagiQ-48-trim, first target

本命student。NAFNet64の構造を縮め、teacher蒸留で画質を戻す。

```text
space: sRGB in/out
width: 48
enc_blk_nums: [2, 2, 4, 6]
middle_blk_num: 8
dec_blk_nums: [2, 2, 2, 2]
params: ~46.7M
compute: ~30.1 GMAC/256 patch
expected speed: 170-230 ms/patch on MPS
target PSNR: 39.5-40.0 dB
```

NAFNet64比で理論MACは約48%。実測速度はメモリ/Metal kernel overheadの影響を受けるが、387 ms/patchから200 ms前後へ落とすことを狙う。

### Quality fallback: NagiQ-56-trim

NagiQ-48-trimが39.5 dB未満なら、widthを56に上げる。

```text
width: 56
enc_blk_nums: [2, 2, 4, 6]
middle_blk_num: 8
dec_blk_nums: [2, 2, 2, 2]
params: ~63.4M
compute: ~41 GMAC/256 patch
expected speed: 230-300 ms/patch
target PSNR: 39.8-40.2 dB
```

### Speed fallback: NagiQ-48-fast

NagiQ-48-trimが遅すぎるが画質が有望なら、blockをさらに削る。

```text
width: 48
enc_blk_nums: [1, 2, 3, 5]
middle_blk_num: 6
dec_blk_nums: [1, 1, 2, 2]
params: ~35.7M
compute: ~23.7 GMAC/256 patch
expected speed: 130-190 ms/patch
target PSNR: 39.0-39.7 dB
```

## Distillation Design

Teacherは既存のNAFNet-width64。オンラインteacherは使わない。事前計算済みteacher PNGを読む。

Training target:

```text
student_output = f(noisy_srgb)
loss_gt        = Charbonnier(student_output, gt_srgb)
loss_teacher   = Charbonnier(student_output, teacher_srgb)
loss_grad      = small gradient/edge loss vs gt_srgb
loss_color     = small chroma/luma balance loss
```

Schedule:

```text
0-30%:   teacher 0.70, gt 0.30
30-80%:  teacher 0.50, gt 0.50
80-100%: teacher 0.30, gt 0.70
```

理由:

- 序盤はteacherの40 dB級写像へ早く寄せる
- 後半はteacherの丸写しによるbiasを避け、GT PSNRに合わせる

## Training Plan For Weak PC

学習速度より画質優先だが、PC負荷を抑えるために段階的に進める。

### Phase 0: speed/profiling only

学習前にrandom inputで推論速度を測る。

Metrics:

- ms/patch for 256x256
- peak memory if available
- params
- theoretical GMAC

Gate:

- Q48-trimが250 ms/patchを大きく超えるなら、Q48-fastを先に試す
- Q48-trimが200 ms/patch前後なら本命として進める

### Phase 1: sanity overfit

少数patchで短時間学習し、lossが正常に下がるかだけ見る。

```text
patch_size: 256
batch_size: 1-2
gradient_accumulation: 4-8
steps: 500-1000
eval: max-patches 128
```

Gate:

- NaNなし
- teacher loss低下
- 128patch PSNRがNagi Mより上に出始める

### Phase 2: 20k screening

```text
steps: 20k
save_every: 2k
eval: 10k, 20k full SIDD Val
```

Gate:

- 10kで38.5 dB未満: 設計見直し
- 20kで39.0 dB未満: Q48-trimは厳しい
- 20kで39.3 dB以上: 続行

### Phase 3: 100k candidate

```text
steps: 100k
eval: 50k, 100k
```

Gate:

- 50kで39.6 dB以上なら続行
- 100kで39.8 dB以上なら40 dB到達候補
- 100kで39.5未満ならQ56-trimへ

### Phase 4: final polishing

```text
SWA / checkpoint averaging
TTA as upper-bound only
small GT-heavy fine-tune
```

TTAは実用速度を落とすので最終モデルの標準推論には使わない。ただし上限確認には使う。

## Evaluation Metrics

Primary:

- SIDD Val mean-patch sRGB PSNR

Secondary:

- SIDD Val SSIM
- ms/patch on MPS for 256x256
- full-image tiled inference time for `tests/sample.jpg`
- residual breakdown: Y/Cb/Cr, low/mid/high luma residual
- worst-scene PSNR
- memory pressure / thermal behavior, observed manually

Acceptance targets:

| stage | PSNR | speed |
| --- | ---: | ---: |
| minimum useful | >=39.0 dB | <=250 ms/patch |
| strong candidate | >=39.6 dB | <=250 ms/patch |
| 40 dB target | >=40.0 dB | <=300 ms/patch |
| teacher parity | >=40.2 dB | faster than 387 ms/patch |

## What To Validate First

1. Implement NagiQ/NAFStudent with configurable width/block counts.
2. Add a small speed benchmark script that reports params, GMAC estimate, and MPS ms/patch.
3. Benchmark Q48-trim, Q48-fast, Q56-trim without training.
4. Only then start training Q48-trim.

## First Local Speed Measurement

Command:

```text
pixi run bench-nagiq --warmup 3 --iters 10
```

Result on MPS, random `1x3x256x256` input:

| preset | params | GMAC | ms/patch | note |
| --- | ---: | ---: | ---: | --- |
| q48-trim | 46.66M | 30.28 | 262.0 | quality-first candidate, slightly above 250ms target |
| q48-fast | 35.71M | 23.83 | 208.0 | speed fallback with enough room |
| q56-trim | 63.43M | 41.08 | 317.0 | quality fallback, still faster than NAFNet64 |
| q40-trim | 32.45M | 21.12 | 213.6 | narrower but not much faster than q48-fast |

Decision after this gate:

```text
Start with q48-trim.
Keep q48-fast as the speed fallback.
Use q56-trim only if q48-trim quality stalls below target.
```

This avoids spending days training a model that is already too slow.

## Decision

First implementation target:

```text
NagiQ-48-trim
```

If speed is too slow:

```text
NagiQ-48-fast
```

If quality is too low but speed is acceptable:

```text
NagiQ-56-trim
```
