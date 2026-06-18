# NAFNet-Fast / NagiQ Redesign From First Principles

## Decision

q48-trim/q48-fast の延長を本筋から外す。次は「小さい NAFNet をゼロから学習」ではなく、40.212 dB を確認済みの NAFNet-width64 teacher から品質を保ったまま構造的に削る。

新しい本命は `NAFNet-Fast`:

1. full NAFNet-width64 を出発点にする
2. block gate / block ablation で、削っても PSNR が落ちにくい block を見つける
3. まず NAFNet 系として高速化・prune する
4. sRGB MSE 主体で短く fine-tune する

NagiQ 化そのものは目的ではない。NAFNet の 40.212 dB を保ったまま推論だけ高速化できるなら、それは `NAFNet-Fast` として成功扱いにする。NagiQ は、独自モデルとして再設計する場合だけの名前にする。

優先順位はユーザー指定どおり、画質 > 推論速度 > 学習速度 > 容量。

## Why The Previous Path Failed

確認済みの主要結果:

| model | result | note |
| --- | ---: | --- |
| NAFNet teacher | 40.212 dB, 387.1 ms/patch | 画質上限基準 |
| Nagi M | 37.463 dB, 58.4 ms/patch | 既存高速基準 |
| q48-fast corrected 12k EMA | 37.206 dB, 243.4 ms/patch | corrected recipe は動くが不足 |
| q48-trim corrected 20k EMA | 37.409 dB, 398.0 ms/patch | q48-fast から +0.203 dB だけ |

q48-trim は q48-fast より重いのに、20k で +0.203 dB しか改善しなかった。これは「同じ recipe のまま容量を足せば 40 dB に近づく」という仮説をかなり弱くする。

また、q48-trim は full validation 実測で 398.0 ms/patch になり、teacher の 387.1 ms/patch より遅い。画質も速度も teacher に負けているので、この方向は本命として成立しない。

## Rejected Shortcut: Naive Teacher Slice

width64 の NagiQ student は teacher checkpoint と state_dict 名・shape が完全一致する。

検証:

```text
q64-fast student keys: 430
exact-copyable keys:   430
copyable ratio:        100%
```

しかし、単純に「同じ名前の重みをコピーして block 数を減らす」だけでは壊れる。

| model | params | GMAC | quick validation |
| --- | ---: | ---: | ---: |
| q64-micro-slice | 46.03M | 30.65 | 6.098 dB on 128 patches |
| q64-fast-slice | 63.37M | 42.11 | 5.549 dB on 32 patches |

理由は、途中 block を丸ごと抜くと後段の feature 分布が teacher と一致しなくなるため。ending まで teacher からコピーされているので、壊れた feature をそのまま RGB に戻してしまう。

したがって、prefix slice は却下。必要なのは、teacher の関数を保ったまま段階的に block を identity 化し、影響が小さい block だけ削る設計。

## What Makes NAFNet Slow

NAFBlock は一見単純だが、各 block に高解像度の 1x1 conv が多い。

主なコスト:

1. `conv1`, `conv3`, `conv4`, `conv5` の pointwise conv
2. 高解像度 encoder/decoder stage の block
3. LayerNorm2d, SimpleGate, SCA などの小さい op が MPS では memory-bound になりやすい
4. block 数が多いほど kernel launch / memory traffic が増える

したがって、params ではなく次の指標で判断する。

| metric | purpose |
| --- | --- |
| GMAC per block/stage | 理論的な削減量 |
| measured ms/patch | MPS 実測速度 |
| PSNR drop per removed block | 品質感度 |
| PSNR per ms | 速度を上げる価値がある削除か |

## NAFNet-Fast Track

最初に試すべきは、モデルを変えずに teacher そのものを速くできるかの確認。

成功条件:

```text
PSNR: unchanged at 40.212 dB
speed: clearly below current teacher 387.1 ms/patch
```

候補:

1. evaluator / inference path の無駄を削る
   - tensor 作成、CPU-GPU 転送、uint8 変換、同期位置を分解計測する
   - model forward だけの時間と full validation loop の時間を分ける

2. PyTorch MPS 実行を詰める
   - `channels_last` の実測
   - repeated allocation を避ける
   - `torch.inference_mode()` と warmup/sync の条件を固定
   - op 別に LayerNorm2d / pointwise conv / PixelShuffle の比率を見る

3. export backend を検討する
   - Core ML / MPS Graph 相当で NAFNet がそのまま速くなるなら最優先
   - ただし PSNR が 40.212 dB から落ちる変換や量子化は採用しない

4. exact-compatible micro-optimization
   - LayerNorm2d 実装の差し替え
   - SCA の `AdaptiveAvgPool2d(1)` 周辺の最適化
   - 不要な padding / crop / contiguous の削減

この track は学習不要なので、最初にやる価値が高い。ここで 250 ms/patch 近辺まで落ちるなら、独自 NagiQ よりも `NAFNet-Fast` を本命にしてよい。

### Initial Measurements

追加した実測スクリプト:

```text
scripts/benchmark_nafnet_fast.py
scripts/ablate_nafnet_blocks.py
scripts/export_nafnet_fast_pruned.py
```

forward breakdown:

| mode | random forward | validation loop | validation forward | result |
| --- | ---: | ---: | ---: | --- |
| baseline | 516.3 ms | 536.7 ms | 520.8 ms | forward支配 |
| channels_last | 548.5 ms | 570.2 ms | 554.1 ms | 遅化、却下 |

stage profile:

| stage | share |
| --- | ---: |
| middle | 16.7% |
| enc0 | 15.0% |
| dec3 | 15.0% |
| enc3 | 13.1% |
| enc2 | 9.9% |
| enc1 | 8.5% |
| dec2 | 8.3% |

単純な backend/format 変更ではなく、block pruning が必要。

single-block ablation の重要結果:

| block | drop | judgment |
| --- | ---: | --- |
| enc0.0 | -36.460 dB | 削除不可 |
| enc0.1 | -27.429 dB | 削除不可 |
| dec3.0 | -6.736 dB | 削除不可 |
| dec3.1 | -2.908 dB | 削除不可 |
| enc3.7 | -0.068 dB on 32 patches | 有望 |
| middle.0 | -0.434 dB on 32 patches | 危険 |
| middle.11 | -0.237 dB on 32 patches | 微妙 |

deep block 16-patch scan では、enc3.1/2/4/5/6/7 が比較的低リスクだった。middle は全体に enc3 より落ちやすい。

multi-block skip:

| skip set | patches | drop |
| --- | ---: | ---: |
| enc3.4 + enc3.7 | 32 | -0.136 dB |
| enc3.2 + enc3.4 + enc3.7 | 32 | -0.236 dB |
| enc3.1 + enc3.2 + enc3.4 | 32 | -0.308 dB |
| enc3.1 + enc3.2 + enc3.4 + enc3.7 | 16 | -0.359 dB |

first physical export:

```text
runs/nafnet_fast_p2/nafnet_fast_p2_enc3_4_7.pt
skip: enc3.4, enc3.7
params: 112.28M
val32: 40.936 dB
```

同一プロセス内の hot-MPS 相対ベンチ:

```text
full NAFNet: 1959 ms/patch
P2:          1327 ms/patch
relative:    about 32% faster
```

絶対速度はMPSの熱状態で悪化しているので採用しない。相対比較として、物理削除により速度改善が出ることは確認できた。

cool-MPS rerun:

```text
random forward, same process:
full NAFNet: 407.9 ms/patch
P2:          390.9 ms/patch
relative:    about 4% faster

SIDD val128:
full NAFNet: 39.553 dB, 420.8 ms/patch
P2:          39.290 dB, 394.5 ms/patch
drop:        -0.263 dB
speedup:     about 6%
```

P2 は品質を大きく壊してはいないが、速度改善は小さい。`NAFNet-Fast` を「爆速」と呼ぶには、2 block prune では足りない。次は次のどちらか:

1. より攻めた prune map を作り、MSE fine-tune で品質を戻す
2. full NAFNet のまま Core ML / MPS Graph など backend 側で高速化する

### Core ML Backend Result

`coremltools` を pixi 依存に追加し、full NAFNet-width64 を `mlprogram` として export できた。

追加したスクリプト:

```text
scripts/export_coreml_nafnet.py
scripts/benchmark_coreml_nafnet.py
```

export:

```text
runs/nafnet_fast_coreml/nafnet_width64_fp32.mlpackage
precision: float32
input/output: 1x3x256x256 tensor
```

Core ML 32/128 patch:

| backend | patches | PSNR | ms/patch | note |
| --- | ---: | ---: | ---: | --- |
| PyTorch MPS teacher | 128 | 39.553 | 420.8 | reference |
| Core ML all | 32 | 41.072 | 227.8 | PSNR matches |
| Core ML cpu_and_gpu | 32 | 41.072 | 187.3 | predict 176.0 ms |
| Core ML cpu_and_gpu | 128 | 39.553 | 194.8 | predict 183.8 ms |
| Core ML cpu_and_ne | random | n/a | 514.3 | slow, not suitable |

Short-run result: Core ML `cpu_and_gpu` keeps exact PSNR and is about 2.16x faster than PyTorch MPS on the same 128 validation patches.

Long-run caveat: after several repeated runs, Core ML runtime/thermal state degraded badly:

```text
random forward: 1396.3 ms/patch
first 64/256 validation: 1429.3 ms/patch
```

That run was stopped. Treat Core ML as a strong short-burst backend, but full 1280-patch sustained speed needs a cooled, single-shot run with progress logging.

pixi tasks:

```text
pixi run export-coreml-nafnet
pixi run bench-coreml-nafnet
```

### Real EXR Pipeline Result

実画像:

```text
samples/coreml_exr_input/sample_cat_noisy.EXR
size: 7728x5152
input: linear float EXR
output: 16-bit TIFF
```

full-res Core ML:

| mode | tile | batch | tiles | elapsed | judgment |
| --- | ---: | ---: | ---: | ---: | --- |
| fp16 full-res | 512 | 1 | 176 | 178.5s | too slow |
| fp16 full-res | 512 | 4 | 176 | 97.7s | best full-res so far |

fast approximation:

| mode | process scale | tile | batch | tiles | elapsed | judgment |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| half-res residual upsample | 0.5 | 512 | 4 | 48 | 27.7s | fast, but leaves some noise |
| 2/3-res residual upsample | 0.6667 | 512 | 4 | 77 | 43.8s | worse than half-res on visual check; reject |

The half-res residual mode reached the speed target, but it is not the
NAFNet-Fast quality path. It is useful as a preview/draft mode only. The main
NAFNet-Fast design should return to full-resolution denoising or a trained
full-resolution approximation.

The 2/3-res residual mode was visually worse than half-res on the sample EXR.
This means the scale-residual family is not monotonic or reliable. It should not
be used as the NAFNet-Fast quality path. Keep half-res residual only as a fast
preview/draft mode.

1024 and 768 Core ML variants did not open a path to 60s:

```text
1024 fp16 batch1: 2222.6 ms/tile
768 fp16 batch2:  1344.3 ms/item
512 fp16 batch4:   555.2 ms/tile on real EXR
```

Large tiles reduce call count, but per-tile cost grows enough that they do not
beat 512 batch4 in practice.

Updated NAFNet-Fast target:

```text
quality path: full-res or trained full-res approximation
speed target: <=60s for 7728x5152
current best full-res quality path: 97.7s
current best fast preview: 27.7s at half-res residual
full-res gap: about 1.63x faster needed
```

Likely paths:

1. Full-res Core ML pipeline optimization
   - keep 512 tile, batch4 as current baseline
   - try batch5/6 if Core ML conversion and memory allow
   - reduce TIFF/input conversion overhead only after predict is no longer dominant

2. Quality-preserving prune + fine-tune
   - P2 prune alone is too small
   - a more aggressive NAFNet-Fast-P needs short MSE fine-tune to recover detail
   - target is 25-40% fewer expensive blocks, not a 2-block trim

3. Two-stage full-res refinement
   - half-res residual gives speed but leaves noise
   - 2/3-res residual was worse than half-res, so do not rely on simple scale tuning
   - add a very small full-res residual cleanup model later, trained specifically
     for remaining chroma/luma noise
   - this becomes a hybrid NAFNet-Fast pipeline, not exact NAFNet

### Return To NAFNet-Fast Design

Scale-residual shortcuts are rejected as the main path:

```text
half-res residual: fast, but leaves noise
2/3-res residual: slower and visually worse than half-res
```

Reasonable interpretation:

- Resizing changes the noise distribution before NAFNet sees it.
- Residual upsampling is not equivalent to full-resolution denoising.
- Fractional resize can introduce/reshape chroma noise and texture artifacts.
- More input pixels at 2/3 scale do not guarantee better denoising, because the
  model is no longer operating on the same degradation distribution.

So the NAFNet-Fast quality path must be full-resolution, or must include a
trained full-resolution cleanup stage.

New design target:

```text
Quality path:
  full-resolution denoising quality, no visible seams

Speed target:
  <=60s for 7728x5152 EXR

Current full-res baseline:
  512 tile, batch4, no overlap: 97.7s

Required improvement:
  1.63x over current full-res baseline
```

Proposed route:

1. Full-res Core ML scheduler
   - keep `512 x 512`, `batch=4` as stable baseline
   - test `batch=5` and `batch=6`; reject if conversion/runtime is unstable
   - use no overlap for speed baseline, then add seam handling only where needed

2. Seam-aware tiling without full overlap
   - visible seams imply each tile's border context is insufficient
   - full overlap everywhere is too expensive
   - instead process a second lightweight seam pass only around tile boundaries
   - possible strategy: crop narrow boundary strips, run NAFNet only on those
     strips with context, and composite them back with feathering

3. Quality-preserving NAFNet-Fast-P
   - P2 prune was too small: 2 blocks removed gave only small speedup
   - next prune must target expensive stage groups, then fine-tune
   - no more blind q48-style scratch training
   - start from teacher, prune, then short MSE/teacher fine-tune

4. Hybrid cleanup model
   - half-res residual can become a preview mode
   - for production speed, train a tiny full-res cleanup model for remaining
     chroma/luma noise after fast approximation
   - this is not exact NAFNet-Fast, but may be the only route to both <=60s and
     acceptable quality on this hardware

## New Architecture Strategy

名前: NAFNet-Fast-P

基本方針:

- width64 は維持する
- teacher と同じ block 構造から始める
- block をいきなり消さず、まず gate で無効化できる形にする
- prune 後も NAFNet 系として扱う
- 必要なら互換実装として NagiQ 形式に export する

full teacher:

```text
width: 64
enc:   [2, 2, 4, 8]
mid:   12
dec:   [2, 2, 2, 2]
total blocks: 36
params: 115.98M
GMAC:   63.24
```

速度候補:

| target | GMAC goal | expected role |
| --- | ---: | --- |
| P-45 | <=45 GMAC | まず 40 dB を狙う本命 |
| P-35 | <=35 GMAC | P-45 が成功した後の高速化 |
| P-safe | quality-first, no fixed GMAC | 40 dB 到達を最優先 |

q64-fast の単純形は 42.11 GMAC で 334.5 ms/patch 程度だった。P-45 は teacher より速く、品質を優先する現実的な第一目標になる。

## Pruning Method

### Stage 0: Calibration Set

validation を直接いじりすぎると判断が濁るので、まず train 由来の固定 calibration set を作る。

```text
calib_train: 256 or 512 random 256x256 crops
contains: noisy, GT, precomputed teacher
space: sRGB
```

validation は 128 patch screen と full 1280 patch final だけに使う。

### Stage 1: Single Block Ablation

full teacher に外部 gate を付ける。

概念:

```python
out = x + gate * (block(x) - x)
```

`gate=1` なら teacher と同じ。`gate=0` ならその block は identity。

各 block を 1 個ずつ `gate=0` にして、calibration set で測る。

記録するもの:

```text
block id
stage
block GMAC
PSNR drop vs GT
MSE increase vs teacher output
saved GMAC / PSNR drop
```

### Stage 2: Greedy Structured Pruning

一度に大量に削らない。1 block ずつ削って、そのたびに全候補を再評価する。

削除ルール:

1. PSNR drop が小さい候補を優先
2. 同程度なら saved GMAC が大きい候補を優先
3. cumulative PSNR が危険域に入ったら止める

hard guard:

```text
calib PSNR drop from teacher <= 0.35 dB for P-45
val128 after export >= 39.0 dB
measured speed < teacher speed
```

ここで 39 dB を割るなら、その prune budget は攻めすぎ。q48-trim のような長期 scratch training には戻らず、削除数を減らす。

### Stage 3: Export

残した block だけを順序を保って `NAFNet-Fast` checkpoint として export する。

checkpoint には pruning map を残す。

```yaml
prune_map:
  encoders:
    0: [0, 1]
    1: [0]
    2: [0, 2, 3]
    3: [1, 3, 5, 7]
  middle_blks: [0, 2, 4, 7, 9, 11]
  decoders:
    0: [0, 1]
    1: [0]
    2: [1]
    3: [0]
```

互換実装として NagiQ 側に載せる場合は、連番 block として再構成してよい。重要なのは、元 teacher で残した block の順序を保つこと。

### First Greedy Search Result

追加した実装:

```text
scripts/search_nafnet_fast_prune.py
pixi task: search-nafnet-prune
```

このスクリプトは full teacher に gate を付け、現在の skip set に候補を1個ずつ追加して再評価する。選択基準は quality first:

```text
primary: cumulative PSNR drop
secondary: stage-profile 由来の推定 saved ms
guard: cumulative drop <= 0.35 dB for P-safe/P-45 screen
```

P2 (`enc3.4, enc3.7`) からの軽量 greedy probe:

| probe | patches | result |
| --- | ---: | --- |
| P2 initial | 8 | -0.033 dB |
| greedy selected | 8 | enc3.5 -> enc3.6 -> middle.7 -> middle.8 |
| P4 check: enc3.4/5/6/7 | 16 | -0.170 dB |
| P4+middle.7 | 16 | -0.222 dB |
| P4+middle.7/8 | 16 | -0.295 dB |

8/16 patch だけなら P4 以上も許容に見えたが、物理 export 後の val128 では P4 が失格になった。

物理 export の結果:

| model | skip | params | val128 PSNR | judgment |
| --- | --- | ---: | ---: | --- |
| P2 | enc3.4, enc3.7 | 112.28M | 39.290 dB | safe but small speed gain |
| P3-middle | enc3.4, enc3.7, middle.4 | 104.91M | 39.141 dB | current best P-safe candidate |
| P4 | enc3.4, enc3.5, enc3.6, enc3.7 | 108.58M | 38.917 dB | reject as first fine-tune base |

結論:

- P4 は 16 patch では良く見えたが、val128 で `>=39.0 dB` guard を割ったので、最初の fine-tune base にはしない。
- P3-middle は guard を通った。P2 より品質は落ちるが、削減幅が大きく、fine-tune で戻す候補として筋が良い。
- MPS 実測msは熱状態で大きく崩れたので、この探索中の速度値は採用しない。速度判断は物理 export 後に cooled PyTorch または Core ML export で取り直す。
- 次は P3-middle を P-safe として、短い MSE fine-tune と Core ML export の両方を検証する。

## Fine-Tuning Recipe

q48 corrected recipe は Charbonnier 中心だったが、SIDD の評価は sRGB MSE 由来の PSNR。次は MSE を主目的にする。

loss:

```text
loss_gt      = MSE(pred, gt)
loss_teacher = MSE(pred, teacher)
loss_grad    = small L1/Charbonnier gradient loss
loss_range   = small soft penalty outside [0, 1]

total = loss_gt + teacher_w * loss_teacher + 0.01 * loss_grad + 0.001 * loss_range
```

teacher weight:

```text
0-2k:   teacher_w = 0.50
2k-8k:  teacher_w = 0.25
8k+:    teacher_w = 0.10
```

理由:

- pruned model は teacher 由来なので、teacher imitation だけを強くしすぎない
- 最終指標は GT に対する PSNR
- teacher は色崩れや局所破綻を防ぐ安定化役にする

training:

```yaml
patch_size: 256
output_space: srgb
randomize_each_access: true
exposure_jitter: null
batch_size: 1
grad_accum_steps: 4
lr: 5.0e-5
lr_min: 1.0e-5
warmup_iters: 500
ema_decay: 0.995
save_every: 1000
keep_best_by: val128_psnr
```

train-time hard clamp は使わない。clamp は評価と画像出力だけ。

## Evaluation Gates

| checkpoint | metric | continue if |
| --- | --- | --- |
| exported, no fine-tune | val128 PSNR | >=39.0 dB |
| 2k fine-tune | val128 PSNR | >=39.4 dB |
| 6k fine-tune | val128 PSNR | >=39.7 dB |
| 12k fine-tune | full val PSNR | >=40.0 dB |

speed gate:

```text
must be faster than teacher full validation: 387.1 ms/patch
preferred P-45 target: <=330 ms/patch
```

best checkpoint は必ず残す。`keep_last` だけではなく、validation best を別名で保存する。

## Linear Space Policy

main loss は sRGB のままにする。

理由:

- SIDD validation の PSNR は sRGB 空間で計算される
- 40 dB 目標では metric と loss を一致させるのが最優先
- linear auxiliary は見た目や暗部安定には効く可能性があるが、主目的にすると sRGB PSNR を外すリスクがある

使うとしても fine-tune 後半の小さい補助項に限定する。

```text
linear_aux_weight <= 0.02
採用条件: val128 sRGB PSNR が上がること
```

## Immediate Work Plan

1. P3-middle を正式な P-safe candidate として扱う
   - checkpoint: `runs/nafnet_fast_p3/nafnet_fast_p3_enc3_4_7_middle_4.pt`
   - val128: 39.141 dB
   - P4 は 38.917 dB なので、最初の fine-tune base から外す

2. prune tooling を育てる
   - done: `scripts/search_nafnet_fast_prune.py`
   - done: `scripts/export_nafnet_fast_pruned.py`
   - next: calibration set を validation 先頭 patch ではなく train crop 固定セットにする
   - next: result JSON から export/eval command を自動生成する

3. `train_q.py` に MSE-based loss option を足す
   - `loss.kind: mse_distill`
   - soft range penalty
   - best checkpoint 保存

4. P3-middle の speed path を確認する
   - cooled PyTorch MPS で P2/P3/full を同一プロセス相対比較
   - Core ML export して 512 batch4 real EXR を再測定
   - seam 対策は speed baseline 後に入れる

5. export model が val128 >=39.0 なので fine-tune 開始条件は満たした
   - 2k で評価
   - 6k で評価
   - full validation は有望な時だけ

## Active P3 Fine-Tune

P3-middle の 2k MSE screen を開始した。

```text
task: pixi run train-nafnet-fast-p3-mse-2k
config: packages/nagi_nr/configs/nagiq_nafnet_fast_p3_mse_2k.yaml
output: runs/nagiq_nafnet_fast_p3_mse_2k
init: runs/nafnet_fast_p3/nafnet_fast_p3_enc3_4_7_middle_4.pt
```

recipe:

```text
loss.kind: mse_distill
teacher_weight: 0.50 fixed
grad_weight: 0.01
range_weight: 0.001
lr: 5e-5 -> 1e-5
batch: 1 x grad_accum 4
val128: every 500 and final
best checkpoint: nagiq_nafnet_fast_p3_mse_2k_best.pt
```

判断:

| point | continue if |
| --- | --- |
| 500 | val128 が initial 39.141 dB から悪化していない |
| 1000 | 39.25 dB 以上なら継続 |
| 2000 | 39.4 dB 近辺なら 6k 延長を検討 |

2k で 39.2 dB 未満なら、prune base は P3 ではなく P2 に戻すか、teacher weight/lr を見直す。

2k result:

| step | val128 PSNR | note |
| ---: | ---: | --- |
| 500 | 39.609 dB | strong recovery from 39.141 initial |
| 1000 | 39.690 dB | still improving |
| 1500 | 39.716 dB | small gain |
| 2000 | 39.732 dB | best/final |

P3-middle MSE fine-tune passed the 2k screen. It recovered +0.591 dB over the
un-finetuned P3 export on val128. The curve is still increasing but the gain
from 1500 to 2000 is small (+0.016 dB), so the next run should be a controlled
extension rather than an open-ended long run.

### Active P3 Controlled Extension

2k best から追加4kの controlled extension を開始した。

```text
task: pixi run train-nafnet-fast-p3-mse-extend4k
config: packages/nagi_nr/configs/nagiq_nafnet_fast_p3_mse_extend4k.yaml
output: runs/nagiq_nafnet_fast_p3_mse_extend4k
init: runs/nagiq_nafnet_fast_p3_mse_2k/nagiq_nafnet_fast_p3_mse_2k_best.pt
```

変更点:

```text
lr: 2e-5 -> 5e-6
teacher_weight: 0.35 -> 0.25
grad_weight: 0.005
ema_decay: 0.997
val128: every 1000 and final
```

狙いは、teacher への固定を少し弱めて GT PSNR を伸ばすこと。1000 step で
39.73 dB を下回るなら、teacher を弱めすぎた可能性があるので止める。

### Inference Speed Track

学習とは別軸で、P3 の推論サイズと Core ML 実行可能性を確認した。

inference-only checkpoint:

```text
source: runs/nagiq_nafnet_fast_p3_mse_extend4k/nagiq_nafnet_fast_p3_mse_extend4k_best.pt
output: runs/nagiq_nafnet_fast_p3_mse_extend4k/nagiq_nafnet_fast_p3_mse_extend4k_best_infer.pt
training ckpt: 1.6GB
inference ckpt: 400MB
Core ML fp16 package: 201MB
```

P3 の構造コスト:

| model | params | 256 GMAC | 512 GMAC | ratio |
| --- | ---: | ---: | ---: | ---: |
| full NAFNet-width64 | 115.98M | 63.24 | 252.90 | 1.000 |
| P3-middle | 104.91M | 58.38 | 233.48 | 0.923 |

結論: P3 は画質回復には成功しているが、理論MAC削減は `7.7%` だけ。P3 だけで
full-res EXR 97.7s -> 60s を達成するのは無理がある。速度改善の本命には、
より大きい構造変更か、別の推論スケジューラ/小型cleanup方式が必要。

Core ML export:

```text
model: runs/nafnet_fast_coreml/nafnet_fast_p3_extend_best_fp16_b4_512.mlpackage
precision: fp16
input: batch4 512x512
size: 201MB
```

学習中に短い Core ML random benchmark を走らせたため、速度値は競合で汚れている:

| model | condition | random 512 batch4 |
| --- | --- | ---: |
| full fp16 b4 512 | training contention | 8125 ms/batch |
| P3 fp16 b4 512 | training contention | 10823 ms/batch |
| full fp16 b4 512 | previous cleaner run | 2122 ms/batch |

したがって、この `8125/10823 ms` は採用しない。重要なのは、学習中の GPU/Core ML
競合で速度測定が完全に壊れることが確認できた点。クリーンな速度判断は、学習停止後
または完走後に次を同一条件で取り直す:

1. full fp16 b4 512 random
2. P3 fp16 b4 512 random
3. P3 fp16 b4 512 real EXR full-res
4. 必要なら P3 fp16 b1 512 / b1 256 も比較

現時点の暫定判断:

- 容量問題は解決可能。Core ML fp16 で約200MB。
- 速度問題は未解決。P3 の削減量は小さすぎる。
- P3 は「画質を保つ候補」であって、「1分切りの速度候補」ではない可能性が高い。

## W48Q Speed-Oriented Design

目標を「40dB絶対」から「画質優先で39dB級、1.5倍速狙い」に緩めたため、
W48Q を開始する。

architecture:

```text
width: 48
enc: [2, 2, 4, 6]
middle: 10
dec: [2, 2, 2, 2]
params: 54.97M
```

theoretical cost:

| model | 256 GMAC | 512 GMAC | full ratio | ideal speed |
| --- | ---: | ---: | ---: | ---: |
| full NAFNet-width64 | 63.24 | 252.90 | 1.000 | 1.00x |
| P3-middle width64 | 58.38 | 233.48 | 0.923 | 1.08x |
| W48Q | 32.10 | 128.38 | 0.508 | 1.97x |

W48Q は 1.5x 速度目標に理論上乗る。P3 は乗らない。

channel surgery:

```text
script: scripts/export_channel_sliced_nagiq.py
source: runs/nagiq_nafnet_fast_p3_mse_extend4k/nagiq_nafnet_fast_p3_mse_extend4k_best_infer.pt
raw sliced output: runs/nafnet_fast_w48q/nafnet_fast_w48q_from_p3_extend_best.pt
```

raw channel-sliced W48Q は val16 `4.945 dB` で崩壊した。slice 済み residual head が
不整合な feature を RGB に戻してしまうため、そのまま学習開始しない。

residual head scale scan:

| ending scale | val16 PSNR |
| ---: | ---: |
| 0 | 22.970 dB |
| 0.0001 | 22.976 dB |
| 0.001 | 22.395 dB |
| 0.01 | 13.940 dB |
| 0.1 | 6.072 dB |
| 1.0 | 4.945 dB |

結論: W48Q は `ending_scale=0` の identity-safe 初期化で始める。内部blockは
channel slice 由来だが、head はゼロから学習させる。

active W48Q screen:

```text
task: pixi run train-nafnet-fast-w48q-mse-2k
config: packages/nagi_nr/configs/nagiq_nafnet_fast_w48q_mse_2k.yaml
output: runs/nagiq_nafnet_fast_w48q_mse_2k
init: runs/nafnet_fast_w48q/nafnet_fast_w48q_from_p3_extend_best_head0.pt
```

judgment:

| point | continue if |
| --- | --- |
| 500 | val128 >= 34 dB |
| 1000 | val128 >= 36.5 dB |
| 2000 | val128 >= 38.0 dB |

W48Q は初期が noisy identity なので、2k で 39dB に届かなくても即失格ではない。
ただし 2k で 38dB 未満なら、W48 単体で画質優先条件を満たすのは厳しいため、
W56 または W48 + cleanup に切り替える。

W48Q 1000 result:

| step | val128 PSNR | judgment |
| ---: | ---: | --- |
| 500 | 21.654 dB | barely above noisy baseline |
| 1000 | 22.210 dB | fail; stop |

W48Q identity-safe head0 did not learn fast enough. This is not a viable path
under the current 2k screen. The failure is informative: width48 may have enough
theoretical speed, but channel-sliced internal weights plus zero head does not
provide a useful 39dB-class initialization. Next options are:

1. W56Q with the same identity-safe surgery, because it keeps more channel
   capacity and may learn faster.
2. W48 with a different distillation strategy: precompute full/P3 residual
   targets and train the head/residual path explicitly before full fine-tune.
3. Hybrid pipeline: keep P3/full-quality model for hard regions/seams and use a
   smaller cleanup/preview model for the broad pass.

## W56Q Quality-First Speed Candidate

W48Q が 1000 step で `22.210 dB` のまま立ち上がらなかったため、次は W56Q を
試す。W56Q は W48Q より速度余裕は小さいが、画質優先条件ではより現実的。

architecture:

```text
width: 56
enc: [2, 2, 4, 6]
middle: 10
dec: [2, 2, 2, 2]
params: 74.73M
```

theoretical cost:

| model | 256 GMAC | 512 GMAC | full ratio | ideal speed |
| --- | ---: | ---: | ---: | ---: |
| full NAFNet-width64 | 63.24 | 252.90 | 1.000 | 1.00x |
| W56Q | 43.56 | 174.21 | 0.689 | 1.45x |

W56Q は 1.5x 目標に少し届かないが、ほぼ近い。画質が十分なら採用候補。

channel surgery:

```text
raw:   runs/nafnet_fast_w56q/nafnet_fast_w56q_from_p3_extend_best.pt
head0: runs/nafnet_fast_w56q/nafnet_fast_w56q_from_p3_extend_best_head0.pt
```

初期評価:

| init | val16 PSNR | note |
| --- | ---: | --- |
| raw channel slice | 5.363 dB | collapse |
| head0 identity-safe | 22.970 dB | safe noisy identity |

W56Q も raw slice は崩壊するため、W48Q と同じく head0 identity-safe から始める。

active W56Q screen:

```text
task: pixi run train-nafnet-fast-w56q-mse-2k
config: packages/nagi_nr/configs/nagiq_nafnet_fast_w56q_mse_2k.yaml
output: runs/nagiq_nafnet_fast_w56q_mse_2k
init: runs/nafnet_fast_w56q/nafnet_fast_w56q_from_p3_extend_best_head0.pt
```

judgment:

| point | continue if |
| --- | --- |
| 500 | val128 > W48Q 500 by a clear margin |
| 1000 | val128 >= 30 dB, preferably much higher |
| 2000 | val128 >= 36 dB to justify extension |

If W56Q also stays near noisy baseline, channel-sliced width reduction is not a
good path. Move to either W56/W48 residual pretraining or hybrid cleanup.

W56Q 1000 result:

| step | val128 PSNR | judgment |
| ---: | ---: | --- |
| 500 | 21.751 dB | barely above noisy baseline |
| 1000 | 23.381 dB | fail; stop |

W56Q improves faster than W48Q but is still far below the 30 dB checkpoint gate.
This means width reduction with identity-safe head0 does not transfer the teacher
mapping effectively. Do not continue this recipe to 2k/6k.

Next design implication:

- channel slicing internal blocks is not enough when the output head must be
  zeroed for stability
- raw sliced heads collapse for W48/W56, so direct surgery cannot preserve the
  denoising function
- if pursuing W48/W56, add an explicit residual/head pretraining stage using
  teacher residual targets, or switch to a hybrid pipeline
- P3 remains the only current high-quality single-model candidate, but it is not
  fast enough for the 1.5x target

## W56Q Residual-First Curriculum

W56Q direct fine-tune failed, so the next test isolates the output residual
learning problem.

```text
task: pixi run train-nafnet-fast-w56q-residual-curriculum-2k
config: packages/nagi_nr/configs/nagiq_nafnet_fast_w56q_residual_curriculum_2k.yaml
output: runs/nagiq_nafnet_fast_w56q_residual_curriculum_2k
init: runs/nafnet_fast_w56q/nafnet_fast_w56q_from_p3_extend_best_head0.pt
```

curriculum:

| step range | trainable | purpose |
| --- | --- | --- |
| 0-499 | ending only | learn teacher residual head from frozen sliced features |
| 500-999 | decoder + ending | adapt upsampling/decoder features |
| 1000-1999 | full model | low-level alignment after residual path exists |

loss:

```text
MSE distill
teacher_weight: 0.95 -> 0.80
grad_weight: 0.005
val128: every 250
```

判断:

- 250/500 で direct W56Q より明確に上なら residual-first は有効
- 500 で noisy baseline 近辺なら head-only では内部特徴が足りない
- 1000 で 30dB 未満なら decoder解凍でも不足
- 2000 で 36dB 未満なら W56単体はこの方式でも厳しい

result:

| step | val128 PSNR | noisy | judgment |
| ---: | ---: | ---: | --- |
| 250 | 21.564 dB | 21.530 dB | no useful lift |
| 500 | 21.577 dB | 21.530 dB | fail; stop |

The head-only residual phase did not learn a useful correction. This rejects the
current W48/W56 channel-slice family:

- raw sliced heads collapse to about 5 dB
- zeroed heads are safe but stay near noisy identity
- direct fine-tune does not rise fast enough
- residual-first curriculum does not fix the missing feature alignment

Next design moves to `NAFNet-Fast-C`: half-resolution NAFNet residual broad pass
plus a small full-resolution cleanup model. The detailed design is in
`docs/nafnet_fast_next_design.md`.

## Stop Conditions

次の場合は、その案を止める。

| condition | action |
| --- | --- |
| exported model val128 <39.0 | prune budget を緩める |
| P-safe でも val128 <39.0 | block pruning が不適合、scratch q48/q56 へ戻らない |
| 6k fine-tune で <39.5 | loss/teacher schedule を見直す |
| full validation speed >= teacher and PSNR < teacher | そのモデルは不採用 |

この設計で重要なのは、学習を始める前に「削っても 40 dB 近傍を保てる構造」を実測で確認すること。q48-trim の失敗は、低い初期値から長く登る方法がこの環境では非効率だと示している。次は 40.212 dB の地点から、どこまで削れるかを論理的に測る。
