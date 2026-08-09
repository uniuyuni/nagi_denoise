#!/bin/zsh
# Auto-resume supervisor for the NagiV2-L Phase 2C confidence-gate fine-tune.
# Relaunches with --resume-latest when the trainer is killed (e.g. OOM under
# system-wide memory pressure). Stops on natural completion, on a fatal
# non-finite-loss abort, or after MAX_RESTARTS.
set -u
cd "$(dirname "$0")/.."

DEVICE="${1:-mps}"
OUT=runs/nagi_v2_l_ft3
MAX_RESTARTS=50
mkdir -p "$OUT"

n=0
while true; do
  echo "[supervisor $(date '+%F %T')] launch #$n device=$DEVICE" >> "$OUT/supervisor.log"
  PYTHONUNBUFFERED=1 pixi run python -m nagi_denoise.train.train_v2 \
    --config configs/nagi_v2_l_ft3.yaml \
    --sidd-root SIDD_Medium_Srgb \
    --output "$OUT" \
    --device "$DEVICE" \
    --ckpt-prefix nagi_v2_l_ft3 \
    --resume-latest >> "$OUT/train.stdout.log" 2>&1
  rc=$?
  echo "[supervisor $(date '+%F %T')] trainer exited rc=$rc" >> "$OUT/supervisor.log"
  if [ -f "$OUT/nagi_v2_l_ft3_final.pt" ]; then
    echo "[supervisor] COMPLETE" >> "$OUT/supervisor.log"; exit 0
  fi
  if grep -q "non-finite loss" "$OUT/train.stdout.log" 2>/dev/null; then
    echo "[supervisor] FATAL non-finite loss; not restarting" >> "$OUT/supervisor.log"; exit 1
  fi
  n=$((n+1))
  if [ $n -gt $MAX_RESTARTS ]; then
    echo "[supervisor] too many restarts; giving up" >> "$OUT/supervisor.log"; exit 1
  fi
  sleep 30
done
