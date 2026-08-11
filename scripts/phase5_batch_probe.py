"""Probe the largest safe (tile, batch_size) combination for tiled MPS
inference, one subprocess per combination with a hard wall-clock timeout.

Isolating each combination in its own subprocess means a hang (MPS
swap-thrash, as flagged in the Phase 5 task notes) only kills that one probe
instead of taking down the whole sweep -- and a real crash/OOM in one
subprocess can't corrupt the MPS state for the next one.

Usage:
    pixi run python scripts/phase5_batch_probe.py --tile 768 --batches 1,2,4,8 --timeout 60
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

_WORKER = r"""
import sys, time, torch
from nagi_denoise.infer import Denoiser
from nagi_denoise.pipeline.denoise import PRODUCTION_WEIGHTS

tile = int(sys.argv[1])
bs = int(sys.argv[2])
n = int(sys.argv[3])

dn = Denoiser.load(str(PRODUCTION_WEIGHTS), device='mps')
x = torch.randn(bs, 3, tile, tile, device='mps')
torch.mps.synchronize()
t0 = time.time()
with torch.inference_mode():
    for _ in range(n):
        y = dn.model(x)
torch.mps.synchronize()
dt = (time.time() - t0) / n
mem = torch.mps.current_allocated_memory() / 1e9
driver_mem = torch.mps.driver_allocated_memory() / 1e9
print(f"RESULT tile={tile} batch={bs} total_ms={dt*1000:.1f} per_tile_ms={dt*1000/bs:.1f} alloc_gb={mem:.2f} driver_gb={driver_mem:.2f}", flush=True)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per subprocess")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    results = []
    for bs in batches:
        print(f"--- probing tile={args.tile} batch={bs} (timeout={args.timeout}s) ---", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _WORKER, str(args.tile), str(bs), str(args.repeat)],
                timeout=args.timeout,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - t0
            if proc.returncode != 0:
                print(f"  FAILED rc={proc.returncode} elapsed={elapsed:.1f}s")
                print("  stderr tail:", proc.stderr[-800:])
                results.append({"tile": args.tile, "batch": bs, "status": "error"})
                break
            print("  " + proc.stdout.strip().replace("\n", "\n  "))
            results.append({"tile": args.tile, "batch": bs, "status": "ok", "stdout": proc.stdout.strip()})
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  TIMEOUT after {elapsed:.1f}s -- treating as unsafe (swap-thrash or too slow), stopping sweep")
            results.append({"tile": args.tile, "batch": bs, "status": "timeout"})
            break

    print("\n=== SUMMARY ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
