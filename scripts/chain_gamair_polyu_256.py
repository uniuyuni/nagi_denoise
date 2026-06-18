"""Start the 256px GAMAIR fine-tune after the 128px PolyU run completes."""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CURRENT_DIR = REPO / "runs" / "nagiq_gamair_s_faststart_128_polyu_5k"
CURRENT_FINAL = CURRENT_DIR / "nagiq_gamair_s_faststart_128_polyu_5k_final.pt"
CURRENT_BEST = CURRENT_DIR / "nagiq_gamair_s_faststart_128_polyu_5k_best.pt"
CURRENT_LOG = CURRENT_DIR / "stdout.log"

NEXT_DIR = REPO / "runs" / "nagiq_gamair_s_polyu_256_ft_10k"
NEXT_FINAL = NEXT_DIR / "nagiq_gamair_s_polyu_256_ft_10k_final.pt"
NEXT_LOG = NEXT_DIR / "stdout.log"

CHAIN_DIR = REPO / "runs" / "chain_gamair_polyu_256"
CHAIN_LOG = CHAIN_DIR / "stdout.log"


def log(message: str) -> None:
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with CHAIN_LOG.open("a", buffering=1) as f:
        f.write(line + "\n")


def current_done() -> bool:
    if not CURRENT_FINAL.exists() or not CURRENT_BEST.exists():
        return False
    try:
        tail = CURRENT_LOG.read_text(errors="replace")[-4096:]
    except OSError:
        return False
    return "done." in tail


def next_already_started() -> bool:
    if NEXT_FINAL.exists():
        return True
    if not NEXT_LOG.exists():
        return False
    try:
        text = NEXT_LOG.read_text(errors="replace")
    except OSError:
        return False
    return "=== launch train-nagiq-gamair_s_polyu_256_ft_10k ===" in text


def main() -> int:
    log("watching GAMAIR-S 128px PolyU run")
    while not current_done():
        time.sleep(60)
    log("128px PolyU run completed")

    if next_already_started():
        log("256px fine-tune already started or completed; not launching again")
        return 0

    cmd = ["pixi", "run", "train-gamair-s-polyu-256-ft-10k"]
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            log("stdout: " + line)
    if proc.stderr:
        for line in proc.stderr.splitlines():
            log("stderr: " + line)
    log(f"launcher exit code: {proc.returncode}")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
