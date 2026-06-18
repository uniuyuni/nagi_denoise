"""Launch a Nagi NR training variant in the background.

Run from pixi outside the Codex sandbox when using MPS:

    pixi run python scripts/launch_training.py l

The child process writes stdout/stderr to runs/nagi_nr_<variant>/stdout.log and
the PID to runs/nagi_nr_<variant>/pid.txt.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "s": "packages/nagi_nr/configs/nagi_nr_s.yaml",
    "m": "packages/nagi_nr/configs/nagi_nr_m.yaml",
    "m2": "packages/nagi_nr/configs/nagi_nr_m2.yaml",
    "m3": "packages/nagi_nr/configs/nagi_nr_m3.yaml",
    "m_luma_ms": "packages/nagi_nr/configs/nagi_nr_m_luma_ms.yaml",
    "m_srgb": "packages/nagi_nr/configs/nagi_nr_m_srgb.yaml",
    "lc": "packages/nagi_nr/configs/nagi_nr_lc.yaml",
    "l": "packages/nagi_nr/configs/nagi_nr_l.yaml",
}

INIT_FROM = {
    "m2": "runs/nagi_nr_m/nagi_nr_m_final.pt",
    "m3": "runs/nagi_nr_m/nagi_nr_m_final.pt",
    "m_luma_ms": "runs/nagi_nr_m/nagi_nr_m_final.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Nagi NR training in the background.")
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--seed", default="0")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    variant = args.variant
    out_dir = repo / "runs" / f"nagi_nr_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "stdout.log"
    pid_path = out_dir / "pid.txt"
    prefix = f"nagi_nr_{variant}"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "nagi_nr.train",
        "--config",
        VARIANTS[variant],
        "--sidd-root",
        "SIDD_Medium_Srgb",
        "--output",
        str(out_dir.relative_to(repo)),
        "--device",
        args.device,
        "--ckpt-prefix",
        prefix,
        "--resume-latest",
        "--seed",
        str(args.seed),
    ]
    if variant in INIT_FROM:
        # Only use --init-from when starting from scratch. If a checkpoint
        # already exists, --resume-latest should own the state restoration.
        if not any(out_dir.glob(f"{prefix}_*.pt")):
            cmd.extend(["--init-from", INIT_FROM[variant]])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_f = open(log_path, "a", buffering=1)
    log_f.write(f"\n=== launch train-{variant} ===\n")
    log_f.write("$ " + " ".join(cmd) + "\n")
    log_f.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=repo,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(f"{proc.pid}\n")
    print(f"started train-{variant} pid={proc.pid}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
