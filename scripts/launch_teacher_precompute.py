"""Launch NAFNet teacher precompute in the background.

This is used from `pixi run` outside the Codex sandbox so the child process can
see MPS. Logs append to runs/teacher_precompute.log and the child PID is written
to runs/teacher_precompute.pid.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    runs = repo / "runs"
    runs.mkdir(exist_ok=True)

    log_path = runs / "teacher_precompute.log"
    pid_path = runs / "teacher_precompute.pid"

    cmd = [
        sys.executable,
        str(repo / "scripts" / "precompute_teacher.py"),
        "--sidd-root",
        "SIDD_Medium_Srgb",
        "--weights",
        "benchmarks/nafnet/NAFNet-SIDD-width64.pth",
        "--device",
        "mps",
    ]

    log_f = open(log_path, "a", buffering=1)
    log_f.write("\n=== resume teacher precompute ===\n")
    log_f.write("$ " + " ".join(cmd) + "\n")
    log_f.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=repo,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(f"{proc.pid}\n")
    print(f"started teacher precompute pid={proc.pid}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
