"""Launch the GAMAIR PolyU 256px chain watcher in the background."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "runs" / "chain_gamair_polyu_256"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"
    pid_path = out_dir / "pid.txt"

    cmd = [sys.executable, "-u", "scripts/chain_gamair_polyu_256.py"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_f = open(log_path, "a", buffering=1)
    log_f.write("\n=== launch chain-gamair-polyu-256 ===\n")
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
    print(f"started chain-gamair-polyu-256 pid={proc.pid}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
