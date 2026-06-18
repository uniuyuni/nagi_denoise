"""Launch ChromaGuard training in the background."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "chromaguard_adaptive_s_1k": "packages/nagi_nr/configs/chromaguard_adaptive_s_1k.yaml",
    "chromaguard_s_1k": "packages/nagi_nr/configs/chromaguard_s_1k.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch ChromaGuard training.")
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--seed", default="0")
    parser.add_argument("--max-iters", default="0")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "runs" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"
    pid_path = out_dir / "pid.txt"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "nagi_nr.train_chromaguard",
        "--config",
        VARIANTS[args.variant],
        "--sidd-root",
        "SIDD_Medium_Srgb",
        "--output",
        str(out_dir.relative_to(repo)),
        "--device",
        args.device,
        "--ckpt-prefix",
        args.variant,
        "--resume-latest",
        "--seed",
        str(args.seed),
    ]
    if int(args.max_iters) > 0:
        cmd.extend(["--max-iters", str(args.max_iters)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== launch {args.variant} ===\n")
    log_f.write("$ " + " ".join(cmd) + "\n")
    log_f.flush()
    proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"started {args.variant} pid={proc.pid}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
