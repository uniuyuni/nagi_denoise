"""Evaluate retained Nagi NR-L checkpoints on SIDD Validation.

Writes a compact summary to runs/eval_l_checkpoints.log. This is intentionally
small and direct so it can be launched via pixi outside the sandbox for MPS.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ckpts = sorted((repo / "runs" / "nagi_nr_l").glob("nagi_nr_l_*.pt"))
    # Keep only retained step checkpoints plus final, in chronological order.
    ckpts = [p for p in ckpts if re.search(r"_(\d{7}|final)\.pt$", p.name)]
    out_path = repo / "runs" / "eval_l_checkpoints.log"

    with open(out_path, "w", buffering=1) as out:
        out.write("# Nagi NR-L retained checkpoint evaluation\n\n")
        out.write("checkpoint,params_m,psnr_noisy,psnr_denoised,delta,total_time_s,ms_per_patch\n")

        best: tuple[float, Path] | None = None
        for ckpt in ckpts:
            print(f"evaluating {ckpt.name}", flush=True)
            cmd = [
                sys.executable,
                "-m",
                "nagi_nr_bench.eval_sidd_val",
                "--model",
                "nagi",
                "--weights",
                str(ckpt.relative_to(repo)),
                "--device",
                "auto",
            ]
            log_path = repo / "runs" / f"eval_{ckpt.stem}.log"
            if log_path.exists() and "PSNR denoised:" in log_path.read_text():
                text = log_path.read_text()
            else:
                proc = subprocess.run(
                    cmd,
                    cwd=repo,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                text = proc.stdout
                log_path.write_text(text)

            params = re.search(r"== nagi \(([^)]+) params\)", text).group(1)
            noisy = float(re.search(r"PSNR noisy\s*:\s*([0-9.]+)", text).group(1))
            den = float(re.search(r"PSNR denoised:\s*([0-9.]+)", text).group(1))
            delta = float(re.search(r"delta \+([0-9.]+)", text).group(1))
            timing = re.search(r"total time\s*:\s*([0-9.]+)s \(([0-9.]+) ms/patch\)", text)
            total_s = float(timing.group(1))
            ms_patch = float(timing.group(2))
            out.write(f"{ckpt.name},{params},{noisy:.3f},{den:.3f},{delta:.3f},{total_s:.1f},{ms_patch:.1f}\n")
            out.flush()

            if best is None or den > best[0]:
                best = (den, ckpt)

        if best is not None:
            out.write(f"\nbest,{best[1].name},{best[0]:.3f}\n")
            print(f"best {best[1].name}: {best[0]:.3f} dB")


if __name__ == "__main__":
    main()
