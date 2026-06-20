"""Launch a NagiPerfect training variant in the background."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "perfect_s_bodypush_4k": "packages/nagi_nr/configs/nagiperfect_s_bodypush_4k.yaml",
    "perfect_s_chromabranch2_1k": "packages/nagi_nr/configs/nagiperfect_s_chromabranch2_1k.yaml",
    "perfect_s_chromabranch_1k": "packages/nagi_nr/configs/nagiperfect_s_chromabranch_1k.yaml",
    "perfect_s_chromadistill3_1k": "packages/nagi_nr/configs/nagiperfect_s_chromadistill3_1k.yaml",
    "perfect_s_chromadistill2_1k": "packages/nagi_nr/configs/nagiperfect_s_chromadistill2_1k.yaml",
    "perfect_s_chromadistill_1k": "packages/nagi_nr/configs/nagiperfect_s_chromadistill_1k.yaml",
    "perfect_s_chromaflat_1k": "packages/nagi_nr/configs/nagiperfect_s_chromaflat_1k.yaml",
    "perfect_s_chromalocal_3k": "packages/nagi_nr/configs/nagiperfect_s_chromalocal_3k.yaml",
    "perfect_s_chromadamp_1k": "packages/nagi_nr/configs/nagiperfect_s_chromadamp_1k.yaml",
    "perfect_s_denoisepush_5k": "packages/nagi_nr/configs/nagiperfect_s_denoisepush_5k.yaml",
    "perfect_s_flatpush_2k": "packages/nagi_nr/configs/nagiperfect_s_flatpush_2k.yaml",
    "perfect_s_flatguard_500": "packages/nagi_nr/configs/nagiperfect_s_flatguard_500.yaml",
    "perfect_s_hlguard_1k": "packages/nagi_nr/configs/nagiperfect_s_hlguard_1k.yaml",
    "perfect_s_lumasmooth_edge_2k": "packages/nagi_nr/configs/nagiperfect_s_lumasmooth_edge_2k.yaml",
    "perfect_s_lumasmooth_2k": "packages/nagi_nr/configs/nagiperfect_s_lumasmooth_2k.yaml",
    "perfect_s_smoothgate_3k": "packages/nagi_nr/configs/nagiperfect_s_smoothgate_3k.yaml",
    "perfect_s_smoothgate_preserve_500": "packages/nagi_nr/configs/nagiperfect_s_smoothgate_preserve_500.yaml",
    "perfect_s_basepush_2k": "packages/nagi_nr/configs/nagiperfect_s_basepush_2k.yaml",
    "perfect_s_pilot_500": "packages/nagi_nr/configs/nagiperfect_s_pilot_500.yaml",
    "perfect_s_stable_2k": "packages/nagi_nr/configs/nagiperfect_s_stable_2k.yaml",
    "perfect_s_strictflat_5k": "packages/nagi_nr/configs/nagiperfect_s_strictflat_5k.yaml",
    "perfect_s_strictflat_2k": "packages/nagi_nr/configs/nagiperfect_s_strictflat_2k.yaml",
    "perfect_s_tailpush_3k": "packages/nagi_nr/configs/nagiperfect_s_tailpush_3k.yaml",
    "perfect_s_weakteacher_500": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_500.yaml",
    "perfect_s_weakteacher_chromahead_800": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_chromahead_800.yaml",
    "perfect_s_weakteacher_chromaaxis_4k": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_chromaaxis_4k.yaml",
    "perfect_s_weakteacher_chromatail_600": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_chromatail_600.yaml",
    "perfect_s_weakteacher_chromaresidual_3k": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_chromaresidual_3k.yaml",
    "perfect_s_weakteacher_guardquality_3k": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_guardquality_3k.yaml",
    "perfect_s_weakteacher_hf_800": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_hf_800.yaml",
    "perfect_s_weakteacher_lumaflat_600": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_lumaflat_600.yaml",
    "perfect_s_weakteacher_smoothhead_1k": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_smoothhead_1k.yaml",
    "perfect_s_weakteacher_strong_500": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_strong_500.yaml",
    "perfect_s_weakteacher_tailquality_12k": "packages/nagi_nr/configs/nagiperfect_s_weakteacher_tailquality_12k.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch NagiPerfect training in the background.")
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--seed", default="0")
    parser.add_argument("--max-iters", default="0")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    variant = args.variant
    out_dir = repo / "runs" / f"nagiperfect_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"
    pid_path = out_dir / "pid.txt"
    prefix = f"nagiperfect_{variant}"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "nagi_nr.train_perfect",
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
    if int(args.max_iters) > 0:
        cmd.extend(["--max-iters", str(args.max_iters)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== launch train-nagiperfect-{variant} ===\n")
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
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"started train-nagiperfect-{variant} pid={proc.pid}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
