"""Launch a NagiQ training variant in the background."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "q40_ab_fixed": "packages/nagi_nr/configs/nagiq_q40_ab_fixed.yaml",
    "q40_ab_random": "packages/nagi_nr/configs/nagiq_q40_ab_random.yaml",
    "q40_micro_overfit": "packages/nagi_nr/configs/nagiq_q40_micro_overfit.yaml",
    "q48_fast_random_screen": "packages/nagi_nr/configs/nagiq_q48_fast_random_screen.yaml",
    "q48_trim_corrected": "packages/nagi_nr/configs/nagiq_q48_trim_corrected.yaml",
    "q48_trim_sanity": "packages/nagi_nr/configs/nagiq_q48_trim_sanity.yaml",
    "q48_trim": "packages/nagi_nr/configs/nagiq_q48_trim.yaml",
    "nafnet_fast_p3_mse_2k": "packages/nagi_nr/configs/nagiq_nafnet_fast_p3_mse_2k.yaml",
    "nafnet_fast_p3_mse_extend4k": "packages/nagi_nr/configs/nagiq_nafnet_fast_p3_mse_extend4k.yaml",
    "nafnet_fast_w48q_mse_2k": "packages/nagi_nr/configs/nagiq_nafnet_fast_w48q_mse_2k.yaml",
    "nafnet_fast_w56q_mse_2k": "packages/nagi_nr/configs/nagiq_nafnet_fast_w56q_mse_2k.yaml",
    "nafnet_fast_w56q_residual_curriculum_2k": "packages/nagi_nr/configs/nagiq_nafnet_fast_w56q_residual_curriculum_2k.yaml",
    "gamair_s_sidd_mse_2k": "packages/nagi_nr/configs/gamair_s_sidd_mse_2k.yaml",
    "gamair_s_sidd_mse_extend10k": "packages/nagi_nr/configs/gamair_s_sidd_mse_extend10k.yaml",
    "gamair_s_faststart_128_10k": "packages/nagi_nr/configs/gamair_s_faststart_128_10k.yaml",
    "gamair_s_faststart_128_polyu_5k": "packages/nagi_nr/configs/gamair_s_faststart_128_polyu_5k.yaml",
    "gamair_s_polyu_256_ft_10k": "packages/nagi_nr/configs/gamair_s_polyu_256_ft_10k.yaml",
    "gamair_s_teacher_restart_256_10k": "packages/nagi_nr/configs/gamair_s_teacher_restart_256_10k.yaml",
    "gamair_s_teacher_restart_256_30k": "packages/nagi_nr/configs/gamair_s_teacher_restart_256_30k.yaml",
    "gamair_m56_faststart_128_2k": "packages/nagi_nr/configs/gamair_m56_faststart_128_2k.yaml",
    "gamair_l_sidd_mse_2k": "packages/nagi_nr/configs/gamair_l_sidd_mse_2k.yaml",
    "realfast_v0_2k": "packages/nagi_nr/configs/nagi_realfast_v0_2k.yaml",
    "realfast_v0_extend1k": "packages/nagi_nr/configs/nagi_realfast_v0_extend1k.yaml",
    "realfast_v0b_2k": "packages/nagi_nr/configs/nagi_realfast_v0b_2k.yaml",
    "realfast_v1_2k": "packages/nagi_nr/configs/nagi_realfast_v1_2k.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch NagiQ training in the background.")
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--seed", default="0")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    variant = args.variant
    out_dir = repo / "runs" / f"nagiq_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"
    pid_path = out_dir / "pid.txt"
    prefix = f"nagiq_{variant}"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "nagi_nr.train_q",
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

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_f = open(log_path, "a", buffering=1)
    log_f.write(f"\n=== launch train-nagiq-{variant} ===\n")
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
    print(f"started train-nagiq-{variant} pid={proc.pid}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
