"""Run small ANE correctness experiments for NAFNet Core ML exports."""
from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "ane_coreml_experiment"
REF_CROP1536 = (
    ROOT
    / "runs/coreml_exr_outputs_full_fp16_native_crop1536/"
    / "sample_cat_noisy_crop_x3096_y1808_w1536_h1536_coreml_nafnet_srgb16.tiff"
)
BROKEN_CROP1536 = (
    ROOT
    / "runs/coreml_exr_outputs_full_fp16_b1_all_native_crop1536/"
    / "sample_cat_noisy_crop_x3096_y1808_w1536_h1536_coreml_nafnet_srgb16.tiff"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    package: Path
    skip_ops: str = ""
    preexisting: bool = False


CANDIDATES = [
    Candidate(
        name="fp16_all_control_broken",
        package=ROOT / "runs/nafnet_fast_coreml/nafnet_width64_fp16.mlpackage",
        preexisting=True,
    ),
    Candidate(
        name="skip_mul",
        package=RUN_ROOT / "nafnet_width64_fp16_skip_mul.mlpackage",
        skip_ops="mul",
    ),
    Candidate(
        name="skip_layernorm",
        package=RUN_ROOT / "nafnet_width64_fp16_skip_layernorm.mlpackage",
        skip_ops="reduce_mean,sub,square,sqrt,real_div",
    ),
    Candidate(
        name="skip_sqrt_div",
        package=RUN_ROOT / "nafnet_width64_fp16_skip_sqrt_div.mlpackage",
        skip_ops="sqrt,real_div",
    ),
]


def run(cmd: list[str], timeout: int | None = None) -> tuple[int | None, float, str]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode, time.perf_counter() - start, proc.stdout
    except subprocess.TimeoutExpired as exc:
        output = ""
        if exc.stdout:
            output += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", "replace")
        if exc.stderr:
            output += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", "replace")
        return None, time.perf_counter() - start, output + "\nTIMEOUT\n"


def export_candidate(candidate: Candidate) -> dict:
    if candidate.preexisting or candidate.package.exists():
        return {"skipped": True, "returncode": 0, "elapsed_sec": 0.0}
    cmd = [
        "pixi",
        "run",
        "python",
        "scripts/export_coreml_nafnet.py",
        "--weights",
        "benchmarks/nafnet/NAFNet-SIDD-width64.pth",
        "--state-key",
        "state_dict",
        "--precision",
        "float16",
        "--height",
        "256",
        "--width",
        "256",
        "--batch",
        "1",
        "--compute-units",
        "all",
        "--output",
        str(candidate.package.relative_to(ROOT)),
    ]
    if candidate.skip_ops:
        cmd.extend(["--fp16-skip-ops", candidate.skip_ops])
    rc, elapsed, output = run(cmd, timeout=180)
    log_path = RUN_ROOT / f"{candidate.name}_export.log"
    log_path.write_text(output, encoding="utf-8")
    return {"skipped": False, "returncode": rc, "elapsed_sec": elapsed, "log": str(log_path)}


def denoise_candidate(candidate: Candidate) -> dict:
    out_dir = RUN_ROOT / candidate.name
    cmd = [
        "pixi",
        "run",
        "python",
        "scripts/denoise_exr_coreml.py",
        "--input",
        "samples/coreml_exr_input/sample_cat_noisy.EXR",
        "--model",
        str(candidate.package.relative_to(ROOT)),
        "--compute-units",
        "all",
        "--tile",
        "256",
        "--batch",
        "1",
        "--overlap",
        "0",
        "--input-space",
        "linear",
        "--crop",
        "3096,1808,256,256",
        "--process-scale",
        "1.0",
        "--progress-every",
        "1",
        "--output-dir",
        str(out_dir.relative_to(ROOT)),
    ]
    rc, elapsed, output = run(cmd, timeout=90)
    log_path = RUN_ROOT / f"{candidate.name}_denoise.log"
    log_path.write_text(output, encoding="utf-8")
    return {"returncode": rc, "elapsed_sec": elapsed, "log": str(log_path), "output_dir": str(out_dir)}


def load_reference_crop() -> np.ndarray:
    ref1536 = tifffile.imread(REF_CROP1536).astype(np.float32) / 65535.0
    return ref1536[:256, :256, :3]


def evaluate_output(candidate: Candidate, denoise: dict, ref: np.ndarray) -> dict:
    out_dir = Path(denoise["output_dir"])
    tif = out_dir / "sample_cat_noisy_crop_x3096_y1808_w256_h256_coreml_nafnet_srgb16.tiff"
    if not tif.exists():
        return {"exists": False, "passed": False}
    arr = tifffile.imread(tif).astype(np.float32) / 65535.0
    arr = arr[..., :3]
    diff = arr - ref
    mse = float(np.mean(diff * diff))
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    sat_any = float((arr >= 0.99).any(axis=2).mean())
    sat_all_values = float((arr >= 0.99).mean())
    med = float(np.percentile(arr, 50))
    mean = [float(x) for x in arr.reshape(-1, 3).mean(axis=0)]
    mean_abs = float(np.mean(np.abs(diff)))
    passed = psnr > 55.0 and sat_any < 0.01 and med < 0.98
    return {
        "exists": True,
        "passed": passed,
        "psnr_vs_cpu_gpu": psnr,
        "mean_abs": mean_abs,
        "sat_any_channel_ge_0_99": sat_any,
        "sat_values_ge_0_99": sat_all_values,
        "p50": med,
        "mean_rgb": mean,
        "output": str(tif),
    }


def write_summary(results: list[dict]) -> None:
    lines = [
        "# ANE Core ML Experiment",
        "",
        "Reference: full fp16 `cpu_and_gpu` crop.",
        "",
        "| candidate | skip ops | export s | denoise s | PSNR | sat any >=0.99 | p50 | result |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        ev = item.get("eval") or {}
        export = item.get("export") or {}
        denoise = item.get("denoise") or {}
        psnr = ev.get("psnr_vs_cpu_gpu")
        sat = ev.get("sat_any_channel_ge_0_99")
        p50 = ev.get("p50")
        result = "PASS" if ev.get("passed") else "FAIL"
        if denoise.get("returncode") is None:
            result = "TIMEOUT"
        lines.append(
            "| {name} | `{skip}` | {export_s:.1f} | {denoise_s:.1f} | {psnr} | {sat} | {p50} | {result} |".format(
                name=item["name"],
                skip=item["skip_ops"] or "-",
                export_s=float(export.get("elapsed_sec") or 0.0),
                denoise_s=float(denoise.get("elapsed_sec") or 0.0),
                psnr="-" if psnr is None else f"{psnr:.3f}",
                sat="-" if sat is None else f"{sat:.4f}",
                p50="-" if p50 is None else f"{p50:.4f}",
                result=result,
            )
        )
    (RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    ref = load_reference_crop()
    results = []
    for candidate in CANDIDATES:
        print(f"== {candidate.name} ==")
        export = export_candidate(candidate)
        if export["returncode"] != 0:
            result = {
                "name": candidate.name,
                "skip_ops": candidate.skip_ops,
                "package": str(candidate.package),
                "export": export,
                "denoise": {},
                "eval": {"exists": False, "passed": False},
            }
            results.append(result)
            write_summary(results)
            continue
        denoise = denoise_candidate(candidate)
        eval_result = evaluate_output(candidate, denoise, ref)
        result = {
            "name": candidate.name,
            "skip_ops": candidate.skip_ops,
            "package": str(candidate.package),
            "export": export,
            "denoise": denoise,
            "eval": eval_result,
        }
        results.append(result)
        write_summary(results)
        print(json.dumps(eval_result, indent=2))
    (RUN_ROOT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results)
    print(RUN_ROOT / "summary.md")


if __name__ == "__main__":
    main()
