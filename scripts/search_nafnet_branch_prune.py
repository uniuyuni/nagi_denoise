"""Greedy search over branch-level NAFNet pruning candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_nafnet_branch_prune import (
    branch_gmac,
    evaluate,
    install_branch_gates,
    load_nafnet,
    load_validation,
    reset_gates,
    resolve_device,
    set_branch,
)


def load_candidates(path: str, max_candidates: int, min_single_drop: float) -> list[str]:
    audit_path = Path(path)
    if not audit_path.exists():
        raise FileNotFoundError(f"candidate audit JSON not found: {audit_path}")
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = [row for row in data["rows"] if float(row["drop"]) >= min_single_drop]
    rows = sorted(rows, key=lambda row: (float(row["drop"]), float(row["saved_gmac"])), reverse=True)
    return [str(row["branch"]) for row in rows[:max_candidates]]


def set_skips(ref_by_name: dict[str, Any], refs: list[Any], skips: set[str]) -> None:
    reset_gates(refs)
    for name in skips:
        set_branch(ref_by_name, name, 0.0)


def saved_gmac(skip: set[str], height: int, width: int) -> float:
    return sum(branch_gmac(name, height, width) for name in skip)


def format_skip(skip: set[str]) -> str:
    return ", ".join(sorted(skip)) if skip else "(none)"


def markdown(args: argparse.Namespace, baseline: dict[str, Any], selected: list[dict[str, Any]], trials: list[dict[str, Any]]) -> str:
    lines = [
        "# NAFNet Branch-Prune Greedy Search",
        "",
        f"Patches: `{baseline['patches']}`",
        f"Baseline PSNR: `{baseline['psnr_out']:.3f} dB`",
        f"Max cumulative drop: `{args.max_drop:.3f} dB`",
        "",
        "## Selected",
        "",
        "| step | added | skip count | PSNR | drop | saved GMAC | ideal speed | teacher MSE |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if selected:
        for row in selected:
            lines.append(
                "| {step} | {candidate} | {count} | {psnr:.3f} | {drop:+.3f} | {saved_gmac:.3f} | {ideal_speed:.3f}x | {teacher_mse:.8f} |".format(
                    **row
                )
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Trials",
            "",
            "| step | candidate | PSNR | drop | saved GMAC | score | teacher MSE |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(trials, key=lambda r: (r["step"], -r["score"])):
        lines.append(
            "| {step} | {candidate} | {psnr:.3f} | {drop:+.3f} | {saved_gmac:.3f} | {score:.3f} | {teacher_mse:.8f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy branch-level NAFNet pruning search.")
    ap.add_argument("--weights", default="benchmarks/nafnet/NAFNet-SIDD-width64.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    ap.add_argument("--candidate-json", default="runs/nafnet_branch_prune_audit/audit.json")
    ap.add_argument("--max-candidates", type=int, default=24)
    ap.add_argument("--min-single-drop", type=float, default=-0.025)
    ap.add_argument("--max-patches", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-drop", type=float, default=0.35)
    ap.add_argument("--cost-bias", type=float, default=0.015)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--teacher-gmac", type=float, default=63.24)
    ap.add_argument("--output-md", default="runs/nafnet_branch_prune_search/greedy.md")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    candidates = load_candidates(args.candidate_json, args.max_candidates, args.min_single_drop)
    if not candidates:
        raise RuntimeError("no candidates survived the single-branch filter")

    device = resolve_device(args.device)
    patches = load_validation(args.noisy_mat, args.gt_mat, args.max_patches)
    model = load_nafnet(args.weights, device)
    refs = install_branch_gates(model)
    ref_by_name = {ref.name: ref for ref in refs}
    unknown = [name for name in candidates if name not in ref_by_name]
    if unknown:
        raise ValueError(f"unknown candidates: {unknown[:8]}")

    print(f"device: {device}")
    print(f"patches: {len(patches)}")
    print(f"candidates: {len(candidates)}")
    baseline = evaluate(model, device, patches, collect_outputs=True)
    teacher_outputs = baseline.pop("outputs")
    print(
        "baseline: psnr={psnr_out:.3f} noisy={psnr_in:.3f} ms={ms_patch:.1f}".format(
            **baseline
        )
    )

    current: set[str] = set()
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for step in range(1, args.max_steps + 1):
        if not remaining:
            break
        print(f"step {step}: evaluating {len(remaining)} candidates", flush=True)
        step_rows: list[dict[str, Any]] = []
        for candidate in remaining:
            skip = set(current)
            skip.add(candidate)
            set_skips(ref_by_name, refs, skip)
            result = evaluate(model, device, patches, teacher_outputs=teacher_outputs)
            saved = saved_gmac(skip, args.height, args.width)
            drop = float(result["psnr_out"] - baseline["psnr_out"])
            score = float(result["psnr_out"]) + args.cost_bias * saved
            row = {
                "step": step,
                "candidate": candidate,
                "skip": sorted(skip),
                "skip_text": format_skip(skip),
                "count": len(skip),
                "psnr": float(result["psnr_out"]),
                "drop": drop,
                "saved_gmac": saved,
                "ideal_speed": args.teacher_gmac / max(args.teacher_gmac - saved, 1e-9),
                "teacher_mse": float(result.get("teacher_mse", 0.0)),
                "score": score,
            }
            step_rows.append(row)
            trials.append(row)
            print(
                f"  {candidate:16s} psnr={row['psnr']:.3f} drop={drop:+.3f} "
                f"saved={saved:.2f} ideal={row['ideal_speed']:.2f}x",
                flush=True,
            )

        viable = [row for row in step_rows if row["drop"] >= -args.max_drop]
        if not viable:
            print("stop: no viable candidate")
            break
        chosen = max(viable, key=lambda row: (row["score"], row["psnr"], row["saved_gmac"]))
        current = set(chosen["skip"])
        remaining = [name for name in remaining if name != chosen["candidate"]]
        selected.append(chosen)
        print(
            "selected step {step}: {candidate} drop={drop:+.3f} saved={saved_gmac:.2f} ideal={ideal_speed:.2f}x".format(
                **chosen
            ),
            flush=True,
        )

    reset_gates(refs)
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output_json) if args.output_json else out_md.with_suffix(".json")
    payload = {
        "args": vars(args),
        "device": str(device),
        "baseline": baseline,
        "selected": selected,
        "trials": trials,
    }
    out_md.write_text(markdown(args, baseline, selected, trials), encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    if selected:
        print(f"recommended branches: {' '.join(selected[-1]['skip'])}")


if __name__ == "__main__":
    main()
