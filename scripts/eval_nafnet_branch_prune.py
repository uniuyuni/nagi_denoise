"""Evaluate a fixed NAFNet branch-prune mask against the full teacher."""
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


def apply_skips(ref_by_name: dict[str, Any], refs: list[Any], skips: list[str]) -> None:
    reset_gates(refs)
    for name in skips:
        if name not in ref_by_name:
            raise ValueError(f"unknown branch {name!r}")
        set_branch(ref_by_name, name, 0.0)


def markdown(payload: dict[str, Any]) -> str:
    b = payload["baseline"]
    p = payload["pruned"]
    lines = [
        "# NAFNet Branch-Prune Fixed Evaluation",
        "",
        f"Device: `{payload['device']}`",
        f"Patches: `{b['patches']}`",
        f"Skipped branches: `{', '.join(payload['skip'])}`",
        "",
        "| metric | baseline | pruned | delta |",
        "| --- | ---: | ---: | ---: |",
        f"| PSNR | {b['psnr_out']:.3f} dB | {p['psnr_out']:.3f} dB | {payload['drop']:+.3f} dB |",
        f"| noisy PSNR | {b['psnr_in']:.3f} dB | {p['psnr_in']:.3f} dB | - |",
        f"| ms/patch | {b['ms_patch']:.1f} | {p['ms_patch']:.1f} | {p['ms_patch'] - b['ms_patch']:+.1f} |",
        f"| teacher MSE | 0 | {p['teacher_mse']:.8f} | - |",
        "",
        "| cost | value |",
        "| --- | ---: |",
        f"| skipped branches | {len(payload['skip'])} |",
        f"| saved GMAC | {payload['saved_gmac']:.3f} |",
        f"| teacher GMAC | {payload['teacher_gmac']:.3f} |",
        f"| remaining GMAC | {payload['remaining_gmac']:.3f} |",
        f"| ideal speed | {payload['ideal_speed']:.3f}x |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate fixed branch-level NAFNet pruning.")
    ap.add_argument("--weights", default="benchmarks/nafnet/NAFNet-SIDD-width64.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--noisy-mat", default="data/ValidationNoisyBlocksSrgb.mat")
    ap.add_argument("--gt-mat", default="data/ValidationGtBlocksSrgb.mat")
    ap.add_argument("--max-patches", type=int, default=128)
    ap.add_argument("--skip", nargs="+", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--teacher-gmac", type=float, default=63.24)
    ap.add_argument("--output-md", default="runs/nafnet_branch_prune_eval/eval.md")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    patches = load_validation(args.noisy_mat, args.gt_mat, args.max_patches)
    model = load_nafnet(args.weights, device)
    refs = install_branch_gates(model)
    ref_by_name = {ref.name: ref for ref in refs}

    print(f"device: {device}")
    print(f"patches: {len(patches)}")
    print(f"skip: {', '.join(args.skip)}")
    baseline = evaluate(model, device, patches, collect_outputs=True)
    teacher_outputs = baseline.pop("outputs")
    print(
        "baseline: psnr={psnr_out:.3f} noisy={psnr_in:.3f} ms={ms_patch:.1f}".format(
            **baseline
        )
    )
    apply_skips(ref_by_name, refs, args.skip)
    pruned = evaluate(model, device, patches, teacher_outputs=teacher_outputs)
    saved = sum(branch_gmac(name, args.height, args.width) for name in args.skip)
    payload: dict[str, Any] = {
        "args": vars(args),
        "device": str(device),
        "skip": list(args.skip),
        "baseline": baseline,
        "pruned": pruned,
        "drop": float(pruned["psnr_out"] - baseline["psnr_out"]),
        "saved_gmac": saved,
        "teacher_gmac": args.teacher_gmac,
        "remaining_gmac": args.teacher_gmac - saved,
        "ideal_speed": args.teacher_gmac / max(args.teacher_gmac - saved, 1e-9),
    }
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output_json) if args.output_json else out_md.with_suffix(".json")
    out_md.write_text(markdown(payload), encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "pruned: psnr={psnr_out:.3f} drop={drop:+.3f} saved={saved:.3f} ideal={ideal:.3f}x".format(
            psnr_out=pruned["psnr_out"],
            drop=payload["drop"],
            saved=saved,
            ideal=payload["ideal_speed"],
        )
    )
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
