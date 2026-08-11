#!/usr/bin/env python
"""Upload the released assets to the Hugging Face Hub. DRY-RUN BY DEFAULT.

Sends four things to ``uniuyuni/nagi_denoise``:

    MODEL_CARD.md                                 -> README.md
    runs/nagi_v2_l_ft2/nagi_v2_l_ft2_final.pt     -> nagi_v2_l_ft2_final.pt
    runs/phase5_speed/coreml/..._fp16.mlpackage/  -> ..._fp16.mlpackage/   (folder)
    runs/phase5_speed/coreml/..._fp32.mlpackage/  -> ..._fp32.mlpackage/   (folder)

The layout matches what ``nagi_denoise.assets`` expects: the checkpoint is a
single file at the repo root (``hf_hub_download``), and each ``.mlpackage`` is
a *folder* of files under its own prefix (``snapshot_download`` with
``allow_patterns``).

Nothing is transferred without ``--yes``. Without it the script prints exactly
what it would send, with sizes, and exits::

    pixi run python scripts/upload_to_hf.py            # dry run, prints the plan
    pixi run python scripts/upload_to_hf.py --yes      # actually uploads

Authentication is the standard Hugging Face one (``hf auth login`` or
``$HF_TOKEN``); this script never handles credentials itself.

Note the licences: the code is Apache-2.0, but the weights are CC BY-NC 4.0
(see ``MODEL_LICENSE``). ``MODEL_CARD.md`` carries that in its front matter,
which is why it must be uploaded as the repo's ``README.md``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nagi_denoise.assets import (  # noqa: E402
    COREML_FP16_PACKAGE,
    COREML_FP32_PACKAGE,
    HF_REPO_ID,
    REPO_COREML_DIR,
    REPO_WEIGHTS,
    WEIGHTS_FILENAME,
)

MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:,.1f} {unit}" if unit != "B" else f"{int(x)} B"
        x /= 1024
    return f"{x:.1f} GB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _plan() -> list[dict]:
    """What would be uploaded, in order. Each entry is one Hub API call."""
    return [
        {
            "kind": "file",
            "local": MODEL_CARD,
            "remote": "README.md",
            "note": "the model card (CC BY-NC 4.0 front matter)",
        },
        {
            "kind": "file",
            "local": REPO_WEIGHTS,
            "remote": WEIGHTS_FILENAME,
            "note": "production PyTorch checkpoint, NagiV2-L 15.43M",
        },
        {
            "kind": "folder",
            "local": REPO_COREML_DIR / COREML_FP16_PACKAGE,
            "remote": COREML_FP16_PACKAGE,
            "note": "Core ML export, tile 768, fp16 (the fast path)",
        },
        {
            "kind": "folder",
            "local": REPO_COREML_DIR / COREML_FP32_PACKAGE,
            "remote": COREML_FP32_PACKAGE,
            "note": "Core ML export, tile 768, fp32",
        },
    ]


def _describe(plan: list[dict], repo_id: str) -> int:
    """Print the plan. Returns the number of missing local assets."""
    print(f"Target Hugging Face repo: {repo_id}")
    print(f"Source tree:              {REPO_ROOT}\n")
    missing = 0
    total = 0
    for item in plan:
        local: Path = item["local"]
        if not local.exists():
            print(f"  MISSING  {item['remote']}")
            print(f"           expected at {local}")
            missing += 1
            continue
        if item["kind"] == "folder":
            files = sorted(f for f in local.rglob("*") if f.is_file())
            size = sum(f.stat().st_size for f in files)
            print(f"  {item['remote']}/  ({_human(size)}, {len(files)} files)  -- {item['note']}")
            for f in files:
                print(f"      {f.relative_to(local)}  ({_human(f.stat().st_size)})")
        else:
            size = local.stat().st_size
            print(f"  {item['remote']}  ({_human(size)})  -- {item['note']}")
            print(f"      from {local}")
        total += size
        print()
    print(f"Total to transfer: {_human(total)}")
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="upload_to_hf.py",
        description="Upload the released assets to the Hugging Face Hub "
                    "(dry run unless --yes is given).",
    )
    ap.add_argument(
        "--yes", action="store_true",
        help="Actually transfer. Without this flag nothing leaves this machine.",
    )
    ap.add_argument("--repo-id", default=HF_REPO_ID, help=f"default: {HF_REPO_ID}")
    ap.add_argument(
        "--commit-message", default="Add NagiV2-L weights, Core ML exports and model card",
    )
    ap.add_argument(
        "--create-repo", action="store_true",
        help="Create the model repo first if it does not exist (needs --yes).",
    )
    args = ap.parse_args()

    plan = _plan()

    print("=" * 74)
    print("DRY RUN -- nothing will be uploaded" if not args.yes else "UPLOADING (--yes given)")
    print("=" * 74)
    missing = _describe(plan, args.repo_id)

    if missing:
        print(f"\n{missing} asset(s) missing locally. Refusing to continue.")
        return 1

    if not args.yes:
        print("\nThis was a dry run. Re-run with --yes to upload.")
        print("Reminder: the weights are CC BY-NC 4.0 (MODEL_LICENSE); the model card "
              "declares that and is uploaded as README.md.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\nhuggingface_hub is required to upload:\n"
              "    pixi add --pypi huggingface_hub    (or: pip install huggingface_hub)")
        return 1

    api = HfApi()
    if args.create_repo:
        print(f"\ncreating repo {args.repo_id} (exist_ok=True) ...")
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)

    for item in plan:
        local: Path = item["local"]
        print(f"\nuploading {item['remote']} ...")
        if item["kind"] == "folder":
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type="model",
                folder_path=str(local),
                path_in_repo=item["remote"],
                commit_message=f"{args.commit_message} ({item['remote']})",
            )
        else:
            api.upload_file(
                repo_id=args.repo_id,
                repo_type="model",
                path_or_fileobj=str(local),
                path_in_repo=item["remote"],
                commit_message=f"{args.commit_message} ({item['remote']})",
            )
        print(f"  done: {item['remote']}")

    print(f"\nAll assets uploaded to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
