"""Check the Pixi/PyTorch/Nagi NR runtime.

This intentionally exits successfully when MPS is merely unavailable so it can
be used inside the Codex sandbox. Pass --require-mps for outside-sandbox checks.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys

import torch

from nagi_nr.devices import describe_devices, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Nagi NR runtime dependencies.")
    parser.add_argument("--require-mps", action="store_true", help="exit non-zero unless MPS is usable")
    args = parser.parse_args()

    print(f"python: {sys.executable}")
    print(f"python version: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '')}")

    devices = describe_devices()
    print(f"mps built: {devices['mps_built']}")
    print(f"mps available: {devices['mps_available']}")
    print(f"cuda available: {devices['cuda_available']}")
    print(f"auto device: {resolve_device('auto')}")

    import nagi_nr
    import nagi_nr_bench

    print(f"nagi_nr: {nagi_nr.__file__}")
    print(f"nagi_nr_bench: {nagi_nr_bench.__file__}")

    if devices["mps_available"]:
        x = torch.ones(4, device="mps")
        print(f"mps tensor check: {float((x * 2).sum().cpu())}")
    elif args.require_mps:
        raise SystemExit(
            "MPS is not available in this process. In Codex, run this pixi task "
            "outside the sandbox / with escalated execution."
        )


if __name__ == "__main__":
    main()
