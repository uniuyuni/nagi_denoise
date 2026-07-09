"""Device selection helpers for CLI and API entry points."""
from __future__ import annotations

from typing import Any

import torch


def describe_devices() -> dict[str, Any]:
    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    return {
        "mps_built": mps_built,
        "mps_available": mps_available,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve "auto" and provide a clear error for sandbox-hidden MPS."""
    if isinstance(requested, torch.device):
        name = requested.type
    else:
        name = str(requested).lower()

    if name == "auto":
        devices = describe_devices()
        if devices["mps_available"]:
            return torch.device("mps")
        if devices["cuda_available"]:
            return torch.device("cuda")
        return torch.device("cpu")

    if name == "mps":
        devices = describe_devices()
        if not devices["mps_built"]:
            raise RuntimeError("This PyTorch build does not include MPS support.")
        if not devices["mps_available"]:
            raise RuntimeError(
                "MPS is not available in this process. On Codex desktop this usually "
                "means the command is running inside the sandbox; run the pixi task "
                "outside the sandbox / with escalated execution, or use --device cpu "
                "for a slow smoke test."
            )

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process.")

    return torch.device(requested)
