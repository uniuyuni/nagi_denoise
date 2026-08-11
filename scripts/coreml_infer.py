"""Compatibility shim -- the Core ML backend now lives in ``nagi_denoise.coreml``.

The Phase 5 scripts in this directory (`bench_coreml_nagi_v2.py`,
`validate_coreml_nagi_v2.py`, `gates_coreml_nagi_v2.py`) import
``CoreMLTiledDenoiser`` / ``apply_highlight_guard_np`` / ``linear_luma_np``
from here. Those now come from the package module so there is exactly one
implementation of the Core ML tiling loop.

New code should import from ``nagi_denoise.coreml`` (or just call
``nagi_denoise.denoise(..., backend="coreml")``) instead of this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nagi_denoise.coreml import (  # noqa: E402,F401
    CoreMLTiledDenoiser,
    apply_highlight_guard_np,
    linear_luma_np,
)

__all__ = ["CoreMLTiledDenoiser", "apply_highlight_guard_np", "linear_luma_np"]
