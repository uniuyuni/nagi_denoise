"""Model architectures.

``NagiV2`` is the production line. ``NagiNR`` is the retired predecessor,
kept only so old checkpoints still load (``Denoiser.load`` auto-detects it)
and so ``nagi_denoise.train.train_nr`` still runs; it is not part of the
public API.
"""

from .nagi_nr import NagiNR
from .nagi_v2 import NagiV2, build_nagi_v2, build_nagi_v2_preset

__all__ = [
    "NagiNR",
    "NagiV2",
    "build_nagi_v2",
    "build_nagi_v2_preset",
]
