"""Nagi Denoise -- HDR-safe blind denoiser for real photographs.

The entry point is :func:`denoise`. It takes and returns a float32 HWC
linear-light RGB numpy array, preserves values above 1.0, and handles
arbitrarily large images with seam-free tiling::

    import numpy as np
    from nagi_denoise import denoise

    out = denoise(img)                      # img: (H, W, 3) float32, linear RGB
    out = denoise(img, backend="coreml")    # ~3.6x faster on Apple Silicon

Everything else in this package is machinery underneath that:

* :class:`Denoiser` -- the lower-level tiled-inference API (model stage only,
  no chroma pass, no highlight guard). Use it when you want the raw model.
* :class:`NagiV2` -- the model architecture.
* :mod:`nagi_denoise.coreml` -- the optional Core ML backend. Importing it
  does not require ``coremltools``; constructing a denoiser from it does.
* :mod:`nagi_denoise.assets` -- where the weights come from.
  :func:`~nagi_denoise.assets.resolve_weights` and
  :func:`~nagi_denoise.assets.resolve_coreml_package` implement one shared
  order: an explicit path, then the ``$NAGI_DENOISE_WEIGHTS`` /
  ``$NAGI_DENOISE_COREML_PACKAGE`` env vars, then the in-repo ``runs/``
  copy, then an already-populated Hugging Face cache, and only if all of
  those miss, a download from ``uniuyuni/nagi_denoise``. Call them directly
  to pre-fetch on purpose; pass ``allow_download=False`` (or set
  ``$NAGI_DENOISE_OFFLINE=1``) to forbid the network outright.
  ``huggingface_hub`` is optional and is imported only if the download
  path is actually reached.
* ``srgb_to_linear`` / ``linear_to_srgb``, ``describe_devices`` /
  ``resolve_device`` -- small helpers shared by the CLIs.

See ``README.md`` for the knobs and measured numbers, and
``docs/architecture.md`` for how the pipeline is put together.
"""

from . import assets
from .assets import AssetNotFoundError, resolve_coreml_package, resolve_weights
from .pipeline.denoise import denoise
from .infer import Denoiser
from .models.nagi_v2 import NagiV2
from .transforms import srgb_to_linear, linear_to_srgb
from .devices import describe_devices, resolve_device

__all__ = [
    # Entry point.
    "denoise",
    # Lower-level API.
    "Denoiser",
    "NagiV2",
    # Asset resolution (local first, Hub last).
    "assets",
    "resolve_weights",
    "resolve_coreml_package",
    "AssetNotFoundError",
    # Helpers.
    "srgb_to_linear",
    "linear_to_srgb",
    "describe_devices",
    "resolve_device",
]
__version__ = "1.0.0"
