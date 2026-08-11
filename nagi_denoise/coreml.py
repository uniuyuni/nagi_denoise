"""Core ML backend for NagiV2 inference (Apple Silicon fast path).

This is the fast path: on a 39.8MP frame the model stage takes ~65s through
Core ML (fp16, ``cpu_and_gpu``) versus ~237s through PyTorch-MPS, a 3.6x
speedup, for a maximum per-pixel difference of ~0.018 against the fp32
PyTorch reference (fp16 rounding only -- see ``docs/architecture.md``).

``coremltools`` is an **optional** dependency. Importing this module never
imports it; it is imported lazily when a :class:`CoreMLTiledDenoiser` is
actually constructed, so ``import nagi_denoise`` works fine on a machine
without Core ML.

Usage::

    from nagi_denoise import denoise
    out = denoise(img, backend="coreml")          # default package, cpu_and_gpu

    # or directly, for the model stage only (no chroma pass / no guard):
    from nagi_denoise.coreml import CoreMLTiledDenoiser
    dn = CoreMLTiledDenoiser()                    # default .mlpackage
    model_out = dn.denoise_array(img, overlap=64)

The exported ``.mlpackage`` is a FIXED tile-size, batch-1 ``mlprogram`` graph
(see ``scripts/export_coreml_nagi_v2.py``). This module wraps it in the exact
same Hann-window tiling scheme as :meth:`nagi_denoise.infer.Denoiser.
_tiled_forward` -- the coordinate grid, stride and window are imported from
``nagi_denoise.infer``, so the stitching geometry cannot drift from the
PyTorch path.

.. warning::
   **Never use ``compute_units="all"`` with an fp16 package.** ``ALL`` lets
   Core ML dispatch to the Apple Neural Engine, which corrupts this graph:
   on every one of the 23 validation tiles in
   ``runs/phase5_speed/coreml/validation_report.json`` the fp16+ANE output
   peaks 1.25x-4.8x higher than the PyTorch reference (worst absolute
   per-pixel error 8.7 on a scene whose true peak is 5.3). The default here
   is ``"cpu_and_gpu"`` and it must stay that way.

The exported graph has the in-graph HDR highlight guard permanently disarmed
(``highlight_protect_strength=0.0`` at export time). The guard is a
*pointwise* blend of the network output toward the collocated input pixel,
masked by a function of the input alone, so applying it once on the assembled
image is mathematically identical to applying it inside every tile before
Hann stitching -- provided the threshold is shared across all tiles, which
``denoise()`` always guarantees by computing it once on the whole image.
:func:`apply_highlight_guard_np` is that post-hoc equivalent.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from .infer import _cosine_window2d, _pad_to_multiple

__all__ = [
    "CoreMLTiledDenoiser",
    "DEFAULT_MLPACKAGE",
    "DEFAULT_COMPUTE_UNITS",
    "COMPUTE_UNITS",
    "apply_highlight_guard_np",
    "default_mlpackage",
    "is_available",
    "linear_luma_np",
    "resolve_compute_units",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default exported package: the production NagiV2-L checkpoint at tile 768,
#: fp16. Override per call, or globally via ``$NAGI_DENOISE_COREML_PACKAGE``.
DEFAULT_MLPACKAGE = (
    _REPO_ROOT / "runs" / "phase5_speed" / "coreml" / "nagi_v2_l_ft2_t768_fp16.mlpackage"
)

#: Environment variable that overrides :data:`DEFAULT_MLPACKAGE`.
MLPACKAGE_ENV_VAR = "NAGI_DENOISE_COREML_PACKAGE"

#: Selectable compute-unit strings. ``"all"`` is listed so it can be
#: benchmarked, but it is NOT safe for fp16 packages -- see the module
#: docstring's warning. It is never a default anywhere.
COMPUTE_UNITS = ("cpu_and_gpu", "cpu_only", "all")

#: The only compute-unit setting validated for production output.
DEFAULT_COMPUTE_UNITS = "cpu_and_gpu"


def is_available() -> bool:
    """True if ``coremltools`` can be imported in this process."""
    try:
        import coremltools  # noqa: F401
    except Exception:  # noqa: BLE001 - any import-time failure means unavailable
        return False
    return True


def default_mlpackage() -> Path:
    """The default ``.mlpackage`` path, *without* consulting the Hub.

    Env var wins, otherwise the in-repo export. This is the purely local
    answer and it never touches the network or needs ``huggingface_hub``; the
    returned path is not guaranteed to exist. For the full resolution chain
    (which additionally falls back to the Hugging Face cache and then to a
    download) use :func:`nagi_denoise.assets.resolve_coreml_package`, which is
    what :class:`CoreMLTiledDenoiser` calls.
    """
    override = os.environ.get(MLPACKAGE_ENV_VAR)
    return Path(override) if override else DEFAULT_MLPACKAGE


def _import_coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The Core ML backend needs the optional 'coremltools' dependency. "
            "Install it (`pip install 'coremltools>=9,<10'`) or use "
            "backend='torch'."
        ) from exc
    return ct


def resolve_compute_units(compute_units: Union[str, Any]) -> Any:
    """Map a compute-unit string to a ``coremltools.ComputeUnit``.

    Accepts an already-resolved ``ct.ComputeUnit`` unchanged. Raises on
    unknown names rather than silently falling back, so a typo can never
    quietly select the ANE.
    """
    ct = _import_coremltools()
    if isinstance(compute_units, ct.ComputeUnit):
        return compute_units
    name = str(compute_units).lower()
    mapping = {
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "all": ct.ComputeUnit.ALL,
    }
    if name not in mapping:
        raise ValueError(
            f"unknown compute_units {compute_units!r}; expected one of {COMPUTE_UNITS}"
        )
    return mapping[name]


def _tile_from_spec(mlmodel) -> Optional[int]:
    """Read the square tile size out of an mlprogram's input shape.

    The exported graph is a static (1, 3, T, T) input. Returns ``None`` if the
    shape cannot be read (e.g. a flexible-shape model), in which case the
    caller must pass ``tile`` explicitly.
    """
    try:
        inputs = mlmodel.get_spec().description.input
        shape = list(inputs[0].type.multiArrayType.shape)
    except Exception:  # noqa: BLE001 - spec layouts vary between versions
        return None
    if len(shape) < 2:
        return None
    h, w = int(shape[-2]), int(shape[-1])
    return h if h == w and h > 0 else None


class CoreMLTiledDenoiser:
    """Tiled Core ML inference matching ``Denoiser._tiled_forward``'s geometry.

    Args:
        mlpackage_path: exported ``.mlpackage``. An explicit path always
            wins. ``None`` resolves it through
            :func:`nagi_denoise.assets.resolve_coreml_package`:
            ``$NAGI_DENOISE_COREML_PACKAGE``, then the in-repo export under
            ``runs/phase5_speed/coreml/``, then an existing Hugging Face
            cache entry, and only then a Hub download.
        allow_download: ``False`` (or ``$NAGI_DENOISE_OFFLINE=1``) forbids
            that last step, so no network access can occur.
        tile: square tile size. ``None`` reads it from the model's input
            shape (the exported graph is static), which is what you want.
        compute_units: ``"cpu_and_gpu"`` (default), ``"cpu_only"``, or
            ``"all"``. **``"all"`` corrupts fp16 output** -- see the module
            docstring.
        size_multiple: the model's required size multiple (8 for NagiV2).
        warmup: run one dummy tile at construction so the one-off Core ML
            graph compile/load cost is not charged to the first real tile.

    This wrapper runs the *model stage only*. It does not apply the highlight
    guard, the chroma pass, or ``input_blend``; use
    ``nagi_denoise.denoise(..., backend="coreml")`` for the full production
    pipeline.
    """

    def __init__(
        self,
        mlpackage_path: Union[str, Path, None] = None,
        tile: Optional[int] = None,
        compute_units: Union[str, Any] = DEFAULT_COMPUTE_UNITS,
        size_multiple: int = 8,
        warmup: bool = True,
        allow_download: bool = True,
    ) -> None:
        ct = _import_coremltools()
        from .assets import resolve_coreml_package

        path = resolve_coreml_package(mlpackage_path, allow_download=allow_download)
        if not path.exists():
            raise FileNotFoundError(
                f"Core ML package not found: {path}. Export one with "
                "`pixi run export-coreml`, download it from "
                "https://huggingface.co/uniuyuni/nagi_denoise, or pass "
                f"coreml_package=... / set ${MLPACKAGE_ENV_VAR}."
            )

        self.path = path
        self.compute_units = resolve_compute_units(compute_units)
        self.size_multiple = int(size_multiple)
        self.model = ct.models.MLModel(str(path), compute_units=self.compute_units)

        resolved_tile = int(tile) if tile is not None else _tile_from_spec(self.model)
        if not resolved_tile:
            raise ValueError(
                f"could not determine the tile size of {path}; pass tile=... explicitly"
            )
        self.tile = int(resolved_tile)

        if warmup:
            dummy = np.zeros((1, 3, self.tile, self.tile), dtype=np.float32)
            self.model.predict({"tile_in": dummy})

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CoreMLTiledDenoiser(path={self.path.name!r}, tile={self.tile}, "
            f"compute_units={self.compute_units})"
        )

    def predict_tile(self, tile_chw: np.ndarray) -> np.ndarray:
        """``tile_chw``: (3, tile, tile) float32 -> (3, tile, tile) float32."""
        x = tile_chw[None].astype(np.float32, copy=False)
        y = self.model.predict({"tile_in": x})["tile_out"]
        return y[0]

    def tiled_forward(
        self,
        img_chw: torch.Tensor,
        overlap: int,
        per_tile_times: Optional[list] = None,
    ) -> torch.Tensor:
        """``img_chw``: (3, H, W) float32 CPU tensor -> (3, H, W) float32 CPU tensor.

        Structurally identical to ``Denoiser._tiled_forward`` at
        ``batch_size=1``: same coordinate grid, same stride, same Hann window,
        same weighted-accumulate-then-normalize. Only the per-tile forward
        call is swapped for a Core ML ``predict``.
        """
        C, H, W = img_chw.shape
        tile = self.tile
        if tile % self.size_multiple != 0:
            tile = (tile // self.size_multiple) * self.size_multiple
        stride = max(1, tile - overlap)

        ys = list(range(0, max(H - tile, 0) + 1, stride))
        xs = list(range(0, max(W - tile, 0) + 1, stride))
        if not ys or ys[-1] + tile < H:
            ys.append(max(0, H - tile))
        if not xs or xs[-1] + tile < W:
            xs.append(max(0, W - tile))

        out = torch.zeros_like(img_chw)
        weight = torch.zeros((1, H, W), dtype=img_chw.dtype)

        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for ty in ys:
            th = min(tile, H - ty)
            for tx in xs:
                tw = min(tile, W - tx)
                groups.setdefault((th, tw), []).append((ty, tx))

        win_cache: dict[tuple[int, int], torch.Tensor] = {}

        for (th, tw), coords in groups.items():
            win = win_cache.get((th, tw))
            if win is None:
                win = _cosine_window2d(th, tw, device="cpu", dtype=img_chw.dtype)[0]  # (1, th, tw)
                win_cache[(th, tw)] = win

            for (ty, tx) in coords:
                patch = img_chw[..., ty : ty + th, tx : tx + tw]
                padded, _ = _pad_to_multiple(patch.unsqueeze(0), self.size_multiple)
                padded_np = padded[0].numpy()

                t0 = time.perf_counter()
                y_np = self.predict_tile(padded_np)
                dt = time.perf_counter() - t0
                if per_tile_times is not None:
                    per_tile_times.append(dt)

                y = torch.from_numpy(y_np)[..., :th, :tw].float()
                out[..., ty : ty + th, tx : tx + tw] += y * win
                weight[..., ty : ty + th, tx : tx + tw] += win

        return out / weight.clamp_min(1e-8)

    def denoise_array(
        self,
        img_hwc: np.ndarray,
        overlap: int = 64,
        per_tile_times: Optional[list] = None,
    ) -> np.ndarray:
        """Denoise a float32 (H, W, 3) linear-light array through Core ML.

        Images smaller than one tile go through a single ``predict`` call, so
        they must not exceed the exported tile size on either axis (the graph
        is static). Anything larger is tiled.
        """
        t = torch.from_numpy(np.ascontiguousarray(img_hwc[..., :3])).permute(2, 0, 1).float()
        H, W = t.shape[-2:]
        if H > self.tile or W > self.tile:
            out = self.tiled_forward(t, overlap=overlap, per_tile_times=per_tile_times)
        else:
            padded, _ = _pad_to_multiple(t.unsqueeze(0), self.size_multiple)
            if padded.shape[-2:] != (self.tile, self.tile):
                raise ValueError(
                    f"the Core ML graph has a fixed {self.tile}x{self.tile} input; an image "
                    f"smaller than one tile must be exactly that size after padding to a "
                    f"multiple of {self.size_multiple} (got {tuple(padded.shape[-2:])}). Use "
                    "backend='torch' for small images, or pad the input yourself."
                )
            t0 = time.perf_counter()
            y_np = self.predict_tile(padded[0].numpy())
            dt = time.perf_counter() - t0
            if per_tile_times is not None:
                per_tile_times.append(dt)
            out = torch.from_numpy(y_np)[..., :H, :W].float()
        return np.ascontiguousarray(out.permute(1, 2, 0).numpy().astype(np.float32, copy=False))


def linear_luma_np(x: np.ndarray) -> np.ndarray:
    """Rec.709 linear luma of an (H, W, 3) array."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def apply_highlight_guard_np(
    denoised: np.ndarray,
    inp: np.ndarray,
    threshold: float,
    transition: float,
    strength: float,
) -> np.ndarray:
    """Numpy re-implementation of ``nagi_v2.input_highlight_guard``.

    Applied once on the assembled (post-stitch) image. See the module
    docstring for why this is exactly equivalent to the in-graph per-tile
    guard when threshold/transition/strength are shared across all tiles of
    an image, which ``denoise()`` always does.
    """
    if strength <= 0.0:
        return denoised
    inp_nonneg = np.clip(inp, 0.0, None)
    y = linear_luma_np(inp_nonneg)
    if transition <= 0.0:
        mask = (y > threshold).astype(np.float32)
    else:
        mask = 1.0 / (1.0 + np.exp(-(y - threshold) / max(transition, 1e-6)))
    mask = (mask * min(max(strength, 0.0), 1.0))[..., None]
    return denoised * (1.0 - mask) + inp_nonneg * mask
