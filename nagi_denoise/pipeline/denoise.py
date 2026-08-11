"""Single production entry point for the NagiV2 denoise pipeline.

``denoise()`` is the function external callers should use; everything else in
this package is machinery underneath it.

The whole pipeline is deliberately two stages:

1. ``Denoiser.load(weights).denoise_array(img, input_space="linear", ...)`` --
   NagiV2-L tiled inference (Hann-window tiling, seam-free). Optionally run
   through Core ML instead (``backend="coreml"``, ~3.6x faster).
2. A single deterministic chroma cleanup pass:
   ``flat_chroma_smoother.smooth_chroma(out, **CH)``.

Both stages operate on float32 linear-light HWC RGB numpy arrays and preserve
HDR values above 1.0. Nothing else is added on top of this -- see
``docs/architecture.md`` for why the legacy ~40-filter stack was retired.
"""
from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path
from typing import Union

import numpy as np
import torch

from ..devices import resolve_device
from ..infer import Denoiser
from .detail_guard import write_exr, write_tiff
from .flat_chroma_smoother import smooth_chroma
from .probe import read_image

# The production checkpoint (NagiV2-L, "Phase 2B"). Verified: SIDD Val
# 39.030 dB, HDR highlight retention 0.9979 on the X-T5 Room crop.
PRODUCTION_WEIGHTS = (
    Path(__file__).resolve().parents[2] / "runs" / "nagi_v2_l_ft2" / "nagi_v2_l_ft2_final.pt"
)

# Deterministic chroma cleanup parameters. Do not add filters beyond this.
CH: dict = dict(
    strength=1.0,
    chroma_sigma=4.5,
    detail_sigma=1.2,
    threshold=0.022,
    transition=0.010,
    highlight_threshold=1.0,
    highlight_transition=0.2,
    hdr_restore_peak_threshold=0.95,
    hdr_restore_threshold=0.85,
    hdr_restore_transition=0.25,
)

_MODEL_CACHE: dict[tuple[str, str], Denoiser] = {}
_COREML_CACHE: dict[tuple[str, str, int], object] = {}
_CACHE_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)

# Highlight-guard defaults (Phase 3 HDR-defect fix). The production
# checkpoint ships with the input-luma highlight guard *disabled*
# (highlight_protect_strength=0.0) -- it must be re-armed at inference time.
#
# The guard works by blending the *noisy input* back in above a luma
# threshold. An earlier attempt scaled the threshold down to each image's
# own peak (min(1.0, 0.7*p99.9)); that engaged over large areas of
# low-dynamic-range scenes and re-introduced noise unevenly -- measured 3.7x
# more flat-region noise on Z7 bird, visible as "some parts still noisy,
# others over-flattened". It was rejected.
#
# The policy below only arms the guard when the image actually contains
# above-SDR content worth protecting. Audited over all 8 diagnostic scenes:
# on true-HDR scenes (p99.9 luma 3.0-4.0) it lifts highlight retention to
# ~1.000 at *identical* flat-region noise, and on low-range scenes it stays
# disarmed, so it can never cause the unevenness above.
HIGHLIGHT_GUARD_MIN_P999 = 1.0
HIGHLIGHT_GUARD_DEFAULT_THRESHOLD = 1.0
HIGHLIGHT_GUARD_DEFAULT_TRANSITION = 0.15
HIGHLIGHT_GUARD_DEFAULT_STRENGTH = 0.85


def _resolve_weights(weights: Union[str, Path, None]) -> str:
    return str(PRODUCTION_WEIGHTS) if weights is None else str(Path(weights))


def _linear_luma_np(x: np.ndarray) -> np.ndarray:
    """Rec.709 linear luma for an (H, W, 3) array. Matches
    ``nagi_denoise.models.nagi_v2.linear_luma`` (same coefficients)."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _resolve_highlight_guard(
    x: np.ndarray,
    highlight_guard: Union[bool, float],
    transition: float,
    strength: float,
) -> dict:
    """Resolve the highlight-guard knob to concrete
    (threshold, transition, strength) values, computing the adaptive
    threshold from the *whole* input image's top-0.1% luma when requested.

    Must be called once per denoise() invocation on the full image -- never
    per tile -- so every tile in a tiled inference pass sees the exact same
    threshold. A per-tile threshold would reintroduce visible seams at tile
    boundaries whenever adjacent tiles disagree on how bright "the image" is.
    """
    if highlight_guard is False:
        return {
            "mode": "disabled",
            "threshold": 0.0,
            "transition": float(transition),
            "strength": 0.0,
            "input_p999_luma": None,
        }

    if highlight_guard is True:
        luma_in = _linear_luma_np(np.clip(x, 0.0, None))
        p999 = float(np.percentile(luma_in, 99.9))
        if p999 <= HIGHLIGHT_GUARD_MIN_P999:
            # No above-SDR content to protect. Arming the guard here would
            # only blend noise back in (see the module-level note).
            return {
                "mode": "conditional-disarmed",
                "threshold": 0.0,
                "transition": float(transition),
                "strength": 0.0,
                "input_p999_luma": p999,
            }
        return {
            "mode": "conditional-armed",
            "threshold": HIGHLIGHT_GUARD_DEFAULT_THRESHOLD,
            "transition": float(transition),
            "strength": float(strength),
            "input_p999_luma": p999,
        }

    # A bare number pins an explicit threshold (adaptive rule bypassed).
    return {
        "mode": "fixed",
        "threshold": float(highlight_guard),
        "transition": float(transition),
        "strength": float(strength),
        "input_p999_luma": None,
    }


def _get_denoiser(weights: str, device: str) -> Denoiser:
    """Return a cached Denoiser for (weights, device), loading it if needed.

    Keyed on the *resolved* device string so "auto" and its concrete
    resolution (e.g. "mps") share a single cache entry / loaded model.
    """
    resolved_device = str(resolve_device(device))
    key = (weights, resolved_device)
    with _CACHE_LOCK:
        dn = _MODEL_CACHE.get(key)
        if dn is None:
            dn = Denoiser.load(weights, device=resolved_device)
            _MODEL_CACHE[key] = dn
        return dn


def _get_coreml_denoiser(package: Union[str, Path, None], compute_units: str, tile: int):
    """Return a cached ``CoreMLTiledDenoiser``, constructing it if needed.

    Imported lazily so that ``coremltools`` is never required to import or use
    the default (PyTorch) path.
    """
    from ..coreml import CoreMLTiledDenoiser, default_mlpackage

    path = str(Path(package)) if package is not None else str(default_mlpackage())
    key = (path, str(compute_units), int(tile))
    with _CACHE_LOCK:
        dn = _COREML_CACHE.get(key)
        if dn is None:
            dn = CoreMLTiledDenoiser(path, tile=None, compute_units=compute_units)
            if int(dn.tile) != int(tile):
                raise ValueError(
                    f"the Core ML package {Path(path).name} is a fixed {dn.tile}x{dn.tile} "
                    f"graph but denoise() was called with tile={tile}. Pass tile={dn.tile}, "
                    "or re-export at the tile size you want "
                    "(`scripts/export_coreml_nagi_v2.py --tile ...`)."
                )
            _COREML_CACHE[key] = dn
        return dn


def denoise(
    img: np.ndarray,
    *,
    weights: Union[str, Path, None] = None,
    device: str = "auto",
    backend: str = "torch",
    coreml_package: Union[str, Path, None] = None,
    coreml_compute_units: str = "cpu_and_gpu",
    tile: int = 768,
    overlap: int = 64,
    chroma_cleanup: bool = True,
    input_blend: float = 0.20,
    detail_strength: float | None = None,
    highlight_guard: Union[bool, float] = True,
    highlight_guard_transition: float = HIGHLIGHT_GUARD_DEFAULT_TRANSITION,
    highlight_guard_strength: float = HIGHLIGHT_GUARD_DEFAULT_STRENGTH,
    batch_size: int = 1,
    amp_dtype: "torch.dtype | None" = None,
) -> np.ndarray:
    """Denoise a float32 HWC linear-light RGB image with the production NagiV2 pipeline.

    Args:
        img: (H, W, 3) float32 numpy array, linear-light RGB. HDR-safe: values
            above 1.0 are preserved, never clipped.
        weights: checkpoint path. ``None`` uses the production checkpoint
            (``runs/nagi_v2_l_ft2/nagi_v2_l_ft2_final.pt``). Used by the
            "torch" backend only -- the "coreml" backend runs whatever
            checkpoint was baked into ``coreml_package``.
        device: "auto" (default), "mps", "cuda", or "cpu". "torch" backend only.
        backend: which engine runs the model stage.
              * ``"torch"`` (default): PyTorch. The reference implementation;
                every quality number in the docs was measured on this path.
              * ``"coreml"``: the exported Core ML graph
                (``nagi_denoise.coreml``). ~3.6x faster on the model stage
                (39.8MP: 65s vs 237s) at the cost of fp16 rounding -- max
                per-pixel difference ~0.018 vs the fp32 torch path on the
                validation tiles. Needs the optional ``coremltools``
                dependency and an exported ``.mlpackage``.
            The two backends stitch tiles with the same code path, so their
            geometry is identical; only the per-tile forward differs.
        coreml_package: path to the ``.mlpackage`` for ``backend="coreml"``.
            ``None`` uses ``nagi_denoise.coreml.default_mlpackage()``
            (overridable with ``$NAGI_DENOISE_COREML_PACKAGE``).
        coreml_compute_units: "cpu_and_gpu" (default), "cpu_only", or "all".
            **Never use "all" with an fp16 package**: it lets Core ML
            dispatch to the Apple Neural Engine, which corrupts this graph --
            peak values run 1.25x-4.8x high on every validation tile. The
            default is the only setting validated for production output.
        tile: square tile size for tiled inference. See ``Denoiser`` for the
            Hann-window seam-free tiling scheme. With ``backend="coreml"``
            this must match the exported graph's fixed tile size (768).
        overlap: overlap in pixels between adjacent tiles.
        chroma_cleanup: if True (default), run the deterministic
            ``smooth_chroma`` stage with the ``CH`` parameters after model
            inference. If False, return the raw model output.
        input_blend: fraction of the original input mixed back into the
            denoised result (0.0-1.0, default 0.20). The model is a strong
            denoiser and on some scenes (verified on K-5 Ice) it removes
            ~74% of the input's high-frequency energy, which reads as flat
            even though the legacy v12 pipeline is equally flat there. A
            small input blend restores apparent detail, but it restores
            noise in proportion -- measured on the Ice detail ROI:
            0% -> HF 0.00242 / chroma-noise 0.00186; 15% -> 0.00316 /
            0.00294; 30% -> 0.00414 / 0.00443. Set 0.0 for the cleanest
            possible output.
        detail_strength: if not None, sets ``model.detail_scale`` before
            inference. This is the live strength knob for a possible future
            generative-detail option; on the production checkpoint the
            confidence gate is closed, so this is currently a no-op by
            design -- it is wired here so it activates automatically if a
            future checkpoint opens the gate.
        highlight_guard: controls the model's input-luma HDR safety valve
            (``NagiV2.highlight_protect_*``, see ``input_highlight_guard`` in
            ``nagi_denoise/models/nagi_v2.py``). The production checkpoint
            ships with this guard *disabled*, which lets the model darken
            the brightest highlights on low-dynamic-range scenes (verified
            defect: up to -38% on the top 1% of luma). Values:
              * ``True`` (default): *conditional*. The guard is armed at a
                fixed luma threshold of 1.0 only when the input actually
                contains above-SDR content, i.e. when the whole image's
                top-0.1% linear luma exceeds
                ``HIGHLIGHT_GUARD_MIN_P999`` (1.0); otherwise it stays
                disarmed. On the 8 diagnostic scenes this lifts highlight
                retention to ~1.000 on true-HDR scenes at *identical*
                flat-region noise, and never engages on low-range scenes.
                (An earlier adaptive variant that scaled the threshold to
                each image's own peak was rejected -- see the module-level
                note: it blended noisy input back in over large areas and
                measured 3.7x more flat-region noise on Z7 bird.)
              * a float: pins an explicit threshold, bypassing the
                conditional rule.
              * ``False``: disables the guard entirely (matches the
                as-shipped checkpoint behavior).
            The decision and threshold are computed once over the full image
            (never per tile) so every tile in a tiled inference pass shares
            the same threshold -- computing it per tile would reintroduce
            seams at tile boundaries.
        highlight_guard_transition: sigmoid transition width (in luma units)
            for the guard blend. Ignored when ``highlight_guard=False``.
        highlight_guard_strength: maximum blend-toward-input strength (0-1)
            above the threshold. Ignored when ``highlight_guard=False``.
        batch_size: number of tiles forwarded together per model call during
            tiled inference. Default 1. Measured on this M1, larger batches
            are *slower*, not faster (b2: 273s, b4: 1014s vs b1: 237s) --
            leave it at 1 unless you are benchmarking other hardware.
            ``backend="torch"`` only.
        amp_dtype: optional ``torch.autocast`` dtype for the model forward,
            e.g. ``torch.float16``. ``None`` (default) keeps the exact fp32
            path. On this M1 it bought nothing (247s vs 237s); use
            ``backend="coreml"`` for the real speedup.
            ``backend="torch"`` only.

    Returns:
        (H, W, 3) float32 numpy array, same color space (linear) as input.

    Note:
        The resolved highlight-guard parameters for the most recent call are
        recorded on ``denoise.last_highlight_guard`` (a dict) for auditing,
        and also emitted via the ``nagi_denoise.pipeline.denoise`` logger.
    """
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(
            f"denoise() expects a (H, W, 3) float32 HWC linear RGB array, got shape {img.shape}"
        )

    backend = str(backend).lower()
    if backend not in ("torch", "coreml"):
        raise ValueError(f"backend must be 'torch' or 'coreml', got {backend!r}")

    x = np.nan_to_num(
        img.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0, copy=True
    )

    # The guard decision is made once, on the whole image, for both backends.
    guard_info = _resolve_highlight_guard(
        x, highlight_guard, highlight_guard_transition, highlight_guard_strength
    )

    if backend == "torch":
        weights_str = _resolve_weights(weights)
        dn = _get_denoiser(weights_str, device)

        if detail_strength is not None:
            dn.model.detail_scale = float(detail_strength)

        # (Re-)apply the highlight guard on every call. The Denoiser instance
        # is cached across calls, so leftover state from a previous call (e.g.
        # a different scene's fixed threshold) must never leak in -- all three
        # attributes are always set explicitly below, regardless of what
        # `highlight_guard` resolved to.
        dn.model.highlight_protect_threshold = float(guard_info["threshold"])
        dn.model.highlight_protect_transition = float(guard_info["transition"])
        dn.model.highlight_protect_strength = float(guard_info["strength"])

        out = dn.denoise_array(
            x, input_space="linear", tile=int(tile), overlap=int(overlap),
            batch_size=int(batch_size), amp_dtype=amp_dtype,
        )
    else:
        # Core ML: the exported graph is fixed, so the torch-only knobs cannot
        # be honoured. Fail loudly rather than silently ignoring them.
        for name, value, default in (
            ("weights", weights, None),
            ("detail_strength", detail_strength, None),
            ("amp_dtype", amp_dtype, None),
            ("batch_size", int(batch_size), 1),
        ):
            if value != default:
                raise ValueError(
                    f"{name}={value!r} is not supported with backend='coreml' "
                    "(the exported graph is static); use backend='torch', or "
                    "re-export the graph you want."
                )

        from ..coreml import apply_highlight_guard_np

        cml = _get_coreml_denoiser(coreml_package, coreml_compute_units, int(tile))
        out = cml.denoise_array(x, overlap=int(overlap))
        # The exported graph has the in-graph guard disarmed; apply the exact
        # post-hoc equivalent here (see nagi_denoise/coreml.py for why the two
        # are identical when the threshold is shared across tiles).
        out = apply_highlight_guard_np(
            out, x,
            threshold=float(guard_info["threshold"]),
            transition=float(guard_info["transition"]),
            strength=float(guard_info["strength"]),
        )
        guard_info = {**guard_info, "applied": "post-hoc (coreml)"}

    if chroma_cleanup:
        out, _stats, _gate = smooth_chroma(out, compute_stats=False, **CH)

    blend = float(input_blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"input_blend must be in [0, 1], got {blend}")
    if blend > 0.0:
        # Mix the (sanitized) input back in. Applied after the chroma pass so
        # the blend fraction refers to the finished result, and using `x`
        # rather than `img` so non-finite input never leaks back out.
        out = out * (1.0 - blend) + x * blend

    out = np.ascontiguousarray(out.astype(np.float32, copy=False))
    assert np.isfinite(out).all(), "denoise() produced non-finite output"

    denoise.last_highlight_guard = guard_info
    _LOGGER.info("highlight_guard resolved: %s", guard_info)

    return out


denoise.last_highlight_guard = None


def main() -> None:
    """Production CLI. Entry point: ``nagi-denoise-pipeline``."""
    ap = argparse.ArgumentParser(
        prog="nagi-denoise-pipeline",
        description="Denoise an EXR/TIFF image with the production NagiV2 pipeline.",
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=None, help="Defaults to the production checkpoint.")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument(
        "--backend", default="torch", choices=["torch", "coreml"],
        help="Model engine. 'torch' (default) is the reference path; 'coreml' is "
             "~3.6x faster on Apple Silicon and needs an exported .mlpackage.",
    )
    ap.add_argument(
        "--coreml-package", default=None,
        help="Path to the .mlpackage for --backend coreml (default: the exported "
             "production package; $NAGI_DENOISE_COREML_PACKAGE overrides).",
    )
    ap.add_argument(
        "--coreml-compute-units", default="cpu_and_gpu",
        choices=["cpu_and_gpu", "cpu_only", "all"],
        help="Core ML compute units. Do NOT use 'all' with an fp16 package: the "
             "Neural Engine corrupts this graph (peak values run 1.25-4.8x high).",
    )
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--no-chroma-cleanup", action="store_true")
    ap.add_argument("--detail-strength", type=float, default=None)
    ap.add_argument("--input-blend", type=float, default=0.20)
    ap.add_argument(
        "--highlight-guard",
        default="adaptive",
        help="'adaptive' (default), 'off', or a float pinning an explicit input-luma threshold.",
    )
    ap.add_argument("--highlight-guard-transition", type=float, default=HIGHLIGHT_GUARD_DEFAULT_TRANSITION)
    ap.add_argument("--highlight-guard-strength", type=float, default=HIGHLIGHT_GUARD_DEFAULT_STRENGTH)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    hg_arg = args.highlight_guard.strip().lower() if isinstance(args.highlight_guard, str) else args.highlight_guard
    if hg_arg in ("adaptive", "auto", "on", "true"):
        highlight_guard: Union[bool, float] = True
    elif hg_arg in ("off", "false", "none", "disabled"):
        highlight_guard = False
    else:
        highlight_guard = float(hg_arg)

    img = read_image(in_path)
    out = denoise(
        img,
        weights=args.weights,
        device=args.device,
        backend=args.backend,
        coreml_package=args.coreml_package,
        coreml_compute_units=args.coreml_compute_units,
        tile=args.tile,
        overlap=args.overlap,
        chroma_cleanup=not args.no_chroma_cleanup,
        detail_strength=args.detail_strength,
        input_blend=args.input_blend,
        highlight_guard=highlight_guard,
        highlight_guard_transition=args.highlight_guard_transition,
        highlight_guard_strength=args.highlight_guard_strength,
    )
    print(f"highlight_guard resolved: {denoise.last_highlight_guard}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".exr":
        write_exr(out_path, out)
    elif suffix in {".tif", ".tiff"}:
        write_tiff(out_path, out)
    else:
        raise ValueError(f"Unsupported output extension {suffix!r}; use .exr or .tif/.tiff")

    print(f"wrote {out_path}")


# Backwards-compatible alias: this CLI used to be reachable only as
# `python -m nagi_denoise.pipeline.denoise`, whose entry function was `_cli`.
_cli = main


if __name__ == "__main__":
    main()
