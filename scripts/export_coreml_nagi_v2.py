"""Phase 5 Stage B: export the production NagiV2-L checkpoint to Core ML.

Exports a FIXED-TILE-SIZE (default 768x768, batch 1) `mlprogram` graph so the
resulting model can be dropped into the exact same Hann-window tiling scheme
used by `nagi_denoise.infer.Denoiser._tiled_forward` (see
`scripts/bench_coreml_nagi_v2.py`, which reimplements that tiling loop around
the Core ML model instead of the PyTorch one).

Why the in-graph HDR highlight guard is left disabled at export time:
`NagiV2.highlight_protect_strength` defaults to 0.0 on the production
checkpoint (`runs/nagi_v2_l_ft2/nagi_v2_l_ft2_final.pt`) -- the production
pipeline (`nagi_denoise/pipeline/denoise.py::denoise`) only arms it
dynamically, per-call, with a threshold computed from the *whole* input
image. That can't be baked into a fixed traced graph. But the guard
(`input_highlight_guard`) is a *pointwise* blend of the network output
against the (collocated) input pixel, gated purely by a function of the
input -- it does not depend on any cross-tile context. Because the
Hann-window tile stitcher is a per-pixel weighted average, blending after
stitching is mathematically identical to blending inside each tile before
stitching (the blend is affine in the denoised value, and the mask is
identical across every tile that covers a given pixel since it only depends
on the collocated input): see docstring math in
`scripts/bench_coreml_nagi_v2.py`. So exporting the graph as
"denoised-without-guard" and applying `input_highlight_guard` once on the
assembled output reproduces the production result exactly, while keeping
the exported graph fixed and reusable across images.

Usage:
    pixi run python scripts/export_coreml_nagi_v2.py --precision fp16
    pixi run python scripts/export_coreml_nagi_v2.py --precision fp32
    pixi run python scripts/export_coreml_nagi_v2.py --precision both
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

import coremltools as ct

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nagi_denoise.infer import Denoiser  # noqa: E402
from nagi_denoise.pipeline.denoise import PRODUCTION_WEIGHTS  # noqa: E402

OUT_DIR = REPO_ROOT / "runs" / "phase5_speed" / "coreml"


@contextlib.contextmanager
def _asinh_via_log_sqrt():
    """Graph-level workaround for a coremltools 9.0 MIL gap: the torch
    frontend's `aten::asinh` handler calls `mb.asinh(...)`, but this
    coremltools version's MIL Builder has no `asinh` op at all (it has
    `asin`, `sinh`, `atanh`, but not `asinh` -- confirmed via
    `coremltools.converters.mil.mil.Builder`), so conversion fails with
    `AttributeError: type object 'Builder' has no attribute 'asinh'`.

    `NagiV2.compress()` is the single call site (`nagi_denoise/models/nagi_v2.py`,
    `torch.asinh(x * self.asinh_k) / self._asinh_norm`). asinh has an exact
    elementary-function identity, asinh(x) = log(x + sqrt(x^2 + 1)), which is
    numerically well-behaved for this model's whole operating range (x up to
    roughly `asinh_k * 10 = 80` for the brightest HDR pixels this pipeline
    ever sees) since `x + sqrt(x^2+1)` is always positive and grows like
    `2x` for large x, well inside fp32 range. This is a like-for-like
    substitution, not an approximation: it changes which primitive ops get
    traced, not the function being computed.

    Scoped as a context manager so the monkeypatch only affects
    `torch.asinh` for the duration of tracing in this export script; the
    installed `nagi_denoise` package (training, production inference) is
    never touched.
    """
    orig_torch_asinh = torch.asinh
    orig_tensor_asinh = torch.Tensor.asinh

    def _replacement(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + torch.sqrt(x * x + 1.0))

    def _replacement_method(self: torch.Tensor) -> torch.Tensor:
        return _replacement(self)

    torch.asinh = _replacement
    torch.Tensor.asinh = _replacement_method
    try:
        yield
    finally:
        torch.asinh = orig_torch_asinh
        torch.Tensor.asinh = orig_tensor_asinh


class _TraceWrapper(torch.nn.Module):
    """Thin wrapper that pins forward() to the plain-tensor-output path.

    NagiV2.forward already returns a single tensor when `return_aux=False`
    (the default), so this only exists to make the traced call-signature
    explicit (single positional tensor in, single tensor out) and to assert
    the highlight guard really is disarmed for this export -- if the loaded
    checkpoint ever ships with a nonzero `highlight_protect_strength`, that
    would silently get baked into the graph, and the post-hoc-guard
    reasoning above would then double-apply the guard. Fail loudly instead.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        assert float(getattr(model, "highlight_protect_strength", 0.0)) == 0.0, (
            "expected highlight_protect_strength == 0.0 at export time (guard is "
            "applied post-hoc in the CoreML benchmarking wrapper); got "
            f"{model.highlight_protect_strength!r}"
        )
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, return_aux=False)


def _load_wrapped_model(weights: str) -> torch.nn.Module:
    dn = Denoiser.load(weights, device="cpu")
    model = dn.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return _TraceWrapper(model)


def _example_input(tile: int, seed: int = 0) -> torch.Tensor:
    """A shape-1x3xTxT example tile with realistic linear-light statistics.

    Not all-zero / all-random-normal: mixes a low-frequency "scene" term with
    high-frequency noise and a few strong highlights so the traced graph
    exercises the asinh compression, the SCA pooling branch, and clamping in
    a representative regime (tracing only needs *a* concrete example for
    shape/dtype; values only matter insofar as NaN/Inf could derail
    coremltools' shape/range inference for ops like asinh).
    """
    g = torch.Generator().manual_seed(seed)
    low = torch.rand((1, 3, tile // 8, tile // 8), generator=g)
    low = torch.nn.functional.interpolate(low, size=(tile, tile), mode="bilinear", align_corners=False)
    noise = torch.randn((1, 3, tile, tile), generator=g) * 0.02
    x = (low * 1.5 + noise).clamp_min(0.0)
    # Sprinkle a few HDR highlight pixels (production HDR frames peak ~10.0).
    x[:, :, :: tile // 6, :: tile // 6] *= 6.0
    return x.float()


def export_one(weights: str, tile: int, precision: str, opset_target: str) -> dict:
    assert precision in ("fp16", "fp32")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"nagi_v2_l_ft2_t{tile}_{precision}.mlpackage"

    wrapped = _load_wrapped_model(weights)
    example = _example_input(tile)

    t0 = time.perf_counter()
    with torch.no_grad(), _asinh_via_log_sqrt():
        traced = torch.jit.trace(wrapped, example, check_trace=True)
    t_trace = time.perf_counter() - t0
    print(f"[{precision}] torch.jit.trace: {t_trace:.1f}s")

    compute_precision = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    deployment_target = getattr(ct.target, opset_target)

    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tile_in", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="tile_out", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=deployment_target,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    t_convert = time.perf_counter() - t0
    print(f"[{precision}] ct.convert: {t_convert:.1f}s")

    mlmodel.save(str(out_path))
    print(f"[{precision}] saved -> {out_path}")

    return {
        "precision": precision,
        "tile": tile,
        "path": str(out_path),
        "trace_seconds": t_trace,
        "convert_seconds": t_convert,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(PRODUCTION_WEIGHTS))
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--precision", choices=["fp16", "fp32", "both"], default="both")
    ap.add_argument(
        "--deployment-target", default="iOS18",
        help="ct.target attribute name (coremltools uses unified iOS-numbered "
        "enums for the macOS deployment target too); iOS18 == macOS15, this "
        "machine's OS (`sw_vers` -> ProductVersion 15.7.7).",
    )
    args = ap.parse_args()

    precisions = ["fp16", "fp32"] if args.precision == "both" else [args.precision]
    results = []
    for prec in precisions:
        try:
            results.append(export_one(args.weights, args.tile, prec, args.deployment_target))
        except Exception as exc:  # noqa: BLE001
            print(f"[{prec}] EXPORT FAILED: {type(exc).__name__}: {exc}")
            raise

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
