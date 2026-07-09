"""Color space and HDR range transforms (all float32, batched)."""
from __future__ import annotations
import torch

_SRGB_GAMMA = 2.4
_SRGB_THRESH_FWD = 0.04045
_SRGB_THRESH_INV = 0.0031308


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB encoded [0,1] tensor -> linear light.

    IEC 61966-2-1 piecewise. Works on any shape, dtype preserved.
    """
    low = x / 12.92
    high = ((x.clamp_min(0.0) + 0.055) / 1.055).pow(_SRGB_GAMMA)
    return torch.where(x <= _SRGB_THRESH_FWD, low, high)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Linear light -> sRGB encoded.

    Gradient-safe. The power branch ``x**(1/2.4)`` has an infinite derivative as
    x -> 0 (exponent < 1). Since ``torch.where`` evaluates BOTH branches, those
    inf gradients poison the selected linear branch via 0*inf = nan during
    backward. We feed the power branch a value floored to a positive constant
    wherever the linear branch is selected, so the forward result is unchanged
    but every element's gradient stays finite. (Previously this function was only
    used at inference / no-grad, so the bug was latent until used inside a loss.)
    """
    is_low = x <= _SRGB_THRESH_INV
    low = x * 12.92
    # Where the low branch wins, replace the power base with 1.0 (finite grad);
    # where the high branch wins, x > _SRGB_THRESH_INV > 0 so the base is safe.
    base = torch.where(is_low, torch.ones_like(x), x.clamp_min(0.0))
    high = 1.055 * base.pow(1.0 / _SRGB_GAMMA) - 0.055
    return torch.where(is_low, low, high)


def asinh_compress(x: torch.Tensor, k: float = 8.0) -> torch.Tensor:
    """Reversible HDR range compression.

    asinh(k * x) / asinh(k). Maps [0, ~10] smoothly to [0, ~1.5].
    Near-linear at small x, log-like at large x. Handles negatives.
    """
    norm = float(torch.asinh(torch.tensor(k)))
    return torch.asinh(x * k) / norm


def asinh_decompress(x_c: torch.Tensor, k: float = 8.0) -> torch.Tensor:
    norm = float(torch.asinh(torch.tensor(k)))
    return torch.sinh(x_c * norm) / k
