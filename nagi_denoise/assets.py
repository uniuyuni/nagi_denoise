"""Asset resolution: find the model weights, downloading only as a last resort.

Both the PyTorch checkpoint and the exported Core ML package go through the
same five-step chain, implemented once in :func:`_resolve_asset` and exposed
as :func:`resolve_weights` / :func:`resolve_coreml_package`. The order is,
stopping at the **first hit**:

1. **An explicit path from the caller** (``weights=`` / ``coreml_package=``).
   Always wins. It is returned verbatim and never falls back to anything --
   if you named a path, a missing file is an error, not a reason to fetch
   236MB from the internet.
2. **The environment variable** -- ``$NAGI_DENOISE_WEIGHTS`` for the
   checkpoint, ``$NAGI_DENOISE_COREML_PACKAGE`` for the package. Same
   contract as (1): honoured verbatim, never falls back.
3. **The in-repo path** under ``runs/`` -- *if it exists*. This is the
   developer / trainer case: someone who cloned the repo with its artifacts,
   or who trained the checkpoint themselves, must never hit the network.
4. **An already-populated Hugging Face cache**, consulted with
   ``local_files_only=True`` so it cannot make a network round-trip. A user
   who downloaded the asset once is, from then on, in the same position as
   (3).
5. **A download from the Hugging Face Hub** (:data:`HF_REPO_ID`). Only ever
   reached when steps 1-4 all missed. It is logged at INFO with the repo id,
   the filename and the destination, so it is never silent.

Steps 1-3 work with no third-party dependency at all. ``huggingface_hub`` is
an **optional** dependency: it is imported lazily, only when steps 4/5 are
actually reached, and its absence there raises an actionable error naming
both the install command and the manual-download URL.

Forbidding the network entirely
-------------------------------

Two equivalent switches, either of which stops the chain after step 4 (the
offline cache lookup) and so guarantees no fetch can happen:

* ``allow_download=False`` on any of the resolver functions, or on
  :func:`nagi_denoise.denoise`;
* ``NAGI_DENOISE_OFFLINE=1`` in the environment (also accepts ``true`` /
  ``yes`` / ``on``), which forces it process-wide.

When the asset genuinely is not present anywhere, offline mode raises
:class:`AssetNotFoundError` (a ``FileNotFoundError``) explaining every place
that was checked and how to supply the file.

Downloads land in the standard Hugging Face cache (``$HF_HOME`` /
``~/.cache/huggingface``), never inside this repository.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "AssetNotFoundError",
    "HF_REPO_ID",
    "COREML_FP16_PACKAGE",
    "COREML_FP32_PACKAGE",
    "COREML_PACKAGE_ENV_VAR",
    "OFFLINE_ENV_VAR",
    "WEIGHTS_ENV_VAR",
    "WEIGHTS_FILENAME",
    "is_offline",
    "resolve_coreml_package",
    "resolve_weights",
]

_LOGGER = logging.getLogger(__name__)

#: Repo root, i.e. the directory containing ``runs/``.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Hugging Face repo hosting the published assets.
HF_REPO_ID = "uniuyuni/nagi_denoise"

#: Human-facing URL, quoted in error messages for the manual-download route.
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_ID}"

#: Production checkpoint (NagiV2-L, "Phase 2B"): filename on the Hub, which is
#: also its basename in ``runs/nagi_v2_l_ft2/``.
WEIGHTS_FILENAME = "nagi_v2_l_ft2_final.pt"

#: In-repo location of the production checkpoint (step 3 for the torch path).
REPO_WEIGHTS = REPO_ROOT / "runs" / "nagi_v2_l_ft2" / WEIGHTS_FILENAME

#: Exported Core ML packages. These are *directories* (``.mlpackage``), so on
#: the Hub they are stored as a folder of files under this prefix.
COREML_FP16_PACKAGE = "nagi_v2_l_ft2_t768_fp16.mlpackage"
COREML_FP32_PACKAGE = "nagi_v2_l_ft2_t768_fp32.mlpackage"

#: In-repo directory holding the exported packages (step 3 for the Core ML path).
REPO_COREML_DIR = REPO_ROOT / "runs" / "phase5_speed" / "coreml"

#: Overrides the checkpoint path globally (step 2 for the torch path).
WEIGHTS_ENV_VAR = "NAGI_DENOISE_WEIGHTS"

#: Overrides the Core ML package path globally (step 2 for the Core ML path).
COREML_PACKAGE_ENV_VAR = "NAGI_DENOISE_COREML_PACKAGE"

#: Set to 1/true/yes/on to forbid Hub downloads process-wide.
OFFLINE_ENV_VAR = "NAGI_DENOISE_OFFLINE"

_TRUTHY = {"1", "true", "yes", "on"}


class AssetNotFoundError(FileNotFoundError):
    """An asset could not be resolved locally and could not be downloaded."""


def is_offline() -> bool:
    """True if ``$NAGI_DENOISE_OFFLINE`` forbids Hub downloads."""
    return os.environ.get(OFFLINE_ENV_VAR, "").strip().lower() in _TRUTHY


def _import_hf_hub(what: str):
    """Import ``huggingface_hub``, or raise an actionable error.

    Only ever called from steps 4/5, so a machine that resolves its assets
    locally never needs the package installed.
    """
    try:
        import huggingface_hub  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:
        raise AssetNotFoundError(
            f"{what} was not found locally, and the optional 'huggingface_hub' "
            "dependency is not installed, so it cannot be fetched from the Hub.\n"
            "Either install it:\n"
            "    pixi add --pypi huggingface_hub      (or: pip install huggingface_hub)\n"
            f"or download the file manually from {HF_REPO_URL} and point at it "
            f"with an explicit path argument or ${WEIGHTS_ENV_VAR} / "
            f"${COREML_PACKAGE_ENV_VAR}."
        ) from exc
    return huggingface_hub


def _missing_error(
    what: str,
    hub_target: str,
    repo_path: Path,
    env_var: str,
    reason: str,
) -> AssetNotFoundError:
    return AssetNotFoundError(
        f"{what} could not be resolved. Checked, in order:\n"
        f"  1. explicit path argument: not given\n"
        f"  2. ${env_var}: not set\n"
        f"  3. in-repo path: {repo_path} does not exist\n"
        f"  4. Hugging Face cache: {reason}\n"
        f"  5. download {hub_target!r} from {HF_REPO_ID}: refused, downloads are "
        f"disabled (allow_download=False or ${OFFLINE_ENV_VAR}=1)\n"
        f"Fix by any one of: downloading it manually from {HF_REPO_URL} and "
        f"setting ${env_var} to it; placing it at {repo_path}; or re-running "
        "with downloads allowed."
    )


def _resolve_asset(
    explicit: Union[str, Path, None],
    env_var: str,
    repo_path: Path,
    hub_target: str,
    what: str,
    allow_download: bool,
    is_dir: bool,
) -> Path:
    """The shared five-step chain. See the module docstring for the order."""
    # 1. Explicit caller path. Wins outright; never falls back to a download.
    if explicit is not None:
        return Path(explicit)

    # 2. Environment override. Same contract as (1): honoured verbatim.
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value)

    # 3. In-repo artifact (developer / trainer case). No network, ever.
    if repo_path.exists():
        return repo_path

    offline = (not allow_download) or is_offline()

    # 4. Already-populated HF cache. `local_files_only=True` guarantees this
    #    step cannot make a network round-trip, so a user who downloaded the
    #    asset once never pays for it again -- and it is consulted even in
    #    offline mode, because reading a local cache is not a fetch.
    try:
        hf = _import_hf_hub(what)
    except AssetNotFoundError:
        if not offline:
            # Downloads were permitted, so the missing optional dependency is
            # the real blocker: report it with its install instructions.
            raise
        hf = None
        cache_miss_reason = "not checked ('huggingface_hub' is not installed)"

    if hf is not None:
        try:
            cached = _hub_fetch(hf, hub_target, is_dir, local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - hub raises several distinct types
            cache_miss_reason = f"miss ({type(exc).__name__})"
        else:
            _LOGGER.debug("%s resolved from the Hugging Face cache: %s", what, cached)
            return cached

    if offline:
        raise _missing_error(what, hub_target, repo_path, env_var, cache_miss_reason)

    # 5. Download. Logged loudly: this is the only step that touches the network.
    _LOGGER.info(
        "downloading %s from the Hugging Face Hub: repo=%s file=%s (this happens "
        "only because it was not found locally)",
        what, HF_REPO_ID, hub_target,
    )
    path = _hub_fetch(hf, hub_target, is_dir, local_files_only=False)
    _LOGGER.info("downloaded %s -> %s", hub_target, path)
    return path


def _hub_fetch(hf, hub_target: str, is_dir: bool, local_files_only: bool) -> Path:
    """One file (``hf_hub_download``) or one folder (``snapshot_download``).

    ``.mlpackage`` assets are directories, so they live on the Hub as a folder
    of files and have to be pulled with a filtered snapshot. Both routes use
    the standard HF cache -- nothing is written into this repository.
    """
    if is_dir:
        root = hf.snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=[f"{hub_target}/*", f"{hub_target}/**"],
            local_files_only=local_files_only,
        )
        path = Path(root) / hub_target
        if not path.exists():
            raise FileNotFoundError(f"{hub_target} not present in {root}")
        return path
    return Path(
        hf.hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hub_target,
            local_files_only=local_files_only,
        )
    )


def resolve_weights(
    weights: Union[str, Path, None] = None,
    *,
    allow_download: bool = True,
) -> Path:
    """Resolve the NagiV2-L PyTorch checkpoint.

    Args:
        weights: an explicit path. Wins over everything else and is returned
            unchanged -- naming a path never triggers a download.
        allow_download: ``False`` forbids step 5 (the Hub fetch) entirely.
            ``$NAGI_DENOISE_OFFLINE=1`` does the same globally.

    Returns:
        A path to ``nagi_v2_l_ft2_final.pt``. See the module docstring for the
        full resolution order.

    Raises:
        AssetNotFoundError: the checkpoint is not present anywhere and either
            downloads are forbidden or ``huggingface_hub`` is not installed.
    """
    return _resolve_asset(
        explicit=weights,
        env_var=WEIGHTS_ENV_VAR,
        repo_path=REPO_WEIGHTS,
        hub_target=WEIGHTS_FILENAME,
        what="the NagiV2-L production checkpoint",
        allow_download=allow_download,
        is_dir=False,
    )


def resolve_coreml_package(
    package: Union[str, Path, None] = None,
    *,
    allow_download: bool = True,
    name: Optional[str] = None,
) -> Path:
    """Resolve an exported Core ML ``.mlpackage``.

    Args:
        package: an explicit path. Wins over everything else, as above.
        allow_download: ``False`` forbids the Hub fetch.
        name: which package to resolve when none is given explicitly --
            :data:`COREML_FP16_PACKAGE` (default, the production package) or
            :data:`COREML_FP32_PACKAGE`.

    Returns:
        A path to the ``.mlpackage`` directory.
    """
    target = name or COREML_FP16_PACKAGE
    return _resolve_asset(
        explicit=package,
        env_var=COREML_PACKAGE_ENV_VAR,
        repo_path=REPO_COREML_DIR / target,
        hub_target=target,
        what=f"the Core ML package {target}",
        allow_download=allow_download,
        is_dir=True,
    )
