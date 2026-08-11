"""Tests for asset resolution: nagi_denoise.assets.

Run with: python tests/test_assets.py  (invoked by `pixi run test`)

The point of these tests is the *order*, and above all the negative
guarantee: a user who already has the asset -- because they trained it,
cloned it, set an env var, or downloaded it once -- must never trigger a
network fetch. Every test here installs a fake ``huggingface_hub`` that
records its calls (or a meta-path hook that makes importing it fail
outright), so any accidental Hub access is an assertion failure rather than
a silent 236MB download.
"""
from __future__ import annotations

import importlib.abc
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nagi_denoise import assets


# --------------------------------------------------------------------------
# Harnesses
# --------------------------------------------------------------------------
class _BlockImport(importlib.abc.MetaPathFinder):
    """Make ``import huggingface_hub`` fail, simulating a machine without it."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "huggingface_hub" or fullname.startswith("huggingface_hub."):
            raise ImportError("huggingface_hub is not installed (simulated)")
        return None


class _hub_blocked:
    """Context manager: huggingface_hub is not importable inside the block."""

    def __enter__(self):
        self._finder = _BlockImport()
        self._saved = {k: v for k, v in sys.modules.items() if k.startswith("huggingface_hub")}
        for k in self._saved:
            del sys.modules[k]
        sys.meta_path.insert(0, self._finder)
        return self

    def __exit__(self, *exc):
        sys.meta_path.remove(self._finder)
        sys.modules.update(self._saved)
        return False


class _FakeHub:
    """Stand-in for huggingface_hub that records calls instead of making them.

    ``cached`` lists targets that a ``local_files_only=True`` lookup should
    succeed for; anything else raises, as the real client does on a cache
    miss.
    """

    def __init__(self, cached=(), root=None):
        self.cached = set(cached)
        self.root = Path(root or tempfile.mkdtemp(prefix="nagi_fakehub_"))
        self.calls: list[tuple[str, str, bool]] = []

    # -- the two functions assets._hub_fetch uses --------------------------
    def hf_hub_download(self, repo_id, filename, local_files_only=False, **_):
        self.calls.append(("file", filename, local_files_only))
        if local_files_only and filename not in self.cached:
            raise FileNotFoundError("not in cache (simulated)")
        p = self.root / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    def snapshot_download(self, repo_id, allow_patterns=None, local_files_only=False, **_):
        target = str(allow_patterns[0]).split("/")[0] if allow_patterns else ""
        self.calls.append(("dir", target, local_files_only))
        if local_files_only and target not in self.cached:
            raise FileNotFoundError("not in cache (simulated)")
        (self.root / target).mkdir(parents=True, exist_ok=True)
        return str(self.root)

    @property
    def network_calls(self):
        """Calls that would have gone over the wire (local_files_only=False)."""
        return [c for c in self.calls if not c[2]]


class _with_hub:
    """Context manager: assets._import_hf_hub returns the given fake."""

    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        self._orig = assets._import_hf_hub
        assets._import_hf_hub = lambda what: self.fake
        return self.fake

    def __exit__(self, *exc):
        assets._import_hf_hub = self._orig
        return False


class _env:
    """Context manager: set/unset environment variables, restoring on exit."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self.kw}
        for k, v in self.kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _clean_env(**kw):
    """The env vars this module cares about, all cleared unless overridden."""
    base = {
        assets.WEIGHTS_ENV_VAR: None,
        assets.COREML_PACKAGE_ENV_VAR: None,
        assets.OFFLINE_ENV_VAR: None,
    }
    base.update(kw)
    return _env(**base)


_MISSING = "definitely_not_a_real_asset.pt"


def _missing_kwargs(**over):
    kw = dict(
        explicit=None,
        env_var="NAGI_DENOISE_TEST_NO_SUCH_VAR",
        repo_path=assets.REPO_ROOT / "runs" / _MISSING,
        hub_target=_MISSING,
        what="a deliberately missing test asset",
        allow_download=True,
        is_dir=False,
    )
    kw.update(over)
    return kw


# --------------------------------------------------------------------------
# 1. Importing the package must not need huggingface_hub
# --------------------------------------------------------------------------
def test_import_works_without_huggingface_hub():
    """`import nagi_denoise` must work on a machine without the optional dep.

    Run in a subprocess with a meta-path hook that makes every
    ``huggingface_hub`` import raise, which is the honest simulation of "not
    installed" -- and stronger than checking sys.modules in-process, where
    another dependency may already have imported it.
    """
    import subprocess
    import textwrap

    script = textwrap.dedent(
        f"""
        import sys, importlib.abc
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == 'huggingface_hub':
                    raise ImportError('huggingface_hub is not installed (simulated)')
                return None
        sys.meta_path.insert(0, Block())
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        import nagi_denoise
        from nagi_denoise import denoise, assets, resolve_weights
        assert 'huggingface_hub' not in sys.modules, 'the hub import must stay lazy'
        print('ok')
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"import failed without huggingface_hub:\n{proc.stderr}"
    assert "ok" in proc.stdout


# --------------------------------------------------------------------------
# 2. Step 1: an explicit path wins, and never falls back
# --------------------------------------------------------------------------
def test_explicit_path_wins_over_everything():
    fake = _FakeHub()
    with _clean_env(**{assets.WEIGHTS_ENV_VAR: "/env/should/be/ignored.pt"}), _with_hub(fake):
        got = assets.resolve_weights("/caller/chosen.pt")
    assert got == Path("/caller/chosen.pt"), got
    assert fake.calls == [], "an explicit path must not consult the Hub at all"


def test_explicit_missing_path_does_not_fall_back_to_download():
    """The central constraint, stated negatively: naming a path that does not
    exist is an error for the caller to fix, never a licence to fetch 236MB."""
    fake = _FakeHub()
    with _clean_env(), _with_hub(fake):
        got = assets.resolve_weights("/no/such/file.pt")
    assert got == Path("/no/such/file.pt")
    assert fake.calls == [], "a missing explicit path must not trigger a download"


def test_explicit_coreml_package_wins():
    fake = _FakeHub()
    with _clean_env(), _with_hub(fake):
        got = assets.resolve_coreml_package("/caller/pkg.mlpackage")
    assert got == Path("/caller/pkg.mlpackage")
    assert fake.calls == []


# --------------------------------------------------------------------------
# 3. Step 2: the environment variables
# --------------------------------------------------------------------------
def test_env_var_beats_the_in_repo_copy():
    fake = _FakeHub()
    with _clean_env(**{assets.WEIGHTS_ENV_VAR: "/env/weights.pt"}), _with_hub(fake):
        assert assets.resolve_weights() == Path("/env/weights.pt")
    assert fake.calls == []


def test_coreml_env_var_beats_the_in_repo_copy():
    fake = _FakeHub()
    with _clean_env(**{assets.COREML_PACKAGE_ENV_VAR: "/env/p.mlpackage"}), _with_hub(fake):
        assert assets.resolve_coreml_package() == Path("/env/p.mlpackage")
    assert fake.calls == []


# --------------------------------------------------------------------------
# 4. Step 3: the in-repo path, resolved with the Hub entirely unavailable
# --------------------------------------------------------------------------
def test_in_repo_path_used_without_the_hub():
    """The developer / trainer case. If ``runs/`` holds the asset, resolution
    must succeed even when huggingface_hub cannot be imported at all."""
    if not assets.REPO_WEIGHTS.exists():
        print("  (skipped: in-repo checkpoint not present)")
        return
    with _clean_env(), _hub_blocked():
        got = assets.resolve_weights()
    assert got == assets.REPO_WEIGHTS, got


def test_in_repo_coreml_package_used_without_the_hub():
    pkg = assets.REPO_COREML_DIR / assets.COREML_FP16_PACKAGE
    if not pkg.exists():
        print("  (skipped: in-repo .mlpackage not present)")
        return
    with _clean_env(), _hub_blocked():
        got = assets.resolve_coreml_package()
    assert got == pkg, got


def test_in_repo_path_is_preferred_over_a_populated_cache():
    """Even when the asset is sitting in the HF cache, the in-repo copy wins,
    so resolution short-circuits before huggingface_hub is touched at all."""
    if not assets.REPO_WEIGHTS.exists():
        print("  (skipped: in-repo checkpoint not present)")
        return
    fake = _FakeHub(cached=[assets.WEIGHTS_FILENAME])
    with _clean_env(), _with_hub(fake):
        assert assets.resolve_weights() == assets.REPO_WEIGHTS
    assert fake.calls == []


# --------------------------------------------------------------------------
# 5. Step 4: a populated cache must not cause a network round-trip
# --------------------------------------------------------------------------
def test_populated_cache_is_used_without_a_network_call():
    fake = _FakeHub(cached=[_MISSING])
    with _clean_env(), _with_hub(fake):
        got = assets._resolve_asset(**_missing_kwargs())
    assert got.name == _MISSING, got
    assert fake.calls == [("file", _MISSING, True)], fake.calls
    assert fake.network_calls == [], "a cached asset must not hit the network"


def test_cache_is_consulted_before_downloading():
    """Cache miss then download: the offline probe must come first, and the
    real fetch only after it fails."""
    fake = _FakeHub(cached=[])
    with _clean_env(), _with_hub(fake):
        got = assets._resolve_asset(**_missing_kwargs())
    assert got.name == _MISSING
    assert [c[2] for c in fake.calls] == [True, False], fake.calls


# --------------------------------------------------------------------------
# 6. Offline mode forbids the download outright
# --------------------------------------------------------------------------
def test_offline_flag_raises_instead_of_fetching():
    fake = _FakeHub(cached=[])
    with _clean_env(), _with_hub(fake):
        try:
            assets._resolve_asset(**_missing_kwargs(allow_download=False))
        except assets.AssetNotFoundError as exc:
            msg = str(exc)
        else:
            raise AssertionError("expected AssetNotFoundError in offline mode")
    assert fake.network_calls == [], "offline mode must make no network call"
    assert assets.HF_REPO_URL in msg, "the error must say where to get it manually"
    assert str(assets.REPO_ROOT / "runs" / _MISSING) in msg, "and where to put it"


def test_offline_env_var_raises_instead_of_fetching():
    fake = _FakeHub(cached=[])
    with _clean_env(**{assets.OFFLINE_ENV_VAR: "1"}), _with_hub(fake):
        assert assets.is_offline()
        try:
            assets._resolve_asset(**_missing_kwargs(allow_download=True))
        except assets.AssetNotFoundError:
            pass
        else:
            raise AssertionError(f"expected AssetNotFoundError with ${assets.OFFLINE_ENV_VAR}=1")
    assert fake.network_calls == []


def test_offline_still_uses_a_populated_cache():
    """Reading a local cache is not a fetch, so offline mode must still allow
    it -- otherwise a user who downloaded once could not work offline."""
    fake = _FakeHub(cached=[_MISSING])
    with _clean_env(**{assets.OFFLINE_ENV_VAR: "1"}), _with_hub(fake):
        got = assets._resolve_asset(**_missing_kwargs(allow_download=True))
    assert got.name == _MISSING
    assert fake.network_calls == []


def test_offline_env_var_accepts_the_usual_spellings():
    for value, expected in [("1", True), ("true", True), ("YES", True), ("on", True),
                            ("0", False), ("false", False), ("", False)]:
        with _env(**{assets.OFFLINE_ENV_VAR: value}):
            assert assets.is_offline() is expected, value


# --------------------------------------------------------------------------
# 7. huggingface_hub is optional: its absence must be an actionable error
# --------------------------------------------------------------------------
def test_missing_hub_gives_actionable_error():
    with _clean_env(), _hub_blocked():
        try:
            assets._resolve_asset(**_missing_kwargs())
        except assets.AssetNotFoundError as exc:
            msg = str(exc)
        else:
            raise AssertionError("expected AssetNotFoundError without huggingface_hub")
    assert "huggingface_hub" in msg
    assert "pip install huggingface_hub" in msg, "must name the install command"
    assert assets.HF_REPO_URL in msg, "must name the manual-download alternative"


# --------------------------------------------------------------------------
# 8. The download step is logged, never silent
# --------------------------------------------------------------------------
def test_download_is_logged_at_info():
    import logging

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    logger = logging.getLogger("nagi_denoise.assets")
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        fake = _FakeHub(cached=[])
        with _clean_env(), _with_hub(fake):
            assets._resolve_asset(**_missing_kwargs())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    infos = [r.getMessage() for r in records if r.levelno >= logging.INFO]
    assert any(assets.HF_REPO_ID in m and _MISSING in m for m in infos), infos
    assert any("downloaded" in m for m in infos), "the destination must be logged too"


# --------------------------------------------------------------------------
# 9. The pipeline is actually wired to the resolver
# --------------------------------------------------------------------------
def test_pipeline_resolve_weights_delegates_to_assets():
    from nagi_denoise.pipeline import denoise as denoise_mod

    fake = _FakeHub()
    with _clean_env(), _with_hub(fake):
        assert denoise_mod._resolve_weights("/caller/x.pt") == str(Path("/caller/x.pt"))
        if assets.REPO_WEIGHTS.exists():
            assert denoise_mod._resolve_weights(None) == str(assets.REPO_WEIGHTS)
    assert fake.calls == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print("all tests passed")
