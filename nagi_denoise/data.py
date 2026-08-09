"""SIDD Medium sRGB dataset loader for Nagi NR.

Expected layout (as downloaded from the SIDD project page):

    <root>/
        Data/
            0001_001_S6_00100_00060_3200_L/
                0001_NOISY_SRGB_010.PNG
                0001_GT_SRGB_010.PNG
                ...
            0002_.../
            ...

Either the parent directory or `<root>/Data` itself can be passed as `root`.

I/O strategy: SIDD images are ~5328x3000 PNG (~25MB each). Decoding one PNG
costs ~0.5-1s, so naive random patch sampling makes training I/O-bound.

This module ships:
  * `SIDDPatchDataset` — caches the last-decoded pair per instance. Random
    crops are drawn per access (`randomize_each_access=True` in training), so
    every iteration sees fresh crops — never a fixed crop list (NagiQ lesson).
  * `ChunkedShuffleSampler` — yields indices grouped by image so the cache hits.
  * `PolyUPatchDataset` — same treatment for PolyU CroppedImages real pairs.
  * `apply_poisson_gaussian_linear` — physically-based heteroscedastic
    Poisson-Gaussian noise in linear light (ISO-ladder a/b, per-image spatial
    correlation via `correlated_standard_normal`, optional chroma-correlated
    component).
  * `MixturePatchDataset` — infinite IterableDataset mixing {SIDD real,
    PolyU real, synthetic} pairs with configurable weights; also carries the
    precomputed NAFNet teacher output for SIDD samples (distillation).
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler
from PIL import Image

from .transforms import srgb_to_linear


LUMA_SRGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)
LUMA_LINEAR = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _linear_to_srgb_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None).astype(np.float32, copy=False)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _luma_np(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(rgb[..., :3].astype(np.float32, copy=False) * weights.reshape(1, 1, 3), axis=2)


def _sigmoid01_np(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False)


def _smoothstep_np(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def _read_exr_rgb(path: Path) -> np.ndarray:
    import OpenEXR

    file = OpenEXR.InputFile(str(path))
    header = file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels = header["channels"]
    names = ["R", "G", "B"] if all(c in channels for c in ("R", "G", "B")) else list(channels)[:3]
    arrs = [np.frombuffer(file.channel(c), dtype=np.float32).reshape(height, width) for c in names]
    return np.stack(arrs, axis=2).astype(np.float32, copy=False)


def _read_float_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".exr":
        arr = _read_exr_rgb(path)
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.max(initial=0.0) > 4.0:
            arr = arr / np.float32(np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 65535.0)
    else:
        with Image.open(path) as im:
            srgb = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
        return srgb_to_linear(torch.from_numpy(srgb).permute(2, 0, 1)).permute(1, 2, 0).numpy()
    return np.nan_to_num(arr[..., :3], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


# ---- Synthetic degradations (sRGB uint8 space) ----
def _jpeg_recompress(img_u8: np.ndarray, q: int) -> np.ndarray:
    """Encode then decode as JPEG to inject compression artifacts."""
    buf = BytesIO()
    Image.fromarray(img_u8).save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    with Image.open(buf) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def _apply_synth_degradation(
    rng: np.random.Generator,
    clean_u8: np.ndarray,
    gauss_sigma_max: float = 30.0,
    poisson_lambda_range: Tuple[float, float] = (10.0, 100.0),
    jpeg_q_range: Tuple[int, int] = (60, 100),
) -> np.ndarray:
    """Pick one of {gauss, poisson, gauss+jpeg, jpeg} and apply it.

    All operations are in sRGB uint8 space (matches how real noise enters
    the image pipeline). Caller converts the result to linear afterwards.
    """
    mode = rng.choice(["gauss", "poisson", "gauss_jpeg", "jpeg"])
    img = clean_u8.astype(np.float32)

    if mode == "gauss":
        sigma = float(rng.uniform(0.0, gauss_sigma_max))
        img = img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)

    elif mode == "poisson":
        lam = float(rng.uniform(*poisson_lambda_range))
        # Map [0,255] -> [0,lam], sample Poisson, map back.
        scale = lam / 255.0
        signal = np.clip(img * scale, 0.0, None)
        img = rng.poisson(signal).astype(np.float32) / max(scale, 1e-8)

    elif mode == "gauss_jpeg":
        sigma = float(rng.uniform(5.0, gauss_sigma_max))
        img = img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
        img_u8 = np.clip(img, 0.0, 255.0).astype(np.uint8)
        q = int(rng.integers(jpeg_q_range[0], jpeg_q_range[1] + 1))
        return _jpeg_recompress(img_u8, q=q)

    else:  # "jpeg"
        q = int(rng.integers(jpeg_q_range[0], jpeg_q_range[1] + 1))
        return _jpeg_recompress(clean_u8, q=q)

    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def find_sidd_pairs(root: str | Path) -> List[Tuple[str, str]]:
    """Discover (noisy_path, gt_path) pairs under a SIDD root."""
    root = Path(root)
    candidates = [root / "Data", root]
    data_dir = next((p for p in candidates if p.is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError(f"SIDD root not found: {root}")

    pairs: List[Tuple[str, str]] = []
    for scene in sorted(data_dir.iterdir()):
        if not scene.is_dir():
            continue
        for noisy in sorted(scene.glob("*NOISY*SRGB*.PNG")):
            gt = scene / noisy.name.replace("NOISY", "GT")
            if gt.exists():
                pairs.append((str(noisy), str(gt)))
    return pairs


def find_polyu_pairs(root: str | Path, split: str = "CroppedImages") -> List[Tuple[str, str]]:
    """Discover PolyU real/mean JPEG pairs.

    Expected layout from PolyU-Real-World-Noisy-Images-Dataset:

        <root>/CroppedImages/*_real.JPG
        <root>/CroppedImages/*_mean.JPG

    The *_real image is noisy and *_mean is the averaged clean target.
    """
    root = Path(root)
    data_dir = root / split
    if not data_dir.is_dir():
        data_dir = root
    pairs: List[Tuple[str, str]] = []
    for noisy in sorted(data_dir.glob("*_real.JPG")):
        gt = noisy.with_name(noisy.name.replace("_real.JPG", "_mean.JPG"))
        if gt.exists():
            pairs.append((str(noisy), str(gt)))
    if not pairs:
        for noisy in sorted(data_dir.glob("*_Real.JPG")):
            gt = noisy.with_name(noisy.name.replace("_Real.JPG", "_mean.JPG"))
            if gt.exists():
                pairs.append((str(noisy), str(gt)))
    return pairs


class SIDDPatchDataset(Dataset):
    """Random patch sampler over SIDD Medium sRGB pairs.

    Each __getitem__ returns one random crop. The reported length is
    `len(pairs) * patches_per_image` so DataLoader shuffling spreads work evenly.
    A single-slot pair cache means consecutive accesses to the same pair_idx
    avoid re-decoding the (huge) source PNGs — use `ChunkedShuffleSampler`
    to ensure adjacent indices share a pair.

    Output: (noisy_linear, clean_linear), both (3, H, W) float32 in linear light.
    With exposure_jitter, values can exceed 1.0 to expose the network to HDR ranges.
    """

    def __init__(
        self,
        root: str | Path,
        patch_size: int = 128,
        patches_per_image: int = 8,
        exposure_jitter: Optional[Tuple[float, float]] = (0.25, 4.0),
        flip_rot: bool = True,
        seed: int = 0,
        pairs: Optional[Iterable[Tuple[str, str]]] = None,
        synth_prob: float = 0.0,
        gauss_sigma_max: float = 30.0,
        poisson_lambda_range: Tuple[float, float] = (10.0, 100.0),
        jpeg_q_range: Tuple[int, int] = (60, 100),
        return_teacher: bool = False,
        output_space: str = "linear",
        randomize_each_access: bool = False,
    ):
        super().__init__()
        self.pairs = list(pairs) if pairs is not None else find_sidd_pairs(root)
        if not self.pairs:
            raise FileNotFoundError(f"No SIDD pairs discovered under {root}")
        if patch_size % 8 != 0:
            raise ValueError("patch_size must be a multiple of 8")

        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.exposure_jitter = exposure_jitter
        self.flip_rot = bool(flip_rot)
        self._base_seed = int(seed)
        # Synthetic-degradation knobs (Phase 2).
        self.synth_prob = float(synth_prob)
        self.gauss_sigma_max = float(gauss_sigma_max)
        self.poisson_lambda_range = (
            float(poisson_lambda_range[0]),
            float(poisson_lambda_range[1]),
        )
        self.jpeg_q_range = (int(jpeg_q_range[0]), int(jpeg_q_range[1]))
        self.return_teacher = bool(return_teacher)
        if output_space not in ("linear", "srgb"):
            raise ValueError(f"output_space must be 'linear' or 'srgb', got {output_space!r}")
        self.output_space = output_space
        self.randomize_each_access = bool(randomize_each_access)
        self._access_counter = 0
        # Single-slot pair cache (per dataset instance; one per worker process).
        self._cache_pair_idx: int = -1
        self._cache_noisy: Optional[np.ndarray] = None
        self._cache_gt: Optional[np.ndarray] = None
        self._cache_teacher: Optional[np.ndarray] = None  # None if no teacher file

    @staticmethod
    def _teacher_path_for(noisy_path: str) -> Path:
        p = Path(noisy_path)
        if "NOISY" not in p.name:
            return p.with_name("__missing_teacher__")
        return p.with_name(p.name.replace("NOISY", "TEACHER"))

    # ---- Length & access ----
    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def _load(self, path: str) -> np.ndarray:
        with Image.open(path) as im:
            # np.array (not asarray) yields an owned, writable buffer.
            arr = np.array(im.convert("RGB"), dtype=np.uint8)
        return arr  # HWC uint8

    def _get_pair(self, pair_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache_pair_idx != pair_idx:
            noisy_path, gt_path = self.pairs[pair_idx]
            self._cache_noisy = self._load(noisy_path)
            self._cache_gt = self._load(gt_path)
            self._cache_teacher = None
            if self.return_teacher:
                tpath = self._teacher_path_for(noisy_path)
                if tpath.exists():
                    try:
                        self._cache_teacher = self._load(str(tpath))
                    except (OSError, Exception):
                        # File may be partially written by precompute_teacher.py;
                        # treat as missing and fall back to GT-only supervision.
                        self._cache_teacher = None
            self._cache_pair_idx = pair_idx
        return self._cache_noisy, self._cache_gt  # type: ignore[return-value]

    def __getitem__(self, idx: int):
        # Per-call RNG so workers stay deterministic-ish across epochs.
        extra = 0
        if self.randomize_each_access:
            self._access_counter += 1
            extra = self._access_counter * 1597334677
        rng = np.random.default_rng((self._base_seed + idx * 2654435761 + extra) & 0xFFFFFFFF)

        pair_idx = idx // self.patches_per_image
        noisy, gt = self._get_pair(pair_idx)

        H, W, _ = noisy.shape
        ps = self.patch_size
        if H < ps or W < ps:
            raise RuntimeError(f"Image too small ({H}x{W}) for patch {ps}")

        y = int(rng.integers(0, H - ps + 1))
        x = int(rng.integers(0, W - ps + 1))
        noisy_p = noisy[y : y + ps, x : x + ps]
        gt_p = gt[y : y + ps, x : x + ps]

        # Phase 2: with prob synth_prob, replace the real noisy patch with a
        # synthetic degradation of the GT patch. The GT (target) is unchanged.
        # This exposes the model to clean-end + non-SIDD noise distributions.
        is_synth = self.synth_prob > 0.0 and rng.random() < self.synth_prob
        if is_synth:
            noisy_p = _apply_synth_degradation(
                rng,
                gt_p,
                gauss_sigma_max=self.gauss_sigma_max,
                poisson_lambda_range=self.poisson_lambda_range,
                jpeg_q_range=self.jpeg_q_range,
            )

        # Teacher (Phase 3 distillation): only valid for REAL noisy patches that
        # have a precomputed teacher image. Synthetic patches are out-of-domain
        # for the teacher, so they fall back to GT-only supervision.
        has_teacher = (
            self.return_teacher and not is_synth and self._cache_teacher is not None
        )
        if self.return_teacher:
            if has_teacher:
                teacher_p = self._cache_teacher[y : y + ps, x : x + ps]
            else:
                teacher_p = gt_p  # placeholder; distill weight will be 0

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div_(255.0)
            if self.output_space == "srgb":
                return t
            return srgb_to_linear(t)

        noisy_t = _to_tensor(noisy_p)
        gt_t = _to_tensor(gt_p)
        views = [noisy_t, gt_t]
        if self.return_teacher:
            teacher_t = _to_tensor(teacher_p)
            views.append(teacher_t)

        # Same exposure jitter to all views (preserves noise/signal ratio).
        if self.exposure_jitter is not None:
            lo, hi = self.exposure_jitter
            # Log-uniform sampling for symmetric brightness coverage.
            scale = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            for v in views:
                v.mul_(scale)

        if self.flip_rot:
            fx = rng.random() < 0.5
            fy = rng.random() < 0.5
            k = int(rng.integers(0, 4))
            for i, v in enumerate(views):
                if fx:
                    v = v.flip(-1)
                if fy:
                    v = v.flip(-2)
                if k:
                    v = torch.rot90(v, k, dims=(-2, -1))
                views[i] = v.contiguous()

        if self.return_teacher:
            return views[0], views[1], views[2], float(has_teacher)
        return views[0], views[1]


class WeakTeacherPatchDataset(Dataset):
    """Random patches from real photos with weak pseudo-teacher supervision.

    The pseudo teacher is not treated as ground truth everywhere. Each sample
    includes a confidence mask that selects flat, non-edge, non-highlight regions
    where teacher-like denoising is least likely to damage real detail.
    """

    def __init__(
        self,
        pairs: Iterable[Tuple[str, str]],
        patch_size: int = 128,
        patches_per_image: int = 32,
        exposure_jitter: Optional[Tuple[float, float]] = None,
        flip_rot: bool = True,
        seed: int = 0,
        randomize_each_access: bool = True,
        structure_sigma: float = 1.2,
        detail_sigma: float = 2.8,
        detail_threshold: float = 0.018,
        detail_transition: float = 0.010,
        edge_sigma: float = 1.0,
        edge_threshold: float = 0.030,
        edge_transition: float = 0.015,
        highlight_threshold: float = 1.0,
        highlight_transition: float = 0.25,
        delta_threshold: float = 0.002,
        delta_transition: float = 0.010,
        min_mask_mean: float = 0.02,
    ):
        super().__init__()
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("WeakTeacherPatchDataset requires at least one pair")
        if patch_size % 8 != 0:
            raise ValueError("patch_size must be a multiple of 8")
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.exposure_jitter = exposure_jitter
        self.flip_rot = bool(flip_rot)
        self._base_seed = int(seed)
        self.randomize_each_access = bool(randomize_each_access)
        self._access_counter = 0
        self.structure_sigma = float(structure_sigma)
        self.detail_sigma = float(detail_sigma)
        self.detail_threshold = float(detail_threshold)
        self.detail_transition = float(detail_transition)
        self.edge_sigma = float(edge_sigma)
        self.edge_threshold = float(edge_threshold)
        self.edge_transition = float(edge_transition)
        self.highlight_threshold = float(highlight_threshold)
        self.highlight_transition = float(highlight_transition)
        self.delta_threshold = float(delta_threshold)
        self.delta_transition = float(delta_transition)
        self.min_mask_mean = float(min_mask_mean)
        self._cache_pair_idx: int = -1
        self._cache_noisy: Optional[np.ndarray] = None
        self._cache_teacher: Optional[np.ndarray] = None
        self._cache_mask: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def _make_mask(self, noisy: np.ndarray, teacher: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

        guide = np.clip(_linear_to_srgb_np(teacher), 0.0, 1.0)
        guide_y = _luma_np(guide, LUMA_SRGB)
        guide_y_linear = _luma_np(np.clip(teacher, 0.0, None), LUMA_LINEAR)
        structure = gaussian_filter(guide_y, sigma=self.structure_sigma, mode="reflect")
        detail = np.abs(structure - gaussian_filter(structure, sigma=self.detail_sigma, mode="reflect"))
        edge = gaussian_gradient_magnitude(structure, sigma=self.edge_sigma, mode="reflect")
        flat = _sigmoid01_np((self.detail_threshold - detail) / max(self.detail_transition, 1.0e-6))
        non_edge = _sigmoid01_np((self.edge_threshold - edge) / max(self.edge_transition, 1.0e-6))
        highlight = _smoothstep_np((guide_y_linear - self.highlight_threshold) / max(self.highlight_transition, 1.0e-6))

        noisy_srgb = np.clip(_linear_to_srgb_np(noisy), 0.0, 1.0)
        delta = np.sqrt(np.sum((guide - noisy_srgb) ** 2, axis=2))
        # Ignore places where teacher is almost identical to input: those patches
        # teach little except identity, and can drown out the useful weak signal.
        changed = _sigmoid01_np((delta - self.delta_threshold) / max(self.delta_transition, 1.0e-6))
        mask = (flat * non_edge * (1.0 - highlight) * changed).astype(np.float32, copy=False)
        return mask[..., None]

    def _get_pair(self, pair_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cache_pair_idx != pair_idx:
            noisy_path, teacher_path = self.pairs[pair_idx]
            noisy = _read_float_rgb(noisy_path)
            teacher = _read_float_rgb(teacher_path)
            if noisy.shape[:2] != teacher.shape[:2]:
                raise ValueError(f"weak teacher shape mismatch: {noisy_path} {noisy.shape} vs {teacher_path} {teacher.shape}")
            self._cache_noisy = noisy
            self._cache_teacher = teacher
            self._cache_mask = self._make_mask(noisy, teacher)
            self._cache_pair_idx = pair_idx
        return self._cache_noisy, self._cache_teacher, self._cache_mask  # type: ignore[return-value]

    def __getitem__(self, idx: int):
        extra = 0
        if self.randomize_each_access:
            self._access_counter += 1
            extra = self._access_counter * 1597334677
        rng = np.random.default_rng((self._base_seed + idx * 2654435761 + extra) & 0xFFFFFFFF)
        pair_idx = idx // self.patches_per_image
        noisy, teacher, mask = self._get_pair(pair_idx)

        H, W, _ = noisy.shape
        ps = self.patch_size
        if H < ps or W < ps:
            raise RuntimeError(f"Weak teacher image too small ({H}x{W}) for patch {ps}")

        for _attempt in range(16):
            y = int(rng.integers(0, H - ps + 1))
            x = int(rng.integers(0, W - ps + 1))
            mask_p = mask[y : y + ps, x : x + ps]
            if float(mask_p.mean()) >= self.min_mask_mean:
                break
        noisy_p = noisy[y : y + ps, x : x + ps]
        teacher_p = teacher[y : y + ps, x : x + ps]

        def _to_tensor_linear(arr: np.ndarray) -> torch.Tensor:
            safe = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            safe = np.clip(safe, 0.0, None)
            return torch.from_numpy(np.ascontiguousarray(safe)).permute(2, 0, 1).float()

        noisy_t = _to_tensor_linear(noisy_p)
        teacher_t = _to_tensor_linear(teacher_p)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_p[..., 0])).unsqueeze(0).float()

        if self.exposure_jitter is not None:
            lo, hi = self.exposure_jitter
            scale = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            noisy_t.mul_(scale)
            teacher_t.mul_(scale)

        if self.flip_rot:
            fx = rng.random() < 0.5
            fy = rng.random() < 0.5
            k = int(rng.integers(0, 4))
            views = [noisy_t, teacher_t, mask_t]
            for i, v in enumerate(views):
                if fx:
                    v = v.flip(-1)
                if fy:
                    v = v.flip(-2)
                if k:
                    v = torch.rot90(v, k, dims=(-2, -1))
                views[i] = v.contiguous()
            noisy_t, teacher_t, mask_t = views

        return noisy_t, teacher_t, mask_t


class ChunkedShuffleSampler(Sampler[int]):
    """Yield dataset indices grouped into per-pair chunks, then shuffle chunks.

    With `chunk_size == patches_per_image`, every chunk contains all the
    patches sampled from one image, so a worker that processes a chunk hits
    its pair cache for `chunk_size - 1` of its `chunk_size` calls.

    `num_pairs` and `patches_per_image` must match the dataset.

    A new chunk shuffle is drawn each iteration. Pass a per-epoch seed via
    `set_epoch` if reproducibility across runs matters.
    """

    def __init__(
        self,
        num_pairs: int,
        patches_per_image: int,
        chunk_size: Optional[int] = None,
        seed: int = 0,
    ):
        if chunk_size is None:
            chunk_size = patches_per_image
        if patches_per_image % chunk_size != 0:
            raise ValueError("patches_per_image must be a multiple of chunk_size")

        self.num_pairs = int(num_pairs)
        self.patches_per_image = int(patches_per_image)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_pairs * self.patches_per_image

    def __iter__(self) -> Iterator[int]:
        chunks_per_pair = self.patches_per_image // self.chunk_size
        total_chunks = self.num_pairs * chunks_per_pair

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(total_chunks, generator=g).tolist()

        for c in order:
            pair_idx = c // chunks_per_pair
            chunk_start = (c % chunks_per_pair) * self.chunk_size
            base = pair_idx * self.patches_per_image + chunk_start
            for offset in range(self.chunk_size):
                yield base + offset
        self.epoch += 1


class PolyUPatchDataset(SIDDPatchDataset):
    """Random patch sampler over PolyU real-world noisy pairs.

    Files under ``<root>/CroppedImages/`` pair ``*_real.JPG`` (noisy) with
    ``*_mean.JPG`` (temporal-mean clean target). Crops/augment/sRGB->linear
    treatment is identical to :class:`SIDDPatchDataset`; only pair discovery
    differs. PolyU crops are 512x512, so keep ``patch_size <= 512``.
    """

    def __init__(self, root: str | Path, split: str = "CroppedImages", **kwargs):
        pairs = find_polyu_pairs(root, split=split)
        if not pairs:
            raise FileNotFoundError(f"No PolyU real/mean pairs discovered under {root}")
        kwargs.pop("pairs", None)
        super().__init__(root=root, pairs=pairs, **kwargs)


# ---- Physically-based synthetic noise (linear space) ----
def sample_poisson_gaussian_params(
    rng: np.random.Generator,
    a_range: Tuple[float, float],
    b_range: Tuple[float, float],
) -> Tuple[float, float]:
    """Sample per-image (a, b) log-uniformly from an ISO-ladder range.

    The heteroscedastic model is ``noisy = clean + sqrt(a*clean + b) * N(0,1)``
    with ``clean`` in linear light normalized so 1.0 = SDR white. ``a`` is the
    shot-noise (Poisson) slope, ``b`` the signal-independent (read) variance.
    """
    a = float(np.exp(rng.uniform(np.log(a_range[0]), np.log(a_range[1]))))
    b = float(np.exp(rng.uniform(np.log(b_range[0]), np.log(b_range[1]))))
    return a, b


def correlated_standard_normal(
    rng: np.random.Generator,
    shape: Tuple[int, ...],
    sigma: float,
) -> np.ndarray:
    """Draw a unit-std, spatially-correlated standard-normal field.

    ``shape`` is ``(C, H, W)`` (or ``(1, H, W)`` for a single-channel field).
    Draws i.i.d. ``N(0,1)``, then — when ``sigma > 0`` — Gaussian-blurs each
    channel independently (``scipy.ndimage.gaussian_filter``, ``mode="reflect"``)
    and renormalizes each channel back to unit std, so the caller's intended
    noise magnitude (``sqrt(a*clean+b)``) is preserved regardless of ``sigma``.

    Calibrated against real-photo demosaiced noise (lag-1 autocorrelation of
    the residual after a 3x3 median): sigma=0.5 -> ~0.005 (still white),
    sigma=0.7 -> ~0.217 (matches Fujifilm X-T5 luma, +0.235), sigma=0.9 ->
    ~0.365 (PolyU range). ``sigma <= 0`` returns plain white noise unchanged.
    """
    field = rng.standard_normal(shape).astype(np.float32)
    if sigma <= 0.0:
        return field
    from scipy.ndimage import gaussian_filter

    out = np.empty_like(field)
    for c in range(shape[0]):
        blurred = gaussian_filter(field[c], sigma=sigma, mode="reflect")
        std = float(blurred.std())
        out[c] = blurred / std if std > 1.0e-8 else blurred
    return out


def apply_poisson_gaussian_linear(
    rng: np.random.Generator,
    clean_linear: torch.Tensor,
    a: float,
    b: float,
    chroma_scale: float = 0.0,
    corr_sigma: float = 0.0,
    chroma_corr_sigma: float = 0.0,
) -> torch.Tensor:
    """Add heteroscedastic Poisson-Gaussian noise to a linear-light CHW tensor.

    ``noisy = clean + sqrt(a*clean + b) * N(0,1)``, all float32 linear. The
    driving noise field is drawn per-channel from ``correlated_standard_normal``
    with spatial correlation ``corr_sigma`` (pixels; 0 = white, matching plain
    i.i.d. Gaussian shot/read noise). This models the fact that real demosaiced
    sensor noise is NOT spatially white (see module-level calibration notes).

    Optionally adds a chroma-correlated component: a single-channel field
    (its own, typically larger, correlation ``chroma_corr_sigma`` — real
    chroma noise is correlated out to lag 3) pushed along a random zero-luma
    chroma axis, with amplitude ``chroma_scale`` relative to the local
    Poisson-Gaussian sigma. Output is clamped to >= 0 like the real
    sRGB-decoded pairs.
    """
    clean = clean_linear.clamp_min(0.0)
    sigma = torch.sqrt(a * clean + b)
    noise = torch.from_numpy(correlated_standard_normal(rng, tuple(clean.shape), corr_sigma))
    noisy = clean + sigma * noise

    if chroma_scale > 0.0:
        _, h, w = clean.shape
        field = torch.from_numpy(correlated_standard_normal(rng, (1, h, w), chroma_corr_sigma))
        axis = rng.standard_normal(3).astype(np.float32)
        axis = axis - float((axis * LUMA_LINEAR).sum())  # remove luma component
        norm = float(np.sqrt((axis * axis).sum()))
        if norm > 1.0e-6:
            axis_t = torch.from_numpy(axis / norm).view(3, 1, 1)
            noisy = noisy + sigma * float(chroma_scale) * field * axis_t

    return noisy.clamp_min(0.0)


def teacher_path_for_noisy(noisy_path: str | Path, teacher_root: Optional[str | Path] = None) -> Path:
    """Resolve the precomputed NAFNet teacher PNG for a SIDD noisy image.

    Default location (written by ``nagi_denoise.pipeline.precompute_teacher``)
    is next to the noisy file with NOISY replaced by TEACHER. With
    ``teacher_root``, the same ``<scene>/<name>`` layout is expected under
    that root instead.
    """
    p = Path(noisy_path)
    if "NOISY" not in p.name:
        return p.with_name("__missing_teacher__")
    name = p.name.replace("NOISY", "TEACHER")
    if teacher_root:
        return Path(teacher_root) / p.parent.name / name
    return p.with_name(name)


class _BurstPairSource:
    """Uniform random (image, position) sampling with burst-amortized decode.

    Every call yields a fresh random crop, but the (expensive to decode) source
    image is re-drawn only every ``burst_length`` calls. Over training this is
    uniform over images and spatial positions while keeping I/O tractable for
    5328x3000 SIDD PNGs. This replaces any notion of a fixed crop list — the
    NagiQ failure mode (training on 1280 frozen crops) is structurally
    impossible here.
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        burst_length: int = 8,
        teacher_root: Optional[str | Path] = None,
        with_teacher: bool = False,
    ):
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("empty pair list for mixture source")
        self.burst_length = max(1, int(burst_length))
        self.teacher_root = teacher_root
        self.with_teacher = bool(with_teacher)
        self._burst_left = 0
        self._noisy: Optional[np.ndarray] = None
        self._clean: Optional[np.ndarray] = None
        self._teacher: Optional[np.ndarray] = None

    @staticmethod
    def _load(path: str | Path) -> np.ndarray:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"), dtype=np.uint8)

    def _decode(self, rng: np.random.Generator) -> None:
        idx = int(rng.integers(0, len(self.pairs)))
        noisy_path, clean_path = self.pairs[idx]
        self._noisy = self._load(noisy_path)
        self._clean = self._load(clean_path)
        self._teacher = None
        if self.with_teacher:
            tpath = teacher_path_for_noisy(noisy_path, self.teacher_root)
            if tpath.exists():
                try:
                    self._teacher = self._load(tpath)
                except Exception:
                    self._teacher = None  # partially-written file; fall back to GT
        self._burst_left = self.burst_length

    def sample(
        self, rng: np.random.Generator, patch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self._burst_left <= 0 or self._noisy is None:
            self._decode(rng)
        self._burst_left -= 1
        H, W, _ = self._noisy.shape
        if H < patch_size or W < patch_size:
            raise RuntimeError(f"Image too small ({H}x{W}) for patch {patch_size}")
        y = int(rng.integers(0, H - patch_size + 1))
        x = int(rng.integers(0, W - patch_size + 1))
        sl = (slice(y, y + patch_size), slice(x, x + patch_size))
        noisy = self._noisy[sl]
        clean = self._clean[sl]
        teacher = self._teacher[sl] if self._teacher is not None else None
        return noisy, clean, teacher


class _BurstCleanSource:
    """Burst-amortized random crops from clean images (for synthetic noise)."""

    def __init__(self, paths: Sequence[str], burst_length: int = 8):
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("empty clean list for synthetic source")
        self.burst_length = max(1, int(burst_length))
        self._burst_left = 0
        self._clean: Optional[np.ndarray] = None

    def _decode(self, rng: np.random.Generator) -> None:
        idx = int(rng.integers(0, len(self.paths)))
        self._clean = _BurstPairSource._load(self.paths[idx])
        self._burst_left = self.burst_length

    def sample(self, rng: np.random.Generator, patch_size: int) -> np.ndarray:
        if self._burst_left <= 0 or self._clean is None:
            self._decode(rng)
        self._burst_left -= 1
        H, W, _ = self._clean.shape
        if H < patch_size or W < patch_size:
            raise RuntimeError(f"Image too small ({H}x{W}) for patch {patch_size}")
        y = int(rng.integers(0, H - patch_size + 1))
        x = int(rng.integers(0, W - patch_size + 1))
        return self._clean[y : y + patch_size, x : x + patch_size]


MIXTURE_SOURCE_IDS: Dict[str, int] = {"sidd": 0, "polyu": 1, "synthetic": 2}
MIXTURE_SOURCE_NAMES: Dict[int, str] = {v: k for k, v in MIXTURE_SOURCE_IDS.items()}


class MixturePatchDataset(IterableDataset):
    """Infinite stream mixing SIDD real, PolyU real, and synthetic-noise pairs.

    Each yielded sample is drawn independently: first the source is chosen by
    the configured weights, then a fresh random crop is taken (see
    ``_BurstPairSource`` for the uniformity/IO tradeoff). There is no fixed
    crop list and no epoch: every iteration sees new random crops.

    Per sample:
      * SIDD / PolyU: real (noisy, clean) crop, sRGB->linear, shared
        log-uniform exposure jitter (preserves the real noise/signal ratio).
      * synthetic: a clean crop (SIDD GT or PolyU mean), sRGB->linear,
        exposure jitter FIRST, then heteroscedastic Poisson-Gaussian noise
        ``noisy = clean + sqrt(a*clean + b) * N(0,1)`` with per-image (a, b)
        log-uniform from the configured ISO-ladder ranges.

    Yields ``(noisy, clean, teacher, has_teacher, source_id)`` where teacher is
    the precomputed NAFNet output (SIDD only, when available; otherwise the
    clean target as a placeholder with ``has_teacher = 0``).
    """

    def __init__(
        self,
        sidd_pairs: Sequence[Tuple[str, str]],
        polyu_pairs: Sequence[Tuple[str, str]],
        weights: Dict[str, float],
        patch_size: int = 256,
        exposure_jitter: Optional[Tuple[float, float]] = (0.25, 4.0),
        flip_rot: bool = True,
        seed: int = 0,
        burst_length: int = 8,
        teacher_root: Optional[str | Path] = None,
        with_teacher: bool = True,
        synth_a_range: Tuple[float, float] = (3.0e-4, 2.0e-2),
        synth_b_range: Tuple[float, float] = (1.0e-6, 1.0e-3),
        synth_chroma_prob: float = 0.5,
        synth_chroma_scale: float = 0.35,
        synth_corr_sigma_range: Tuple[float, float] = (0.0, 1.0),
        synth_chroma_corr_sigma_range: Tuple[float, float] = (0.8, 2.0),
    ):
        super().__init__()
        if patch_size % 8 != 0:
            raise ValueError("patch_size must be a multiple of 8")
        self.patch_size = int(patch_size)
        self.exposure_jitter = exposure_jitter
        self.flip_rot = bool(flip_rot)
        self.seed = int(seed)
        self.burst_length = int(burst_length)
        self.teacher_root = teacher_root
        self.with_teacher = bool(with_teacher)
        self.synth_a_range = (float(synth_a_range[0]), float(synth_a_range[1]))
        self.synth_b_range = (float(synth_b_range[0]), float(synth_b_range[1]))
        self.synth_chroma_prob = float(synth_chroma_prob)
        self.synth_chroma_scale = float(synth_chroma_scale)
        # Per-image spatial correlation (pixels) for the synthetic noise field.
        # Sampled UNIFORMLY (not log-uniform): the range legitimately includes
        # 0 (white noise, e.g. matching SIDD's near-white real noise), and
        # log-uniform is undefined at 0. Uniform sampling also matches how the
        # calibration was done (sigma swept linearly: 0.5/0.7/0.9 -> lag1
        # 0.005/0.217/0.365) so training sees a representative spread across
        # the white-to-correlated range rather than concentrating near 0.
        self.synth_corr_sigma_range = (float(synth_corr_sigma_range[0]), float(synth_corr_sigma_range[1]))
        # Chroma field gets its own (typically larger) correlation range;
        # also sampled uniformly for the same reason.
        self.synth_chroma_corr_sigma_range = (
            float(synth_chroma_corr_sigma_range[0]),
            float(synth_chroma_corr_sigma_range[1]),
        )

        weights = {k: float(v) for k, v in dict(weights).items() if float(v) > 0.0}
        unknown = set(weights) - set(MIXTURE_SOURCE_IDS)
        if unknown:
            raise ValueError(f"unknown mixture sources: {sorted(unknown)}")
        if not weights:
            raise ValueError("mixture weights are all zero")
        if "sidd" in weights and not sidd_pairs:
            raise ValueError("mixture requests sidd but no SIDD pairs found")
        if "polyu" in weights and not polyu_pairs:
            raise ValueError("mixture requests polyu but no PolyU pairs found")
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

        self._sidd_pairs = list(sidd_pairs)
        self._polyu_pairs = list(polyu_pairs)
        clean_paths = [gt for _, gt in self._sidd_pairs] + [mean for _, mean in self._polyu_pairs]
        if "synthetic" in self.weights and not clean_paths:
            raise ValueError("mixture requests synthetic but no clean images found")
        self._clean_paths = clean_paths

        # Count available precomputed teacher files (informational; the trainer
        # warns and zeroes the distill term when this is 0).
        self.num_teacher_files = 0
        if self.with_teacher:
            self.num_teacher_files = sum(
                1 for noisy, _ in self._sidd_pairs if teacher_path_for_noisy(noisy, teacher_root).exists()
            )

    # ---- Sample pipeline ----
    def _to_linear(self, arr_u8: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(arr_u8)).permute(2, 0, 1).float().div_(255.0)
        return srgb_to_linear(t)

    def _augment(self, rng: np.random.Generator, views: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self.flip_rot:
            return [v.contiguous() for v in views]
        fx = rng.random() < 0.5
        fy = rng.random() < 0.5
        k = int(rng.integers(0, 4))
        out = []
        for v in views:
            if fx:
                v = v.flip(-1)
            if fy:
                v = v.flip(-2)
            if k:
                v = torch.rot90(v, k, dims=(-2, -1))
            out.append(v.contiguous())
        return out

    def _exposure_scale(self, rng: np.random.Generator) -> float:
        if self.exposure_jitter is None:
            return 1.0
        lo, hi = self.exposure_jitter
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    def _sample(self, rng: np.random.Generator, sources: Dict[str, object]):
        names = list(self.weights.keys())
        probs = np.array([self.weights[n] for n in names], dtype=np.float64)
        source_name = names[int(rng.choice(len(names), p=probs))]
        source_id = MIXTURE_SOURCE_IDS[source_name]
        ps = self.patch_size
        scale = self._exposure_scale(rng)

        if source_name == "synthetic":
            clean_u8 = sources["synthetic"].sample(rng, ps)
            clean = self._to_linear(clean_u8) * scale  # exposure jitter FIRST
            a, b = sample_poisson_gaussian_params(rng, self.synth_a_range, self.synth_b_range)
            chroma_scale = self.synth_chroma_scale if rng.random() < self.synth_chroma_prob else 0.0
            corr_sigma = float(rng.uniform(*self.synth_corr_sigma_range))
            chroma_corr_sigma = float(rng.uniform(*self.synth_chroma_corr_sigma_range)) if chroma_scale > 0.0 else 0.0
            noisy = apply_poisson_gaussian_linear(
                rng, clean, a, b,
                chroma_scale=chroma_scale,
                corr_sigma=corr_sigma,
                chroma_corr_sigma=chroma_corr_sigma,
            )
            teacher = clean
            has_teacher = 0.0
            views = self._augment(rng, [noisy, clean])
            noisy, clean = views
            teacher = clean
        else:
            noisy_u8, clean_u8, teacher_u8 = sources[source_name].sample(rng, ps)
            noisy = self._to_linear(noisy_u8) * scale
            clean = self._to_linear(clean_u8) * scale
            if teacher_u8 is not None:
                teacher = self._to_linear(teacher_u8) * scale
                has_teacher = 1.0
                noisy, clean, teacher = self._augment(rng, [noisy, clean, teacher])
            else:
                has_teacher = 0.0
                noisy, clean = self._augment(rng, [noisy, clean])
                teacher = clean

        return noisy, clean, teacher, np.float32(has_teacher), np.int64(source_id)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = int(info.id) if info is not None else 0
        rng = np.random.default_rng((self.seed * 1000003 + worker_id * 7919 + 17) & 0xFFFFFFFF)
        sources: Dict[str, object] = {}
        if "sidd" in self.weights:
            sources["sidd"] = _BurstPairSource(
                self._sidd_pairs,
                burst_length=self.burst_length,
                teacher_root=self.teacher_root,
                with_teacher=self.with_teacher,
            )
        if "polyu" in self.weights:
            sources["polyu"] = _BurstPairSource(self._polyu_pairs, burst_length=self.burst_length)
        if "synthetic" in self.weights:
            sources["synthetic"] = _BurstCleanSource(self._clean_paths, burst_length=self.burst_length)
        while True:
            yield self._sample(rng, sources)
