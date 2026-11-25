# src/em_denoise_refactor/data/n2n_dataset.py
"""
Noise2Noise (N2N) paired overlap dataset reading directly from sbem-n2n-overlaps outputs.

Reads variable-sized `(2, H_ov, W_ov)` uint8 2-page TIFF pairs from byte-offset
indexed USTAR `.tar` shards using `manifest.parquet` (queried via DuckDB).

Invariants
----------
- `page 0`: first-exposed tile region (1 electron-beam exposure).
- `page 1`: second-exposed tile region (2 electron-beam exposures).
- Native `(Y, X)` orientation is preserved; random crops of `(patch_size, patch_size)`
  are extracted identically from both pages.
- Returns `(input_tensor, target_tensor, pair_id)` matching the 3-tuple contract of
  `DenoisingTrainer` and `DenoisingEvaluator`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import shutil
import struct
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndi_shift
from skimage.registration import phase_cross_correlation
import tifffile
import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from em_denoise_refactor.config import normalize_hw_pair

logger = logging.getLogger("train_and_infer")


def _estimate_integer_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    max_shift: int = 16,
    blur_sigma: float = 1.0,
) -> Tuple[int, int]:
    """
    Estimate integer (dy, dx) shift such that mov[y - dy, x - dx] aligns with ref[y, x]
    (equivalently, slicing pair_arr[1] at y0 - dy, x0 - dx aligns it to ref at y0, x0).

    Uses Hann-windowed FFT cross-correlation on Gaussian-smoothed proxies so shot
    noise does not bias the peak, bounded to [-max_shift, +max_shift].
    """
    a = ref.astype(np.float32)
    b = mov.astype(np.float32)
    if blur_sigma > 0.0:
        a = gaussian_filter(a, sigma=blur_sigma)
        b = gaussian_filter(b, sigma=blur_sigma)

    a = a - a.mean()
    b = b - b.mean()

    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    a *= win
    b *= win

    cc = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=(h, w))
    cc_shifted = np.fft.fftshift(cc)

    cy, cx = h // 2, w // 2
    r = min(max_shift, cy, cx)
    roi = cc_shifted[cy - r : cy + r + 1, cx - r : cx + r + 1]
    iy, ix = np.unravel_index(np.argmax(roi), roi.shape)
    return int(iy - r), int(ix - r)


def _estimate_subpixel_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    max_shift: int = 16,
    blur_sigma: float = 1.0,
    upsample_factor: int = 10,
) -> Tuple[float, float]:
    """
    Estimate sub-pixel (dy, dx) shift such that `scipy.ndimage.shift(mov, (dy, dx))`
    aligns `mov` with `ref`.

    Uses a two-stage approach: coarse integer alignment via `_estimate_integer_shift`
    (bounded to `[-max_shift, +max_shift]`), followed by Hann-windowed sub-pixel
    cross-correlation on the overlapping interior so boundary windowing does not
    bias the fractional peak.
    """
    idy, idx = _estimate_integer_shift(
        ref, mov, max_shift=max_shift, blur_sigma=blur_sigma
    )
    h, w = ref.shape
    ry0, ry1 = max(0, idy), h + min(0, idy)
    rx0, rx1 = max(0, idx), w + min(0, idx)
    my0, my1 = max(0, -idy), h + min(0, -idy)
    mx0, mx1 = max(0, -idx), w + min(0, -idx)

    if (ry1 - ry0) < 16 or (rx1 - rx0) < 16:
        return float(idy), float(idx)

    a = ref[ry0:ry1, rx0:rx1].astype(np.float32)
    b = mov[my0:my1, mx0:mx1].astype(np.float32)
    if blur_sigma > 0.0:
        a = gaussian_filter(a, sigma=blur_sigma)
        b = gaussian_filter(b, sigma=blur_sigma)

    a = a - a.mean()
    b = b - b.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    fshift, _, _ = phase_cross_correlation(
        a * win,
        b * win,
        upsample_factor=upsample_factor,
        normalization=None,
    )
    dy = float(idy + fshift[0])
    dx = float(idx + fshift[1])
    if abs(dy) > max_shift or abs(dx) > max_shift:
        return float(idy), float(idx)
    return dy, dx


def _compute_pair_quality(
    crop0: np.ndarray,
    crop1: np.ndarray,
    metric: str = "ncc",
    blur_sigma: float = 1.5,
) -> float:
    """
    Compute structural alignment score between two noisy crops on smoothed proxies.

    Parameters
    ----------
    metric : str
        "ncc" (zero-mean normalized cross-correlation in [-1, 1]) or
        "ssim" (structural similarity on smoothed [0, 255] proxies).
    blur_sigma : float
        Gaussian smoothing sigma applied before scoring to suppress shot noise.
    """
    s0 = crop0.astype(np.float32)
    s1 = crop1.astype(np.float32)
    if blur_sigma > 0.0:
        s0 = gaussian_filter(s0, sigma=blur_sigma)
        s1 = gaussian_filter(s1, sigma=blur_sigma)

    if metric == "ssim":
        from skimage.metrics import structural_similarity as ssim

        return float(ssim(s0, s1, data_range=255.0))

    a = s0 - s0.mean()
    b = s1 - s1.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom < 1e-8:
        return 0.0
    return float(np.sum(a * b) / denom)


def _build_section_split_filter(
    sections: List[int],
    strategy: str = "block",
    block_size: int = 100,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    guard_band_size: int = 5,
    origin: Optional[int] = None,
) -> Dict[str, set[int]]:
    """
    Partition sorted section numbers into train/val/test sets without Z-leakage.

    Matches the splitting semantics of `SplittingConfig` in `config.py`.
    When `origin` is provided (e.g., `0`), block boundaries are anchored at that
    absolute section index rather than `sorted_secs[0]`, preventing anchor drift
    across filtered queries or superset cache slicing.
    """
    if not sections:
        return {"train": set(), "val": set(), "test": set()}

    sorted_secs = sorted(set(sections))
    splits: Dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}

    if strategy == "mod10":
        for s in sorted_secs:
            rem = s % 10
            if rem < 8:
                splits["train"].add(s)
            elif rem == 8:
                splits["val"].add(s)
            else:
                splits["test"].add(s)
        return splits

    # Default: "block" volumetric splitting with guard bands
    anchor = int(origin) if origin is not None else sorted_secs[0]
    train_cut = train_ratio
    val_cut = train_ratio + val_ratio

    for s in sorted_secs:
        offset = s - anchor
        block_idx = offset // max(block_size, 1)
        pos_in_block = offset % max(block_size, 1)

        # Guard band at the boundary of each block
        if guard_band_size > 0 and (
            pos_in_block < guard_band_size
            or pos_in_block >= block_size - guard_band_size
        ):
            continue

        # Deterministic assignment per block using a 10-block cycle
        cycle_pos = (block_idx % 10) / 10.0
        if cycle_pos < train_cut:
            splits["train"].add(s)
        elif cycle_pos < val_cut:
            splits["val"].add(s)
        else:
            splits["test"].add(s)

    return splits


class _NpyChunkWriter:
    """Constant-RAM sequential writer for standard NumPy .npy v1.0 (N, C, H, W) uint8 files."""

    HEADER_LEN = 128

    def __init__(
        self,
        path: Path,
        patch_size: int | Sequence[int],
        num_channels: int = 2,
    ) -> None:
        self.path = Path(path)
        self.patch_h, self.patch_w = normalize_hw_pair(patch_size, "patch_size")
        self.num_channels = int(num_channels)
        if self.num_channels <= 0:
            raise ValueError(f"num_channels must be > 0, got {num_channels}")
        self.patch_size: int | Tuple[int, int] = (
            self.patch_h
            if self.patch_h == self.patch_w
            else (self.patch_h, self.patch_w)
        )
        self.count = 0
        self._fh = open(self.path, "wb")
        self._fh.write(b"\x00" * self.HEADER_LEN)

    def append(self, chunk: np.ndarray) -> None:
        arr = np.ascontiguousarray(chunk, dtype=np.uint8)
        if arr.ndim != 4 or arr.shape[1:] != (
            self.num_channels,
            self.patch_h,
            self.patch_w,
        ):
            raise ValueError(
                f"Expected chunk shape (N, {self.num_channels}, {self.patch_h}, {self.patch_w}), "
                f"got {arr.shape}"
            )
        if arr.shape[0] == 0:
            return
        self._fh.write(arr.tobytes())
        self.count += int(arr.shape[0])

    def finalize(self) -> None:
        try:
            header_dict = (
                f"{{'descr': '|u1', 'fortran_order': False, "
                f"'shape': ({self.count}, {self.num_channels}, {self.patch_h}, {self.patch_w}), }}"
            ).encode("latin1")
            dict_target_len = self.HEADER_LEN - 10
            pad_len = dict_target_len - len(header_dict) - 1
            if pad_len < 0:
                raise ValueError(
                    f"NPY header exceeded {self.HEADER_LEN} bytes ({len(header_dict)})"
                )
            dict_block = header_dict + (b" " * pad_len) + b"\n"
            header = (
                b"\x93NUMPY\x01\x00" + struct.pack("<H", len(dict_block)) + dict_block
            )
            assert len(header) == self.HEADER_LEN
            self._fh.seek(0)
            self._fh.write(header)
            self._fh.flush()
        finally:
            self._fh.close()

    def abort(self) -> None:
        try:
            if not self._fh.closed:
                self._fh.close()
        except OSError:
            pass


def _compute_source_signature(overlaps_dir: Path) -> Dict[str, Any]:
    """Return a lightweight fingerprint of manifest.parquet (or .tar shards) without heavy I/O."""
    manifest_path = overlaps_dir / "manifest.parquet"
    if manifest_path.exists():
        st = manifest_path.stat()
        return {
            "kind": "manifest",
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    tar_files = sorted(overlaps_dir.glob("*.tar"))
    if not tar_files and (overlaps_dir / "shards").is_dir():
        tar_files = sorted((overlaps_dir / "shards").glob("*.tar"))
    if not tar_files:
        return {"kind": "missing"}
    return {
        "kind": "tar_list",
        "files": [
            [
                str(p.relative_to(overlaps_dir)),
                int(p.stat().st_size),
                int(p.stat().st_mtime_ns),
            ]
            for p in tar_files
        ],
    }


def _hash_payload(payload: Dict[str, Any]) -> str:
    """Compute short deterministic SHA-256 hex digest for cache metadata."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class N2NOverlapPairDataset(Dataset):
    """
    PyTorch Dataset for Noise2Noise training on `sbem-n2n-overlaps` `.tar` shards.

    Parameters
    ----------
    overlaps_dir : Path | str
        Root output directory of `sbem-n2n-overlaps` containing `manifest.parquet`
        and `shards/`.
    split : str
        One of `"train"`, `"val"`, or `"test"`.
    patch_size : int | Sequence[int]
        Crop size `(patch_h, patch_w)` (or square scalar `ps`) extracted from `(H_ov, W_ov)`.
    first_sec : Optional[int]
        Minimum `section_num` to include (inclusive).
    last_sec : Optional[int]
        Maximum `section_num` to include (inclusive).
    split_strategy : str
        `"block"` or `"mod10"`.
    block_size : int
        Block size in sections for `"block"` splitting.
    train_ratio : float
        Train fraction for `"block"` splitting.
    val_ratio : float
        Validation fraction for `"block"` splitting.
    guard_band_size : int
        Buffer sections at block boundaries for `"block"` splitting.
    orientation : str
        `"all"`, `"horizontal"`, or `"vertical"` (filters `manifest.parquet`).
    max_pairs : Optional[int]
        Maximum number of overlap pairs to sample for this split.
    pair_direction : str
        `"random_swap"` (50% 0->1, 50% 1->0 on train; 0->1 on val/test),
        `"first_to_second"` (always page 0 -> page 1), or
        `"second_to_first"` (always page 1 -> page 0).
    cache_in_ram : bool
        If True, pre-loads decoded `(2, H_ov, W_ov)` uint8 pairs into RAM grouped
        by shard for fast sequential disk/SMB reads.
    augment : bool
        If True, applies synchronized random horizontal/vertical flips.
    augment_rot90 : bool
        If True and `augment` is True, also applies random 90-degree rotations
        (disabled by default to preserve SBF-SEM scan-axis noise anisotropy).
    seed : int
        Random seed for deterministic pair sampling.
    """

    def __init__(
        self,
        overlaps_dir: Path | str,
        split: str = "train",
        patch_size: int | Sequence[int] = 128,
        first_sec: Optional[int] = None,
        last_sec: Optional[int] = None,
        split_strategy: str = "block",
        block_size: int = 100,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        guard_band_size: int = 5,
        split_origin: Optional[int] = None,
        orientation: str = "all",
        max_pairs: Optional[int] = None,
        pair_direction: str = "random_swap",
        cache_in_ram: bool = True,
        augment: bool = False,
        augment_rot90: bool = False,
        seed: int = 42,
        align_patches: bool = False,
        align_subpixel: bool = False,
        align_max_shift_px: int = 16,
        align_blur_sigma: float = 1.0,
        patch_sampling: str = "random",
        tile_stride: Optional[int | Sequence[int]] = None,
        trim_enabled: bool = False,
        v_trim_top_px: int = 50,
        v_trim_left_px: int = 500,
        v_trim_bottom_px: int = 0,
        v_trim_right_px: int = 0,
        h_trim_left_px: int = 32,
        h_trim_top_px: int = 0,
        h_trim_bottom_px: int = 0,
        h_trim_right_px: int = 0,
        quality_filter_enabled: bool = False,
        quality_metric: str = "ncc",
        min_pair_quality: float = 0.60,
        quality_blur_sigma: float = 1.5,
        min_patch_std: float = 0.0,
        crop_cache_dir: Optional[Path | str] = None,
        max_tiles_per_stripe: Optional[int] = None,
        prep_workers: int = 1,
    ) -> None:
        self.overlaps_dir = Path(overlaps_dir)
        self.split = split.lower().strip()
        self.patch_h, self.patch_w = normalize_hw_pair(patch_size, "patch_size")
        self.patch_size: int | Tuple[int, int] = (
            self.patch_h
            if self.patch_h == self.patch_w
            else (self.patch_h, self.patch_w)
        )
        self.pair_direction = pair_direction.lower().strip()
        self.cache_in_ram = cache_in_ram
        self.augment = augment
        self.augment_rot90 = bool(augment_rot90)
        if self.augment_rot90 and self.patch_h != self.patch_w:
            logger.warning(
                f"augment_rot90=True is incompatible with rectangular patches "
                f"({self.patch_h}x{self.patch_w}) because 90/270-deg rotation transposes H/W; "
                "auto-disabling augment_rot90."
            )
            self.augment_rot90 = False
        self.align_patches = align_patches
        self.align_subpixel = bool(align_subpixel)
        self.align_max_shift_px = align_max_shift_px
        self.align_blur_sigma = align_blur_sigma
        self.patch_sampling = patch_sampling.lower().strip()
        if tile_stride is not None:
            self.stride_h, self.stride_w = normalize_hw_pair(tile_stride, "tile_stride")
        else:
            self.stride_h, self.stride_w = self.patch_h, self.patch_w
        self.tile_stride: int | Tuple[int, int] = (
            self.stride_h
            if self.stride_h == self.stride_w
            else (self.stride_h, self.stride_w)
        )
        self.trim_enabled = trim_enabled
        self.v_trim = (
            int(v_trim_top_px),
            int(v_trim_bottom_px),
            int(v_trim_left_px),
            int(v_trim_right_px),
        )
        self.h_trim = (
            int(h_trim_top_px),
            int(h_trim_bottom_px),
            int(h_trim_left_px),
            int(h_trim_right_px),
        )
        self.quality_filter_enabled = quality_filter_enabled
        self.quality_metric = quality_metric.lower().strip()
        self.min_pair_quality = float(min_pair_quality)
        self.quality_blur_sigma = float(quality_blur_sigma)
        self.min_patch_std = float(min_patch_std)
        self.crop_cache_dir = (
            Path(crop_cache_dir)
            if crop_cache_dir is not None and str(crop_cache_dir).strip() != ""
            else None
        )
        self.max_tiles_per_stripe = (
            int(max_tiles_per_stripe) if max_tiles_per_stripe is not None else None
        )
        self.prep_workers = max(1, int(prep_workers))
        self.is_val_or_test = self.split in ("val", "test")
        self.rejected_oob_count: int = 0
        self.rejected_flat_count: int = 0
        self.cached_pairs: Optional[List[Tuple[np.ndarray, str]]] = None
        self.tile_index: Optional[List[Tuple[int, int, int]]] = None
        self.cached_tile_crops: Optional[np.ndarray] = None
        self.cached_tile_ids: Optional[List[str]] = None
        self.cached_tile_sections: Optional[np.ndarray] = None
        self._mmap_crops_path: Optional[Path] = None
        self.cache_slice_indices: Optional[np.ndarray] = None
        self.records: List[Tuple[str, str, int, int, int, int, int]] = []

        if (
            self.align_patches
            and self.align_subpixel
            and self.pair_direction != "first_to_second"
        ):
            logger.warning(
                f"align_subpixel=True with pair_direction='{self.pair_direction}' "
                "feeds cubic-interpolated noise as input in some/all samples, which can "
                "introduce a mild noise-smoothing mismatch vs raw inference images "
                "(pair_direction='first_to_second' is recommended)."
            )

        if self.quality_metric not in ("ncc", "ssim"):
            raise ValueError(
                f"quality_metric must be 'ncc' or 'ssim', got '{quality_metric}'"
            )

        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        if self.pair_direction not in (
            "random_swap",
            "first_to_second",
            "second_to_first",
        ):
            raise ValueError(
                f"pair_direction must be 'random_swap', 'first_to_second', or "
                f"'second_to_first', got '{pair_direction}'"
            )
        if self.patch_sampling not in ("random", "tile"):
            raise ValueError(
                f"patch_sampling must be 'random' or 'tile', got '{patch_sampling}'"
            )
        if (
            self.quality_filter_enabled
            and self.quality_metric == "ssim"
            and min(self.patch_h, self.patch_w) < 7
        ):
            raise ValueError(
                f"SSIM quality filtering requires patch_h, patch_w >= 7, got ({self.patch_h}, {self.patch_w})"
            )

        tile_streaming = self.patch_sampling == "tile" and (
            self.cache_in_ram or self.crop_cache_dir is not None
        )

        prep_fingerprint, prep_hash, exact_key = self._build_cache_keys(
            first_sec=first_sec,
            last_sec=last_sec,
            split_strategy=split_strategy,
            block_size=block_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            guard_band_size=guard_band_size,
            split_origin=split_origin,
            orientation=orientation,
            max_pairs=max_pairs,
            seed=seed,
        )

        if tile_streaming and self.crop_cache_dir is not None:
            if self._try_load_disk_cache(
                prep_hash=prep_hash,
                exact_key=exact_key,
                first_sec=first_sec,
                last_sec=last_sec,
                max_pairs=max_pairs,
                seed=seed,
            ):
                return

        manifest_path = self.overlaps_dir / "manifest.parquet"
        record_limit = None if tile_streaming else max_pairs
        if manifest_path.exists():
            self.records = self._load_records_from_manifest(
                manifest_path=manifest_path,
                patch_size=patch_size,
                first_sec=first_sec,
                last_sec=last_sec,
                split_strategy=split_strategy,
                block_size=block_size,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                guard_band_size=guard_band_size,
                split_origin=split_origin,
                orientation=orientation,
                max_pairs=record_limit,
                seed=seed,
                order_by_shard=tile_streaming,
            )
        else:
            self.records = self._load_records_from_tar_dir(
                first_sec=first_sec,
                last_sec=last_sec,
                split_strategy=split_strategy,
                block_size=block_size,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                guard_band_size=guard_band_size,
                split_origin=split_origin,
                orientation=orientation,
                max_pairs=record_limit,
                seed=seed,
            )

        if tile_streaming:
            self._stream_and_materialize_tiles(
                max_pairs=max_pairs,
                seed=seed,
                prep_fingerprint=prep_fingerprint,
                prep_hash=prep_hash,
                exact_key=exact_key,
                first_sec=first_sec,
                last_sec=last_sec,
            )
            return

        if self.cache_in_ram:
            self._preload_into_ram()

        if self.patch_sampling == "tile":
            self.tile_index = self._build_tile_index()
            total_tiles = len(self.tile_index)
            if (
                max_pairs is not None and total_tiles > int(max_pairs)
            ) or self.quality_filter_enabled:
                rng = random.Random(seed)
                rng.shuffle(self.tile_index)

            if self.quality_filter_enabled:
                self.tile_index = self._filter_tiles_by_quality(
                    self.tile_index, max_pairs=max_pairs
                )
            elif max_pairs is not None and total_tiles > int(max_pairs):
                self.tile_index = self.tile_index[: int(max_pairs)]

            if self.cache_in_ram and self.cached_tile_crops is None:
                self.tile_index = self._materialize_tile_crops(self.tile_index)

            logger.info(
                f"N2N '{self.split}' tile mode: {len(self.records)} stripes -> "
                f"{len(self.tile_index)}/{total_tiles} patches ({self.patch_h}x{self.patch_w}, stride={self.tile_stride})"
            )

    def _load_records_from_manifest(
        self,
        manifest_path: Path,
        patch_size: int | Sequence[int],
        first_sec: Optional[int],
        last_sec: Optional[int],
        split_strategy: str,
        block_size: int,
        train_ratio: float,
        val_ratio: float,
        guard_band_size: int,
        orientation: str,
        max_pairs: Optional[int],
        seed: int,
        order_by_shard: bool = False,
        split_origin: Optional[int] = None,
    ) -> List[Tuple[str, str, int, int, int, int, int]]:
        """Query manifest.parquet via DuckDB and return byte-offset records."""
        ph, pw = normalize_hw_pair(patch_size, "patch_size")
        con = duckdb.connect()
        if self.trim_enabled:
            vt, vb, vl, vr = self.v_trim
            ht, hb, hl, hr = self.h_trim
            where_clauses = [
                f"(CASE WHEN lower(m.orientation) = 'vertical' "
                f"THEN m.height - {vt + vb} ELSE m.height - {ht + hb} END) >= {ph}",
                f"(CASE WHEN lower(m.orientation) = 'vertical' "
                f"THEN m.width - {vl + vr} ELSE m.width - {hl + hr} END) >= {pw}",
            ]
        else:
            where_clauses = [
                f"m.height >= {ph}",
                f"m.width >= {pw}",
            ]
        if first_sec is not None:
            where_clauses.append(f"m.section_num >= {int(first_sec)}")
        if last_sec is not None:
            where_clauses.append(f"m.section_num <= {int(last_sec)}")
        if orientation.lower() in ("horizontal", "vertical"):
            where_clauses.append(f"lower(m.orientation) = '{orientation.lower()}'")

        where_sql = " AND ".join(where_clauses)
        manifest_str = str(manifest_path).replace("'", "''")

        sec_rows = con.execute(
            f"SELECT DISTINCT m.section_num FROM read_parquet('{manifest_str}') m "
            f"WHERE {where_sql} ORDER BY m.section_num"
        ).fetchall()
        all_sections = [int(r[0]) for r in sec_rows]

        split_map = _build_section_split_filter(
            sections=all_sections,
            strategy=split_strategy,
            block_size=block_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            guard_band_size=guard_band_size,
            origin=split_origin,
        )
        target_sections = sorted(split_map[self.split])
        if not target_sections:
            raise ValueError(
                f"No sections assigned to split '{self.split}' under strategy "
                f"'{split_strategy}' in {manifest_path}"
            )

        con.execute("CREATE TEMP TABLE split_secs (section_num INTEGER)")
        con.executemany(
            "INSERT INTO split_secs VALUES (?)", [(s,) for s in target_sections]
        )

        order_sql = (
            "ORDER BY m.shard_relpath, m.tif_byte_offset"
            if order_by_shard
            else f"ORDER BY hash(m.pair_id || '_{int(seed)}')"
        )
        limit_sql = f"LIMIT {int(max_pairs)}" if max_pairs is not None else ""
        query = f"""
            SELECT
                m.pair_id,
                m.shard_relpath,
                m.tif_byte_offset,
                m.tif_byte_length,
                m.height,
                m.width,
                m.section_num
            FROM read_parquet('{manifest_str}') m
            INNER JOIN split_secs s ON m.section_num = s.section_num
            WHERE {where_sql}
            {order_sql}
            {limit_sql}
        """
        rows = con.execute(query).fetchall()
        con.close()

        if not rows:
            raise ValueError(
                f"No overlap pairs matched criteria for split '{self.split}' in {manifest_path}"
            )

        return [
            (
                str(r[0]),
                str(r[1]),
                int(r[2]),
                int(r[3]),
                int(r[4]),
                int(r[5]),
                int(r[6]),
            )
            for r in rows
        ]

    def _load_records_from_tar_dir(
        self,
        first_sec: Optional[int],
        last_sec: Optional[int],
        split_strategy: str,
        block_size: int,
        train_ratio: float,
        val_ratio: float,
        guard_band_size: int,
        orientation: str,
        max_pairs: Optional[int],
        seed: int,
        split_origin: Optional[int] = None,
    ) -> List[Tuple[str, str, int, int, int, int, int]]:
        """Scan local .tar shards directly when manifest.parquet is not present."""
        tar_files = sorted(self.overlaps_dir.glob("*.tar"))
        if not tar_files and (self.overlaps_dir / "shards").is_dir():
            tar_files = sorted((self.overlaps_dir / "shards").glob("*.tar"))
        if not tar_files:
            raise FileNotFoundError(
                f"Neither manifest.parquet nor *.tar shards found in {self.overlaps_dir}"
            )

        name_re = re.compile(r"s(\d+)_g\d+_t\d+_t\d+_([hv])\.tif$")
        orient_filter = orientation.lower().strip()
        candidates: List[Tuple[int, str, str, int, int]] = []
        all_sections: set[int] = set()

        for tar_path in tar_files:
            rel_path = str(tar_path.relative_to(self.overlaps_dir))
            with tarfile.open(tar_path, "r") as tf:
                for member in tf:
                    if not member.isfile() or not member.name.endswith(".tif"):
                        continue
                    match = name_re.search(member.name)
                    if not match:
                        continue
                    sec_num = int(match.group(1))
                    orient = "horizontal" if match.group(2) == "h" else "vertical"

                    if first_sec is not None and sec_num < first_sec:
                        continue
                    if last_sec is not None and sec_num > last_sec:
                        continue
                    if (
                        orient_filter in ("horizontal", "vertical")
                        and orient != orient_filter
                    ):
                        continue

                    pair_id = Path(member.name).stem
                    all_sections.add(sec_num)
                    candidates.append(
                        (
                            sec_num,
                            pair_id,
                            rel_path,
                            int(member.offset_data),
                            int(member.size),
                        )
                    )

        split_map = _build_section_split_filter(
            sections=sorted(all_sections),
            strategy=split_strategy,
            block_size=block_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            guard_band_size=guard_band_size,
            origin=split_origin,
        )
        target_sections = split_map[self.split]
        if not target_sections:
            raise ValueError(
                f"No sections assigned to split '{self.split}' under strategy "
                f"'{split_strategy}' in {self.overlaps_dir}"
            )

        filtered = [
            (pair_id, rel_path, offset, length, 0, 0, sec_num)
            for sec_num, pair_id, rel_path, offset, length in candidates
            if sec_num in target_sections
        ]
        if not filtered:
            raise ValueError(
                f"No overlap pairs matched split '{self.split}' in {self.overlaps_dir}"
            )

        rng = random.Random(seed)
        rng.shuffle(filtered)
        if max_pairs is not None:
            filtered = filtered[: int(max_pairs)]
        return filtered

    def _trim_pair(self, arr: np.ndarray, pair_id: str) -> np.ndarray:
        """Slice off configured distorted margins based on stripe orientation (_h vs _v)."""
        if not self.trim_enabled:
            return arr
        if pair_id.endswith("_v"):
            top, bottom, left, right = self.v_trim
        else:
            top, bottom, left, right = self.h_trim
        _, h, w = arr.shape
        y_end = h - bottom if bottom > 0 else h
        x_end = w - right if right > 0 else w
        if top >= y_end or left >= x_end:
            return arr[:, 0:0, 0:0]
        return arr[:, top:y_end, left:x_end]

    def _preload_into_ram(self) -> None:
        """Read selected pairs grouped by shard in ascending byte-offset order."""
        logger.info(
            f"Caching {len(self.records)} N2N '{self.split}' overlap pairs into RAM..."
        )
        shard_groups: Dict[str, List[Tuple[int, int, int, str]]] = defaultdict(list)
        for idx, (pair_id, shard_rel, offset, length, _, _, _) in enumerate(
            self.records
        ):
            shard_groups[shard_rel].append((offset, length, idx, pair_id))

        cached: List[Optional[Tuple[np.ndarray, str]]] = [None] * len(self.records)
        kept_counts = {"_h": 0, "_v": 0}
        dropped_counts = {"_h": 0, "_v": 0}
        ph, pw = self.patch_h, self.patch_w

        pbar = tqdm(
            total=len(self.records),
            desc=f"N2N RAM Cache ({self.split})",
            bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for shard_rel, items in shard_groups.items():
            shard_path = self.overlaps_dir / shard_rel
            items.sort(key=lambda x: x[0])
            with open(shard_path, "rb") as f:
                for offset, length, idx, pair_id in items:
                    f.seek(offset)
                    raw_bytes = f.read(length)
                    arr = tifffile.imread(io.BytesIO(raw_bytes))
                    if arr.ndim != 3 or arr.shape[0] != 2:
                        raise ValueError(
                            f"Expected (2, H, W) TIFF for {pair_id}, got shape {arr.shape}"
                        )
                    arr = self._trim_pair(arr, pair_id)
                    orient_key = "_v" if pair_id.endswith("_v") else "_h"
                    if arr.shape[1] < ph or arr.shape[2] < pw:
                        dropped_counts[orient_key] += 1
                    else:
                        kept_counts[orient_key] += 1
                        cached[idx] = (arr, pair_id)
                    pbar.update(1)
        pbar.close()

        valid_pairs = [
            (rec, c) for rec, c in zip(self.records, cached) if c is not None
        ]
        if not valid_pairs:
            raise ValueError(
                f"No pairs remained after ROI trimming (patch_size={ph}x{pw}) for split '{self.split}'"
            )
        self.records = [rec for rec, _ in valid_pairs]
        self.cached_pairs = [c for _, c in valid_pairs]

        if self.trim_enabled:
            logger.info(
                f"N2N '{self.split}' ROI trim summary (patch_size={ph}x{pw}): "
                f"_h kept={kept_counts['_h']} dropped={dropped_counts['_h']} | "
                f"_v kept={kept_counts['_v']} dropped={dropped_counts['_v']}"
            )

    def _read_pair_from_disk(self, idx: int) -> Tuple[np.ndarray, str]:
        pair_id, shard_rel, offset, length, _, _, _ = self.records[idx]
        shard_path = self.overlaps_dir / shard_rel
        with open(shard_path, "rb") as f:
            f.seek(offset)
            raw_bytes = f.read(length)
        arr = tifffile.imread(io.BytesIO(raw_bytes))
        arr = self._trim_pair(arr, pair_id)
        return arr, pair_id

    @staticmethod
    def _axis_tile_coords(length: int, ps: int, stride: int, margin: int) -> List[int]:
        """Compute tile start coordinates along one axis, inset by `margin` when possible."""
        if margin <= 0:
            coords = list(range(0, length - ps + 1, stride))
            if coords[-1] != length - ps:
                coords.append(length - ps)
            return coords

        m = min(margin, (length - ps) // 2)
        lo = m
        hi = length - ps - m
        if hi - lo < stride:
            return [(length - ps) // 2]
        coords = list(range(lo, hi + 1, stride))
        if coords[-1] != hi:
            coords.append(hi)
        return coords

    def _build_tile_index(self) -> List[Tuple[int, int, int]]:
        """Generate (pair_idx, y0, x0) grid coordinates for each stripe."""
        ph, pw = self.patch_h, self.patch_w
        sh, sw = self.stride_h, self.stride_w
        tile_margin = 0 if not self.align_patches else (6 if self.align_subpixel else 2)
        tiles: List[Tuple[int, int, int]] = []

        for pair_idx in range(len(self.records)):
            if self.cached_pairs is not None:
                _, h, w = self.cached_pairs[pair_idx][0].shape
            else:
                _, _, _, _, h, w, _ = self.records[pair_idx]
                if h == 0 or w == 0:
                    arr, _ = self._read_pair_from_disk(pair_idx)
                    _, h, w = arr.shape

            if h < ph or w < pw:
                continue

            ys = self._axis_tile_coords(h, ph, sh, tile_margin)
            xs = self._axis_tile_coords(w, pw, sw, tile_margin)

            for y0 in ys:
                for x0 in xs:
                    tiles.append((pair_idx, y0, x0))

        if not tiles:
            raise ValueError(
                f"No valid {ph}x{pw} tiles generated for split '{self.split}'"
            )
        return tiles

    def _build_cache_keys(
        self,
        first_sec: Optional[int],
        last_sec: Optional[int],
        split_strategy: str,
        block_size: int,
        train_ratio: float,
        val_ratio: float,
        guard_band_size: int,
        orientation: str,
        max_pairs: Optional[int],
        seed: int,
        split_origin: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        """Build preprocessing fingerprint, its hash, and the exact cache directory key."""
        source_sig = (
            _compute_source_signature(self.overlaps_dir)
            if self.crop_cache_dir is not None
            else {"kind": "disabled"}
        )
        # Preserve scalar int serialization for square configs so existing square caches stay valid.
        patch_fp: int | List[int] = (
            int(self.patch_h)
            if self.patch_h == self.patch_w
            else [int(self.patch_h), int(self.patch_w)]
        )
        stride_fp: int | List[int] = (
            int(self.stride_h)
            if self.stride_h == self.stride_w
            else [int(self.stride_h), int(self.stride_w)]
        )
        prep_fingerprint: Dict[str, Any] = {
            "schema_version": 1,
            "overlaps_dir": str(self.overlaps_dir.resolve()),
            "source_signature": source_sig,
            "split": self.split,
            "patch_size": patch_fp,
            "tile_stride": stride_fp,
            "orientation": str(orientation).lower().strip(),
            "trim_enabled": bool(self.trim_enabled),
            "v_trim": list(self.v_trim),
            "h_trim": list(self.h_trim),
            "align_patches": bool(self.align_patches),
            "align_subpixel": bool(self.align_subpixel),
            "align_max_shift_px": int(self.align_max_shift_px),
            "align_blur_sigma": float(self.align_blur_sigma),
            "quality_filter_enabled": bool(self.quality_filter_enabled),
            "quality_metric": self.quality_metric,
            "min_pair_quality": float(self.min_pair_quality),
            "quality_blur_sigma": float(self.quality_blur_sigma),
            "min_patch_std": float(self.min_patch_std),
            "split_strategy": str(split_strategy),
            "block_size": int(block_size),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "guard_band_size": int(guard_band_size),
            "seed": int(seed),
            "max_tiles_per_stripe": (
                int(self.max_tiles_per_stripe)
                if self.max_tiles_per_stripe is not None
                else None
            ),
        }
        if split_origin is not None:
            prep_fingerprint["split_origin"] = int(split_origin)
        prep_hash = _hash_payload(prep_fingerprint)
        exact_payload = {
            **prep_fingerprint,
            "first_sec": int(first_sec) if first_sec is not None else None,
            "last_sec": int(last_sec) if last_sec is not None else None,
            "max_pairs": int(max_pairs) if max_pairs is not None else None,
        }
        exact_key = _hash_payload(exact_payload)
        return prep_fingerprint, prep_hash, exact_key

    @staticmethod
    def _is_valid_cache_dir(cache_dir: Path) -> bool:
        return (
            cache_dir.is_dir()
            and (cache_dir / "metadata.json").is_file()
            and (cache_dir / "crops.npy").is_file()
            and (cache_dir / "sections.npy").is_file()
            and (cache_dir / "ids.txt").is_file()
        )

    def _activate_mmap_cache(
        self,
        cache_dir: Path,
        sections: np.ndarray,
        slice_indices: Optional[np.ndarray],
        is_superset: bool,
        meta: Dict[str, Any],
    ) -> None:
        """Bind a validated on-disk .npy mmap crop cache onto this dataset."""
        crops_path = cache_dir / "crops.npy"
        all_ids = (cache_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
        crops_mmap = np.load(crops_path, mmap_mode="r")
        if (
            crops_mmap.ndim != 4
            or crops_mmap.shape[1:] != (2, self.patch_h, self.patch_w)
            or len(all_ids) != crops_mmap.shape[0]
            or len(sections) != crops_mmap.shape[0]
        ):
            raise ValueError(
                f"Corrupt crop cache in {cache_dir}: shape={crops_mmap.shape}, "
                f"expected (*, 2, {self.patch_h}, {self.patch_w}), "
                f"ids={len(all_ids)}, sections={len(sections)}"
            )

        self._mmap_crops_path = crops_path
        self.cached_tile_crops = crops_mmap
        if slice_indices is not None:
            self.cache_slice_indices = np.asarray(slice_indices, dtype=np.int64)
            self.cached_tile_ids = [all_ids[int(i)] for i in self.cache_slice_indices]
            self.cached_tile_sections = sections[self.cache_slice_indices]
        else:
            self.cache_slice_indices = None
            self.cached_tile_ids = all_ids
            self.cached_tile_sections = sections

        self.rejected_oob_count = int(meta.get("rejected_oob_count", 0))
        self.rejected_flat_count = int(meta.get("rejected_flat_count", 0))
        self.records = []
        self.cached_pairs = None

        size_gb = crops_path.stat().st_size / 1e9
        n_active = len(self.cached_tile_ids)
        mode_tag = "superset mmap slice" if is_superset else "exact mmap hit"
        sec_lo = (
            int(self.cached_tile_sections.min())
            if len(self.cached_tile_sections) > 0
            else None
        )
        sec_hi = (
            int(self.cached_tile_sections.max())
            if len(self.cached_tile_sections) > 0
            else None
        )
        logger.info(
            f"N2N '{self.split}' loaded disk cache ({mode_tag}) from {cache_dir}: "
            f"{n_active}/{crops_mmap.shape[0]} crops ({crops_mmap.shape[2]}x{crops_mmap.shape[3]}, "
            f"{size_gb:.2f} GB on disk), sections=[{sec_lo}, {sec_hi}]."
        )

    def _try_load_disk_cache(
        self,
        prep_hash: str,
        exact_key: str,
        first_sec: Optional[int],
        last_sec: Optional[int],
        max_pairs: Optional[int],
        seed: int,
    ) -> bool:
        """Check for an exact or compatible superset disk crop cache and load via mmap."""
        assert self.crop_cache_dir is not None
        if not self.crop_cache_dir.exists():
            return False

        exact_dir = self.crop_cache_dir / f"{self.split}_{exact_key}"
        if self._is_valid_cache_dir(exact_dir):
            try:
                meta = json.loads(
                    (exact_dir / "metadata.json").read_text(encoding="utf-8")
                )
                if (
                    meta.get("schema_version") == 1
                    and meta.get("prep_fingerprint_hash") == prep_hash
                ):
                    sections = np.load(exact_dir / "sections.npy")
                    self._activate_mmap_cache(
                        cache_dir=exact_dir,
                        sections=sections,
                        slice_indices=None,
                        is_superset=False,
                        meta=meta,
                    )
                    return True
            except Exception as exc:
                logger.warning(f"Ignoring invalid cache at {exact_dir}: {exc}")

        best_candidate: Optional[
            Tuple[int, int, Path, np.ndarray, np.ndarray, Dict[str, Any]]
        ] = None
        for cand_dir in sorted(self.crop_cache_dir.glob(f"{self.split}_*")):
            if cand_dir == exact_dir or not self._is_valid_cache_dir(cand_dir):
                continue
            try:
                meta = json.loads(
                    (cand_dir / "metadata.json").read_text(encoding="utf-8")
                )
            except Exception:
                continue
            if (
                meta.get("schema_version") != 1
                or meta.get("prep_fingerprint_hash") != prep_hash
            ):
                continue

            c_first = meta.get("first_sec")
            c_last = meta.get("last_sec")
            covers_first = c_first is None or (
                first_sec is not None and int(c_first) <= int(first_sec)
            )
            covers_last = c_last is None or (
                last_sec is not None and int(c_last) >= int(last_sec)
            )
            if not (covers_first and covers_last):
                continue

            c_max = meta.get("max_pairs_requested")
            if max_pairs is None and c_max is not None:
                continue

            sections = np.load(cand_dir / "sections.npy")
            mask = np.ones(len(sections), dtype=bool)
            if first_sec is not None:
                mask &= sections >= int(first_sec)
            if last_sec is not None:
                mask &= sections <= int(last_sec)
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                continue
            if max_pairs is not None and len(idx) < int(max_pairs):
                continue

            is_uncapped = 1 if c_max is None else 0
            score = (is_uncapped, int(len(idx)))
            if best_candidate is None or score > best_candidate[:2]:
                best_candidate = (*score, cand_dir, sections, idx, meta)

        if best_candidate is None:
            return False

        _, _, cand_dir, sections, idx, meta = best_candidate
        if max_pairs is not None and len(idx) > int(max_pairs):
            rng_np = np.random.default_rng(int(seed))
            idx = np.sort(rng_np.choice(idx, size=int(max_pairs), replace=False))

        self._activate_mmap_cache(
            cache_dir=cand_dir,
            sections=sections,
            slice_indices=idx,
            is_superset=True,
            meta=meta,
        )
        return True

    def _is_patch_above_std(self, crop0: np.ndarray) -> bool:
        """Pure check returning True if crop0 meets the optional min_patch_std threshold."""
        if self.min_patch_std <= 0.0:
            return True
        arr = crop0.astype(np.float32)
        if self.quality_blur_sigma > 0.0:
            arr = gaussian_filter(arr, sigma=self.quality_blur_sigma)
        return float(arr.std()) >= self.min_patch_std

    def _passes_flat_filter(self, crop0: np.ndarray) -> bool:
        """Return True if crop0 meets the optional min_patch_std texture threshold."""
        return self._is_patch_above_std(crop0)

    def _align_crop_pair(
        self, pair_arr: np.ndarray, y0: int, x0: int
    ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], bool]:
        """
        Thread-safe crop extraction and alignment without mutating instance counters.

        Returns `(crops_tuple_or_None, was_rejected_oob)`.
        """
        _, h, w = pair_arr.shape
        ph, pw = self.patch_h, self.patch_w
        crop0 = pair_arr[0, y0 : y0 + ph, x0 : x0 + pw]
        crop1 = pair_arr[1, y0 : y0 + ph, x0 : x0 + pw]

        if not self.align_patches:
            return (crop0, crop1), False

        if self.align_subpixel:
            dy_f, dx_f = _estimate_subpixel_shift(
                crop0,
                crop1,
                max_shift=self.align_max_shift_px,
                blur_sigma=self.align_blur_sigma,
            )
            pad = math.ceil(max(abs(dy_f), abs(dx_f))) + 3
            y_lo, y_hi = y0 - pad, y0 + ph + pad
            x_lo, x_hi = x0 - pad, x0 + pw + pad
            if y_lo < 0 or y_hi > h or x_lo < 0 or x_hi > w:
                return None, True
            img1_pad = pair_arr[1, y_lo:y_hi, x_lo:x_hi].astype(np.float32)
            shifted_pad = ndi_shift(
                img1_pad, shift=(dy_f, dx_f), order=3, mode="nearest"
            )
            crop1_shifted = np.clip(
                shifted_pad[pad : pad + ph, pad : pad + pw], 0.0, 255.0
            )
            return (crop0, crop1_shifted), False

        dy, dx = _estimate_integer_shift(
            crop0,
            crop1,
            max_shift=self.align_max_shift_px,
            blur_sigma=self.align_blur_sigma,
        )
        y1 = y0 - dy
        x1 = x0 - dx
        if y1 < 0 or y1 + ph > h or x1 < 0 or x1 + pw > w:
            return None, True
        return (crop0, pair_arr[1, y1 : y1 + ph, x1 : x1 + pw]), False

    def _extract_aligned_crops(
        self, pair_arr: np.ndarray, y0: int, x0: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract (patch_h, patch_w) crops at (y0, x0) and optionally align crop1.

        Returns `None` (incrementing `self.rejected_oob_count`) if the estimated shift
        or sub-pixel padding would exceed the stripe boundary — never clamps.
        """
        crops, is_oob = self._align_crop_pair(pair_arr, y0, x0)
        if is_oob:
            self.rejected_oob_count += 1
            return None
        return crops

    def _process_stripe_for_tiles(
        self,
        pair_idx: int,
        sec_num: int,
        pair_arr: np.ndarray,
        pair_id: str,
        stripe_seed: int,
        per_stripe_cap: Optional[int],
        skip_coords: Optional[set[Tuple[int, int]]] = None,
    ) -> Tuple[
        List[Tuple[int, int, int, int, np.ndarray, np.ndarray, str, Optional[float]]],
        int,
        int,
        int,
    ]:
        """Extract, align, flat-filter, and quality-filter tiles from one stripe."""
        _, h, w = pair_arr.shape
        ph, pw = self.patch_h, self.patch_w
        sh, sw = self.stride_h, self.stride_w
        tile_margin = 0 if not self.align_patches else (6 if self.align_subpixel else 2)
        ys = self._axis_tile_coords(h, ph, sh, tile_margin)
        xs = self._axis_tile_coords(w, pw, sw, tile_margin)
        coords = [(y0, x0) for y0 in ys for x0 in xs]
        random.Random(stripe_seed).shuffle(coords)

        kept: List[
            Tuple[int, int, int, int, np.ndarray, np.ndarray, str, Optional[float]]
        ] = []
        rejected_quality = 0
        rejected_oob = 0
        rejected_flat = 0

        for y0, x0 in coords:
            if skip_coords is not None and (y0, x0) in skip_coords:
                continue
            crops, is_oob = self._align_crop_pair(pair_arr, y0, x0)
            if crops is None:
                if is_oob:
                    rejected_oob += 1
                continue
            crop0, crop1 = crops
            if not self._is_patch_above_std(crop0):
                rejected_flat += 1
                continue

            score: Optional[float] = None
            if self.quality_filter_enabled:
                score = _compute_pair_quality(
                    crop0,
                    crop1,
                    metric=self.quality_metric,
                    blur_sigma=self.quality_blur_sigma,
                )
                if score < self.min_pair_quality:
                    rejected_quality += 1
                    continue

            c0_u8 = np.rint(np.clip(crop0, 0.0, 255.0)).astype(np.uint8)
            c1_u8 = np.rint(np.clip(crop1, 0.0, 255.0)).astype(np.uint8)
            kept.append(
                (
                    pair_idx,
                    sec_num,
                    y0,
                    x0,
                    c0_u8,
                    c1_u8,
                    f"{pair_id}_y{y0}_x{x0}",
                    score,
                )
            )
            if per_stripe_cap is not None and len(kept) >= per_stripe_cap:
                break

        return kept, rejected_quality, rejected_oob, rejected_flat

    def _stream_and_materialize_tiles(
        self,
        max_pairs: Optional[int],
        seed: int,
        prep_fingerprint: Dict[str, Any],
        prep_hash: str,
        exact_key: str,
        first_sec: Optional[int],
        last_sec: Optional[int],
    ) -> None:
        """
        Single-pass shard-streaming crop materializer.

        Reads stripes from `.tar` shards in small sequential chunks (~180 MB RAM),
        aligns/filters tiles deterministically (optionally with `prep_workers` threads),
        enforces per-shard uniform quotas across all shards when `max_pairs` is capped,
        and writes surviving uint8 crops directly into either RAM or a persistent `.npy`
        memory-mapped cache directory.
        """
        ph, pw = self.patch_h, self.patch_w
        target_cap = int(max_pairs) if max_pairs is not None else None
        num_records = len(self.records)

        shard_groups: Dict[str, List[Tuple[int, str, int, int, int]]] = defaultdict(
            list
        )
        for pair_idx, (pair_id, shard_rel, offset, length, _, _, sec_num) in enumerate(
            self.records
        ):
            shard_groups[shard_rel].append((pair_idx, pair_id, offset, length, sec_num))

        shard_list = sorted(shard_groups.keys())
        random.Random(int(seed)).shuffle(shard_list)
        num_shards = len(shard_list)

        if target_cap is not None:
            per_stripe_cap: Optional[int] = (
                max(1, self.max_tiles_per_stripe)
                if self.max_tiles_per_stripe is not None
                else max(1, math.ceil(1.5 * target_cap / max(num_records, 1)))
            )
        else:
            per_stripe_cap = (
                max(1, self.max_tiles_per_stripe)
                if self.max_tiles_per_stripe is not None
                else None
            )

        shard_shuffled_items: Dict[str, List[Tuple[int, str, int, int, int]]] = {}
        for s_idx, shard_rel in enumerate(shard_list):
            items = list(shard_groups[shard_rel])
            random.Random(int(seed) ^ ((s_idx + 1) * 0x9E3779B9)).shuffle(items)
            shard_shuffled_items[shard_rel] = items

        # Initialize sink: disk mmap writer or compact RAM buffer
        use_disk_cache = self.crop_cache_dir is not None
        tmp_cache_dir: Optional[Path] = None
        target_cache_dir: Optional[Path] = None
        npy_writer: Optional[_NpyChunkWriter] = None
        disk_batch_crops: List[Tuple[np.ndarray, np.ndarray]] = []
        ram_buf: Optional[np.ndarray] = None
        ram_chunks: List[np.ndarray] = []

        if use_disk_cache:
            assert self.crop_cache_dir is not None
            self.crop_cache_dir.mkdir(parents=True, exist_ok=True)
            target_cache_dir = self.crop_cache_dir / f"{self.split}_{exact_key}"
            tmp_cache_dir = (
                self.crop_cache_dir / f".{self.split}_{exact_key}.tmp.{os.getpid()}"
            )
            if tmp_cache_dir.exists():
                shutil.rmtree(tmp_cache_dir, ignore_errors=True)
            tmp_cache_dir.mkdir(parents=True, exist_ok=True)
            npy_writer = _NpyChunkWriter(
                tmp_cache_dir / "crops.npy", patch_size=(ph, pw)
            )
        elif target_cap is not None:
            ram_buf = np.empty((target_cap, 2, ph, pw), dtype=np.uint8)

        kept_ids: List[str] = []
        kept_sections: List[int] = []
        scores_kept: List[float] = []
        used_coords: Dict[int, set[Tuple[int, int]]] = defaultdict(set)
        trim_seen: set[int] = set()
        trim_kept_counts = {"_h": 0, "_v": 0}
        trim_dropped_counts = {"_h": 0, "_v": 0}
        rejected_quality = 0
        rejected_oob = 0
        rejected_flat = 0
        shards_used: set[str] = set()
        kept_count = 0
        chunk_size = 64

        def _flush_disk_batch() -> None:
            if npy_writer is not None and disk_batch_crops:
                arr_b = np.empty((len(disk_batch_crops), 2, ph, pw), dtype=np.uint8)
                for j, (c0_b, c1_b) in enumerate(disk_batch_crops):
                    arr_b[j, 0] = c0_b
                    arr_b[j, 1] = c1_b
                npy_writer.append(arr_b)
                disk_batch_crops.clear()

        def _emit_item(
            shard_rel: str,
            item: Tuple[
                int, int, int, int, np.ndarray, np.ndarray, str, Optional[float]
            ],
        ) -> None:
            nonlocal kept_count
            pair_idx, sec_num, y0, x0, c0_u8, c1_u8, tile_id, score = item
            if use_disk_cache:
                disk_batch_crops.append((c0_u8, c1_u8))
                if len(disk_batch_crops) >= 256:
                    _flush_disk_batch()
            elif ram_buf is not None:
                ram_buf[kept_count, 0] = c0_u8
                ram_buf[kept_count, 1] = c1_u8
            else:
                pair_u8 = np.empty((1, 2, ph, pw), dtype=np.uint8)
                pair_u8[0, 0] = c0_u8
                pair_u8[0, 1] = c1_u8
                ram_chunks.append(pair_u8)

            kept_ids.append(tile_id)
            kept_sections.append(int(sec_num))
            if target_cap is not None:
                used_coords[pair_idx].add((y0, x0))
            shards_used.add(shard_rel)
            if score is not None:
                scores_kept.append(score)
            kept_count += 1

        def _run_pass(
            pass_name: str,
            pass_per_stripe_cap: Optional[int],
            use_shard_quotas: bool,
            use_skip_coords: bool,
        ) -> None:
            nonlocal rejected_quality, rejected_oob, rejected_flat
            pbar = tqdm(
                total=target_cap if target_cap is not None else num_records,
                desc=f"N2N Stream ({self.split}, {pass_name})",
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
            if target_cap is not None and kept_count > 0:
                pbar.update(kept_count)

            pool: Optional[ThreadPoolExecutor] = (
                ThreadPoolExecutor(max_workers=self.prep_workers)
                if self.prep_workers > 1
                else None
            )
            try:
                for s_idx, shard_rel in enumerate(shard_list):
                    if target_cap is not None and kept_count >= target_cap:
                        break
                    shard_quota: Optional[int] = None
                    if use_shard_quotas and target_cap is not None:
                        shard_quota = target_cap // num_shards + (
                            1 if s_idx < (target_cap % num_shards) else 0
                        )
                        if shard_quota <= 0:
                            continue

                    items = shard_shuffled_items[shard_rel]
                    shard_path = self.overlaps_dir / shard_rel
                    shard_kept = 0

                    with open(shard_path, "rb") as f:
                        for c_start in range(0, len(items), chunk_size):
                            if target_cap is not None and kept_count >= target_cap:
                                break
                            if shard_quota is not None and shard_kept >= shard_quota:
                                break

                            raw_chunk = items[c_start : c_start + chunk_size]
                            # Read in ascending byte-offset order for sequential I/O, then restore shuffled order
                            read_order = sorted(
                                range(len(raw_chunk)),
                                key=lambda pos: raw_chunk[pos][2],
                            )
                            trimmed_slots: List[
                                Optional[Tuple[int, int, np.ndarray, str]]
                            ] = [None] * len(raw_chunk)

                            for pos in read_order:
                                pair_idx, pair_id, offset, length, sec_num = raw_chunk[
                                    pos
                                ]
                                f.seek(offset)
                                raw_bytes = f.read(length)
                                arr = tifffile.imread(io.BytesIO(raw_bytes))
                                if arr.ndim != 3 or arr.shape[0] != 2:
                                    raise ValueError(
                                        f"Expected (2, H, W) TIFF for {pair_id}, got shape {arr.shape}"
                                    )
                                arr = self._trim_pair(arr, pair_id)
                                orient_key = "_v" if pair_id.endswith("_v") else "_h"
                                if arr.shape[1] < ph or arr.shape[2] < pw:
                                    if pair_idx not in trim_seen:
                                        trim_seen.add(pair_idx)
                                        trim_dropped_counts[orient_key] += 1
                                    continue
                                if pair_idx not in trim_seen:
                                    trim_seen.add(pair_idx)
                                    trim_kept_counts[orient_key] += 1
                                trimmed_slots[pos] = (pair_idx, sec_num, arr, pair_id)

                            work_items = [
                                slot for slot in trimmed_slots if slot is not None
                            ]
                            if not work_items:
                                if target_cap is None:
                                    pbar.update(len(raw_chunk))
                                continue

                            if pool is not None and len(work_items) > 1:
                                futures = [
                                    pool.submit(
                                        self._process_stripe_for_tiles,
                                        pair_idx,
                                        sec_num,
                                        arr,
                                        pair_id,
                                        int(seed) ^ ((pair_idx + 1) * 0x9E3779B9),
                                        pass_per_stripe_cap,
                                        used_coords.get(pair_idx)
                                        if use_skip_coords
                                        else None,
                                    )
                                    for pair_idx, sec_num, arr, pair_id in work_items
                                ]
                                results = [fut.result() for fut in futures]
                            else:
                                results = [
                                    self._process_stripe_for_tiles(
                                        pair_idx,
                                        sec_num,
                                        arr,
                                        pair_id,
                                        int(seed) ^ ((pair_idx + 1) * 0x9E3779B9),
                                        pass_per_stripe_cap,
                                        used_coords.get(pair_idx)
                                        if use_skip_coords
                                        else None,
                                    )
                                    for pair_idx, sec_num, arr, pair_id in work_items
                                ]

                            for kept_list, rq, ro, rf in results:
                                rejected_quality += rq
                                rejected_oob += ro
                                rejected_flat += rf
                                for item in kept_list:
                                    if (
                                        target_cap is not None
                                        and kept_count >= target_cap
                                    ):
                                        break
                                    if (
                                        shard_quota is not None
                                        and shard_kept >= shard_quota
                                    ):
                                        break
                                    _emit_item(shard_rel, item)
                                    shard_kept += 1
                                    if target_cap is not None:
                                        pbar.update(1)

                            if target_cap is None:
                                pbar.update(len(raw_chunk))
            finally:
                if pool is not None:
                    pool.shutdown(wait=True)
                pbar.close()

        try:
            _run_pass(
                pass_name="pass1",
                pass_per_stripe_cap=per_stripe_cap,
                use_shard_quotas=(target_cap is not None),
                use_skip_coords=False,
            )

            if target_cap is not None and kept_count < target_cap:
                logger.info(
                    f"N2N '{self.split}' streaming pass1 collected {kept_count}/{target_cap}; "
                    f"running shortfall fallback without per-shard/per-stripe caps."
                )
                _run_pass(
                    pass_name="pass2_fallback",
                    pass_per_stripe_cap=None,
                    use_shard_quotas=False,
                    use_skip_coords=True,
                )

            used_coords.clear()
            _flush_disk_batch()
            if kept_count == 0:
                raise ValueError(
                    f"No valid {ph}x{pw} tiles survived streaming materialization for "
                    f"split '{self.split}' (rejected_quality={rejected_quality}, "
                    f"rejected_oob={rejected_oob}, rejected_flat={rejected_flat})."
                )

            if self.trim_enabled:
                logger.info(
                    f"N2N '{self.split}' ROI trim summary (patch_size={ph}x{pw}): "
                    f"_h kept={trim_kept_counts['_h']} dropped={trim_dropped_counts['_h']} | "
                    f"_v kept={trim_kept_counts['_v']} dropped={trim_dropped_counts['_v']}"
                )

            mean_score_str = (
                f", mean_score={np.mean(scores_kept):.3f}" if scores_kept else ""
            )
            logger.info(
                f"N2N '{self.split}' stream summary (shards={len(shards_used)}/{num_shards}): "
                f"kept={kept_count}, rejected_quality={rejected_quality}, "
                f"rejected_oob={rejected_oob}, rejected_flat={rejected_flat}{mean_score_str}"
            )

            self.rejected_oob_count = rejected_oob
            self.rejected_flat_count = rejected_flat
            sections_arr = np.asarray(kept_sections, dtype=np.int32)

            if use_disk_cache:
                assert (
                    npy_writer is not None
                    and tmp_cache_dir is not None
                    and target_cache_dir is not None
                )
                npy_writer.finalize()
                np.save(tmp_cache_dir / "sections.npy", sections_arr)
                (tmp_cache_dir / "ids.txt").write_text(
                    "\n".join(kept_ids) + "\n", encoding="utf-8"
                )
                meta = {
                    "schema_version": 1,
                    "prep_fingerprint_hash": prep_hash,
                    "prep_fingerprint": prep_fingerprint,
                    "split": self.split,
                    "first_sec": int(first_sec) if first_sec is not None else None,
                    "last_sec": int(last_sec) if last_sec is not None else None,
                    "actual_min_sec": int(sections_arr.min()),
                    "actual_max_sec": int(sections_arr.max()),
                    "max_pairs_requested": target_cap,
                    "num_crops": int(kept_count),
                    "patch_size": (int(ph) if ph == pw else [int(ph), int(pw)]),
                    "patch_h": int(ph),
                    "patch_w": int(pw),
                    "shards_covered": int(len(shards_used)),
                    "total_shards": int(num_shards),
                    "rejected_quality_count": int(rejected_quality),
                    "rejected_oob_count": int(rejected_oob),
                    "rejected_flat_count": int(rejected_flat),
                    "mean_quality_score": (
                        float(np.mean(scores_kept)) if scores_kept else None
                    ),
                }
                (tmp_cache_dir / "metadata.json").write_text(
                    json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
                )
                if not target_cache_dir.exists():
                    try:
                        os.replace(tmp_cache_dir, target_cache_dir)
                    except OSError:
                        if not self._is_valid_cache_dir(target_cache_dir):
                            raise
                if tmp_cache_dir.exists():
                    shutil.rmtree(tmp_cache_dir, ignore_errors=True)
                self._activate_mmap_cache(
                    cache_dir=target_cache_dir,
                    sections=sections_arr,
                    slice_indices=None,
                    is_superset=False,
                    meta=meta,
                )
            else:
                if ram_buf is not None:
                    final_buf = (
                        ram_buf
                        if kept_count == len(ram_buf)
                        else ram_buf[:kept_count].copy()
                    )
                else:
                    final_buf = np.concatenate(ram_chunks, axis=0)
                    ram_chunks.clear()
                self.cached_tile_crops = final_buf
                self.cached_tile_ids = kept_ids
                self.cached_tile_sections = sections_arr
                self.cached_pairs = None
                self.records = []
                gc.collect()
                logger.info(
                    f"N2N '{self.split}' streamed {kept_count} aligned ({ph}x{pw}) "
                    f"tile crops into RAM ({final_buf.nbytes / 1e6:.1f} MB) with zero full-stripe preload."
                )
        except Exception:
            if npy_writer is not None:
                npy_writer.abort()
            if tmp_cache_dir is not None and tmp_cache_dir.exists():
                shutil.rmtree(tmp_cache_dir, ignore_errors=True)
            raise

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if state.get("_mmap_crops_path") is not None:
            state["cached_tile_crops"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _store_materialized_crops(
        self, crops: List[Tuple[np.ndarray, np.ndarray, str]]
    ) -> None:
        """Pack pre-aligned (patch_h, patch_w) uint8 crop pairs into a compact array and drop full stripes."""
        n = len(crops)
        ph, pw = self.patch_h, self.patch_w
        buf = np.empty((n, 2, ph, pw), dtype=np.uint8)
        ids: List[str] = []
        for i, (c0, c1, pid) in enumerate(crops):
            buf[i, 0] = np.rint(np.clip(c0, 0.0, 255.0)).astype(np.uint8)
            buf[i, 1] = np.rint(np.clip(c1, 0.0, 255.0)).astype(np.uint8)
            ids.append(pid)
        self.cached_tile_crops = buf
        self.cached_tile_ids = ids
        self.cached_pairs = None
        gc.collect()
        logger.info(
            f"N2N '{self.split}' materialized {n} aligned ({ph}x{pw}) tile crops "
            f"({buf.nbytes / 1e6:.1f} MB) and freed full stripe cache."
        )

    def _materialize_tile_crops(
        self, tiles: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        """Extract and align crops once for all selected tiles, then free full stripes."""
        crops: List[Tuple[np.ndarray, np.ndarray, str]] = []
        kept_tiles: List[Tuple[int, int, int]] = []
        rejected_oob = 0
        rejected_flat = 0
        for pair_idx, y0, x0 in tiles:
            if self.cached_pairs is not None:
                pair_arr, pair_id = self.cached_pairs[pair_idx]
            else:
                pair_arr, pair_id = self._read_pair_from_disk(pair_idx)
            res = self._extract_aligned_crops(pair_arr, y0, x0)
            if res is None:
                rejected_oob += 1
                continue
            crop0, crop1 = res
            if not self._passes_flat_filter(crop0):
                self.rejected_flat_count += 1
                rejected_flat += 1
                continue
            kept_tiles.append((pair_idx, y0, x0))
            crops.append((crop0, crop1, f"{pair_id}_y{y0}_x{x0}"))
        if not crops:
            raise ValueError(
                f"No valid {self.patch_h}x{self.patch_w} crops survived alignment/flat "
                f"filtering for split '{self.split}' (rejected_oob={rejected_oob}, rejected_flat={rejected_flat})."
            )
        if rejected_oob > 0 or rejected_flat > 0:
            logger.info(
                f"N2N '{self.split}' tile extraction filtered: kept={len(crops)}, "
                f"rejected_oob={rejected_oob}, rejected_flat={rejected_flat}"
            )
        self._store_materialized_crops(crops)
        return kept_tiles

    def _filter_tiles_by_quality(
        self,
        candidates: List[Tuple[int, int, int]],
        max_pairs: Optional[int] = None,
    ) -> List[Tuple[int, int, int]]:
        """Filter candidate tiles by smoothed NCC/SSIM, stopping once max_pairs is reached."""
        kept: List[Tuple[int, int, int]] = []
        kept_crops: List[Tuple[np.ndarray, np.ndarray, str]] = []
        rejected = 0
        rejected_oob = 0
        rejected_flat = 0
        scores_kept: List[float] = []
        target_cap = int(max_pairs) if max_pairs is not None else len(candidates)

        pbar = tqdm(
            candidates,
            desc=f"N2N Quality Filter ({self.split}, {self.quality_metric}>={self.min_pair_quality:.2f})",
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for pair_idx, y0, x0 in pbar:
            if self.cached_pairs is not None:
                pair_arr, pair_id = self.cached_pairs[pair_idx]
            else:
                pair_arr, pair_id = self._read_pair_from_disk(pair_idx)

            res = self._extract_aligned_crops(pair_arr, y0, x0)
            if res is None:
                rejected_oob += 1
                continue
            crop0, crop1 = res
            if not self._passes_flat_filter(crop0):
                self.rejected_flat_count += 1
                rejected_flat += 1
                continue

            score = _compute_pair_quality(
                crop0,
                crop1,
                metric=self.quality_metric,
                blur_sigma=self.quality_blur_sigma,
            )
            if score >= self.min_pair_quality:
                kept.append((pair_idx, y0, x0))
                if self.cache_in_ram:
                    kept_crops.append((crop0, crop1, f"{pair_id}_y{y0}_x{x0}"))
                scores_kept.append(score)
                if len(kept) >= target_cap:
                    break
            else:
                rejected += 1
        pbar.close()

        if not kept:
            raise ValueError(
                f"No tiles passed quality filter ({self.quality_metric} >= {self.min_pair_quality}, "
                f"min_patch_std={self.min_patch_std}) for split '{self.split}' "
                f"(rejected_quality={rejected}, rejected_oob={rejected_oob}, rejected_flat={rejected_flat})."
            )

        logger.info(
            f"N2N '{self.split}' quality filter ({self.quality_metric} >= {self.min_pair_quality:.2f}, "
            f"sigma={self.quality_blur_sigma}): kept={len(kept)}, rejected_quality={rejected}, "
            f"rejected_oob={rejected_oob}, rejected_flat={rejected_flat}, "
            f"mean_score={np.mean(scores_kept):.3f}"
        )
        if self.cache_in_ram:
            self._store_materialized_crops(kept_crops)
        return kept

    def __len__(self) -> int:
        if self.cache_slice_indices is not None:
            return len(self.cache_slice_indices)
        if self.cached_tile_ids is not None:
            return len(self.cached_tile_ids)
        if self.tile_index is not None:
            return len(self.tile_index)
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.cached_tile_ids is not None:
            if self.cached_tile_crops is None and self._mmap_crops_path is not None:
                self.cached_tile_crops = np.load(self._mmap_crops_path, mmap_mode="r")
            if self.cached_tile_crops is not None:
                crop_idx = (
                    int(self.cache_slice_indices[idx])
                    if self.cache_slice_indices is not None
                    else idx
                )
                return _format_pair_tensors(
                    crop0=self.cached_tile_crops[crop_idx, 0],
                    crop1=self.cached_tile_crops[crop_idx, 1],
                    pair_direction=self.pair_direction,
                    is_val_or_test=self.is_val_or_test,
                    augment=self.augment,
                    augment_rot90=self.augment_rot90,
                ) + (self.cached_tile_ids[idx],)

        ph, pw = self.patch_h, self.patch_w

        if self.tile_index is not None:
            n_tiles = len(self.tile_index)
            max_tile_tries = min(n_tiles, 256)
            for offset in range(max_tile_tries):
                pair_idx, y0, x0 = self.tile_index[(idx + offset) % n_tiles]
                if self.cached_pairs is not None:
                    pair_arr, pair_id = self.cached_pairs[pair_idx]
                else:
                    pair_arr, pair_id = self._read_pair_from_disk(pair_idx)
                res = self._extract_aligned_crops(pair_arr, y0, x0)
                if res is None:
                    continue
                crop0, crop1 = res
                if not self._passes_flat_filter(crop0):
                    self.rejected_flat_count += 1
                    continue
                return _format_pair_tensors(
                    crop0=crop0,
                    crop1=crop1,
                    pair_direction=self.pair_direction,
                    is_val_or_test=self.is_val_or_test,
                    augment=self.augment,
                    augment_rot90=self.augment_rot90,
                ) + (f"{pair_id}_y{y0}_x{x0}",)
            raise RuntimeError(
                f"Failed to sample a valid in-bounds tile for split '{self.split}' after {max_tile_tries} attempts."
            )

        # Random mode (tile_index is None)
        n_records = len(self.records)
        for pair_offset in range(min(n_records, 32)):
            pair_idx = (idx + pair_offset) % n_records
            if self.cached_pairs is not None:
                pair_arr, pair_id = self.cached_pairs[pair_idx]
            else:
                pair_arr, pair_id = self._read_pair_from_disk(pair_idx)

            _, h, w = pair_arr.shape
            if h < ph or w < pw:
                continue
            if self.is_val_or_test:
                y0 = (h - ph) // 2
                x0 = (w - pw) // 2
                res = self._extract_aligned_crops(pair_arr, y0, x0)
                if res is None:
                    continue
                crop0, crop1 = res
                if not self._passes_flat_filter(crop0):
                    self.rejected_flat_count += 1
                    continue
                return _format_pair_tensors(
                    crop0=crop0,
                    crop1=crop1,
                    pair_direction=self.pair_direction,
                    is_val_or_test=self.is_val_or_test,
                    augment=self.augment,
                    augment_rot90=self.augment_rot90,
                ) + (pair_id,)

            if self.quality_filter_enabled:
                best_score = -float("inf")
                best_crops: Optional[Tuple[np.ndarray, np.ndarray]] = None
                for _ in range(16):
                    cy0 = random.randint(0, h - ph) if h > ph else 0
                    cx0 = random.randint(0, w - pw) if w > pw else 0
                    res = self._extract_aligned_crops(pair_arr, cy0, cx0)
                    if res is None:
                        continue
                    c0, c1 = res
                    if not self._passes_flat_filter(c0):
                        self.rejected_flat_count += 1
                        continue
                    score = _compute_pair_quality(
                        c0,
                        c1,
                        metric=self.quality_metric,
                        blur_sigma=self.quality_blur_sigma,
                    )
                    if score >= self.min_pair_quality:
                        best_crops = (c0, c1)
                        break
                    if score > best_score:
                        best_score = score
                        best_crops = (c0, c1)
                if best_crops is not None:
                    crop0, crop1 = best_crops
                    return _format_pair_tensors(
                        crop0=crop0,
                        crop1=crop1,
                        pair_direction=self.pair_direction,
                        is_val_or_test=self.is_val_or_test,
                        augment=self.augment,
                        augment_rot90=self.augment_rot90,
                    ) + (pair_id,)
            else:
                for _ in range(16):
                    y0 = random.randint(0, h - ph) if h > ph else 0
                    x0 = random.randint(0, w - pw) if w > pw else 0
                    res = self._extract_aligned_crops(pair_arr, y0, x0)
                    if res is None:
                        continue
                    crop0, crop1 = res
                    if not self._passes_flat_filter(crop0):
                        self.rejected_flat_count += 1
                        continue
                    return _format_pair_tensors(
                        crop0=crop0,
                        crop1=crop1,
                        pair_direction=self.pair_direction,
                        is_val_or_test=self.is_val_or_test,
                        augment=self.augment,
                        augment_rot90=self.augment_rot90,
                    ) + (pair_id,)

        raise RuntimeError(
            f"Failed to sample a valid aligned patch for split '{self.split}'."
        )


def _format_pair_tensors(
    crop0: np.ndarray,
    crop1: np.ndarray,
    pair_direction: str,
    is_val_or_test: bool,
    augment: bool,
    augment_rot90: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select input/target direction, normalize uint8 -> float32 [0, 1], and augment."""
    if pair_direction == "second_to_first":
        inp_np, tgt_np = crop1, crop0
    elif pair_direction == "first_to_second" or is_val_or_test:
        inp_np, tgt_np = crop0, crop1
    else:
        if random.random() < 0.5:
            inp_np, tgt_np = crop0, crop1
        else:
            inp_np, tgt_np = crop1, crop0

    inp_t = torch.from_numpy(inp_np.astype(np.float32) / 255.0).unsqueeze(0)
    tgt_t = torch.from_numpy(tgt_np.astype(np.float32) / 255.0).unsqueeze(0)

    if augment:
        if random.random() < 0.5:
            inp_t = torch.flip(inp_t, dims=[2])
            tgt_t = torch.flip(tgt_t, dims=[2])
        if random.random() < 0.5:
            inp_t = torch.flip(inp_t, dims=[1])
            tgt_t = torch.flip(tgt_t, dims=[1])
        if augment_rot90 and inp_t.shape[1] == inp_t.shape[2]:
            k = random.randint(0, 3)
            if k > 0:
                inp_t = torch.rot90(inp_t, k, dims=[1, 2])
                tgt_t = torch.rot90(tgt_t, k, dims=[1, 2])

    return inp_t, tgt_t


class N2NStreamingShardDataset(IterableDataset):
    """
    Constant-RAM shard-sequential streaming IterableDataset for 2M+ N2N overlap pairs.

    Reads `manifest.parquet` once in `__init__` to partition sections and list shards,
    then streams each `.tar` shard in ascending byte-offset order through a local
    shuffle buffer. Safe for multi-worker DataLoaders (`num_workers > 0`).
    """

    def __init__(
        self,
        overlaps_dir: Path | str,
        split: str = "train",
        patch_size: int | Sequence[int] = 128,
        first_sec: Optional[int] = None,
        last_sec: Optional[int] = None,
        split_strategy: str = "block",
        block_size: int = 100,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        guard_band_size: int = 5,
        split_origin: Optional[int] = None,
        orientation: str = "all",
        max_pairs: Optional[int] = None,
        pair_direction: str = "random_swap",
        augment: bool = False,
        augment_rot90: bool = False,
        seed: int = 42,
        shuffle_buffer_size: int = 2048,
    ) -> None:
        super().__init__()
        self.overlaps_dir = Path(overlaps_dir)
        self.split = split.lower().strip()
        self.patch_h, self.patch_w = normalize_hw_pair(patch_size, "patch_size")
        self.patch_size: int | Tuple[int, int] = (
            self.patch_h
            if self.patch_h == self.patch_w
            else (self.patch_h, self.patch_w)
        )
        self.pair_direction = pair_direction.lower().strip()
        self.augment = augment
        self.augment_rot90 = bool(augment_rot90)
        if self.augment_rot90 and self.patch_h != self.patch_w:
            logger.warning(
                f"augment_rot90=True is incompatible with rectangular patches "
                f"({self.patch_h}x{self.patch_w}) because 90/270-deg rotation transposes H/W; "
                "auto-disabling augment_rot90."
            )
            self.augment_rot90 = False
        self.seed = int(seed)
        self.shuffle_buffer_size = (
            0 if self.split in ("val", "test") else int(shuffle_buffer_size)
        )
        self.is_val_or_test = self.split in ("val", "test")
        self._epoch = 0

        self.manifest_path = self.overlaps_dir / "manifest.parquet"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"N2NStreamingShardDataset requires manifest.parquet at: {self.manifest_path}"
            )

        where_clauses = [
            f"m.height >= {self.patch_h}",
            f"m.width >= {self.patch_w}",
        ]
        if first_sec is not None:
            where_clauses.append(f"m.section_num >= {int(first_sec)}")
        if last_sec is not None:
            where_clauses.append(f"m.section_num <= {int(last_sec)}")
        if orientation.lower() in ("horizontal", "vertical"):
            where_clauses.append(f"lower(m.orientation) = '{orientation.lower()}'")
        self.where_sql = " AND ".join(where_clauses)
        manifest_str = str(self.manifest_path).replace("'", "''")

        con = duckdb.connect()
        sec_rows = con.execute(
            f"SELECT DISTINCT m.section_num FROM read_parquet('{manifest_str}') m "
            f"WHERE {self.where_sql} ORDER BY m.section_num"
        ).fetchall()
        all_sections = [int(r[0]) for r in sec_rows]

        split_map = _build_section_split_filter(
            sections=all_sections,
            strategy=split_strategy,
            block_size=block_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            guard_band_size=guard_band_size,
            origin=split_origin,
        )
        self.target_sections = sorted(split_map[self.split])
        if not self.target_sections:
            raise ValueError(
                f"No sections assigned to split '{self.split}' under strategy '{split_strategy}'"
            )

        con.execute("CREATE TEMP TABLE split_secs (section_num INTEGER)")
        con.executemany(
            "INSERT INTO split_secs VALUES (?)", [(s,) for s in self.target_sections]
        )
        shard_rows = con.execute(
            f"SELECT m.shard_relpath, COUNT(*) AS cnt "
            f"FROM read_parquet('{manifest_str}') m "
            f"INNER JOIN split_secs s ON m.section_num = s.section_num "
            f"WHERE {self.where_sql} "
            f"GROUP BY m.shard_relpath ORDER BY m.shard_relpath"
        ).fetchall()
        con.close()

        self.shards: List[Tuple[str, int]] = [
            (str(r[0]), int(r[1])) for r in shard_rows
        ]
        raw_total = sum(cnt for _, cnt in self.shards)
        self.max_pairs = int(max_pairs) if max_pairs is not None else None
        self.total_pairs = (
            min(raw_total, self.max_pairs) if self.max_pairs else raw_total
        )

        logger.info(
            f"N2N Streaming '{self.split}': {len(self.shards)} shards, "
            f"{self.total_pairs} pairs (raw={raw_total}, buffer={self.shuffle_buffer_size})"
        )

    def __len__(self) -> int:
        return self.total_pairs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        epoch_seed = self.seed + self._epoch
        self._epoch += 1
        rng = random.Random(epoch_seed + worker_id * 1009)

        shard_list = [s for s, _ in self.shards]
        if not self.is_val_or_test:
            rng.shuffle(shard_list)
        worker_shards = shard_list[worker_id::num_workers]
        if not worker_shards:
            return

        worker_cap: Optional[int] = None
        if self.max_pairs is not None:
            worker_cap = self.max_pairs // num_workers + (
                1 if worker_id < (self.max_pairs % num_workers) else 0
            )

        manifest_str = str(self.manifest_path).replace("'", "''")
        con = duckdb.connect()
        con.execute("CREATE TEMP TABLE split_secs (section_num INTEGER)")
        con.executemany(
            "INSERT INTO split_secs VALUES (?)", [(s,) for s in self.target_sections]
        )

        buf: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
        yielded = 0
        ph, pw = self.patch_h, self.patch_w

        try:
            for shard_rel in worker_shards:
                if worker_cap is not None and yielded >= worker_cap:
                    break
                shard_esc = shard_rel.replace("'", "''")
                rows = con.execute(
                    f"SELECT m.pair_id, m.tif_byte_offset, m.tif_byte_length "
                    f"FROM read_parquet('{manifest_str}') m "
                    f"INNER JOIN split_secs s ON m.section_num = s.section_num "
                    f"WHERE {self.where_sql} AND m.shard_relpath = '{shard_esc}' "
                    f"ORDER BY m.tif_byte_offset ASC"
                ).fetchall()

                shard_path = self.overlaps_dir / shard_rel
                with open(shard_path, "rb") as f:
                    for pair_id, offset, length in rows:
                        if worker_cap is not None and yielded + len(buf) >= worker_cap:
                            break
                        f.seek(int(offset))
                        raw_bytes = f.read(int(length))
                        arr = tifffile.imread(io.BytesIO(raw_bytes))
                        _, h, w = arr.shape
                        if h < ph or w < pw:
                            continue

                        if self.is_val_or_test:
                            y0 = (h - ph) // 2
                            x0 = (w - pw) // 2
                        else:
                            y0 = rng.randint(0, h - ph) if h > ph else 0
                            x0 = rng.randint(0, w - pw) if w > pw else 0

                        crop0 = arr[0, y0 : y0 + ph, x0 : x0 + pw]
                        crop1 = arr[1, y0 : y0 + ph, x0 : x0 + pw]
                        inp_t, tgt_t = _format_pair_tensors(
                            crop0=crop0,
                            crop1=crop1,
                            pair_direction=self.pair_direction,
                            is_val_or_test=self.is_val_or_test,
                            augment=self.augment,
                            augment_rot90=self.augment_rot90,
                        )
                        item = (inp_t, tgt_t, str(pair_id))

                        if self.shuffle_buffer_size <= 1:
                            yield item
                            yielded += 1
                        else:
                            buf.append(item)
                            if len(buf) >= self.shuffle_buffer_size:
                                pop_i = rng.randrange(len(buf))
                                buf[pop_i], buf[-1] = buf[-1], buf[pop_i]
                                yield buf.pop()
                                yielded += 1
        finally:
            con.close()

        if not self.is_val_or_test:
            rng.shuffle(buf)
        for item in buf:
            if worker_cap is not None and yielded >= worker_cap:
                break
            yield item
            yielded += 1
