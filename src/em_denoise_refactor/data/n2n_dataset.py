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

import gc
import io
import logging
import random
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
import tifffile
import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

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
) -> Dict[str, set[int]]:
    """
    Partition sorted section numbers into train/val/test sets without Z-leakage.

    Matches the splitting semantics of `SplittingConfig` in `config.py`.
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
    min_s = sorted_secs[0]
    train_cut = train_ratio
    val_cut = train_ratio + val_ratio

    for s in sorted_secs:
        offset = s - min_s
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
    patch_size : int
        Square crop size `(patch_size, patch_size)` extracted from `(H_ov, W_ov)`.
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
        patch_size: int = 128,
        first_sec: Optional[int] = None,
        last_sec: Optional[int] = None,
        split_strategy: str = "block",
        block_size: int = 100,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        guard_band_size: int = 5,
        orientation: str = "all",
        max_pairs: Optional[int] = None,
        pair_direction: str = "random_swap",
        cache_in_ram: bool = True,
        augment: bool = False,
        augment_rot90: bool = False,
        seed: int = 42,
        align_patches: bool = False,
        align_max_shift_px: int = 16,
        align_blur_sigma: float = 1.0,
        patch_sampling: str = "random",
        tile_stride: Optional[int] = None,
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
    ) -> None:
        self.overlaps_dir = Path(overlaps_dir)
        self.split = split.lower().strip()
        self.patch_size = patch_size
        self.pair_direction = pair_direction.lower().strip()
        self.cache_in_ram = cache_in_ram
        self.augment = augment
        self.augment_rot90 = augment_rot90
        self.align_patches = align_patches
        self.align_max_shift_px = align_max_shift_px
        self.align_blur_sigma = align_blur_sigma
        self.patch_sampling = patch_sampling.lower().strip()
        self.tile_stride = tile_stride if tile_stride is not None else patch_size
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
        self.is_val_or_test = self.split in ("val", "test")

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
        if self.tile_stride <= 0:
            raise ValueError(f"tile_stride must be > 0, got {self.tile_stride}")

        manifest_path = self.overlaps_dir / "manifest.parquet"
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
                orientation=orientation,
                max_pairs=max_pairs,
                seed=seed,
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
                orientation=orientation,
                max_pairs=max_pairs,
                seed=seed,
            )

        self.cached_pairs: Optional[List[Tuple[np.ndarray, str]]] = None
        if self.cache_in_ram:
            self._preload_into_ram()

        self.tile_index: Optional[List[Tuple[int, int, int]]] = None
        self.cached_tile_crops: Optional[np.ndarray] = None
        self.cached_tile_ids: Optional[List[str]] = None
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
                self._materialize_tile_crops(self.tile_index)

            logger.info(
                f"N2N '{self.split}' tile mode: {len(self.records)} stripes -> "
                f"{len(self.tile_index)}/{total_tiles} patches ({self.patch_size}x{self.patch_size}, stride={self.tile_stride})"
            )

    def _load_records_from_manifest(
        self,
        manifest_path: Path,
        patch_size: int,
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
    ) -> List[Tuple[str, str, int, int, int, int]]:
        """Query manifest.parquet via DuckDB and return byte-offset records."""
        con = duckdb.connect()
        if self.trim_enabled:
            vt, vb, vl, vr = self.v_trim
            ht, hb, hl, hr = self.h_trim
            where_clauses = [
                f"(CASE WHEN lower(m.orientation) = 'vertical' "
                f"THEN m.height - {vt + vb} ELSE m.height - {ht + hb} END) >= {int(patch_size)}",
                f"(CASE WHEN lower(m.orientation) = 'vertical' "
                f"THEN m.width - {vl + vr} ELSE m.width - {hl + hr} END) >= {int(patch_size)}",
            ]
        else:
            where_clauses = [
                f"m.height >= {int(patch_size)}",
                f"m.width >= {int(patch_size)}",
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

        limit_sql = f"LIMIT {int(max_pairs)}" if max_pairs is not None else ""
        query = f"""
            SELECT
                m.pair_id,
                m.shard_relpath,
                m.tif_byte_offset,
                m.tif_byte_length,
                m.height,
                m.width
            FROM read_parquet('{manifest_str}') m
            INNER JOIN split_secs s ON m.section_num = s.section_num
            WHERE {where_sql}
            ORDER BY hash(m.pair_id || '_{int(seed)}')
            {limit_sql}
        """
        rows = con.execute(query).fetchall()
        con.close()

        if not rows:
            raise ValueError(
                f"No overlap pairs matched criteria for split '{self.split}' in {manifest_path}"
            )

        return [
            (str(r[0]), str(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5]))
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
    ) -> List[Tuple[str, str, int, int, int, int]]:
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
        )
        target_sections = split_map[self.split]
        if not target_sections:
            raise ValueError(
                f"No sections assigned to split '{self.split}' under strategy "
                f"'{split_strategy}' in {self.overlaps_dir}"
            )

        filtered = [
            (pair_id, rel_path, offset, length, 0, 0)
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
        for idx, (pair_id, shard_rel, offset, length, _, _) in enumerate(self.records):
            shard_groups[shard_rel].append((offset, length, idx, pair_id))

        cached: List[Optional[Tuple[np.ndarray, str]]] = [None] * len(self.records)
        kept_counts = {"_h": 0, "_v": 0}
        dropped_counts = {"_h": 0, "_v": 0}
        ps = self.patch_size

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
                    if arr.shape[1] < ps or arr.shape[2] < ps:
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
                f"No pairs remained after ROI trimming (patch_size={ps}) for split '{self.split}'"
            )
        self.records = [rec for rec, _ in valid_pairs]
        self.cached_pairs = [c for _, c in valid_pairs]

        if self.trim_enabled:
            logger.info(
                f"N2N '{self.split}' ROI trim summary (patch_size={ps}): "
                f"_h kept={kept_counts['_h']} dropped={dropped_counts['_h']} | "
                f"_v kept={kept_counts['_v']} dropped={dropped_counts['_v']}"
            )

    def _read_pair_from_disk(self, idx: int) -> Tuple[np.ndarray, str]:
        pair_id, shard_rel, offset, length, _, _ = self.records[idx]
        shard_path = self.overlaps_dir / shard_rel
        with open(shard_path, "rb") as f:
            f.seek(offset)
            raw_bytes = f.read(length)
        arr = tifffile.imread(io.BytesIO(raw_bytes))
        arr = self._trim_pair(arr, pair_id)
        return arr, pair_id

    def _build_tile_index(self) -> List[Tuple[int, int, int]]:
        """Generate (pair_idx, y0, x0) grid coordinates covering 100% of each stripe."""
        ps = self.patch_size
        stride = self.tile_stride
        tiles: List[Tuple[int, int, int]] = []

        for pair_idx in range(len(self.records)):
            if self.cached_pairs is not None:
                _, h, w = self.cached_pairs[pair_idx][0].shape
            else:
                _, _, _, _, h, w = self.records[pair_idx]
                if h == 0 or w == 0:
                    arr, _ = self._read_pair_from_disk(pair_idx)
                    _, h, w = arr.shape

            if h < ps or w < ps:
                continue

            ys = list(range(0, h - ps + 1, stride))
            if ys[-1] != h - ps:
                ys.append(h - ps)

            xs = list(range(0, w - ps + 1, stride))
            if xs[-1] != w - ps:
                xs.append(w - ps)

            for y0 in ys:
                for x0 in xs:
                    tiles.append((pair_idx, y0, x0))

        if not tiles:
            raise ValueError(
                f"No valid {ps}x{ps} tiles generated for split '{self.split}'"
            )
        return tiles

    def _extract_aligned_crops(
        self, pair_arr: np.ndarray, y0: int, x0: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract (ps, ps) crops at (y0, x0) and optionally align crop1 via integer shift."""
        _, h, w = pair_arr.shape
        ps = self.patch_size
        crop0 = pair_arr[0, y0 : y0 + ps, x0 : x0 + ps]
        crop1 = pair_arr[1, y0 : y0 + ps, x0 : x0 + ps]

        if self.align_patches:
            dy, dx = _estimate_integer_shift(
                crop0,
                crop1,
                max_shift=self.align_max_shift_px,
                blur_sigma=self.align_blur_sigma,
            )
            if dy != 0 or dx != 0:
                y1 = min(max(0, y0 - dy), h - ps)
                x1 = min(max(0, x0 - dx), w - ps)
                crop1 = pair_arr[1, y1 : y1 + ps, x1 : x1 + ps]
        return crop0, crop1

    def _store_materialized_crops(
        self, crops: List[Tuple[np.ndarray, np.ndarray, str]]
    ) -> None:
        """Pack pre-aligned (ps, ps) uint8 crop pairs into a compact array and drop full stripes."""
        n = len(crops)
        ps = self.patch_size
        buf = np.empty((n, 2, ps, ps), dtype=np.uint8)
        ids: List[str] = []
        for i, (c0, c1, pid) in enumerate(crops):
            buf[i, 0] = c0
            buf[i, 1] = c1
            ids.append(pid)
        self.cached_tile_crops = buf
        self.cached_tile_ids = ids
        self.cached_pairs = None
        gc.collect()
        logger.info(
            f"N2N '{self.split}' materialized {n} aligned ({ps}x{ps}) tile crops "
            f"({buf.nbytes / 1e6:.1f} MB) and freed full stripe cache."
        )

    def _materialize_tile_crops(self, tiles: List[Tuple[int, int, int]]) -> None:
        """Extract and align crops once for all selected tiles, then free full stripes."""
        crops: List[Tuple[np.ndarray, np.ndarray, str]] = []
        for pair_idx, y0, x0 in tiles:
            if self.cached_pairs is not None:
                pair_arr, pair_id = self.cached_pairs[pair_idx]
            else:
                pair_arr, pair_id = self._read_pair_from_disk(pair_idx)
            crop0, crop1 = self._extract_aligned_crops(pair_arr, y0, x0)
            crops.append((crop0, crop1, f"{pair_id}_y{y0}_x{x0}"))
        self._store_materialized_crops(crops)

    def _filter_tiles_by_quality(
        self,
        candidates: List[Tuple[int, int, int]],
        max_pairs: Optional[int] = None,
    ) -> List[Tuple[int, int, int]]:
        """Filter candidate tiles by smoothed NCC/SSIM, stopping once max_pairs is reached."""
        kept: List[Tuple[int, int, int]] = []
        kept_crops: List[Tuple[np.ndarray, np.ndarray, str]] = []
        rejected = 0
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

            crop0, crop1 = self._extract_aligned_crops(pair_arr, y0, x0)
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
                f"No tiles passed quality filter ({self.quality_metric} >= {self.min_pair_quality}) "
                f"for split '{self.split}' (rejected {rejected} tiles)."
            )

        logger.info(
            f"N2N '{self.split}' quality filter ({self.quality_metric} >= {self.min_pair_quality:.2f}, "
            f"sigma={self.quality_blur_sigma}): kept={len(kept)}, rejected={rejected}, "
            f"mean_score={np.mean(scores_kept):.3f}"
        )
        if self.cache_in_ram:
            self._store_materialized_crops(kept_crops)
        return kept

    def __len__(self) -> int:
        if self.cached_tile_ids is not None:
            return len(self.cached_tile_ids)
        if self.tile_index is not None:
            return len(self.tile_index)
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.cached_tile_crops is not None and self.cached_tile_ids is not None:
            return _format_pair_tensors(
                crop0=self.cached_tile_crops[idx, 0],
                crop1=self.cached_tile_crops[idx, 1],
                pair_direction=self.pair_direction,
                is_val_or_test=self.is_val_or_test,
                augment=self.augment,
                augment_rot90=self.augment_rot90,
            ) + (self.cached_tile_ids[idx],)

        if self.tile_index is not None:
            pair_idx, y0, x0 = self.tile_index[idx]
        else:
            pair_idx = idx
            y0, x0 = -1, -1

        if self.cached_pairs is not None:
            pair_arr, pair_id = self.cached_pairs[pair_idx]
        else:
            pair_arr, pair_id = self._read_pair_from_disk(pair_idx)

        _, h, w = pair_arr.shape
        ps = self.patch_size

        if self.tile_index is None:
            if self.is_val_or_test:
                y0 = (h - ps) // 2
                x0 = (w - ps) // 2
                crop0, crop1 = self._extract_aligned_crops(pair_arr, y0, x0)
            elif self.quality_filter_enabled:
                best_score = -float("inf")
                best_crops: Optional[Tuple[np.ndarray, np.ndarray]] = None
                for _ in range(8):
                    cy0 = random.randint(0, h - ps) if h > ps else 0
                    cx0 = random.randint(0, w - ps) if w > ps else 0
                    c0, c1 = self._extract_aligned_crops(pair_arr, cy0, cx0)
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
                crop0, crop1 = best_crops  # type: ignore[misc]
            else:
                y0 = random.randint(0, h - ps) if h > ps else 0
                x0 = random.randint(0, w - ps) if w > ps else 0
                crop0, crop1 = self._extract_aligned_crops(pair_arr, y0, x0)
        else:
            pair_id = f"{pair_id}_y{y0}_x{x0}"
            crop0, crop1 = self._extract_aligned_crops(pair_arr, y0, x0)

        return _format_pair_tensors(
            crop0=crop0,
            crop1=crop1,
            pair_direction=self.pair_direction,
            is_val_or_test=self.is_val_or_test,
            augment=self.augment,
            augment_rot90=self.augment_rot90,
        ) + (pair_id,)


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
        if augment_rot90:
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
        patch_size: int = 128,
        first_sec: Optional[int] = None,
        last_sec: Optional[int] = None,
        split_strategy: str = "block",
        block_size: int = 100,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        guard_band_size: int = 5,
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
        self.patch_size = patch_size
        self.pair_direction = pair_direction.lower().strip()
        self.augment = augment
        self.augment_rot90 = augment_rot90
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
            f"m.height >= {int(patch_size)}",
            f"m.width >= {int(patch_size)}",
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
        ps = self.patch_size

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
                        if h < ps or w < ps:
                            continue

                        if self.is_val_or_test:
                            y0 = (h - ps) // 2
                            x0 = (w - ps) // 2
                        else:
                            y0 = rng.randint(0, h - ps) if h > ps else 0
                            x0 = rng.randint(0, w - ps) if w > ps else 0

                        crop0 = arr[0, y0 : y0 + ps, x0 : x0 + ps]
                        crop1 = arr[1, y0 : y0 + ps, x0 : x0 + ps]
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
