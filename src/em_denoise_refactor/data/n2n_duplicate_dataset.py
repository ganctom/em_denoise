# src/em_denoise_refactor/data/n2n_duplicate_dataset.py

"""
PyTorch Dataset for Noise2Noise (N2N) training on duplicated SBEMimage tiles.

Scans a directory of raw full-resolution SBEMimage tiles matching
`..._g{G}_t{T}_s{S}.tif`, groups consecutive section numbers per tile position
into duplicate clusters, builds unordered or consecutive pairs, splits
reproducibly at the requested granularity (`group`, `image`, or `patch`),
and extracts paired patches compatible with `N2NOverlapPairDataset`.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import json
import logging
import math
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import zlib

import numpy as np
from scipy.ndimage import shift as ndi_shift
import tifffile
import torch
from torch.utils.data import Dataset, get_worker_info

from em_denoise_refactor.data.n2n_dataset import (
    _compute_pair_quality,
    _estimate_integer_shift,
    _estimate_subpixel_shift,
    _format_pair_tensors,
)

logger = logging.getLogger("train_and_infer")

_TILE_NAME_RE = re.compile(r"_g(\d+)_t(\d+)_s(\d+)\.tif$", re.IGNORECASE)


def _partition_items_by_ratio(
    items: List[Any],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    unit_name: str,
) -> Dict[str, List[Any]]:
    """Deterministically shuffle items and partition them into train/val/test."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    active_splits = sum(1 for r in (train_ratio, val_ratio, test_ratio) if r > 0)
    n_total = len(shuffled)
    if n_total < active_splits:
        raise ValueError(
            f"Need at least {active_splits} {unit_name}(s) for the requested splits, "
            f"got {n_total}."
        )

    n_train = max(1, int(round(n_total * train_ratio))) if train_ratio > 0 else 0
    n_val = max(1, int(round(n_total * val_ratio))) if val_ratio > 0 else 0
    n_test = n_total - n_train - n_val

    while test_ratio > 0 and n_test < 1:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
            n_test += 1
        elif n_val > 1:
            n_val -= 1
            n_test += 1
        else:
            raise ValueError(
                f"Cannot allocate non-empty test split from {n_total} {unit_name}(s)."
            )

    while val_ratio > 0 and n_val < 1:
        if n_train > 1:
            n_train -= 1
            n_val += 1
        else:
            raise ValueError(
                f"Cannot allocate non-empty val split from {n_total} {unit_name}(s)."
            )

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _assign_patch_split(
    pair_id: str,
    y0: int,
    x0: int,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
) -> str:
    """Deterministically assign a patch coordinate to train, val, or test."""
    key = f"{pair_id}_{y0}_{x0}_{split_seed}".encode("utf-8")
    u = (zlib.crc32(key) & 0xFFFFFFFF) / 4294967296.0
    if u < train_ratio:
        return "train"
    if u < train_ratio + val_ratio:
        return "val"
    return "test"


class N2NDuplicateTileDataset(Dataset):
    """
    PyTorch Dataset for Noise2Noise training on duplicated SBEMimage tiles.

    Returns `(input_tensor, target_tensor, pair_id)` matching `N2NOverlapPairDataset`.
    """

    def __init__(
        self,
        duplicates_dir: Path | str,
        split: str = "train",
        patch_size: int = 128,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        pair_mode: str = "all",
        split_unit: str = "group",
        split_seed: int = 42,
        pair_direction: str = "random_swap",
        max_pairs: Optional[int] = None,
        cache_in_ram: bool = False,
        max_cache_tiles: int = 32,
        augment: bool = False,
        augment_rot90: bool = False,
        seed: int = 42,
        align_patches: bool = False,
        align_subpixel: bool = False,
        align_max_shift_px: int = 16,
        align_blur_sigma: float = 1.0,
        patch_sampling: str = "random",
        tile_stride: Optional[int] = None,
        trim_enabled: bool = False,
        quality_filter_enabled: bool = False,
        quality_metric: str = "ssim",
        min_pair_quality: float = 0.80,
        quality_blur_sigma: float = 1.5,
        max_quality_attempts: int = 32,
    ) -> None:
        self.duplicates_dir = Path(duplicates_dir)
        self.split = split.lower().strip()
        self.patch_size = int(patch_size)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.pair_mode = pair_mode.lower().strip()
        self.split_unit = split_unit.lower().strip()
        self.split_seed = int(split_seed)
        self.pair_direction = pair_direction.lower().strip()
        self.max_pairs = int(max_pairs) if max_pairs is not None else None
        self.cache_in_ram = bool(cache_in_ram)
        self.max_cache_tiles = max(2, int(max_cache_tiles))
        self.augment = bool(augment)
        self.augment_rot90 = bool(augment_rot90)
        self.seed = int(seed)
        self.align_patches = bool(align_patches)
        self.align_subpixel = bool(align_subpixel)
        self.align_max_shift_px = int(align_max_shift_px)
        self.align_blur_sigma = float(align_blur_sigma)
        self.patch_sampling = patch_sampling.lower().strip()
        self.tile_stride = (
            int(tile_stride) if tile_stride is not None else self.patch_size
        )
        self.quality_filter_enabled = bool(quality_filter_enabled)
        self.quality_metric = quality_metric.lower().strip()
        self.min_pair_quality = float(min_pair_quality)
        self.quality_blur_sigma = float(quality_blur_sigma)
        self.max_quality_attempts = max(1, int(max_quality_attempts))
        self.is_val_or_test = self.split in ("val", "test")

        self._validate_args()

        if trim_enabled:
            logger.warning(
                "n2n.trim_enabled=True applies to overlaps only and is ignored for source='duplicates'."
            )

        # Stats & caches
        self._tile_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._tile_shapes: Dict[Path, Tuple[int, int]] = {}
        self._rng = random.Random(self.seed)
        self._worker_rngs: Dict[int, random.Random] = {}
        self.rejected_oob_count: int = 0
        self.rejected_quality_count: int = 0
        self.kept_quality_count: int = 0
        self._quality_scores_sum: float = 0.0
        self._corrupt_paths: Set[Path] = set()

        # Scan, group, build pairs, and split
        groups, self.excluded_singletons = self._scan_and_group()
        all_pairs = self._build_pairs(groups)
        self.split_summary: Dict[str, Any] = {
            "split_unit": self.split_unit,
            "split_seed": self.split_seed,
            "pair_mode": self.pair_mode,
            "groups": {},
        }
        self.pairs = self._apply_split(groups, all_pairs)

        # Validate tile dimensions against patch_size before training starts
        self._validate_tile_dimensions()

        self._random_epoch_len: Optional[int] = None
        if self.max_pairs is not None and self.patch_sampling == "random":
            rng = random.Random(self.seed)
            rng.shuffle(self.pairs)
            if len(self.pairs) > self.max_pairs:
                self.pairs = self.pairs[: self.max_pairs]
            else:
                self._random_epoch_len = self.max_pairs

        if self.cache_in_ram:
            self._preload_into_ram()

        self.tile_index: Optional[List[Tuple[int, int, int]]] = None
        if self.patch_sampling == "tile":
            self.tile_index = self._build_tile_index()
            if self.max_pairs is not None:
                rng = random.Random(self.seed)
                rng.shuffle(self.tile_index)
                if len(self.tile_index) > self.max_pairs:
                    self.tile_index = self.tile_index[: self.max_pairs]

        logger.info(
            f"N2N duplicates '{self.split}': {len(self.pairs)} pairs "
            f"(split_unit={self.split_unit}, pair_mode={self.pair_mode}, "
            f"patch_size={self.patch_size}, sampling={self.patch_sampling})"
        )

    def _validate_args(self) -> None:
        if not self.duplicates_dir.is_dir():
            raise FileNotFoundError(
                f"duplicates_dir does not exist: {self.duplicates_dir}"
            )
        if self.split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got '{self.split}'"
            )
        if self.pair_mode not in ("all", "neighbours"):
            raise ValueError(
                f"pair_mode must be 'all' or 'neighbours', got '{self.pair_mode}'"
            )
        if self.split_unit not in ("group", "image", "patch"):
            raise ValueError(
                f"split_unit must be 'group', 'image', or 'patch', got '{self.split_unit}'"
            )
        if self.pair_direction not in (
            "random_swap",
            "first_to_second",
            "second_to_first",
        ):
            raise ValueError(
                f"pair_direction must be 'random_swap', 'first_to_second', or "
                f"'second_to_first', got '{self.pair_direction}'"
            )
        if self.quality_metric not in ("ncc", "ssim"):
            raise ValueError(
                f"quality_metric must be 'ncc' or 'ssim', got '{self.quality_metric}'"
            )
        if self.patch_sampling not in ("random", "tile"):
            raise ValueError(
                f"patch_sampling must be 'random' or 'tile', got '{self.patch_sampling}'"
            )
        if self.tile_stride <= 0:
            raise ValueError(f"tile_stride must be > 0, got {self.tile_stride}")

    def _scan_and_group(
        self,
    ) -> Tuple[Dict[str, List[Tuple[int, Path]]], int]:
        """Scan duplicates_dir and group consecutive sections per (grid, tile)."""
        by_tile: Dict[Tuple[int, int], List[Tuple[int, Path]]] = defaultdict(list)
        skipped_files: List[str] = []

        for entry in sorted(self.duplicates_dir.iterdir()):
            if not entry.is_file() or entry.name.startswith("."):
                continue
            m = _TILE_NAME_RE.search(entry.name)
            if not m:
                skipped_files.append(entry.name)
                continue
            grid_num = int(m.group(1))
            tile_id = int(m.group(2))
            sec_num = int(m.group(3))
            by_tile[(grid_num, tile_id)].append((sec_num, entry))

        if skipped_files:
            logger.warning(
                f"Skipped {len(skipped_files)} non-matching file(s) in "
                f"{self.duplicates_dir}: {skipped_files}"
            )

        groups: Dict[str, List[Tuple[int, Path]]] = {}
        excluded_singletons = 0

        for (grid_num, tile_id), items in sorted(by_tile.items()):
            items.sort(key=lambda x: x[0])
            cluster: List[Tuple[int, Path]] = [items[0]]
            for item in items[1:]:
                if item[0] == cluster[-1][0] + 1:
                    cluster.append(item)
                else:
                    if len(cluster) >= 2:
                        gid = (
                            f"g{grid_num:04d}_t{tile_id:04d}_"
                            f"s{cluster[0][0]:05d}_s{cluster[-1][0]:05d}"
                        )
                        groups[gid] = cluster
                    else:
                        excluded_singletons += 1
                    cluster = [item]

            if len(cluster) >= 2:
                gid = (
                    f"g{grid_num:04d}_t{tile_id:04d}_"
                    f"s{cluster[0][0]:05d}_s{cluster[-1][0]:05d}"
                )
                groups[gid] = cluster
            else:
                excluded_singletons += 1

        if excluded_singletons > 0:
            logger.info(
                f"Excluded {excluded_singletons} single-section tile image(s) "
                f"without duplicates."
            )

        if not groups:
            raise ValueError(
                f"No duplicate tile groups (>=2 consecutive sections) found in "
                f"{self.duplicates_dir}"
            )

        return groups, excluded_singletons

    def _build_pairs(
        self, groups: Dict[str, List[Tuple[int, Path]]]
    ) -> List[Tuple[str, str, Path, Path, int, int]]:
        """Build (pair_id, group_id, path_a, path_b, sec_a, sec_b) records."""
        pairs: List[Tuple[str, str, Path, Path, int, int]] = []
        for gid, items in sorted(groups.items()):
            n = len(items)
            if self.pair_mode == "neighbours":
                index_pairs = [(i, i + 1) for i in range(n - 1)]
            else:
                index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

            for i, j in index_pairs:
                sec_a, path_a = items[i]
                sec_b, path_b = items[j]
                pair_id = f"{gid}_p{sec_a:05d}_{sec_b:05d}"
                pairs.append((pair_id, gid, path_a, path_b, sec_a, sec_b))
        return pairs

    def _apply_split(
        self,
        groups: Dict[str, List[Tuple[int, Path]]],
        all_pairs: List[Tuple[str, str, Path, Path, int, int]],
    ) -> List[Tuple[str, str, Path, Path, int, int]]:
        """Assign pairs to self.split according to split_unit."""
        if self.split_unit == "group":
            partition = _partition_items_by_ratio(
                sorted(groups.keys()),
                self.train_ratio,
                self.val_ratio,
                self.test_ratio,
                self.split_seed,
                unit_name="group",
            )
            for sp, gids in partition.items():
                for gid in gids:
                    self.split_summary["groups"][gid] = sp
            target_groups = set(partition[self.split])
            return [p for p in all_pairs if p[1] in target_groups]

        if self.split_unit == "image":
            logger.warning(
                "split_unit='image': validation may not be independent of training "
                "because the same tile position appears across consecutive sections."
            )
            all_images = sorted({p[2] for p in all_pairs} | {p[3] for p in all_pairs})
            partition = _partition_items_by_ratio(
                all_images,
                self.train_ratio,
                self.val_ratio,
                self.test_ratio,
                self.split_seed,
                unit_name="image",
            )
            target_images = set(partition[self.split])
            kept = [
                p for p in all_pairs if p[2] in target_images and p[3] in target_images
            ]
            if not kept:
                raise ValueError(
                    f"No duplicate pairs had both images assigned to split '{self.split}' "
                    f"under split_unit='image'."
                )
            return kept

        # split_unit == "patch"
        logger.warning(
            "split_unit='patch': validation may not be independent of training "
            "because patches are drawn from the same images."
        )
        for gid in groups:
            self.split_summary["groups"][gid] = "shared(patch)"
        return all_pairs

    def _get_tile_shape(self, path: Path) -> Tuple[int, int]:
        if path in self._tile_shapes:
            return self._tile_shapes[path]
        with tifffile.TiffFile(path) as tf:
            shape = tf.pages[0].shape
        h, w = int(shape[-2]), int(shape[-1])
        self._tile_shapes[path] = (h, w)
        return h, w

    def _validate_tile_dimensions(self) -> None:
        """Ensure tile dimensions can accommodate patch_size."""
        for _, _, path_a, _, _, _ in self.pairs:
            try:
                h, w = self._get_tile_shape(path_a)
                if h < self.patch_size or w < self.patch_size:
                    raise ValueError(
                        f"patch_size ({self.patch_size}) exceeds tile dimensions "
                        f"({h}x{w}) in {path_a.name}"
                    )
                return
            except ValueError:
                raise
            except Exception:
                continue

    def _preload_into_ram(self) -> None:
        unique_paths = sorted({p[2] for p in self.pairs} | {p[3] for p in self.pairs})
        logger.info(
            f"Preloading {len(unique_paths)} duplicate tiles for '{self.split}' into RAM..."
        )
        for p in unique_paths:
            try:
                self._load_tile(p, allow_evict=False)
            except Exception as exc:
                self._handle_corrupt_tile(p, exc)

    def _load_tile(self, path: Path, allow_evict: bool = True) -> np.ndarray:
        if path in self._tile_cache:
            self._tile_cache.move_to_end(path)
            return self._tile_cache[path]

        arr = tifffile.imread(path)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D grayscale tile for {path}, got {arr.shape}")

        self._tile_shapes[path] = (int(arr.shape[0]), int(arr.shape[1]))
        self._tile_cache[path] = arr
        if allow_evict and not self.cache_in_ram:
            while len(self._tile_cache) > self.max_cache_tiles:
                self._tile_cache.popitem(last=False)
        return arr

    def _handle_corrupt_tile(self, bad_path: Path, exc: Exception) -> None:
        logger.warning(f"Skipping corrupt/unreadable tile {bad_path}: {exc}")
        self._corrupt_paths.add(bad_path)
        self._tile_cache.pop(bad_path, None)
        kept_indices = [
            i for i, p in enumerate(self.pairs) if p[2] != bad_path and p[3] != bad_path
        ]
        old_to_new = {old_i: new_i for new_i, old_i in enumerate(kept_indices)}
        self.pairs = [self.pairs[i] for i in kept_indices]
        if self.tile_index is not None:
            self.tile_index = [
                (old_to_new[p_idx], y0, x0)
                for p_idx, y0, x0 in self.tile_index
                if p_idx in old_to_new
            ]

    def _build_tile_index(self) -> List[Tuple[int, int, int]]:
        ps = self.patch_size
        stride = self.tile_stride
        tiles: List[Tuple[int, int, int]] = []
        for pair_idx, (pair_id, _, path_a, _, _, _) in enumerate(self.pairs):
            try:
                h, w = self._get_tile_shape(path_a)
            except Exception as exc:
                self._handle_corrupt_tile(path_a, exc)
                continue
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
                    if (
                        self.split_unit == "patch"
                        and _assign_patch_split(
                            pair_id,
                            y0,
                            x0,
                            self.split_seed,
                            self.train_ratio,
                            self.val_ratio,
                        )
                        != self.split
                    ):
                        continue
                    tiles.append((pair_idx, y0, x0))
        if not tiles:
            raise ValueError(
                f"No valid {ps}x{ps} tiles generated for split '{self.split}'."
            )
        return tiles

    def _get_rng(self) -> random.Random:
        winfo = get_worker_info()
        if winfo is None:
            return self._rng
        if winfo.id not in self._worker_rngs:
            self._worker_rngs[winfo.id] = random.Random(self.seed + winfo.id * 1009)
        return self._worker_rngs[winfo.id]

    def get_split_summary(self) -> Dict[str, Any]:
        return self.split_summary

    def save_split_summary(self, path: Path | str) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.split_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return out_path

    def get_filter_stats(self) -> Dict[str, Any]:
        mean_score = (
            self._quality_scores_sum / self.kept_quality_count
            if self.kept_quality_count > 0
            else 0.0
        )
        return {
            "kept": self.kept_quality_count,
            "rejected_quality": self.rejected_quality_count,
            "rejected_oob": self.rejected_oob_count,
            "mean_score": mean_score,
        }

    def _sample_patch_for_pair(
        self,
        pair_id: str,
        img0: np.ndarray,
        img1: np.ndarray,
        rng: random.Random,
        fixed_yx: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
        h0, w0 = img0.shape
        h1, w1 = img1.shape
        h, w = min(h0, h1), min(w0, w1)
        ps = self.patch_size
        if h < ps or w < ps:
            return None

        max_attempts = 1 if fixed_yx is not None else self.max_quality_attempts
        for _ in range(max_attempts):
            if fixed_yx is not None:
                y0, x0 = fixed_yx
            else:
                y0 = rng.randint(0, h - ps) if h > ps else 0
                x0 = rng.randint(0, w - ps) if w > ps else 0
                if self.split_unit == "patch":
                    found_coord = False
                    for _ in range(64):
                        if (
                            _assign_patch_split(
                                pair_id,
                                y0,
                                x0,
                                self.split_seed,
                                self.train_ratio,
                                self.val_ratio,
                            )
                            == self.split
                        ):
                            found_coord = True
                            break
                        y0 = rng.randint(0, h - ps) if h > ps else 0
                        x0 = rng.randint(0, w - ps) if w > ps else 0
                    if not found_coord:
                        continue

            crop0 = img0[y0 : y0 + ps, x0 : x0 + ps]
            crop1 = img1[y0 : y0 + ps, x0 : x0 + ps]

            if self.align_patches:
                if self.align_subpixel:
                    dy_f, dx_f = _estimate_subpixel_shift(
                        crop0,
                        crop1,
                        max_shift=self.align_max_shift_px,
                        blur_sigma=self.align_blur_sigma,
                    )
                    pad = math.ceil(max(abs(dy_f), abs(dx_f))) + 4
                    y_lo = y0 - pad
                    y_hi = y0 + ps + pad
                    x_lo = x0 - pad
                    x_hi = x0 + ps + pad
                    if y_lo < 0 or y_hi > h1 or x_lo < 0 or x_hi > w1:
                        self.rejected_oob_count += 1
                        continue
                    img1_pad = img1[y_lo:y_hi, x_lo:x_hi].astype(np.float32)
                    shifted_pad = ndi_shift(
                        img1_pad, shift=(dy_f, dx_f), order=3, mode="nearest"
                    )
                    crop1 = np.clip(
                        shifted_pad[pad : pad + ps, pad : pad + ps], 0.0, 255.0
                    )
                else:
                    dy, dx = _estimate_integer_shift(
                        crop0,
                        crop1,
                        max_shift=self.align_max_shift_px,
                        blur_sigma=self.align_blur_sigma,
                    )
                    y1 = y0 - dy
                    x1 = x0 - dx
                    if y1 < 0 or y1 + ps > h1 or x1 < 0 or x1 + ps > w1:
                        self.rejected_oob_count += 1
                        continue
                    crop1 = img1[y1 : y1 + ps, x1 : x1 + ps]

            if self.quality_filter_enabled:
                score = _compute_pair_quality(
                    crop0,
                    crop1,
                    metric=self.quality_metric,
                    blur_sigma=self.quality_blur_sigma,
                )
                if score < self.min_pair_quality:
                    self.rejected_quality_count += 1
                    continue
                self.kept_quality_count += 1
                self._quality_scores_sum += score

            return crop0, crop1, y0, x0

        return None

    def __len__(self) -> int:
        if self.tile_index is not None:
            return len(self.tile_index)
        if self._random_epoch_len is not None and self.pairs:
            return (
                min(self._random_epoch_len, len(self.pairs))
                if self._corrupt_paths
                else self._random_epoch_len
            )
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        rng = self._get_rng()
        if not self.pairs:
            raise RuntimeError("No valid duplicate pairs remain in dataset.")

        attempts = (
            max(1, min(len(self), 256))
            if self.tile_index is not None
            else max(1, len(self.pairs))
        )
        cur_idx = idx % len(self)

        for _ in range(attempts):
            if self.tile_index is not None:
                if not self.tile_index:
                    raise RuntimeError("No valid tiles remain in dataset.")
                pair_idx, y0, x0 = self.tile_index[cur_idx % len(self.tile_index)]
                fixed_yx: Optional[Tuple[int, int]] = (y0, x0)
            else:
                pair_idx = cur_idx % len(self.pairs)
                fixed_yx = None

            pair_id, _, path_a, path_b, _, _ = self.pairs[pair_idx]

            try:
                img0 = self._load_tile(path_a)
            except Exception as exc:
                self._handle_corrupt_tile(path_a, exc)
                if not self.pairs:
                    raise RuntimeError("All pairs failed due to corrupt tiles.")
                cur_idx = rng.randrange(len(self))
                continue

            try:
                img1 = self._load_tile(path_b)
            except Exception as exc:
                self._handle_corrupt_tile(path_b, exc)
                if not self.pairs:
                    raise RuntimeError("All pairs failed due to corrupt tiles.")
                cur_idx = rng.randrange(len(self))
                continue

            sampled = self._sample_patch_for_pair(
                pair_id=pair_id,
                img0=img0,
                img1=img1,
                rng=rng,
                fixed_yx=fixed_yx,
            )
            if sampled is None:
                if fixed_yx is None:
                    logger.warning(
                        f"Skipping pair {pair_id}: no patch passed alignment/quality filter "
                        f"after {self.max_quality_attempts} attempts."
                    )
                else:
                    logger.debug(
                        f"Skipping tile {pair_id} at {fixed_yx}: failed alignment/quality filter."
                    )
                cur_idx = (cur_idx + 1) % len(self)
                continue

            crop0, crop1, sy0, sx0 = sampled
            out_id = f"{pair_id}_y{sy0}_x{sx0}"
            return _format_pair_tensors(
                crop0=crop0,
                crop1=crop1,
                pair_direction=self.pair_direction,
                is_val_or_test=self.is_val_or_test,
                augment=self.augment,
                augment_rot90=self.augment_rot90,
            ) + (out_id,)

        raise RuntimeError(
            f"Failed to sample a valid patch for split '{self.split}' after {attempts} pair attempts."
        )
