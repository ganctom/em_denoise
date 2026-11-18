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
import tifffile
import torch
from torch.utils.data import Dataset
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
        self.is_val_or_test = self.split in ("val", "test")

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
        if self.patch_sampling == "tile":
            self.tile_index = self._build_tile_index()
            total_tiles = len(self.tile_index)
            if max_pairs is not None and total_tiles > int(max_pairs):
                rng = random.Random(seed)
                rng.shuffle(self.tile_index)
                self.tile_index = self.tile_index[: int(max_pairs)]
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

    def _preload_into_ram(self) -> None:
        """Read selected pairs grouped by shard in ascending byte-offset order."""
        logger.info(
            f"Caching {len(self.records)} N2N '{self.split}' overlap pairs into RAM..."
        )
        shard_groups: Dict[str, List[Tuple[int, int, int, str]]] = defaultdict(list)
        for idx, (pair_id, shard_rel, offset, length, _, _) in enumerate(self.records):
            shard_groups[shard_rel].append((offset, length, idx, pair_id))

        cached: List[Optional[Tuple[np.ndarray, str]]] = [None] * len(self.records)

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
                    cached[idx] = (arr, pair_id)
                    pbar.update(1)
        pbar.close()

        self.cached_pairs = [c for c in cached if c is not None]

    def _read_pair_from_disk(self, idx: int) -> Tuple[np.ndarray, str]:
        pair_id, shard_rel, offset, length, _, _ = self.records[idx]
        shard_path = self.overlaps_dir / shard_rel
        with open(shard_path, "rb") as f:
            f.seek(offset)
            raw_bytes = f.read(length)
        arr = tifffile.imread(io.BytesIO(raw_bytes))
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

    def __len__(self) -> int:
        if self.tile_index is not None:
            return len(self.tile_index)
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
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
            else:
                y0 = random.randint(0, h - ps) if h > ps else 0
                x0 = random.randint(0, w - ps) if w > ps else 0
        else:
            pair_id = f"{pair_id}_y{y0}_x{x0}"

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

        # Select input/target direction
        if self.pair_direction == "second_to_first":
            inp_np, tgt_np = crop1, crop0
        elif self.pair_direction == "first_to_second" or self.is_val_or_test:
            inp_np, tgt_np = crop0, crop1
        else:
            # random_swap during training
            if random.random() < 0.5:
                inp_np, tgt_np = crop0, crop1
            else:
                inp_np, tgt_np = crop1, crop0

        inp_t = torch.from_numpy(inp_np.astype(np.float32) / 255.0).unsqueeze(0)
        tgt_t = torch.from_numpy(tgt_np.astype(np.float32) / 255.0).unsqueeze(0)

        if self.augment:
            if random.random() < 0.5:
                inp_t = torch.flip(inp_t, dims=[2])
                tgt_t = torch.flip(tgt_t, dims=[2])
            if random.random() < 0.5:
                inp_t = torch.flip(inp_t, dims=[1])
                tgt_t = torch.flip(tgt_t, dims=[1])
            if self.augment_rot90:
                k = random.randint(0, 3)
                if k > 0:
                    inp_t = torch.rot90(inp_t, k, dims=[1, 2])
                    tgt_t = torch.rot90(tgt_t, k, dims=[1, 2])

        return inp_t, tgt_t, pair_id
