"""True 2.5D Noise2Noise overlap dataset reading directly from sbem-n2n-overlaps shards.

Produces `(input_tensor, target_tensor, sample_id)` where:
  - `input_tensor`: `(2*R + 1, H, W)` float32 in `[0, 1]` ordered strictly in ascending Z:
    `[z-R, ..., z-1, z_first, z+1, ..., z+R]` (all virgin single-exposure `page 0` slices).
  - `target_tensor`: `(1, H, W)` float32 in `[0, 1]` containing `z_second` (`page 1` of center slice `z`).

Supports `z_context_view` (`"full"`, `"center_only"`, `"center_filled"`) and `z_eval_radius`
for unconfounded same-patch 2D baselines and frozen `R=2` intersection evaluation.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, BinaryIO, Dict, List, Mapping, Optional, Sequence, Tuple

import duckdb
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndi_shift
import tifffile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from em_denoise_refactor.config import normalize_hw_pair
from em_denoise_refactor.data.n2n_dataset import (
    N2NOverlapPairDataset,
    _NpyChunkWriter,
    _build_section_split_filter,
    _compute_pair_quality,
    _compute_source_signature,
    _estimate_integer_shift,
    _estimate_subpixel_shift,
    _hash_payload,
)
from em_denoise_refactor.data.z_registration import (
    ShiftResult,
    check_tile_consistency,
    estimate_prior_centered_shift,
    filter_section_consensus,
    get_registration_constants_fingerprint,
    is_confident_lock,
    refine_patch_local_shift,
)

logger = logging.getLogger("train_and_infer")

StripeKey = Tuple[
    int, int, int, str
]  # (grid_num, anchor_tile_id, neighbor_tile_id, orientation)


@dataclass(frozen=True)
class _StripeMeta:
    pair_id: str
    section_num: int
    grid_num: int
    anchor_tile_id: int
    neighbor_tile_id: int
    orientation: str
    first_exposed_tile_id: int
    anchor_exposed_first: bool
    height: int
    width: int
    fy0: int
    fx0: int
    shard_relpath: str
    byte_offset: int
    byte_length: int

    @property
    def key(self) -> StripeKey:
        return (
            self.grid_num,
            self.anchor_tile_id,
            self.neighbor_tile_id,
            self.orientation,
        )


def assert_disjoint_z_supports(
    split_sections: Mapping[str, Sequence[int] | np.ndarray],
    radius: int,
) -> None:
    """Assert that expanded Z-support sets `{s + d : d in [-R, R]}` are pairwise disjoint.

    Turns the block-splitting guard-band guarantee into a runtime invariant across
    cached, freshly built, and superset-sliced datasets.
    """
    r = int(radius)
    supports: Dict[str, set[int]] = {}
    for split_name, secs in split_sections.items():
        if secs is None or len(secs) == 0:
            continue
        unique_secs = {int(s) for s in np.unique(np.asarray(secs, dtype=np.int64))}
        supports[split_name] = {s + d for s in unique_secs for d in range(-r, r + 1)}

    names = sorted(supports.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = supports[a] & supports[b]
            if overlap:
                sample = sorted(overlap)[:10]
                raise RuntimeError(
                    f"Z-support leakage detected between splits '{a}' and '{b}' "
                    f"(radius={r}): {len(overlap)} overlapping sections (sample={sample}). "
                    "Ensure splitting.strategy='block', splitting.origin is identical across splits, "
                    "and splitting.guard_band_size >= z_context_radius."
                )


class _ShardHandleLRU:
    """Bounded LRU cache of open binary `.tar` shard file handles."""

    def __init__(self, base_dir: Path, max_handles: int = 4) -> None:
        self.base_dir = base_dir
        self.max_handles = max(1, int(max_handles))
        self._handles: OrderedDict[str, BinaryIO] = OrderedDict()

    def read_tif(
        self, shard_relpath: str, byte_offset: int, byte_length: int
    ) -> np.ndarray:
        fh = self._handles.get(shard_relpath)
        if fh is None:
            if len(self._handles) >= self.max_handles:
                _, old_fh = self._handles.popitem(last=False)
                old_fh.close()
            fh = open(self.base_dir / shard_relpath, "rb")
            self._handles[shard_relpath] = fh
        else:
            self._handles.move_to_end(shard_relpath)
        fh.seek(byte_offset)
        raw = fh.read(byte_length)
        return tifffile.imread(io.BytesIO(raw))

    def close(self) -> None:
        for fh in self._handles.values():
            try:
                fh.close()
            except OSError:
                pass
        self._handles.clear()


def _stable_seed(base_seed: int, *parts: object) -> int:
    """Deterministic cross-process integer seed from arbitrary hashable parts."""
    payload = "|".join(str(p) for p in parts)
    return int(_hash_payload({"s": int(base_seed), "p": payload}), 16)


class N2N25DOverlapDataset(Dataset):
    """True 2.5D Noise2Noise dataset reading directly from `sbem-n2n-overlaps` shards."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        overlaps_dir: Path | str,
        split: str = "train",
        patch_size: int | Sequence[int] = 128,
        z_context_radius: int = 1,
        z_context_view: str = "full",
        z_eval_radius: Optional[int] = None,
        z_neighbor_dropout_prob: float = 0.15,
        z_flip_augment: bool = False,
        first_sec: Optional[int] = None,
        last_sec: Optional[int] = None,
        split_strategy: str = "block",
        block_size: int = 100,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        guard_band_size: int = 5,
        split_origin: int = 0,
        orientation: str = "all",
        max_pairs: Optional[int] = None,
        cache_in_ram: bool = True,
        augment: bool = False,
        augment_rot90: bool = False,
        seed: int = 42,
        align_patches: bool = True,
        align_subpixel: bool = False,
        align_max_shift_px: int = 8,
        align_blur_sigma: float = 1.0,
        tile_stride: Optional[int | Sequence[int]] = None,
        trim_enabled: bool = True,
        v_trim_top_px: int = 25,
        v_trim_left_px: int = 500,
        v_trim_bottom_px: int = 0,
        v_trim_right_px: int = 0,
        h_trim_left_px: int = 24,
        h_trim_top_px: int = 0,
        h_trim_bottom_px: int = 0,
        h_trim_right_px: int = 0,
        quality_filter_enabled: bool = True,
        quality_metric: str = "ncc",
        min_pair_quality: float = 0.60,
        quality_blur_sigma: float = 1.5,
        min_patch_std: float = 0.0,
        crop_cache_dir: Optional[Path | str] = None,
        max_tiles_per_stripe: Optional[int] = None,
        prep_workers: int = 1,
        z_align_prior_search_px: int = 32,
        z_align_min_dog_ncc: float = 0.15,
        z_align_min_psr: float = 7.5,
        z_align_high_ncc_override: float = 0.45,
        z_align_high_ncc_min_psr: float = 6.0,
        z_align_missed_cut_max_ncc: float = 0.85,
        z_align_closure_tol_px: int = 2,
        z_align_tile_consistency_tol_px: int = 2,
        z_align_consensus_tol_px: float = 12.0,
        z_align_local_refine_px: int = 0,
    ) -> None:
        self.overlaps_dir = Path(overlaps_dir)
        self.split = split.lower().strip()
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        if isinstance(z_context_radius, bool) or z_context_radius not in (1, 2):
            raise ValueError(
                f"N2N25DOverlapDataset requires z_context_radius in (1, 2), got {z_context_radius!r}"
            )
        self.z_context_radius = int(z_context_radius)
        self.num_cache_channels = (
            2 * self.z_context_radius + 2
        )  # [z-R..z_first..z+R, z_second]
        self.z_context_view = z_context_view.lower().strip()
        if self.z_context_view not in ("full", "center_only", "center_filled"):
            raise ValueError(f"Invalid z_context_view: '{z_context_view}'")
        if z_eval_radius is not None:
            if not (0 <= int(z_eval_radius) <= self.z_context_radius):
                raise ValueError(
                    f"z_eval_radius must be in [0, {self.z_context_radius}], got {z_eval_radius}"
                )
            self.eff_radius = int(z_eval_radius)
        else:
            self.eff_radius = self.z_context_radius

        if split_strategy.lower().strip() != "block":
            raise ValueError(
                f"2.5D N2N requires split_strategy='block' to prevent Z-leakage, got '{split_strategy}'"
            )
        if int(guard_band_size) < self.z_context_radius:
            raise ValueError(
                f"2.5D N2N requires guard_band_size >= {self.z_context_radius}, got {guard_band_size}"
            )
        if split_origin is None:
            raise ValueError(
                "2.5D N2N requires an explicit integer split_origin (e.g. 0)."
            )

        self.patch_h, self.patch_w = normalize_hw_pair(patch_size, "patch_size")
        self.patch_size: int | Tuple[int, int] = (
            self.patch_h
            if self.patch_h == self.patch_w
            else (self.patch_h, self.patch_w)
        )
        if tile_stride is not None:
            self.stride_h, self.stride_w = normalize_hw_pair(tile_stride, "tile_stride")
        else:
            self.stride_h, self.stride_w = self.patch_h, self.patch_w
        self.tile_stride: int | Tuple[int, int] = (
            self.stride_h
            if self.stride_h == self.stride_w
            else (self.stride_h, self.stride_w)
        )

        self.z_neighbor_dropout_prob = float(z_neighbor_dropout_prob)
        self.z_flip_augment = bool(z_flip_augment)
        self.cache_in_ram = bool(cache_in_ram)
        self.augment = bool(augment)
        self.augment_rot90 = bool(augment_rot90)
        if self.augment_rot90 and self.patch_h != self.patch_w:
            self.augment_rot90 = False
        self.is_val_or_test = self.split in ("val", "test")

        self.align_patches = bool(align_patches)
        self.align_subpixel = bool(align_subpixel)
        self.align_max_shift_px = int(align_max_shift_px)
        self.align_blur_sigma = float(align_blur_sigma)
        self.trim_enabled = bool(trim_enabled)
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
        self.quality_filter_enabled = bool(quality_filter_enabled)
        self.quality_metric = quality_metric.lower().strip()
        if self.quality_metric not in ("ncc", "ssim"):
            raise ValueError(
                f"quality_metric must be 'ncc' or 'ssim', got '{quality_metric}'"
            )
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

        self.z_align_prior_search_px = int(z_align_prior_search_px)
        self.z_align_min_dog_ncc = float(z_align_min_dog_ncc)
        self.z_align_min_psr = float(z_align_min_psr)
        self.z_align_high_ncc_override = float(z_align_high_ncc_override)
        self.z_align_high_ncc_min_psr = float(z_align_high_ncc_min_psr)
        self.z_align_missed_cut_max_ncc = float(z_align_missed_cut_max_ncc)
        self.z_align_closure_tol_px = int(z_align_closure_tol_px)
        self.z_align_tile_consistency_tol_px = int(z_align_tile_consistency_tol_px)
        self.z_align_consensus_tol_px = float(z_align_consensus_tol_px)
        self.z_align_local_refine_px = int(z_align_local_refine_px)
        self.split_origin = int(split_origin)

        self.cached_tile_crops: Optional[np.ndarray] = None
        self.cached_tile_ids: List[str] = []
        self.cached_tile_sections: np.ndarray = np.empty((0,), dtype=np.int32)
        self._mmap_crops_path: Optional[Path] = None
        self.cache_slice_indices: Optional[np.ndarray] = None
        self.exposure_bias_stats: Dict[str, Optional[float]] = {
            "mean_target_minus_input_255": None,
            "std_ratio_target_over_input": None,
        }
        self.rejection_counts: Dict[str, int] = {}

        prep_fingerprint, prep_hash, exact_key = self._build_cache_keys(
            first_sec=first_sec,
            last_sec=last_sec,
            split_strategy=split_strategy,
            block_size=block_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            guard_band_size=guard_band_size,
            split_origin=self.split_origin,
            orientation=orientation,
            max_pairs=max_pairs,
            seed=seed,
        )

        if self.crop_cache_dir is not None and self._try_load_disk_cache(
            prep_hash=prep_hash,
            exact_key=exact_key,
            first_sec=first_sec,
            last_sec=last_sec,
            max_pairs=max_pairs,
            seed=seed,
        ):
            return

        manifest_path = self.overlaps_dir / "manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"N2N25DOverlapDataset requires manifest.parquet at: {manifest_path}"
            )

        self._materialize_25d_crops(
            manifest_path=manifest_path,
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
            prep_fingerprint=prep_fingerprint,
            prep_hash=prep_hash,
            exact_key=exact_key,
        )

    def _build_cache_keys(
        self,
        first_sec: Optional[int],
        last_sec: Optional[int],
        split_strategy: str,
        block_size: int,
        train_ratio: float,
        val_ratio: float,
        guard_band_size: int,
        split_origin: int,
        orientation: str,
        max_pairs: Optional[int],
        seed: int,
    ) -> Tuple[Dict[str, Any], str, str]:
        source_sig = (
            _compute_source_signature(self.overlaps_dir)
            if self.crop_cache_dir is not None
            else {"kind": "disabled"}
        )
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
            "schema_version": self.SCHEMA_VERSION,
            "overlaps_dir": str(self.overlaps_dir.resolve()),
            "source_signature": source_sig,
            "split": self.split,
            "z_context_radius": int(self.z_context_radius),
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
            "split_origin": int(split_origin),
            "seed": int(seed),
            "max_tiles_per_stripe": (
                int(self.max_tiles_per_stripe)
                if self.max_tiles_per_stripe is not None
                else None
            ),
            # Complete 2.5D registration fingerprint (M3)
            "z_align_prior_search_px": int(self.z_align_prior_search_px),
            "z_align_min_dog_ncc": float(self.z_align_min_dog_ncc),
            "z_align_min_psr": float(self.z_align_min_psr),
            "z_align_high_ncc_override": float(self.z_align_high_ncc_override),
            "z_align_high_ncc_min_psr": float(self.z_align_high_ncc_min_psr),
            "z_align_missed_cut_max_ncc": float(self.z_align_missed_cut_max_ncc),
            "z_align_closure_tol_px": int(self.z_align_closure_tol_px),
            "z_align_tile_consistency_tol_px": int(
                self.z_align_tile_consistency_tol_px
            ),
            "z_align_consensus_tol_px": float(self.z_align_consensus_tol_px),
            "z_align_local_refine_px": int(self.z_align_local_refine_px),
            **get_registration_constants_fingerprint(),
        }
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
        crops_path = cache_dir / "crops.npy"
        all_ids = (cache_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
        crops_mmap = np.load(crops_path, mmap_mode="r")
        expected_shape = (self.num_cache_channels, self.patch_h, self.patch_w)
        if (
            crops_mmap.ndim != 4
            or crops_mmap.shape[1:] != expected_shape
            or len(all_ids) != crops_mmap.shape[0]
            or len(sections) != crops_mmap.shape[0]
        ):
            raise ValueError(
                f"Corrupt 2.5D crop cache in {cache_dir}: shape={crops_mmap.shape}, "
                f"expected (*, {expected_shape}), ids={len(all_ids)}, sections={len(sections)}"
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

        self.exposure_bias_stats = {
            "mean_target_minus_input_255": meta.get("mean_target_minus_input_255"),
            "std_ratio_target_over_input": meta.get("std_ratio_target_over_input"),
        }
        self.rejection_counts = dict(meta.get("rejection_counts", {}))

        size_gb = crops_path.stat().st_size / 1e9
        n_active = len(self.cached_tile_ids)
        mode_tag = "superset mmap slice" if is_superset else "exact mmap hit"
        sec_lo = int(self.cached_tile_sections.min()) if n_active > 0 else None
        sec_hi = int(self.cached_tile_sections.max()) if n_active > 0 else None
        logger.info(
            f"N2N-2.5D (R={self.z_context_radius}, view={self.z_context_view}) '{self.split}' "
            f"loaded disk cache ({mode_tag}) from {cache_dir}: {n_active}/{crops_mmap.shape[0]} "
            f"crops ({self.num_cache_channels}x{self.patch_h}x{self.patch_w}, {size_gb:.2f} GB), "
            f"sections=[{sec_lo}, {sec_hi}]."
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
        assert self.crop_cache_dir is not None
        if not self.crop_cache_dir.exists():
            return False

        exact_dir = (
            self.crop_cache_dir
            / f"{self.split}_25d_r{self.z_context_radius}_{exact_key}"
        )
        if self._is_valid_cache_dir(exact_dir):
            try:
                meta = json.loads(
                    (exact_dir / "metadata.json").read_text(encoding="utf-8")
                )
                if (
                    meta.get("schema_version") == self.SCHEMA_VERSION
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
                logger.warning(f"Ignoring invalid 2.5D cache at {exact_dir}: {exc}")

        best_candidate: Optional[
            Tuple[int, int, Path, np.ndarray, np.ndarray, Dict[str, Any]]
        ] = None
        for cand_dir in sorted(
            self.crop_cache_dir.glob(f"{self.split}_25d_r{self.z_context_radius}_*")
        ):
            if cand_dir == exact_dir or not self._is_valid_cache_dir(cand_dir):
                continue
            try:
                meta = json.loads(
                    (cand_dir / "metadata.json").read_text(encoding="utf-8")
                )
            except Exception:
                continue
            if (
                meta.get("schema_version") != self.SCHEMA_VERSION
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

    def _trim_box(self, orient: str, h: int, w: int) -> Tuple[int, int, int, int]:
        """Return (y0, y1, x0, x1) clean trim box for the center stripe."""
        if not self.trim_enabled:
            return 0, h, 0, w
        top, bot, left, right = self.v_trim if orient == "vertical" else self.h_trim
        return top, h - bot, left, w - right

    def _reg_margins(self, orient: str) -> Tuple[int, int, int, int]:
        """Return (mask_left, mask_right, mask_top, mask_bottom) for page-0 DoG-NCC (M7).

        Note: on vertical stripes, `v_trim_top_px` reflects top-edge flyback on `page 1`
        (bottom tile's top edge), NOT `page 0` (upper tile's bottom edge), so `mask_top`
        is only applied for horizontal stripes.
        """
        if not self.trim_enabled:
            return 0, 0, 0, 0
        if orient == "vertical":
            _, bot, left, right = self.v_trim
            return left, right, 0, bot
        top, bot, left, right = self.h_trim
        return left, right, top, bot

    def _align_target_in_trim_box(
        self, arr_c_trimmed: np.ndarray, y0_t: int, x0_t: int
    ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], bool]:
        """Exact 2D target alignment inside the clean center trim box (H1)."""
        _, h, w = arr_c_trimmed.shape
        ph, pw = self.patch_h, self.patch_w
        if y0_t < 0 or y0_t + ph > h or x0_t < 0 or x0_t + pw > w:
            return None, True
        crop0 = arr_c_trimmed[0, y0_t : y0_t + ph, x0_t : x0_t + pw]
        crop1 = arr_c_trimmed[1, y0_t : y0_t + ph, x0_t : x0_t + pw]
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
            y_lo, y_hi = y0_t - pad, y0_t + ph + pad
            x_lo, x_hi = x0_t - pad, x0_t + pw + pad
            if y_lo < 0 or y_hi > h or x_lo < 0 or x_hi > w:
                return None, True
            img1_pad = arr_c_trimmed[1, y_lo:y_hi, x_lo:x_hi].astype(np.float32)
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
        y1 = y0_t - dy
        x1 = x0_t - dx
        if y1 < 0 or y1 + ph > h or x1 < 0 or x1 + pw > w:
            return None, True
        return (crop0, arr_c_trimmed[1, y1 : y1 + ph, x1 : x1 + pw]), False

    def _is_patch_above_std(self, crop0: np.ndarray) -> bool:
        if self.min_patch_std <= 0.0:
            return True
        arr = crop0.astype(np.float32)
        if self.quality_blur_sigma > 0.0:
            arr = gaussian_filter(arr, sigma=self.quality_blur_sigma)
        return float(arr.std()) >= self.min_patch_std

    def _query_manifest_sections(
        self,
        manifest_path: Path,
        first_sec: Optional[int],
        last_sec: Optional[int],
        split_strategy: str,
        block_size: int,
        train_ratio: float,
        val_ratio: float,
        guard_band_size: int,
        orientation: str,
    ) -> Tuple[
        Dict[str, List[int]],
        Dict[int, Dict[StripeKey, _StripeMeta]],
        Dict[int, List[StripeKey]],
    ]:
        """Deduplicate manifest (H2), partition center sections (C4), and load required metadata."""
        ph, pw = self.patch_h, self.patch_w
        r = self.z_context_radius
        manifest_str = str(manifest_path).replace("'", "''")
        orient_f = orientation.lower().strip()

        con = duckdb.connect()
        try:
            # H2: Deduplicate on (section_num, grid_num, anchor_tile_id, neighbor_tile_id, orientation)
            con.execute(
                f"""
                CREATE TEMP TABLE m_dedup AS
                SELECT
                    pair_id, section_num, grid_num, anchor_tile_id, neighbor_tile_id,
                    lower(orientation) AS orientation,
                    first_exposed_tile_id, anchor_exposed_first,
                    height, width,
                    first_exposed_crop_y0 AS fy0,
                    first_exposed_crop_x0 AS fx0,
                    shard_relpath, tif_byte_offset, tif_byte_length
                FROM (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY section_num, grid_num, anchor_tile_id, neighbor_tile_id, lower(orientation)
                            ORDER BY pair_id
                        ) AS rn
                    FROM read_parquet('{manifest_str}')
                )
                WHERE rn = 1
                """
            )

            # Determine center section candidates (filtered by first_sec/last_sec and trim-aware center dims)
            if self.trim_enabled:
                vt, vb, vl, vr = self.v_trim
                ht, hb, hl, hr = self.h_trim
                dim_sql = (
                    f"(CASE WHEN c.orientation = 'vertical' THEN c.height - {vt + vb} ELSE c.height - {ht + hb} END) >= {ph} "
                    f"AND (CASE WHEN c.orientation = 'vertical' THEN c.width - {vl + vr} ELSE c.width - {hl + hr} END) >= {pw}"
                )
            else:
                dim_sql = f"c.height >= {ph} AND c.width >= {pw}"

            c_where = [dim_sql]
            if first_sec is not None:
                c_where.append(f"c.section_num >= {int(first_sec)}")
            if last_sec is not None:
                c_where.append(f"c.section_num <= {int(last_sec)}")
            if orient_f in ("horizontal", "vertical"):
                c_where.append(f"c.orientation = '{orient_f}'")
            c_where_sql = " AND ".join(c_where)

            # Build self-join across all offsets d in [-R..R]\{0} with H2 frame invariants
            join_clauses = []
            nb_dim_clauses = []
            for d in range(-r, r + 1):
                if d == 0:
                    continue
                alias = f"nb_{'m' if d < 0 else 'p'}{abs(d)}"
                join_clauses.append(
                    f"""
                    INNER JOIN m_dedup {alias}
                      ON {alias}.section_num = c.section_num + ({d})
                     AND {alias}.grid_num = c.grid_num
                     AND {alias}.anchor_tile_id = c.anchor_tile_id
                     AND {alias}.neighbor_tile_id = c.neighbor_tile_id
                     AND {alias}.orientation = c.orientation
                     AND {alias}.first_exposed_tile_id = c.first_exposed_tile_id
                     AND {alias}.anchor_exposed_first = c.anchor_exposed_first
                    """
                )
                nb_dim_clauses.append(
                    f"{alias}.height >= {ph} AND {alias}.width >= {pw}"
                )

            joins_sql = "\n".join(join_clauses)
            nb_dims_sql = " AND ".join(nb_dim_clauses)

            sec_rows = con.execute(
                f"""
                SELECT DISTINCT c.section_num
                FROM m_dedup c
                {joins_sql}
                WHERE {c_where_sql} AND {nb_dims_sql}
                ORDER BY c.section_num
                """
            ).fetchall()
            all_sections = [int(row[0]) for row in sec_rows]
            if not all_sections:
                raise ValueError(
                    f"No complete 2.5D (R={r}) overlap triplets found in {manifest_path} "
                    f"for sections [{first_sec}, {last_sec}], orientation='{orientation}', patch={ph}x{pw}."
                )

            split_map = _build_section_split_filter(
                sections=all_sections,
                strategy=split_strategy,
                block_size=block_size,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                guard_band_size=guard_band_size,
                origin=self.split_origin,
            )
            target_sections = sorted(split_map[self.split])
            if not target_sections:
                raise ValueError(
                    f"No sections assigned to split '{self.split}' under block split "
                    f"(origin={self.split_origin}, block_size={block_size}, guard={guard_band_size})."
                )

            con.execute("CREATE TEMP TABLE split_secs (section_num INTEGER)")
            con.executemany(
                "INSERT INTO split_secs VALUES (?)", [(s,) for s in target_sections]
            )

            # Fetch valid center triplet keys & their center shard
            center_rows = con.execute(
                f"""
                SELECT
                    c.section_num, c.grid_num, c.anchor_tile_id, c.neighbor_tile_id,
                    c.orientation, c.shard_relpath
                FROM m_dedup c
                INNER JOIN split_secs s ON c.section_num = s.section_num
                {joins_sql}
                WHERE {c_where_sql} AND {nb_dims_sql}
                ORDER BY c.shard_relpath, c.section_num, c.grid_num, c.anchor_tile_id, c.neighbor_tile_id, c.orientation
                """
            ).fetchall()
            if not center_rows:
                raise ValueError(
                    f"No valid 2.5D center triplets found for split '{self.split}'."
                )

            shard_to_secs: Dict[str, set[int]] = defaultdict(set)
            center_keys_by_sec: Dict[int, List[StripeKey]] = defaultdict(list)
            needed_secs: set[int] = set()
            for sec, grid, anc, nb, orient, shard_rel in center_rows:
                s_int = int(sec)
                key: StripeKey = (int(grid), int(anc), int(nb), str(orient))
                shard_to_secs[str(shard_rel)].add(s_int)
                center_keys_by_sec[s_int].append(key)
                for d in range(-r, r + 1):
                    needed_secs.add(s_int + d)

            # Fetch ALL deduplicated stripes for needed_secs (both _v and _h so tile consistency C3 works!)
            con.execute("CREATE TEMP TABLE needed_secs (section_num INTEGER)")
            con.executemany(
                "INSERT INTO needed_secs VALUES (?)",
                [(s,) for s in sorted(needed_secs)],
            )

            meta_rows = con.execute(
                """
                SELECT
                    m.pair_id, m.section_num, m.grid_num, m.anchor_tile_id, m.neighbor_tile_id,
                    m.orientation, m.first_exposed_tile_id, m.anchor_exposed_first,
                    m.height, m.width, m.fy0, m.fx0,
                    m.shard_relpath, m.tif_byte_offset, m.tif_byte_length
                FROM m_dedup m
                INNER JOIN needed_secs ns ON m.section_num = ns.section_num
                ORDER BY m.section_num, m.shard_relpath, m.tif_byte_offset
                """
            ).fetchall()
        finally:
            con.close()

        stripes_by_sec: Dict[int, Dict[StripeKey, _StripeMeta]] = defaultdict(dict)
        for row in meta_rows:
            sm = _StripeMeta(
                pair_id=str(row[0]),
                section_num=int(row[1]),
                grid_num=int(row[2]),
                anchor_tile_id=int(row[3]),
                neighbor_tile_id=int(row[4]),
                orientation=str(row[5]),
                first_exposed_tile_id=int(row[6]),
                anchor_exposed_first=bool(row[7]),
                height=int(row[8]),
                width=int(row[9]),
                fy0=int(row[10]),
                fx0=int(row[11]),
                shard_relpath=str(row[12]),
                byte_offset=int(row[13]),
                byte_length=int(row[14]),
            )
            stripes_by_sec[sm.section_num][sm.key] = sm

        shard_sections = {sh: sorted(secs) for sh, secs in shard_to_secs.items()}
        return shard_sections, stripes_by_sec, center_keys_by_sec

    def _register_single_pair(
        self,
        ref_p0: np.ndarray,
        mov_p0: np.ndarray,
        prior_dy: int,
        prior_dx: int,
        orient: str,
    ) -> ShiftResult:
        ml, mr, mt, mb = self._reg_margins(orient)
        return estimate_prior_centered_shift(
            ref=ref_p0,
            mov=mov_p0,
            prior_dy=prior_dy,
            prior_dx=prior_dx,
            search_half_px=self.z_align_prior_search_px,
            mask_left_px=ml,
            mask_right_px=mr,
            mask_top_px=mt,
            mask_bottom_px=mb,
        )

    def _materialize_25d_crops(
        self,
        manifest_path: Path,
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
        prep_fingerprint: Dict[str, Any],
        prep_hash: str,
        exact_key: str,
    ) -> None:
        r = self.z_context_radius
        ph, pw = self.patch_h, self.patch_w
        sh, sw = self.stride_h, self.stride_w
        target_cap = int(max_pairs) if max_pairs is not None else None

        shard_sections, stripes_by_sec, center_keys_by_sec = (
            self._query_manifest_sections(
                manifest_path=manifest_path,
                first_sec=first_sec,
                last_sec=last_sec,
                split_strategy=split_strategy,
                block_size=block_size,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                guard_band_size=guard_band_size,
                orientation=orientation,
            )
        )

        # Estimate disk size warning for uncapped caches (M3)
        total_center_stripes = sum(len(v) for v in center_keys_by_sec.values())
        bytes_per_crop = self.num_cache_channels * ph * pw
        if self.crop_cache_dir is not None and target_cap is None:
            est_gb = total_center_stripes * 10 * bytes_per_crop / 1e9
            if est_gb > 50.0:
                logger.warning(
                    f"Uncapped 2.5D cache (R={r}, {self.num_cache_channels}ch {ph}x{pw}) over "
                    f"{total_center_stripes} stripes may require ~{est_gb:.1f} GB on disk; "
                    "consider setting n2n.max_train_pairs."
                )

        shard_list = sorted(shard_sections.keys())
        random.Random(int(seed)).shuffle(shard_list)
        num_shards = len(shard_list)

        if target_cap is not None:
            per_stripe_cap: Optional[int] = (
                max(1, self.max_tiles_per_stripe)
                if self.max_tiles_per_stripe is not None
                else max(1, math.ceil(1.5 * target_cap / max(total_center_stripes, 1)))
            )
        else:
            per_stripe_cap = (
                max(1, self.max_tiles_per_stripe)
                if self.max_tiles_per_stripe is not None
                else None
            )

        use_disk_cache = self.crop_cache_dir is not None
        tmp_cache_dir: Optional[Path] = None
        target_cache_dir: Optional[Path] = None
        npy_writer: Optional[_NpyChunkWriter] = None
        disk_batch: List[np.ndarray] = []
        ram_buf: Optional[np.ndarray] = None
        ram_chunks: List[np.ndarray] = []

        if use_disk_cache:
            assert self.crop_cache_dir is not None
            self.crop_cache_dir.mkdir(parents=True, exist_ok=True)
            target_cache_dir = (
                self.crop_cache_dir / f"{self.split}_25d_r{r}_{exact_key}"
            )
            tmp_cache_dir = (
                self.crop_cache_dir
                / f".{self.split}_25d_r{r}_{exact_key}.tmp.{os.getpid()}"
            )
            if tmp_cache_dir.exists():
                shutil.rmtree(tmp_cache_dir, ignore_errors=True)
            tmp_cache_dir.mkdir(parents=True, exist_ok=True)
            npy_writer = _NpyChunkWriter(
                tmp_cache_dir / "crops.npy",
                patch_size=(ph, pw),
                num_channels=self.num_cache_channels,
            )
        elif target_cap is not None:
            ram_buf = np.empty(
                (target_cap, self.num_cache_channels, ph, pw), dtype=np.uint8
            )

        counters: Dict[str, int] = {
            "rejected_z_lock": 0,
            "rejected_z_tile_consistency": 0,
            "rejected_z_consensus": 0,
            "rejected_z_closure": 0,
            "rejected_missed_cut": 0,
            "rejected_z_overlap": 0,
            "rejected_quality": 0,
            "rejected_oob": 0,
            "rejected_flat": 0,
        }
        kept_ids: List[str] = []
        kept_sections: List[int] = []
        scores_kept: List[float] = []
        used_coords: Dict[Tuple[int, StripeKey], set[Tuple[int, int]]] = defaultdict(
            set
        )
        kept_count = 0
        # M2 exposure bias running accumulators
        bias_diff_sum = 0.0
        bias_c0_std_sum = 0.0
        bias_c1_std_sum = 0.0

        def _flush_disk_batch() -> None:
            if npy_writer is not None and disk_batch:
                npy_writer.append(np.stack(disk_batch, axis=0))
                disk_batch.clear()

        def _emit(
            sec_num: int,
            skey: StripeKey,
            y0: int,
            x0: int,
            stack_u8: np.ndarray,
            tile_id: str,
            score: Optional[float],
        ) -> None:
            nonlocal kept_count, bias_diff_sum, bias_c0_std_sum, bias_c1_std_sum
            if use_disk_cache:
                disk_batch.append(stack_u8)
                if len(disk_batch) >= 256:
                    _flush_disk_batch()
            elif ram_buf is not None:
                ram_buf[kept_count] = stack_u8
            else:
                ram_chunks.append(stack_u8[None, ...])

            kept_ids.append(tile_id)
            kept_sections.append(int(sec_num))
            if target_cap is not None:
                used_coords[(sec_num, skey)].add((y0, x0))
            if score is not None:
                scores_kept.append(score)
            c0_f = stack_u8[r].astype(np.float32)
            c1_f = stack_u8[-1].astype(np.float32)
            bias_diff_sum += float(np.mean(c1_f - c0_f))
            bias_c0_std_sum += float(np.std(c0_f))
            bias_c1_std_sum += float(np.std(c1_f))
            kept_count += 1

        tile_margin = 0 if not self.align_patches else (6 if self.align_subpixel else 2)
        eff_margin = max(tile_margin, self.z_align_local_refine_px)

        # Canonical hop cache persists across Pass 1 and Pass 2 (H4: center-independent verdicts!)
        # hop_cache[(s0, s1)] -> Dict[StripeKey, Tuple[Optional[ShiftResult], Optional[str]]]
        hop_cache: Dict[
            Tuple[int, int],
            Dict[StripeKey, Tuple[Optional[ShiftResult], Optional[str]]],
        ] = {}
        # Closure skip cache: skip_cache[(s_a, s_b, skey)] -> ShiftResult
        skip_cache: Dict[Tuple[int, int, StripeKey], ShiftResult] = {}

        handles = _ShardHandleLRU(self.overlaps_dir, max_handles=4)
        pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=self.prep_workers)
            if self.prep_workers > 1
            else None
        )

        # Sliding section pixel buffer: sec_buf[sec][skey] = (2, H, W) uint8 or (1, H, W)
        sec_buf: Dict[int, Dict[StripeKey, np.ndarray]] = {}

        def _ensure_section_loaded(sec: int) -> None:
            if sec in sec_buf:
                return
            meta_map = stripes_by_sec.get(sec, {})
            if not meta_map:
                sec_buf[sec] = {}
                return
            # Read in ascending (shard_relpath, byte_offset) order for sequential I/O
            ordered = sorted(
                meta_map.values(), key=lambda sm: (sm.shard_relpath, sm.byte_offset)
            )
            loaded: Dict[StripeKey, np.ndarray] = {}
            for sm in ordered:
                arr = handles.read_tif(sm.shard_relpath, sm.byte_offset, sm.byte_length)
                if arr.ndim != 3 or arr.shape[0] != 2:
                    raise ValueError(
                        f"Expected (2, H, W) TIFF for {sm.pair_id}, got shape {arr.shape}"
                    )
                loaded[sm.key] = arr
            sec_buf[sec] = loaded

        def _evict_sections_outside(keep_secs: set[int]) -> None:
            for s in list(sec_buf.keys()):
                if s not in keep_secs:
                    del sec_buf[s]

        def _compute_canonical_hop(
            s0: int, s1: int
        ) -> Dict[StripeKey, Tuple[Optional[ShiftResult], Optional[str]]]:
            """Compute canonical 1-step directed hop (s0 -> s1) ONCE deterministically (H4)."""
            assert s1 == s0 + 1
            if (s0, s1) in hop_cache:
                return hop_cache[(s0, s1)]

            _ensure_section_loaded(s0)
            _ensure_section_loaded(s1)
            m0 = stripes_by_sec.get(s0, {})
            m1 = stripes_by_sec.get(s1, {})
            b0 = sec_buf.get(s0, {})
            b1 = sec_buf.get(s1, {})

            # Deterministic sorted stripe keys sharing exact (grid, anchor, neighbor, orient) & same tile frame
            common_keys = sorted(
                k
                for k in (set(m0.keys()) & set(m1.keys()))
                if k in b0
                and k in b1
                and m0[k].first_exposed_tile_id == m1[k].first_exposed_tile_id
                and m0[k].anchor_exposed_first == m1[k].anchor_exposed_first
            )

            def _reg_job(k: StripeKey) -> Tuple[StripeKey, ShiftResult]:
                sm0, sm1 = m0[k], m1[k]
                res = self._register_single_pair(
                    ref_p0=b0[k][0],
                    mov_p0=b1[k][0],
                    prior_dy=sm1.fy0 - sm0.fy0,
                    prior_dx=sm1.fx0 - sm0.fx0,
                    orient=sm0.orientation,
                )
                return k, res

            if pool is not None and len(common_keys) > 1:
                raw_results = list(pool.map(_reg_job, common_keys))
            else:
                raw_results = [_reg_job(k) for k in common_keys]

            # Serial deterministic verdict assembly on main thread (H4)
            verdicts: Dict[StripeKey, Tuple[Optional[ShiftResult], Optional[str]]] = {}
            locked: Dict[StripeKey, ShiftResult] = {}
            for k, res in raw_results:
                if is_confident_lock(
                    res,
                    min_dog_ncc=self.z_align_min_dog_ncc,
                    min_psr=self.z_align_min_psr,
                    high_ncc_override=self.z_align_high_ncc_override,
                    high_ncc_min_psr=self.z_align_high_ncc_min_psr,
                ):
                    locked[k] = res
                else:
                    verdicts[k] = (None, "rejected_z_lock")

            # M4: Missed-cut check on section-median locked hop DoG-NCC
            if locked:
                med_ncc = float(np.median([res.dog_ncc for res in locked.values()]))
                if med_ncc > self.z_align_missed_cut_max_ncc:
                    logger.warning(
                        f"Missed-cut anomaly detected on hop s{s0}->s{s1}: "
                        f"section-median DoG-NCC={med_ncc:.3f} > {self.z_align_missed_cut_max_ncc:.3f}; "
                        f"dropping all {len(locked)} stripes on this hop."
                    )
                    for k in locked:
                        verdicts[k] = (None, "rejected_missed_cut")
                    hop_cache[(s0, s1)] = verdicts
                    return verdicts
                if med_ncc > 0.78:
                    logger.info(
                        f"High section-median hop DoG-NCC={med_ncc:.3f} on s{s0}->s{s1} "
                        f"(below missed-cut ceiling {self.z_align_missed_cut_max_ncc:.3f})."
                    )

            # C3 #3: Per-first_exposed_tile_id _v/_h physical shift consistency
            tile_phys: Dict[object, Tuple[int, int, int, int]] = {}
            for k, res in locked.items():
                sm0, sm1 = m0[k], m1[k]
                pdy = res.dy + (sm0.fy0 - sm1.fy0)
                pdx = res.dx + (sm0.fx0 - sm1.fx0)
                tile_phys[k] = (sm0.grid_num, sm0.first_exposed_tile_id, pdy, pdx)

            passed_tile = check_tile_consistency(
                tile_phys, tol_px=self.z_align_tile_consistency_tol_px
            )
            after_tile: Dict[StripeKey, Tuple[int, int]] = {}
            for k in locked:
                if k not in passed_tile:
                    verdicts[k] = (None, "rejected_z_tile_consistency")
                else:
                    _, _, pdy, pdx = tile_phys[k]
                    after_tile[k] = (pdy, pdx)

            # C3 outer sanity check: section-median consensus
            passed_cons = filter_section_consensus(
                after_tile, tol_px=self.z_align_consensus_tol_px
            )
            for k in after_tile:
                if k not in passed_cons:
                    verdicts[k] = (None, "rejected_z_consensus")
                else:
                    verdicts[k] = (locked[k], None)

            hop_cache[(s0, s1)] = verdicts
            return verdicts

        def _get_skip_shift(s_a: int, s_b: int, skey: StripeKey) -> ShiftResult:
            ck = (s_a, s_b, skey)
            if ck in skip_cache:
                return skip_cache[ck]
            sm_a = stripes_by_sec[s_a][skey]
            sm_b = stripes_by_sec[s_b][skey]
            res = self._register_single_pair(
                ref_p0=sec_buf[s_a][skey][0],
                mov_p0=sec_buf[s_b][skey][0],
                prior_dy=sm_b.fy0 - sm_a.fy0,
                prior_dx=sm_b.fx0 - sm_a.fx0,
                orient=sm_a.orientation,
            )
            skip_cache[ck] = res
            return res

        def _extract_stripe_patches(
            sec_z: int,
            skey: StripeKey,
            shifts_from_z: Dict[int, ShiftResult],
            stripe_cap: Optional[int],
            use_skip_coords: bool,
        ) -> Tuple[
            List[Tuple[int, int, np.ndarray, str, Optional[float]]], Dict[str, int]
        ]:
            """Extract aligned 2.5D patches from one validated stripe triplet/quintet."""
            local_rej = {
                "rejected_z_overlap": 0,
                "rejected_quality": 0,
                "rejected_oob": 0,
                "rejected_flat": 0,
            }
            sm_c = stripes_by_sec[sec_z][skey]
            arr_c = sec_buf[sec_z][skey]
            _, h_c, w_c = arr_c.shape

            # H1: Trim box applies strictly to center slice z (not pre-cropped off z+-d)
            c_y0, c_y1, c_x0, c_x1 = self._trim_box(sm_c.orientation, h_c, w_c)
            if c_y1 - c_y0 < ph or c_x1 - c_x0 < pw:
                local_rej["rejected_z_overlap"] += 1
                return [], local_rej

            roi_y0, roi_y1, roi_x0, roi_x1 = c_y0, c_y1, c_x0, c_x1
            nb_p0: Dict[int, np.ndarray] = {}
            for d in range(-r, r + 1):
                if d == 0:
                    continue
                p0_d = sec_buf[sec_z + d][skey][0]
                nb_p0[d] = p0_d
                hd, wd = p0_d.shape
                sd = shifts_from_z[d]
                roi_y0 = max(roi_y0, sd.dy)
                roi_y1 = min(roi_y1, hd + sd.dy)
                roi_x0 = max(roi_x0, sd.dx)
                roi_x1 = min(roi_x1, wd + sd.dx)

            roi_h = roi_y1 - roi_y0
            roi_w = roi_x1 - roi_x0
            if roi_h < ph or roi_w < pw:
                local_rej["rejected_z_overlap"] += 1
                return [], local_rej

            ys = [
                roi_y0 + y
                for y in N2NOverlapPairDataset._axis_tile_coords(
                    roi_h, ph, sh, eff_margin
                )
            ]
            xs = [
                roi_x0 + x
                for x in N2NOverlapPairDataset._axis_tile_coords(
                    roi_w, pw, sw, eff_margin
                )
            ]
            coords = [(y0, x0) for y0 in ys for x0 in xs]
            random.Random(_stable_seed(seed, "coords", sec_z, sm_c.pair_id)).shuffle(
                coords
            )

            # H1: Trimmed center view so target alignment OOB is strictly bounded by trim box
            arr_c_trimmed = arr_c[:, c_y0:c_y1, c_x0:c_x1]
            skip_set = used_coords.get((sec_z, skey)) if use_skip_coords else None

            kept_items: List[Tuple[int, int, np.ndarray, str, Optional[float]]] = []
            for y0, x0 in coords:
                if skip_set is not None and (y0, x0) in skip_set:
                    continue

                crops_pair, is_oob = self._align_target_in_trim_box(
                    arr_c_trimmed, y0 - c_y0, x0 - c_x0
                )
                if crops_pair is None:
                    if is_oob:
                        local_rej["rejected_oob"] += 1
                    continue
                crop_z0, crop_z1 = crops_pair

                if not self._is_patch_above_std(crop_z0):
                    local_rej["rejected_flat"] += 1
                    continue

                score: Optional[float] = None
                if self.quality_filter_enabled:
                    score = _compute_pair_quality(
                        crop_z0,
                        crop_z1,
                        metric=self.quality_metric,
                        blur_sigma=self.quality_blur_sigma,
                    )
                    if score < self.min_pair_quality:
                        local_rej["rejected_quality"] += 1
                        continue

                # Extract aligned integer-sliced neighbor crops (zero interpolation on inputs)
                stack_u8 = np.empty((self.num_cache_channels, ph, pw), dtype=np.uint8)
                nb_ok = True
                for idx_ch, d in enumerate(range(-r, r + 1)):
                    if d == 0:
                        stack_u8[idx_ch] = np.rint(np.clip(crop_z0, 0.0, 255.0)).astype(
                            np.uint8
                        )
                        continue
                    sd = shifts_from_z[d]
                    base_ny0 = y0 - sd.dy
                    base_nx0 = x0 - sd.dx
                    refined = refine_patch_local_shift(
                        ref_patch=crop_z0,
                        mov_full=nb_p0[d],
                        base_y0=base_ny0,
                        base_x0=base_nx0,
                        max_shift=self.z_align_local_refine_px,
                        blur_sigma=self.align_blur_sigma,
                    )
                    if refined is None:
                        nb_ok = False
                        local_rej["rejected_oob"] += 1
                        break
                    ny0, nx0 = refined
                    stack_u8[idx_ch] = nb_p0[d][ny0 : ny0 + ph, nx0 : nx0 + pw]

                if not nb_ok:
                    continue

                stack_u8[-1] = np.rint(np.clip(crop_z1, 0.0, 255.0)).astype(np.uint8)
                kept_items.append(
                    (y0, x0, stack_u8, f"{sm_c.pair_id}_y{y0}_x{x0}", score)
                )
                if stripe_cap is not None and len(kept_items) >= stripe_cap:
                    break

            return kept_items, local_rej

        def _run_pass(
            pass_name: str,
            pass_stripe_cap: Optional[int],
            use_quotas: bool,
            use_skip_coords: bool,
        ) -> None:
            pbar = tqdm(
                total=target_cap if target_cap is not None else total_center_stripes,
                desc=f"N2N-2.5D Stream ({self.split}, R={r}, {pass_name})",
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
            if target_cap is not None and kept_count > 0:
                pbar.update(kept_count)

            try:
                for s_idx, shard_rel in enumerate(shard_list):
                    if target_cap is not None and kept_count >= target_cap:
                        break
                    shard_quota: Optional[int] = None
                    if use_quotas and target_cap is not None:
                        shard_quota = target_cap // num_shards + (
                            1 if s_idx < (target_cap % num_shards) else 0
                        )
                        if shard_quota <= 0:
                            continue

                    secs_in_shard = shard_sections[shard_rel]
                    # H3: per-section sub-quota prevents exhausting quota on first sections of shard
                    sec_quota: Optional[int] = (
                        max(1, math.ceil(shard_quota / max(len(secs_in_shard), 1)))
                        if shard_quota is not None
                        else None
                    )
                    shard_kept = 0

                    for sec_z in secs_in_shard:
                        if target_cap is not None and kept_count >= target_cap:
                            break
                        if shard_quota is not None and shard_kept >= shard_quota:
                            break

                        window_secs = set(range(sec_z - r, sec_z + r + 1))
                        _evict_sections_outside(window_secs)
                        for ws in sorted(window_secs):
                            _ensure_section_loaded(ws)

                        # Compute all canonical 1-step hops needed for sec_z once (H4)
                        for s_hop in range(sec_z - r, sec_z + r):
                            _compute_canonical_hop(s_hop, s_hop + 1)

                        # Candidate center stripes on sec_z in deterministic shuffled order
                        cand_keys = list(center_keys_by_sec[sec_z])
                        random.Random(_stable_seed(seed, "sec_keys", sec_z)).shuffle(
                            cand_keys
                        )

                        # Validate hops + cycle closure per candidate stripe (serially & deterministically)
                        valid_triplets: List[
                            Tuple[StripeKey, Dict[int, ShiftResult]]
                        ] = []
                        for skey in cand_keys:
                            hop_prev_res, hop_prev_rej = hop_cache[
                                (sec_z - 1, sec_z)
                            ].get(skey, (None, "rejected_z_lock"))
                            if hop_prev_res is None:
                                if not use_skip_coords:
                                    counters[hop_prev_rej or "rejected_z_lock"] += 1
                                continue

                            hop_next_res, hop_next_rej = hop_cache[
                                (sec_z, sec_z + 1)
                            ].get(skey, (None, "rejected_z_lock"))
                            if hop_next_res is None:
                                if not use_skip_coords:
                                    counters[hop_next_rej or "rejected_z_lock"] += 1
                                continue

                            # Shifts from z: z->z-1 is -hop(z-1->z), z->z+1 is hop(z->z+1)
                            shifts_from_z: Dict[int, ShiftResult] = {
                                -1: hop_prev_res.negated(),
                                1: hop_next_res,
                            }

                            # C3 #2: Cycle closure verification
                            if r == 1:
                                skip = _get_skip_shift(sec_z - 1, sec_z + 1, skey)
                                # shift(z-1->z) + shift(z->z+1) == shift(z-1->z+1)
                                c_err = max(
                                    abs((hop_prev_res.dy + hop_next_res.dy) - skip.dy),
                                    abs((hop_prev_res.dx + hop_next_res.dx) - skip.dx),
                                )
                                if (
                                    not is_confident_lock(
                                        skip,
                                        min_dog_ncc=min(0.12, self.z_align_min_dog_ncc),
                                        min_psr=min(6.0, self.z_align_high_ncc_min_psr),
                                        high_ncc_override=self.z_align_high_ncc_override,
                                        high_ncc_min_psr=self.z_align_high_ncc_min_psr,
                                    )
                                    or c_err > self.z_align_closure_tol_px
                                ):
                                    if not use_skip_coords:
                                        counters["rejected_z_closure"] += 1
                                    continue
                            else:
                                # R == 2: chained 1-hop + direct z->z+-2 closure check
                                h_m2_res, h_m2_rej = hop_cache[
                                    (sec_z - 2, sec_z - 1)
                                ].get(skey, (None, "rejected_z_lock"))
                                if h_m2_res is None:
                                    if not use_skip_coords:
                                        counters[h_m2_rej or "rejected_z_lock"] += 1
                                    continue
                                h_p2_res, h_p2_rej = hop_cache[
                                    (sec_z + 1, sec_z + 2)
                                ].get(skey, (None, "rejected_z_lock"))
                                if h_p2_res is None:
                                    if not use_skip_coords:
                                        counters[h_p2_rej or "rejected_z_lock"] += 1
                                    continue

                                chain_m2 = ShiftResult(
                                    dy=-(hop_prev_res.dy + h_m2_res.dy),
                                    dx=-(hop_prev_res.dx + h_m2_res.dx),
                                    dog_ncc=min(hop_prev_res.dog_ncc, h_m2_res.dog_ncc),
                                    psr=min(hop_prev_res.psr, h_m2_res.psr),
                                    valid=True,
                                )
                                chain_p2 = ShiftResult(
                                    dy=hop_next_res.dy + h_p2_res.dy,
                                    dx=hop_next_res.dx + h_p2_res.dx,
                                    dog_ncc=min(hop_next_res.dog_ncc, h_p2_res.dog_ncc),
                                    psr=min(hop_next_res.psr, h_p2_res.psr),
                                    valid=True,
                                )

                                dir_m2 = _get_skip_shift(sec_z, sec_z - 2, skey)
                                dir_p2 = _get_skip_shift(sec_z, sec_z + 2, skey)
                                err_m2 = max(
                                    abs(chain_m2.dy - dir_m2.dy),
                                    abs(chain_m2.dx - dir_m2.dx),
                                )
                                err_p2 = max(
                                    abs(chain_p2.dy - dir_p2.dy),
                                    abs(chain_p2.dx - dir_p2.dx),
                                )
                                lock_m2 = is_confident_lock(
                                    dir_m2,
                                    min_dog_ncc=min(0.12, self.z_align_min_dog_ncc),
                                    min_psr=min(6.0, self.z_align_high_ncc_min_psr),
                                    high_ncc_override=self.z_align_high_ncc_override,
                                    high_ncc_min_psr=self.z_align_high_ncc_min_psr,
                                )
                                lock_p2 = is_confident_lock(
                                    dir_p2,
                                    min_dog_ncc=min(0.12, self.z_align_min_dog_ncc),
                                    min_psr=min(6.0, self.z_align_high_ncc_min_psr),
                                    high_ncc_override=self.z_align_high_ncc_override,
                                    high_ncc_min_psr=self.z_align_high_ncc_min_psr,
                                )
                                if (
                                    not lock_m2
                                    or not lock_p2
                                    or err_m2 > self.z_align_closure_tol_px
                                    or err_p2 > self.z_align_closure_tol_px
                                ):
                                    if not use_skip_coords:
                                        counters["rejected_z_closure"] += 1
                                    continue

                                shifts_from_z[-2] = chain_m2
                                shifts_from_z[2] = chain_p2

                            valid_triplets.append((skey, shifts_from_z))

                        # Extract patches (optionally in parallel across stripes, emitted in deterministic order)
                        if pool is not None and len(valid_triplets) > 1:
                            futs = [
                                pool.submit(
                                    _extract_stripe_patches,
                                    sec_z,
                                    skey,
                                    smap,
                                    pass_stripe_cap,
                                    use_skip_coords,
                                )
                                for skey, smap in valid_triplets
                            ]
                            ext_results = [f.result() for f in futs]
                        else:
                            ext_results = [
                                _extract_stripe_patches(
                                    sec_z, skey, smap, pass_stripe_cap, use_skip_coords
                                )
                                for skey, smap in valid_triplets
                            ]

                        sec_kept = 0
                        for (skey, _), (kept_list, lrej) in zip(
                            valid_triplets, ext_results
                        ):
                            for rk, rv in lrej.items():
                                counters[rk] += rv
                            for y0, x0, stack_u8, tile_id, score in kept_list:
                                if target_cap is not None and kept_count >= target_cap:
                                    break
                                if (
                                    shard_quota is not None
                                    and shard_kept >= shard_quota
                                ):
                                    break
                                if sec_quota is not None and sec_kept >= sec_quota:
                                    break
                                _emit(sec_z, skey, y0, x0, stack_u8, tile_id, score)
                                shard_kept += 1
                                sec_kept += 1
                                if target_cap is not None:
                                    pbar.update(1)
                            if (
                                (target_cap is not None and kept_count >= target_cap)
                                or (
                                    shard_quota is not None
                                    and shard_kept >= shard_quota
                                )
                                or (sec_quota is not None and sec_kept >= sec_quota)
                            ):
                                break

                        if target_cap is None:
                            pbar.update(len(cand_keys))

                    # Evict section buffers between shards
                    sec_buf.clear()
            finally:
                pbar.close()

        try:
            _run_pass(
                pass_name="pass1",
                pass_stripe_cap=per_stripe_cap,
                use_quotas=(target_cap is not None),
                use_skip_coords=False,
            )
            if target_cap is not None and kept_count < target_cap:
                logger.info(
                    f"N2N-2.5D '{self.split}' pass1 collected {kept_count}/{target_cap}; "
                    "running pass2 shortfall fallback without per-shard/per-section caps."
                )
                _run_pass(
                    pass_name="pass2_fallback",
                    pass_stripe_cap=None,
                    use_quotas=False,
                    use_skip_coords=True,
                )

            _flush_disk_batch()
            if kept_count == 0:
                raise ValueError(
                    f"No valid 2.5D (R={r}) {ph}x{pw} crops survived materialization for "
                    f"split '{self.split}' (counters={counters})."
                )

            mean_bias = bias_diff_sum / kept_count
            std_ratio = bias_c1_std_sum / max(bias_c0_std_sum, 1e-8)
            self.exposure_bias_stats = {
                "mean_target_minus_input_255": float(mean_bias),
                "std_ratio_target_over_input": float(std_ratio),
            }
            self.rejection_counts = dict(counters)

            mean_score_str = (
                f", mean_score={np.mean(scores_kept):.3f}" if scores_kept else ""
            )
            logger.info(
                f"N2N-2.5D (R={r}) '{self.split}' summary: kept={kept_count}{mean_score_str} | "
                f"exposure_bias(p1-p0)={mean_bias:+.2f} gray, std_ratio={std_ratio:.3f} | "
                f"rejections={counters}"
            )

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
                    "schema_version": self.SCHEMA_VERSION,
                    "prep_fingerprint_hash": prep_hash,
                    "prep_fingerprint": prep_fingerprint,
                    "split": self.split,
                    "z_context_radius": r,
                    "num_channels": self.num_cache_channels,
                    "first_sec": int(first_sec) if first_sec is not None else None,
                    "last_sec": int(last_sec) if last_sec is not None else None,
                    "actual_min_sec": int(sections_arr.min()),
                    "actual_max_sec": int(sections_arr.max()),
                    "max_pairs_requested": target_cap,
                    "num_crops": int(kept_count),
                    "patch_size": int(ph) if ph == pw else [int(ph), int(pw)],
                    "patch_h": int(ph),
                    "patch_w": int(pw),
                    "rejection_counts": counters,
                    "mean_quality_score": float(np.mean(scores_kept))
                    if scores_kept
                    else None,
                    "mean_target_minus_input_255": float(mean_bias),
                    "std_ratio_target_over_input": float(std_ratio),
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
                gc.collect()
                logger.info(
                    f"N2N-2.5D (R={r}) '{self.split}' materialized {kept_count} aligned "
                    f"({self.num_cache_channels}x{ph}x{pw}) crops into RAM "
                    f"({final_buf.nbytes / 1e6:.1f} MB)."
                )
        except Exception:
            if npy_writer is not None:
                npy_writer.abort()
            if tmp_cache_dir is not None and tmp_cache_dir.exists():
                shutil.rmtree(tmp_cache_dir, ignore_errors=True)
            raise
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
            handles.close()
            sec_buf.clear()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if state.get("_mmap_crops_path") is not None:
            state["cached_tile_crops"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        if self.cache_slice_indices is not None:
            return len(self.cache_slice_indices)
        return len(self.cached_tile_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.cached_tile_crops is None and self._mmap_crops_path is not None:
            self.cached_tile_crops = np.load(self._mmap_crops_path, mmap_mode="r")
        assert self.cached_tile_crops is not None

        crop_idx = (
            int(self.cache_slice_indices[idx])
            if self.cache_slice_indices is not None
            else idx
        )
        raw = self.cached_tile_crops[crop_idx]  # (2*R + 2, H, W) uint8
        r = self.z_context_radius
        eff_r = self.eff_radius

        tgt_t = torch.from_numpy(raw[-1:].astype(np.float32) / 255.0)
        if self.z_context_view == "center_only":
            inp_t = torch.from_numpy(raw[r : r + 1].astype(np.float32) / 255.0)
        elif self.z_context_view == "center_filled":
            center_f = torch.from_numpy(raw[r : r + 1].astype(np.float32) / 255.0)
            inp_t = center_f.repeat(2 * eff_r + 1, 1, 1)
        else:
            ch_lo = r - eff_r
            ch_hi = r + eff_r + 1
            inp_t = torch.from_numpy(raw[ch_lo:ch_hi].astype(np.float32) / 255.0)

        if self.augment and not self.is_val_or_test:
            # Synchronized spatial flips (preserve scan-axis noise anisotropy)
            if random.random() < 0.5:
                inp_t = torch.flip(inp_t, dims=[2])
                tgt_t = torch.flip(tgt_t, dims=[2])
            if random.random() < 0.5:
                inp_t = torch.flip(inp_t, dims=[1])
                tgt_t = torch.flip(tgt_t, dims=[1])
            if self.augment_rot90 and inp_t.shape[1] == inp_t.shape[2]:
                k = random.randint(0, 3)
                if k > 0:
                    inp_t = torch.rot90(inp_t, k, dims=[1, 2])
                    tgt_t = torch.rot90(tgt_t, k, dims=[1, 2])

            if self.z_context_view == "full" and inp_t.shape[0] > 1:
                # Optional Z-reversal (default False per M1 SBEM subsurface BSE asymmetry)
                if self.z_flip_augment and random.random() < 0.5:
                    inp_t = torch.flip(inp_t, dims=[0])

                # Z-neighbor center-fill dropout
                c_center = inp_t.shape[0] // 2
                if self.z_neighbor_dropout_prob > 0.0:
                    inp_t = inp_t.clone()
                    for ch in range(inp_t.shape[0]):
                        if (
                            ch != c_center
                            and random.random() < self.z_neighbor_dropout_prob
                        ):
                            inp_t[ch] = inp_t[c_center]
                    # For R == 2, occasionally mirror volume-boundary one-sided drop
                    if c_center == 2:
                        u = random.random()
                        if u < 0.05:
                            inp_t[0] = inp_t[c_center]
                            inp_t[1] = inp_t[c_center]
                        elif u < 0.10:
                            inp_t[3] = inp_t[c_center]
                            inp_t[4] = inp_t[c_center]

        return inp_t, tgt_t, self.cached_tile_ids[idx]
