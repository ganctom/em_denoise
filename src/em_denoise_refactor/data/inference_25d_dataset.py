"""Phase 4: Consecutive raw-tile directory 2.5D inference dataset (`Custom25DTIFFStackDataset`).

Groups `.tif`/`.tiff` files by tile template, orders sections ascending
(`[z-R, ..., z, ..., z+R]`), maintains a bounded LRU tile cache (`num_workers=0`
enforced), and performs block-wise inter-section registration with per-block
confidence + cycle closure + center-fill fallback on boundaries/isolated/damaged slices.
"""

from __future__ import annotations

from collections import OrderedDict
import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from em_denoise_refactor.data.z_registration import (
    ShiftResult,
    estimate_prior_centered_shift,
    is_confident_lock,
)

logger = logging.getLogger("train_and_infer")

_SBEM_SEC_RE = re.compile(r"^(.*_s)(\d+)(\.tiff?)$", flags=re.IGNORECASE)
_GENERIC_SEC_RE = re.compile(r"^(.*?)(\d+)(\.tiff?)$", flags=re.IGNORECASE)


def _parse_tile_and_section(fname: str) -> Tuple[Tuple[str, str], int, str]:
    """Parse `(group_key, section_num, fname)` from a raw tile filename."""
    m = _SBEM_SEC_RE.match(fname)
    if m is not None:
        return (m.group(1), m.group(3).lower()), int(m.group(2)), fname
    m2 = _GENERIC_SEC_RE.match(fname)
    if m2 is not None:
        return (m2.group(1), m2.group(3).lower()), int(m2.group(2)), fname
    return (fname, ""), 0, fname


def _to_float01(arr: np.ndarray) -> np.ndarray:
    if arr.ndim > 2:
        arr = arr[0]
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr_f = arr.astype(np.float32)
    if arr_f.max() > 1.0:
        arr_f = arr_f / 255.0
    return np.clip(arr_f, 0.0, 1.0)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim > 2:
        arr = arr[0]
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)
    if arr_f.max() <= 1.0:
        arr_f = arr_f * 255.0
    return np.rint(np.clip(arr_f, 0.0, 255.0)).astype(np.uint8)


class Custom25DTIFFStackDataset(Dataset):
    """Dataset for 2.5D inference on a directory of consecutive raw `.tif`/`.tiff` tiles."""

    def __init__(
        self,
        image_dir: Path | str,
        z_context_radius: int = 1,
        z_register: bool = True,
        z_register_max_shift_px: int = 64,
        z_register_block_size: int = 1024,
        min_dog_ncc: float = 0.15,
        min_psr: float = 7.0,
        high_ncc_override: float = 0.45,
        high_ncc_min_psr: float = 5.5,
        closure_tol_px: int = 3,
    ) -> None:
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Custom image directory not found: {self.image_dir}"
            )
        if z_context_radius not in (1, 2):
            raise ValueError(
                f"Custom25DTIFFStackDataset requires z_context_radius in (1, 2), got {z_context_radius}"
            )

        self.z_context_radius = int(z_context_radius)
        self.z_register = bool(z_register)
        self.max_shift_px = int(z_register_max_shift_px)
        self.block_size = max(256, int(z_register_block_size))
        self.min_dog_ncc = float(min_dog_ncc)
        self.min_psr = float(min_psr)
        self.high_ncc_override = float(high_ncc_override)
        self.high_ncc_min_psr = float(high_ncc_min_psr)
        self.closure_tol_px = int(closure_tol_px)

        all_files = sorted(
            [f.name for f in self.image_dir.glob("*.tif")]
            + [f.name for f in self.image_dir.glob("*.tiff")]
        )
        if not all_files:
            raise FileNotFoundError(
                f"No .tif or .tiff images found in custom directory: {self.image_dir}"
            )

        # Group by tile template and sort ascending by section_num (M1 channel convention)
        self.groups: Dict[Tuple[str, str], Dict[int, str]] = {}
        samples: List[Tuple[Tuple[str, str], int, str]] = []
        for fn in all_files:
            gkey, sec, fname = _parse_tile_and_section(fn)
            self.groups.setdefault(gkey, {})[sec] = fname
            samples.append((gkey, sec, fname))
        samples.sort(key=lambda x: (x[0], x[1], x[2]))
        self.samples = samples
        self.filenames = [s[2] for s in samples]

        # Bounded LRU raw tile cache (M6: requires num_workers=0)
        self._max_lru = 2 * self.z_context_radius + 3
        self._lru: OrderedDict[str, np.ndarray] = OrderedDict()

        # Telemetry counters
        self.stats: Dict[str, int] = {
            "blocks_total": 0,
            "blocks_locked": 0,
            "blocks_center_filled": 0,
            "missing_neighbors": 0,
        }

    def _load_raw_u8(self, fname: str) -> np.ndarray:
        arr = self._lru.get(fname)
        if arr is not None:
            self._lru.move_to_end(fname)
            return arr
        raw = tifffile.imread(str(self.image_dir / fname))
        u8 = _to_uint8(raw)
        if len(self._lru) >= self._max_lru:
            self._lru.popitem(last=False)
        self._lru[fname] = u8
        return u8

    def _reg_block(self, ref_blk: np.ndarray, mov_blk: np.ndarray) -> ShiftResult:
        h, w = ref_blk.shape
        ds = 4 if min(h, w) >= 1024 else 2
        return estimate_prior_centered_shift(
            ref=ref_blk,
            mov=mov_blk,
            prior_dy=0,
            prior_dx=0,
            search_half_px=self.max_shift_px,
            coarse_downsample=ds,
        )

    def _is_ok(self, res: ShiftResult) -> bool:
        return is_confident_lock(
            res,
            min_dog_ncc=self.min_dog_ncc,
            min_psr=self.min_psr,
            high_ncc_override=self.high_ncc_override,
            high_ncc_min_psr=self.high_ncc_min_psr,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError(
                "Custom25DTIFFStackDataset requires DataLoader(num_workers=0) to preserve "
                "its bounded LRU tile cache."
            )

        gkey, sec_z, fname_z = self.samples[idx]
        group_map = self.groups[gkey]
        r = self.z_context_radius

        img_z_u8 = self._load_raw_u8(fname_z)
        h, w = img_z_u8.shape
        img_z_f = _to_float01(img_z_u8)

        # Initialize all channels with center slice (safe fallback for missing/failed neighbors)
        stack = np.repeat(img_z_f[None, ...], 2 * r + 1, axis=0)

        # Load available neighbor images in [-R..R]\{0}
        nb_u8: Dict[int, np.ndarray] = {}
        nb_f: Dict[int, np.ndarray] = {}
        for d in range(-r, r + 1):
            if d == 0:
                continue
            nb_fname = group_map.get(sec_z + d)
            if nb_fname is None:
                self.stats["missing_neighbors"] += 1
                continue
            arr_u8 = self._load_raw_u8(nb_fname)
            if arr_u8.shape != (h, w):
                logger.warning(
                    f"Shape mismatch for neighbor {nb_fname} {arr_u8.shape} != {fname_z} {(h, w)}; "
                    "using center-fill."
                )
                self.stats["missing_neighbors"] += 1
                continue
            nb_u8[d] = arr_u8
            nb_f[d] = _to_float01(arr_u8)

        if not nb_u8:
            # Isolated single tile (e.g., 0x_test_dataset): immediate center-fill
            inp_t = torch.from_numpy(stack)
            tgt_t = torch.from_numpy(img_z_f[None, ...])
            return inp_t, tgt_t, fname_z

        if not self.z_register:
            for ch_idx, d in enumerate(range(-r, r + 1)):
                if d != 0 and d in nb_f:
                    stack[ch_idx] = nb_f[d]
            return (
                torch.from_numpy(stack),
                torch.from_numpy(img_z_f[None, ...]),
                fname_z,
            )

        # Block-wise registration (H5) with per-block closure & center-fill fallback
        pad_ctx = self.max_shift_px + 16
        bs = min(self.block_size, h, w)
        ys = list(range(0, h, bs))
        xs = list(range(0, w, bs))

        for y0 in ys:
            y1 = min(h, y0 + bs)
            for x0 in xs:
                x1 = min(w, x0 + bs)
                self.stats["blocks_total"] += 1

                # Context window for registration
                cy0, cy1 = max(0, y0 - pad_ctx), min(h, y1 + pad_ctx)
                cx0, cx1 = max(0, x0 - pad_ctx), min(w, x1 + pad_ctx)
                ref_blk = img_z_u8[cy0:cy1, cx0:cx1]

                block_shifts: Dict[int, ShiftResult] = {}
                for d in (-1, 1):
                    if d in nb_u8:
                        res = self._reg_block(ref_blk, nb_u8[d][cy0:cy1, cx0:cx1])
                        if self._is_ok(res):
                            block_shifts[d] = res

                # R=1 cycle closure when both -1 and +1 locked
                if -1 in block_shifts and 1 in block_shifts:
                    skip = self._reg_block(
                        nb_u8[-1][cy0:cy1, cx0:cx1], nb_u8[1][cy0:cy1, cx0:cx1]
                    )
                    # shift(z->z+1) - shift(z->z-1) == shift(z-1->z+1)
                    c_err = max(
                        abs((block_shifts[1].dy - block_shifts[-1].dy) - skip.dy),
                        abs((block_shifts[1].dx - block_shifts[-1].dx) - skip.dx),
                    )
                    if not self._is_ok(skip) or c_err > self.closure_tol_px:
                        block_shifts.pop(-1, None)
                        block_shifts.pop(1, None)

                # R=2 chained 1-hop registration
                if r == 2:
                    for s_mid, s_far in ((-1, -2), (1, 2)):
                        if s_mid in block_shifts and s_far in nb_u8:
                            hop2 = self._reg_block(
                                nb_u8[s_mid][cy0:cy1, cx0:cx1],
                                nb_u8[s_far][cy0:cy1, cx0:cx1],
                            )
                            if self._is_ok(hop2):
                                block_shifts[s_far] = ShiftResult(
                                    dy=block_shifts[s_mid].dy + hop2.dy,
                                    dx=block_shifts[s_mid].dx + hop2.dx,
                                    dog_ncc=min(
                                        block_shifts[s_mid].dog_ncc, hop2.dog_ncc
                                    ),
                                    psr=min(block_shifts[s_mid].psr, hop2.psr),
                                    valid=True,
                                )

                # Copy integer-shifted neighbor blocks into stack; uncovered edge pixels stay center-filled
                for ch_idx, d in enumerate(range(-r, r + 1)):
                    if d == 0 or d not in block_shifts:
                        if d != 0 and d in nb_u8:
                            self.stats["blocks_center_filled"] += 1
                        continue
                    self.stats["blocks_locked"] += 1
                    sd = block_shifts[d]
                    # dst[y, x] = src[y - dy, x - dx]
                    src_y0 = max(0, y0 - sd.dy)
                    src_y1 = min(h, y1 - sd.dy)
                    src_x0 = max(0, x0 - sd.dx)
                    src_x1 = min(w, x1 - sd.dx)
                    if src_y1 > src_y0 and src_x1 > src_x0:
                        dst_y0 = src_y0 + sd.dy
                        dst_y1 = src_y1 + sd.dy
                        dst_x0 = src_x0 + sd.dx
                        dst_x1 = src_x1 + sd.dx
                        stack[ch_idx, dst_y0:dst_y1, dst_x0:dst_x1] = nb_f[d][
                            src_y0:src_y1, src_x0:src_x1
                        ]

        return torch.from_numpy(stack), torch.from_numpy(img_z_f[None, ...]), fname_z
