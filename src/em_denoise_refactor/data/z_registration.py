"""Shared inter-section registration and geometric consistency utilities for 2.5D N2N."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.ndimage import gaussian_filter

from em_denoise_refactor.data.n2n_dataset import _estimate_integer_shift

# Bump whenever the registration algorithm or default filter constants change
# so persistent 2.5D crop caches never silently reuse stale alignments.
REGISTRATION_ALGO_VERSION = 1

DOG_LOW_SIGMA = 1.5
DOG_HIGH_SIGMA = 16.0
COARSE_DOWNSAMPLE = 2
FINE_PAD_PX = 6
FINE_SEARCH_PX = 5
MIN_OVERLAP_PX = 48
CONSENSUS_MIN_SAMPLES = 5


@dataclass(frozen=True)
class ShiftResult:
    """Integer inter-section registration result.

    Coordinate convention
    ---------------------
    `ref[y, x]` corresponds to `mov[y - dy, x - dx]`.
    Equivalently, shifting `mov` by `(+dy, +dx)` aligns `mov` onto `ref`, and
    reversing direction `(mov -> ref)` is exact negation `(-dy, -dx)`.

    Metrics
    -------
    - `psr`: Peak-to-Sidelobe Ratio computed strictly on the coarse-stage fixed
      lag window (portable across stripes with identical search window).
    - `dog_ncc`: Difference-of-Gaussians normalized cross-correlation from the
      fine 1x stage at `(dy, dx)`.
    """

    dy: int
    dx: int
    dog_ncc: float
    psr: float
    valid: bool

    def negated(self) -> "ShiftResult":
        """Return exact reverse-direction shift (mov -> ref)."""
        return ShiftResult(
            dy=-self.dy,
            dx=-self.dx,
            dog_ncc=self.dog_ncc,
            psr=self.psr,
            valid=self.valid,
        )


def _make_integral(arr: np.ndarray) -> np.ndarray:
    """Compute 2D prefix sum table with leading zero row/column."""
    h, w = arr.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    np.cumsum(arr, axis=0, out=ii[1:, 1:])
    np.cumsum(ii[1:, 1:], axis=1, out=ii[1:, 1:])
    return ii


def _box_sum_integral(
    ii: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
) -> np.ndarray:
    """Evaluate exact 2D box sums over broadcast grids of (y0, y1) and (x0, x1)."""
    return (
        ii[y1[:, None], x1[None, :]]
        - ii[y0[:, None], x1[None, :]]
        - ii[y1[:, None], x0[None, :]]
        + ii[y0[:, None], x0[None, :]]
    )


def _prepare_masked_dog(
    img: np.ndarray,
    mask_left_px: int = 0,
    mask_right_px: int = 0,
    mask_top_px: int = 0,
    mask_bottom_px: int = 0,
    downsample: int = 1,
    dog_low: float = DOG_LOW_SIGMA,
    dog_high: float = DOG_HIGH_SIGMA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute DoG-bandpassed masked float32 image and 2D integral tables."""
    arr = img.astype(np.float32)
    h0, w0 = arr.shape
    mask = np.ones((h0, w0), dtype=np.float32)

    if mask_left_px > 0 and mask_left_px < w0 - MIN_OVERLAP_PX:
        mask[:, :mask_left_px] = 0.0
    if mask_right_px > 0 and mask_right_px < w0 - MIN_OVERLAP_PX:
        mask[:, w0 - mask_right_px :] = 0.0
    if mask_top_px > 0 and mask_top_px < h0 - MIN_OVERLAP_PX:
        mask[:mask_top_px, :] = 0.0
    if mask_bottom_px > 0 and mask_bottom_px < h0 - MIN_OVERLAP_PX:
        mask[h0 - mask_bottom_px :, :] = 0.0

    if downsample > 1:
        arr = gaussian_filter(arr, sigma=downsample * 0.7)[::downsample, ::downsample]
        mask = mask[::downsample, ::downsample]
        s_low = max(0.8, dog_low / downsample)
        s_high = max(3.0, dog_high / downsample)
    else:
        s_low = dog_low
        s_high = dog_high

    bp = (gaussian_filter(arr, sigma=s_low) - gaussian_filter(arr, sigma=s_high)) * mask
    ii_sq = _make_integral(bp * bp)
    ii_m = _make_integral(mask)
    return bp, ii_sq, ii_m


def masked_dog_ncc_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    center_dy: int = 0,
    center_dx: int = 0,
    half_dy: int = 32,
    half_dx: int = 32,
    mask_left_px: int = 0,
    mask_right_px: int = 0,
    mask_top_px: int = 0,
    mask_bottom_px: int = 0,
    min_overlap_h: int = MIN_OVERLAP_PX,
    min_overlap_w: int = MIN_OVERLAP_PX,
    downsample: int = 1,
    dog_low: float = DOG_LOW_SIGMA,
    dog_high: float = DOG_HIGH_SIGMA,
) -> ShiftResult:
    """Exact zero-padded masked DoG-NCC shift over a fixed lag box.

    Uses a single `rfft2`/`irfft2` pair plus O(1) 2D integral-image energy
    normalization to eliminate circular FFT wrap-around bias.
    """
    a_bp, a_sq, a_m = _prepare_masked_dog(
        ref,
        mask_left_px=mask_left_px,
        mask_right_px=mask_right_px,
        mask_top_px=mask_top_px,
        mask_bottom_px=mask_bottom_px,
        downsample=downsample,
        dog_low=dog_low,
        dog_high=dog_high,
    )
    b_bp, b_sq, b_m = _prepare_masked_dog(
        mov,
        mask_left_px=mask_left_px,
        mask_right_px=mask_right_px,
        mask_top_px=mask_top_px,
        mask_bottom_px=mask_bottom_px,
        downsample=downsample,
        dog_low=dog_low,
        dog_high=dog_high,
    )

    if downsample > 1:
        c_dy = int(round(center_dy / downsample))
        c_dx = int(round(center_dx / downsample))
        h_dy = max(1, half_dy // downsample)
        h_dx = max(1, half_dx // downsample)
        min_oh = max(16, min_overlap_h // downsample)
        min_ow = max(16, min_overlap_w // downsample)
        excl = 3
    else:
        c_dy = int(center_dy)
        c_dx = int(center_dx)
        h_dy = max(1, int(half_dy))
        h_dx = max(1, int(half_dx))
        min_oh = int(min_overlap_h)
        min_ow = int(min_overlap_w)
        excl = 5

    ha, wa = a_bp.shape
    hb, wb = b_bp.shape
    fh = next_fast_len(ha + hb - 1)
    fw = next_fast_len(wa + wb - 1)

    cc = irfft2(rfft2(a_bp, s=(fh, fw)) * np.conj(rfft2(b_bp, s=(fh, fw))), s=(fh, fw))

    dys = np.arange(c_dy - h_dy, c_dy + h_dy + 1)
    dxs = np.arange(c_dx - h_dx, c_dx + h_dx + 1)
    num_roi = cc[np.ix_(dys % fh, dxs % fw)]

    ay0 = np.maximum(0, dys)
    ay1 = np.minimum(ha, hb + dys)
    ax0 = np.maximum(0, dxs)
    ax1 = np.minimum(wa, wb + dxs)

    by0 = np.maximum(0, -dys)
    by1 = np.minimum(hb, ha - dys)
    bx0 = np.maximum(0, -dxs)
    bx1 = np.minimum(wb, wa - dxs)

    ov_h = np.maximum(0, ay1 - ay0)
    ov_w = np.maximum(0, ax1 - ax0)
    valid = (ov_h[:, None] >= min_oh) & (ov_w[None, :] >= min_ow)

    ay0_c, ay1_c = np.minimum(ay0, ha), np.maximum(ay0, np.minimum(ay1, ha))
    ax0_c, ax1_c = np.minimum(ax0, wa), np.maximum(ax0, np.minimum(ax1, wa))
    by0_c, by1_c = np.minimum(by0, hb), np.maximum(by0, np.minimum(by1, hb))
    bx0_c, bx1_c = np.minimum(bx0, wb), np.maximum(bx0, np.minimum(bx1, wb))

    ea_roi = np.maximum(_box_sum_integral(a_sq, ay0_c, ay1_c, ax0_c, ax1_c), 0.0)
    eb_roi = np.maximum(_box_sum_integral(b_sq, by0_c, by1_c, bx0_c, bx1_c), 0.0)
    ma_roi = _box_sum_integral(a_m, ay0_c, ay1_c, ax0_c, ax1_c)
    mb_roi = _box_sum_integral(b_m, by0_c, by1_c, bx0_c, bx1_c)
    min_area = float(min_oh * min_ow)

    denom = np.sqrt(ea_roi * eb_roi)
    good = valid & (ma_roi >= min_area) & (mb_roi >= min_area) & (denom > 1e-6)
    if not np.any(good):
        return ShiftResult(
            dy=int(center_dy), dx=int(center_dx), dog_ncc=-1.0, psr=0.0, valid=False
        )

    ncc = np.full(num_roi.shape, -2.0, dtype=np.float32)
    ncc[good] = (num_roi[good] / denom[good]).astype(np.float32)

    by, bx = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    peak_ncc = float(ncc[by, bx])
    best_dy = int(dys[by]) * downsample
    best_dx = int(dxs[bx]) * downsample

    bg = good.copy()
    bg[
        max(0, by - excl) : min(ncc.shape[0], by + excl + 1),
        max(0, bx - excl) : min(ncc.shape[1], bx + excl + 1),
    ] = False
    if int(np.sum(bg)) > 20:
        bg_vals = ncc[bg]
        psr = float((peak_ncc - bg_vals.mean()) / (bg_vals.std() + 1e-6))
    else:
        psr = 0.0

    return ShiftResult(dy=best_dy, dx=best_dx, dog_ncc=peak_ncc, psr=psr, valid=True)


def estimate_prior_centered_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    prior_dy: int = 0,
    prior_dx: int = 0,
    search_half_px: int = 32,
    mask_left_px: int = 0,
    mask_right_px: int = 0,
    mask_top_px: int = 0,
    mask_bottom_px: int = 0,
    coarse_downsample: int = COARSE_DOWNSAMPLE,
) -> ShiftResult:
    """Two-stage prior-centered masked DoG-NCC registration.

    Stage 1 (coarse): searches `[prior - search_half_px, prior + search_half_px]`
    at `coarse_downsample` and records `psr` strictly from this fixed lag window.
    Stage 2 (fine): refines at 1x resolution within `+-FINE_SEARCH_PX` around the
    coarse peak on the unmasked overlapping interior and records `dog_ncc`.
    """
    coarse = masked_dog_ncc_shift(
        ref=ref,
        mov=mov,
        center_dy=prior_dy,
        center_dx=prior_dx,
        half_dy=search_half_px,
        half_dx=search_half_px,
        mask_left_px=mask_left_px,
        mask_right_px=mask_right_px,
        mask_top_px=mask_top_px,
        mask_bottom_px=mask_bottom_px,
        downsample=coarse_downsample,
    )
    if not coarse.valid:
        return coarse

    h_r, w_r = ref.shape
    h_m, w_m = mov.shape
    c_dy, c_dx = coarse.dy, coarse.dx

    x_min_r = (
        mask_left_px
        if (mask_left_px > 0 and mask_left_px < w_r - MIN_OVERLAP_PX)
        else 0
    )
    x_max_r = (
        w_r - mask_right_px
        if (mask_right_px > 0 and mask_right_px < w_r - MIN_OVERLAP_PX)
        else w_r
    )
    y_min_r = (
        mask_top_px if (mask_top_px > 0 and mask_top_px < h_r - MIN_OVERLAP_PX) else 0
    )
    y_max_r = (
        h_r - mask_bottom_px
        if (mask_bottom_px > 0 and mask_bottom_px < h_r - MIN_OVERLAP_PX)
        else h_r
    )

    x_min_m = (
        mask_left_px
        if (mask_left_px > 0 and mask_left_px < w_m - MIN_OVERLAP_PX)
        else 0
    )
    x_max_m = (
        w_m - mask_right_px
        if (mask_right_px > 0 and mask_right_px < w_m - MIN_OVERLAP_PX)
        else w_m
    )
    y_min_m = (
        mask_top_px if (mask_top_px > 0 and mask_top_px < h_m - MIN_OVERLAP_PX) else 0
    )
    y_max_m = (
        h_m - mask_bottom_px
        if (mask_bottom_px > 0 and mask_bottom_px < h_m - MIN_OVERLAP_PX)
        else h_m
    )

    ry0 = max(y_min_r, y_min_m + c_dy - FINE_PAD_PX)
    ry1 = min(y_max_r, y_max_m + c_dy + FINE_PAD_PX)
    rx0 = max(x_min_r, x_min_m + c_dx - FINE_PAD_PX)
    rx1 = min(x_max_r, x_max_m + c_dx + FINE_PAD_PX)

    my0 = max(y_min_m, y_min_r - c_dy - FINE_PAD_PX)
    my1 = min(y_max_m, y_max_r - c_dy + FINE_PAD_PX)
    mx0 = max(x_min_m, x_min_r - c_dx - FINE_PAD_PX)
    mx1 = min(x_max_m, x_max_r - c_dx + FINE_PAD_PX)

    if (
        (ry1 - ry0) < MIN_OVERLAP_PX
        or (rx1 - rx0) < MIN_OVERLAP_PX
        or (my1 - my0) < MIN_OVERLAP_PX
        or (mx1 - mx0) < MIN_OVERLAP_PX
    ):
        return coarse

    exp_dy = c_dy - (ry0 - my0)
    exp_dx = c_dx - (rx0 - mx0)
    fine = masked_dog_ncc_shift(
        ref=ref[ry0:ry1, rx0:rx1],
        mov=mov[my0:my1, mx0:mx1],
        center_dy=exp_dy,
        center_dx=exp_dx,
        half_dy=FINE_SEARCH_PX,
        half_dx=FINE_SEARCH_PX,
        downsample=1,
    )
    if not fine.valid:
        return coarse

    return ShiftResult(
        dy=int((ry0 - my0) + fine.dy),
        dx=int((rx0 - mx0) + fine.dx),
        dog_ncc=fine.dog_ncc,
        psr=coarse.psr,
        valid=True,
    )


def is_confident_lock(
    res: ShiftResult,
    min_dog_ncc: float = 0.15,
    min_psr: float = 7.5,
    high_ncc_override: float = 0.45,
    high_ncc_min_psr: float = 6.0,
) -> bool:
    """Evaluate calibrated confidence gate (Phase 0 ROC defaults)."""
    if not res.valid:
        return False
    if res.psr >= min_psr and res.dog_ncc >= min_dog_ncc:
        return True
    if res.dog_ncc >= high_ncc_override and res.psr >= high_ncc_min_psr:
        return True
    return False


def check_tile_consistency(
    tile_phys_shifts: Mapping[object, Tuple[int, int, int, int]],
    tol_px: int = 2,
) -> set[object]:
    """Return stripe keys passing per-first_exposed_tile_id _v/_h shift consistency.

    Parameters
    ----------
    tile_phys_shifts : Mapping[stripe_key, (grid_num, first_exposed_tile_id, phys_dy, phys_dx)]
        Locked stripes for a single canonical 1-step hop `(s -> s+1)`.
    tol_px : int
        Max Chebyshev disagreement (px) allowed between stripes sharing the same
        `(grid_num, first_exposed_tile_id)`. If `tol_px < 0`, disables check.
    """
    if tol_px < 0:
        return set(tile_phys_shifts.keys())

    by_tile: Dict[Tuple[int, int], list[Tuple[object, int, int]]] = {}
    for key, (grid_num, ftid, pdy, pdx) in tile_phys_shifts.items():
        by_tile.setdefault((int(grid_num), int(ftid)), []).append(
            (key, int(pdy), int(pdx))
        )

    accepted: set[object] = set()
    for items in by_tile.values():
        if len(items) == 1:
            accepted.add(items[0][0])
            continue
        dys = [it[1] for it in items]
        dxs = [it[2] for it in items]
        if (max(dys) - min(dys)) <= tol_px and (max(dxs) - min(dxs)) <= tol_px:
            for key, _, _ in items:
                accepted.add(key)
    return accepted


def filter_section_consensus(
    phys_shifts: Mapping[object, Tuple[int, int]],
    tol_px: float = 12.0,
    min_samples: int = CONSENSUS_MIN_SAMPLES,
) -> set[object]:
    """Return stripe keys whose physical shift agrees with section median within `tol_px`."""
    if tol_px <= 0.0 or len(phys_shifts) < min_samples:
        return set(phys_shifts.keys())

    arr_dy = np.array([v[0] for v in phys_shifts.values()], dtype=np.float32)
    arr_dx = np.array([v[1] for v in phys_shifts.values()], dtype=np.float32)
    med_dy = float(np.median(arr_dy))
    med_dx = float(np.median(arr_dx))

    return {
        key
        for key, (pdy, pdx) in phys_shifts.items()
        if max(abs(pdy - med_dy), abs(pdx - med_dx)) <= tol_px
    }


def refine_patch_local_shift(
    ref_patch: np.ndarray,
    mov_full: np.ndarray,
    base_y0: int,
    base_x0: int,
    max_shift: int = 4,
    blur_sigma: float = 1.0,
) -> Optional[Tuple[int, int]]:
    """Refine integer neighbor slice position `(ny0, nx0)` around `(base_y0, base_x0)`.

    Returns `(ny0, nx0)` strictly in-bounds of `mov_full`, or `None` on OOB (never clamps).
    """
    ph, pw = ref_patch.shape
    hm, wm = mov_full.shape
    if base_y0 < 0 or base_y0 + ph > hm or base_x0 < 0 or base_x0 + pw > wm:
        return None
    if max_shift <= 0:
        return base_y0, base_x0

    mov_base = mov_full[base_y0 : base_y0 + ph, base_x0 : base_x0 + pw]
    ldy, ldx = _estimate_integer_shift(
        ref_patch, mov_base, max_shift=max_shift, blur_sigma=blur_sigma
    )
    ny0 = base_y0 - ldy
    nx0 = base_x0 - ldx
    if ny0 < 0 or ny0 + ph > hm or nx0 < 0 or nx0 + pw > wm:
        return None
    return int(ny0), int(nx0)


def get_registration_constants_fingerprint() -> Dict[str, object]:
    """Return all internal registration constants for persistent cache fingerprinting."""
    return {
        "registration_algo_version": REGISTRATION_ALGO_VERSION,
        "dog_low_sigma": DOG_LOW_SIGMA,
        "dog_high_sigma": DOG_HIGH_SIGMA,
        "coarse_downsample": COARSE_DOWNSAMPLE,
        "fine_pad_px": FINE_PAD_PX,
        "fine_search_px": FINE_SEARCH_PX,
        "min_overlap_px": MIN_OVERLAP_PX,
        "consensus_min_samples": CONSENSUS_MIN_SAMPLES,
    }


__all__ = [
    "REGISTRATION_ALGO_VERSION",
    "ShiftResult",
    "masked_dog_ncc_shift",
    "estimate_prior_centered_shift",
    "is_confident_lock",
    "check_tile_consistency",
    "filter_section_consensus",
    "refine_patch_local_shift",
    "get_registration_constants_fingerprint",
]
