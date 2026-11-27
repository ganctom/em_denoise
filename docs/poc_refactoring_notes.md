# PoC / Production Refactoring Notes (from PoP Phase)

This document tracks architectural decisions, dead/experimental code to remove, and data-pipeline lessons learned during the Proof-of-Principle (PoP) phase of `em-denoise` (specifically Tier 3 True 2.5D Noise2Noise on `feat/n2n-25d`), to be applied during the Proof-of-Concept (PoC) codebase refactoring.

---

## 1. Remove `z_align_local_refine_px` / Per-Patch Local Z-Refinement

- **Status in PoP:** Retained at default `0` (no-op) solely to preserve `prep_fingerprint` cache hashes and backward compatibility with existing PoP checkpoints/YAMLs.
- **PoC Action:** Delete `z_align_local_refine_px` from `N2NConfig` (`config.py`), `refine_patch_local_shift` from `z_registration.py`, and its call site in `n2n_25d_dataset.py`.
- **Empirical Rationale (Arm 1 vs Arm 2 A/B):**
  - Per-patch `128×128` local Z-refinement (`local_refine=4`) gets pulled by oblique membranes and shot noise, degrading own-distribution clean EstPSNR by `−0.045 dB` (`31.491` vs `31.536 dB`) and softening fine synaptic vesicles and membranes visually (`s05329` ROI Laplacian energy `50.46` vs `54.59` for Arm 1 and `54.84` for Arm 3).
  - Natural SBEM stripe-level elastic variation, oblique membrane drift, and `≤ 2 px` cycle-closure tolerance already provide the right amount of natural jitter without artificial shift-jitter augmentation.
  - Keep **stripe-rigid (`local_refine=0`)** for training and **block-wise rigid + strict `≤ 2 px` closure/NCC gating** (with `center_filled` fallback) for raw-tile inference.

---

## 2. 2.5D Overlap Cache Data-Distribution Fixes

- **Left-margin beam-darkening trim:** `v_trim_left_px: 500` + `h_trim_left_px: 24` introduced `2.3×` more severe target-darkening patches (`mean(tgt - inp) < -5 gray`: `2.02%` vs `0.88%` in `04_n2n_ovs_100k`) and an `87.5%` quality rejection rate (with `_h` contributing only `~1.4%` of kept crops). Use `v_trim_left_px: 1500` (and `orientation: "vertical"` or stricter `_h` trim) when rebuilding caches.
- **High-contrast / dark tissue diversity:** The PoP `25d_r1` cache had `0.10%` high-contrast (`std > 45`) patches vs `6.18%` in `04_n2n_ovs_100k`, causing a `−5.45 dB` drop on `std > 45` patches and making `GroupNorm` sensitive to small-tile local intensity/variance. Audit why `std > 45` edge/dark tiles were dropped during 2.5D registration and preserve contrast diversity in the PoC cache builder.
- **Tighter thin/missed-cut guard:** Lower training `z_align_missed_cut_max_ncc` from `0.85` to `~0.72` to match the validated `0.70` raw-tile inference thin-cut guard (which eliminated thin-cut over-averaging on `s20804–s20814`).

---

## 3. Inference Tiling & `GroupNorm`

- **Keep `inference.tiled: false` as default** for fully-convolutional UNets (`ResidualUNet`, `ConnectomicsDenoisingNet`) on `3072×4096` tiles — full-frame runs ~4× faster and avoids small-tile `GroupNorm` local-variance shifts.
- **Keep `_denoise_tiled` only for:**
  1. Fixed-size architectures (`COSDDModel` via `model.crop_size`), and
  2. VRAM-limited giant slabs (`> 4096×4096`), using `inference.tile_size >= 1024` (never `128×128`).

---

## 4. Pipeline Placement (Pre-SOFIMA vs Post-SOFIMA)

- **Denoise *before* SOFIMA on raw uninterpolated tiles:** True 2.5D N2N requires independent `(z_first, z_second)` overlap pairs, avoids SOFIMA sub-pixel interpolation domain shift (spatially varying noise low-passing), and provides SOFIMA with higher-SNR inputs for cleaner flow estimation.
- **Radius selection:** `R=1` (3-slice) captures ~92% of the total 2.5D gain (`+0.453 dB` over 2D) with simpler operational requirements (3 consecutive intact sections vs 5); `R=2` (5-slice) adds a statistically significant `+0.039 dB` (`+0.492 dB` total) at +50% cache size and slightly higher `z+1` bleed (`10.6%` vs `9.6%`).
