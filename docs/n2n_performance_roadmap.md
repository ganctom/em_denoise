# Noise2Noise (N2N) Performance Improvement Roadmap for SBF-SEM

This document consolidates the **Tier 1 – Tier 3 optimization roadmap**, **anti-priorities (what NOT to spend time tuning)**, **empirical lessons from cluster runs `04`–`07`**, and **new connectomics-specific ideas** (including fixing false vesicle-to-membrane mergers observed in Neuroglancer).

> **Core Experimental Rule:** Change **one variable at a time** against a fixed cached dataset (`n2n.crop_cache_dir`) and evaluate on both quantitative validation metrics *and* visual/residual inspection of dense neuropil (synaptic vesicles, thin spine necks) plus smooth somata.

---

## Current Baseline Status

Already implemented and verified on `feat/n2n`:
- Single-pass shard-streaming crop materializer + persistent `.npy` `mmap` cache (`crop_cache_dir`) with `< 1 s` reloads, DuckDB SQL trim pushdown, and uniform per-shard quotas across all shards.
- Strict OOB crop rejection (no edge clamping), inset tile grids, optional two-stage sub-pixel registration (`align_subpixel`), and flat-patch filtering (`min_patch_std`).
- **Stabilized `ResidualUNet` (`f2ad4bc`):** Normalized `1×1` `shortcut`, `Downsample`, and `Upsample` transitions with `GroupNorm` plus zero-initialized `final_conv`, eliminating `bfloat16` skip-path activation explosions at `base_dim=48` and `64` (start fresh runs; pre-`f2ad4bc` checkpoints are not `state_dict`-compatible).
- Clean training loop: cosine LR with linear warmup, noise-corrected `EstPSNR`, and 3-panel `Raw | Denoised | Residual` previews (`mean`, `std`, `ac_x`, `ac_y`).

---

## Tier 1 — Highest Tangible Impact (2D Data, Physics & Loss)

### 1. Scale Up Unique Training Data & Section Coverage
- **Why it matters:** Moving from `5k` local patches to **`100k–1M` unique patches** across the stack exposes the network to the full diversity of organelle morphologies, membrane orientations, staining variations, and local charging regimes.
- **How to run:**
  - Benchmark regime (`1250–8000`, constant imaging/cutting thickness): `100k–300k` patches (`3.3–9.8 GB` cache).
  - Full stack (`1250–22800`, 1,090 shards): build a `500k–1M` patch cache (`16.4–32.8 GB`), and use `sections.npy` instant slicing (`first_sec` / `last_sec`) or `training.samples_per_epoch: 200000` to keep epochs ~10 minutes long.

### 2. Preserve Scan-Axis Noise Anisotropy (`augment_rot90: false`)
- **Why it matters:** SBF-SEM detector bandwidth correlation and charging streaks run along the horizontal fast-scan axis (`X`), not `Y`. Rotating patches by 90° mixes `X` and `Y` noise statistics and weakens horizontal streak suppression.
- **Recommendation:** Keep `augment_rot90: false` (horizontal + vertical flips only, both of which preserve scan-axis orientation).

### 3. Pure MSE vs. Pure Charbonnier Loss (`loss_type: "enhanced"`)
- **Pure MSE (`loss_type: "composite"`, `mse_weight: 1.0`, others `0.0`):** Estimates the conditional **mean** $\mathbb{E}[y \mid x]$. Produces the smoothest cytoplasm, but averages uncertain 1–2 px gaps between dark synaptic vesicles into faint dark bridges.
- **Pure Charbonnier (`loss_type: "enhanced"`, `charbonnier_weight: 1.0`, `ssim/laplacian/ffl/espi/edge: 0.0`):** Approximates the conditional **median**. Produces crisper membrane edges and better separation between closely packed vesicles, at the cost of slightly higher sensitivity if target noise is skewed.
- **Evaluation tip:** Do not compare raw `Val` loss values across MSE and Charbonnier (different scales); compare `Val PSNR`/`SSIM`, residual `ac_x`/`ac_y`, and Neuroglancer vesicle clusters.

### 4. Registration Precision (`align_subpixel`) & Quality Threshold (`min_pair_quality`)
- **`align_subpixel: false` vs `true` (with `pair_direction: "first_to_second"`):**
  - Integer alignment leaves up to $\pm 0.5\text{ px}$ residual shift between `crop0` and `crop1`, which smears 1–2 px inter-vesicle gaps in the target and encourages vesicle bridging.
  - Note: cubic spline shifting on `crop1` slightly reduces $\sigma_y^2$, clamping `EstPSNR` to `100.00 dB` — judge sub-pixel runs by `Val` loss progression and visual sharpness on `0x_test_dataset`.
- **`min_pair_quality` sweep (`0.75` vs `0.80` vs `0.83`):**
  - Disabling QA completely (`05_n2n_ovs_100k_ssim_false`) dropped peak `EstPSNR` (`30.72 dB`, `SSIM = 0.1898` vs `04_100k`'s `31.14 dB`, `SSIM = 0.2129`), confirming structural QA filtering helps.
  - Higher (`0.83`) rejects more elastically sheared patches → sharper membranes, but biases sampling away from low-contrast soma interiors.
  - Lower (`0.75`) retains more homogeneous cytoplasm → smoother somata, slightly more elastic blur.

---

## Tier 2 — Model Architecture, Capacity & Inference Upgrades

### 5. Network Architecture & Width (`connectomics` vs Stabilized `residual_unet`)
- **Stabilized `ResidualUNet` (`base_dim: 48` or `64`, fixed in `f2ad4bc`):**
  - Previously, `ResidualUNet` had an unnormalized skip path (`shortcut`, `Downsample`, `Upsample`), causing `bfloat16` activation explosions (`10¹¹–10¹⁶`) in runs `06_dim64`, `06_dim48`, and `05_ssim_false`.
  - With `GroupNorm` now added to `1×1` `shortcut`, `Downsample`, and `Upsample` plus zero-initialized `final_conv`, `ResidualUNet` keeps activations `O(1)` at `base_dim=48` and `64`.
  - Recommended settings: `base_dim: 48` (`lr: 0.0003`, ~3.8h/30 ep) or `base_dim: 64` (`lr: 0.0002`–`0.0003`, bump `#SBATCH --time=06:30:00` for 30 ep).
- **`model_type: "connectomics"` (NAFNet + `LaplacianEdgeAttention`):**
  - Lightweight alternative (`base_dim: 64` ≈ `898k` params; `base_dim: 96` ≈ `2.0M` params) with 2 downsampling levels (smaller receptive field than a 4-level UNet, which can reduce long-range membrane hallucination across vesicles, though a 3-level variant may help on large somata).
- **Longer Cosine Schedule (`epochs: 50–80`):** Fine connectomics detail (cristae, lipid bilayers, inter-vesicle gaps) emerges in the final low-LR tail (`lr < 5e-5`) where high-variance N2N target noise averages out.

### 6. Recover Horizontal-Pair (`_h`) Overlap Stripes
- Currently `h_trim_left_px: 90` leaves `< 128 px` width on `_h` stripes (`median width = 152 px`), so 100% of `_h` stripes (`996k` stripes / `887 GB`) are skipped.
- Inspect raw `_h` stripes: if charging distortion is confined to the leftmost `20–24 px`, setting `h_trim_left_px: 24` recovers hundreds of thousands of vertical-seam patches with complementary scan geometry.
- **Hard constraint on square `patch_size`:** Do **not** set `patch_size: 256` on `_v` overlaps — trimmed `_v` median height is `~156 px`, which would drop ~100% of `_v` stripes.

### 7. "Free" Inference & Weight-Stability Gains (Low Effort, High Return)
- **4× Flip Test-Time Augmentation (TTA):** Average predictions over `[orig, flip_x, flip_y, flip_xy]` during tiled inference (no 90° rotations, preserving scan-axis physics). Typically yields `+0.15–0.35 dB` and suppresses asymmetric 1-pixel false bridges between vesicles with **zero retraining**.
- **Exponential Moving Average (EMA) of Weights (`decay = 0.999`):** Maintains a shadow copy of weights during training for validation/inference. Strongly reduces late-epoch weight jitter caused by noisy N2N targets.

---

## Tier 3 — Breaking the 2D Information Limit (Solving False Vesicle Mergers)

### 8. True 2.5D Noise2Noise (`[z-1, z_first, z+1] -> z_second`)
- **Why 2D hits a fundamental ceiling:** In heavy noise, 3–4 dark synaptic vesicles (~40 nm, 3–5 px) aligned by chance are locally indistinguishable in 2D from a continuous dark plasma membrane. A 2D denoiser bridges them in ~1 out of 10 slices even when slices $z-5 \dots z-1$ and $z+1 \dots z+5$ clearly show no membrane.
- **Why True 2.5D N2N is the gold standard:**
  - **Input:** 3-channel (or 5-channel) registered stack `[z-1, z_first, z+1]` (all single-exposure).
  - **Target:** `z_second` (second exposure of center slice $z$).
  - **Preserves single-slice vesicles:** Unlike `[z-1, z+1] -> z` (DeepInterpolation/SUPPORT, which **erases** 40 nm vesicles because they span only 1–2 sections!), `z_first` is directly visible in the input and supervised at the exact same Z-plane by `z_second`.
  - **Eliminates false membrane bridges:** Real membranes persist across $z\pm 1$; transient vesicle alignments in $z$ do not.
  - **Z-neighbor dropout (10–15%):** Randomly replace `z-1` or `z+1` with `z_first` during training so knife marks, folds, or missing neighbors in $z\pm 1$ never corrupt slice $z$ at inference.
  - **Compute cost:** Almost identical to 2D (only the first `3×3` conv changes from `1 → C` to `3 → C` channels, with global residual skip applied to center channel `x[:, C//2 : C//2 + 1]`).

---

## Fresh Connectomics-Specific Ideas Worth Exploring

### 9. Rectangular Patches (`128 × 256` or `128 × 384`) on `_v` Stripes
- While `_v` stripes are constrained vertically (`~156 px` tall), they are **`~2,100 px` wide** after trimming!
- Fully convolutional U-Nets/NAFNets naturally accept rectangular `(H, W) = (128, 256)` patches. This doubles spatial context *and* captures long-range horizontal fast-scan line artifacts without dropping narrow `_v` stripes.

### 10. Post-Hoc Raw-Residual Blending (`alpha` Blending) for Neuroglancer & Segmentation
- Computing $I_{\text{out}} = \alpha \, I_{\text{denoised}} + (1 - \alpha) \, I_{\text{raw}}$ with $\alpha \in [0.85, 0.92]$ at export time costs zero compute and is widely used in connectomics pipelines (FFN / LSD proofreading):
  - Restores subtle stochastic grain inside ambiguous 1–2 px inter-vesicle gaps so neither human proofreaders nor 3D segmentation models mistake a smooth N2N bridge for a high-confidence membrane.

### 11. Inter-Exposure Photometric Drift Check (`page 0` vs `page 1`)
- Because `page 1` has seen **two** electron-beam passes while `page 0` has seen **one**, mild beam-induced brightening/darkening (radiation bleaching or local charging offset) can introduce a small systematic gray-level bias (`mean(residual) != 0`).
- Check `mean=` in your `previews/epoch_XXX_comparison.png`. If `|mean| > 1.0` gray level (out of 255), matching `crop1`'s very low-frequency mean (`gaussian_filter(sigma=32)`) to `crop0` during crop materialization prevents subtle brightness shifts at inference.

### 12. Regime-Aware Models Across Section `8000`
- Because imaging parameters and section thickness changed above section `8000`, mixing `1250–8000` and `8000–22800` blindly could slightly compromise optimal calibration for either regime.
- Using `crop_cache_dir` + `sections.npy`, you can benchmark:
  1. **Specialist checkpoint** trained strictly on `1250–8000` vs
  2. **Generalist checkpoint** trained on `1250–22800` (optionally fine-tuned for 5 epochs on `1250–8000`), verifying on `0x_test_dataset` that adding `> 8000` data helps rather than shifts contrast.

---

## What NOT to Spend Time On

1. **Hand-tuning multi-term composite loss weights (`ssim_weight`, `laplacian_weight`, `ffl_weight`, `espi_weight`, `edge_weight`) on N2N pairs:**
   - Noise2Noise mathematical unbiasedness ($\arg\min_\theta \mathbb{E}[\mathcal{L}(f_\theta(x), y)] = \arg\min_\theta \mathbb{E}[\mathcal{L}(f_\theta(x), x_{\text{clean}})]$) holds strictly for **MSE** (mean) and approximately for **L1/Charbonnier** (median under symmetric noise).
   - Non-linear structural/derivative/frequency losses (`SSIM`, `Laplacian`, `Sobel`, `FFL`) evaluated against a **noisy** target $y$ fit target noise gradients and introduce systematic bias/halos. Keep them at **`0.0`**.
2. **Withholding the center slice in 3D self-supervision (`[z-1, z+1] -> z`):**
   - At SBF-SEM Z-resolution (~25–30 nm), omitting slice $z$ erases single-slice synaptic vesicles (~40 nm) and blurs oblique membranes.
3. **Micro-tuning insensitive hyperparameters:**
   - `warmup_epochs` (`2` vs `4`), `lr_min` (`1e-6` vs `1e-7`), `align_blur_sigma` (`1.0` vs `1.2`), or `quality_blur_sigma` (`1.5` vs `1.8`) have negligible impact compared to data diversity, registration quality, architecture stability, and 2.5D context.
4. **Square `patch_size > 128` on overlap stripes:**
   - Rejects almost all `_v` and `_h` overlap stripes due to physical tile-overlap width (`~152–181 px`).

---

## Recommended Next Experiment Queue

| Priority | Experiment | Single Change vs Baseline (`04_n2n_ovs_100k`) | Primary Question Answered |
|---|---|---|---|
| **1** | `07_n2n_ovs_100k_CONN` | `model_type: "connectomics"`, `base_dim: 64` | Does NAFNet + edge attention beat `residual_unet` `base_dim=32` without skip-path instability? |
| **2** | `04_n2n_ovs_100k_SUBPX` | `align_subpixel: true` | Does removing $\pm 0.5\text{ px}$ integer target jitter sharpen inter-vesicle gaps? |
| **3** | `05b_ab_pure_charbonnier` | `loss_type: "enhanced"`, `charbonnier_weight: 1.0` (others `0.0`) | Does median-like L1/Charbonnier reduce MSE membrane bridging between vesicles? |
| **4** | Stabilized `ResidualUNet` (`dim=48` / `64`) | `model_type: "residual_unet"`, `base_dim: 48` or `64` (ready on `f2ad4bc`) | Does 2.25×–4× wider multi-scale UNet capacity improve bouton/vesicle separation? |
| **5** | 4× Flip TTA + EMA | Inference-time 4-flip average + weight EMA | How much "free" vesicle crispness & mottling reduction do we get without changing data? |
| **6** | **2.5D N2N (`[z-1, z, z+1]`)** | 3-slice registered input → `z_second` target | Eliminates cross-slice inconsistent false vesicle mergers in Neuroglancer. |

---

## Cluster Scaling & Multi-GPU Compute Acceleration

Currently `src/em_denoise_refactor/execution/train_and_infer.py` is **single-GPU only** (requesting `--gres=gpu:4` today leaves 3 GPUs idle). Below is the structured path to speed up training on multi-GPU cluster hardware.

### 1. Immediate Zero-Code Wins (Single-GPU & Parallel Sweeps)
- **Prefer A100 over A40 (`#SBATCH --gres=gpu:a10080g:1`):** 2D U-Nets are memory-bandwidth bound; A100 80 GB provides ~2.9× higher memory bandwidth (`2.0 TB/s` vs `696 GB/s`) and faster `bfloat16` Tensor Cores (~2× faster epochs).
- **Enable `torch.compile()` (`training.compile_model: true`):** Fuses `GroupNorm + GELU + Conv` kernels on CUDA for ~25–45% throughput gain after a ~45 s Epoch 1 warmup.
- **Scale `batch_size` (`128 → 256`) and `num_workers` (`4 → 8`):** Better saturates A40/A100 compute (scale `learning_rate` gently by $\sim\sqrt{2}$ when doubling batch size).
- **Parallelize ablation sweeps:** Run 4 independent 1-GPU jobs simultaneously (all safely sharing the same read-only `crop_cache` via `mmap`) for exact 4× experiment throughput with zero multi-GPU communication overhead.

### 2. True Multi-GPU Training (`torchrun` + PyTorch `DDP`)
For heavy single runs (`ResidualUNet` `base_dim=64`, `500k–1M` patches, or 2.5D training), **DistributedDataParallel (DDP)** via `torchrun` provides ~3.5×–3.8× speedup on 4 GPUs:
- **Architecture compatibility:** `ResidualUNet` and `ConnectomicsDenoisingNet` use `GroupNorm`/`LayerNorm` (no `BatchNorm`), so no `SyncBatchNorm` overhead is needed.
- **Cache-safe startup:** Rank 0 builds `crop_cache` if missing, followed by `dist.barrier()`, then all ranks open `crops.npy` read-only via `mmap_mode="r"` in `< 0.1 s`.
- **Implementation checklist (~90 lines in `train_and_infer.py`, backward-compatible with 1-GPU / MPS):**
  1. Detect `WORLD_SIZE` / `LOCAL_RANK`, call `dist.init_process_group("nccl")` and `torch.cuda.set_device(local_rank)`.
  2. Shard `train_loader` and `val_loader` with `DistributedSampler` (calling `sampler.set_epoch(epoch)` each epoch).
  3. Wrap `model` in `DistributedDataParallel(model, device_ids=[local_rank])`.
  4. `dist.all_reduce` validation sums across ranks for exact global `Val` / `PSNR` / `EstPSNR` / `SSIM`; gate logging, TensorBoard, checkpoints, previews, and final test evaluation to rank 0.
  5. Launch via `torchrun --standalone --nproc_per_node=4 -m em_denoise_refactor.execution.train_and_infer --config "$CONFIG"`.

### 3. Additional CUDA Pipeline Optimizations
- **Eliminate per-batch `.item()` CPU↔GPU syncs** inside `train_epoch` and `loss.py` (accumulate loss tensors on GPU and sync only at epoch end or progress intervals).
- **Use `channels_last` memory format** (`model.to(memory_format=torch.channels_last)`) on CUDA for faster `bfloat16` Tensor Core convolutions.
- **Stage `crop_cache` to local node NVMe (`$TMPDIR`):** Copy small-to-medium caches (`3–30 GB`) to `$TMPDIR/crop_cache` at the top of `run_n2n_ovs.sh` so random `DataLoader` reads hit local SSD instead of `/tachyon` network storage.
