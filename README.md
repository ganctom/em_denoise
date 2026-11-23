# em-denoise

PyTorch training, evaluation, and tiled inference package for 2D electron microscopy (SBF-SEM) image denoising, located under `src/em_denoise_refactor/`.

It supports:
- **Noise2Noise (N2N)** on `sbem-n2n-overlaps` `.tar` shards (with DuckDB `manifest.parquet` querying, single-pass shard streaming, persistent `.npy` memory-mapped crop caching, sub-pixel registration, and structural QA filtering) and on duplicate tile acquisitions.
- **Noise2Void / StructN2V** blind-spot self-supervised denoising (`random`, `horizontal`, and `vertical` stripe masking).
- **Supervised / Synthetic-Noise** denoising via parametric or empirically calibrated SBF-SEM noise synthesizers and LMDB/TIFF patch stores.
- **COSDD** unsupervised Ladder VAE + PixelCNN denoising.
- **Classical Baselines** (Gaussian, Bilateral, Non-Local Means) for quantitative and visual benchmarking.

---

## Table of Contents

1. [Installation & Environment](#1-installation--environment)
2. [Repository Layout](#2-repository-layout)
3. [Quick Start](#3-quick-start)
4. [Training Paradigms & Data Pipelines](#4-training-paradigms--data-pipelines)
5. [CLI Reference](#5-cli-reference)
6. [Common Operational Workflows](#6-common-operational-workflows)
7. [Configuration Reference](#7-configuration-reference)
8. [Outputs, Metrics & Diagnostics](#8-outputs-metrics--diagnostics)
9. [Best Practices & Troubleshooting](#9-best-practices--troubleshooting)
10. [Development & Testing](#10-development--testing)

---

## 1. Installation & Environment

Requires **Python >= 3.12** (managed with `uv` via `pyproject.toml`).

```bash
git clone <repo-url>
cd em-denoise
uv sync --dev
```

Activate the virtual environment or invoke `.venv/bin/python` directly:

```bash
export PATH="$(pwd)/.venv/bin:$PATH"
```

> **Cluster / SLURM note:** Compute nodes often have read-only home directories or no internet access. Expose `$REPO_DIR/.venv/bin` on `PATH` and run `python -m ...` directly rather than `uv run` on compute nodes (see `run_n2n_slurm.sh`).

---

## 2. Repository Layout

```text
em-denoise/
├── pyproject.toml
├── run_n2n_slurm.sh                              # SLURM submission script template
└── src/em_denoise_refactor/
    ├── config.py                                 # PipelineConfig & dataclass schemas
    ├── config.yaml                               # Default synthetic/LMDB config
    ├── config_n2n_overlaps_baseline.yaml         # Local MPS N2N overlaps baseline
    ├── config_n2n_slurm.yaml                     # Cluster CUDA N2N overlaps config
    ├── config_n2n_dupl_baseline.yaml             # Local N2N duplicate-tile baseline
    ├── data/
    │   ├── n2n_dataset.py                        # N2NOverlapPairDataset & N2NStreamingShardDataset
    │   ├── n2n_duplicate_dataset.py              # N2NDuplicateTileDataset
    │   ├── n2v_dataset.py                        # N2VDatasetWrapper (blind-spot masking)
    │   └── dataset.py                            # Synthesizers & legacy datasets
    ├── models/
    │   ├── factory.py                            # build_model() registry
    │   ├── residual_unet.py                      # ResidualUNet, ModernUNet
    │   ├── connectomics.py                       # ConnectomicsDenoisingNet (NAFBlock + edge attention)
    │   ├── n2v.py                                # N2VUNet
    │   ├── cosdd/                                # COSDDModel
    │   └── loss.py                               # CompositeDenoisingLoss, EnhancedCompositeLoss, DynamicEMLoss, SSIMLoss
    ├── execution/
    │   ├── train_and_infer.py                    # Main training, validation, resume & tiled inference CLI
    │   ├── run_classical_baselines.py            # Gaussian / Bilateral / NLM CLI
    │   ├── dataset_prep.py                       # Legacy TIFF/LMDB patch extraction CLI
    │   ├── calibration.py                        # Empirical noise model calibration CLI
    │   └── convert_folder_to_lmdb.py             # TIFF folder -> LMDB conversion CLI
    └── tests/                                    # Pytest suite (synthetic fixtures)
```

---

## 3. Quick Start

### Local macOS (MPS) N2N Overlap Baseline

```bash
.venv/bin/python -m em_denoise_refactor.execution.train_and_infer \
  --config src/em_denoise_refactor/config_n2n_overlaps_baseline.yaml
```

### Cluster (SLURM / CUDA) N2N Training

1. Edit `src/em_denoise_refactor/config_n2n_slurm.yaml` (paths, `first_sec`/`last_sec`, `n2n.crop_cache_dir`, `max_train_pairs`).
2. Update `REPO_DIR` and `#SBATCH` directives in `run_n2n_slurm.sh` if needed, then submit:

```bash
sbatch run_n2n_slurm.sh
```

---

## 4. Training Paradigms & Data Pipelines

### 4.1 Noise2Noise on Overlap Shards (`n2n.enabled: true`, `n2n.source: "overlaps"`)

Reads paired `(2, H, W)` uint8 TIFF overlap stripes (`page 0`: single exposure, `page 1`: double exposure) stored in byte-offset-indexed USTAR `.tar` shards with `manifest.parquet`.

#### Key Data Stages (in `N2NOverlapPairDataset`)
1. **Volumetric Z-Splitting (`splitting.strategy`)**:
   - `"mod10"`: `sec % 10 < 8` → `train`, `== 8` → `val`, `== 9` → `test`.
   - `"block"`: Contiguous blocks of `block_size` sections with `guard_band_size` excluded boundary sections to prevent Z-leakage.
2. **SQL Trim Pushdown**: When `manifest.parquet` exists and `trim_enabled: true`, DuckDB filters stripes by post-trim dimensions per orientation (`_v` vs `_h`) before reading any `.tar` bytes.
3. **Single-Pass Shard-Streaming Materialization** (`patch_sampling: "tile"` with `cache_in_ram: true` or `crop_cache_dir` set):
   - Streams `.tar` shards in 64-stripe chunks in ascending byte-offset order (~180 MB stripe working RAM), avoiding full-stripe RAM preloading.
   - Assigns uniform per-shard quotas across all shards in `[first_sec, last_sec]` when `max_train_pairs` / `max_val_pairs` is capped, with a Pass 2 fallback if any shard has a shortfall.
   - Parallelizes crop alignment and QA filtering across `n2n.prep_workers` threads deterministically.
4. **Margin Trimming, Registration & QA Filtering**:
   - Trims distorted acquisition borders (`v_trim_*` for `_v`, `h_trim_*` for `_h`).
   - Uses an inset/centered tile grid (`margin=2` integer, `margin=6` sub-pixel) and strictly rejects out-of-bounds shifted crops (never clamping edge pixels).
   - `align_patches: true` estimates shift via Hann-windowed FFT cross-correlation; `align_subpixel: true` adds upsampled `phase_cross_correlation` + cubic spline shift on `crop1` (use `pair_direction: "first_to_second"` so the network input remains raw single-exposure noise).
   - `quality_filter_enabled: true` scores smoothed proxies via `"ssim"` or `"ncc"` and rejects distorted regions below `min_pair_quality`.
   - `min_patch_std > 0` rejects flat/empty epoxy patches whose smoothed standard deviation (0–255 scale) falls below threshold.

#### Persistent Disk-Backed `.npy` Memmap Cache (`n2n.crop_cache_dir`)

For multi-run benchmarking and large section ranges (hundreds of thousands to millions of patches):
- Set `n2n.crop_cache_dir` to a writable scratch path (never inside the raw shard directory).
- On the first run, curated `(N, 2, ps, ps)` `uint8` crops are streamed directly to `<crop_cache_dir>/<split>_<hash>/crops.npy` along with `sections.npy`, `ids.txt`, and `metadata.json` (written atomically via a temp directory).
- Subsequent runs with matching preprocessing settings open `crops.npy` via `np.load(..., mmap_mode="r")` in `< 0.1 s` with `< 4 GB` RAM, even across `num_workers > 0` DataLoader workers.
- **Instant sub-range slicing:** A cache built over a wide range (e.g. `1250–22800` with `max_pairs: null`) automatically serves narrower `first_sec..last_sec` queries (e.g. `1250–8000`) via `sections.npy` without rebuilding.
- **Automatic invalidation:** `metadata.json` hashes `overlaps_dir`, `manifest.parquet` stat, `patch_size`, `trim_*`, `align_*`, `quality_*`, `min_patch_std`, and split parameters. Changing training hyperparameters (`model_type`, `base_dim`, `loss`, `learning_rate`, `batch_size`, `pair_direction`, `samples_per_epoch`) reuses the same cache.
- **Epoch pacing:** Set `training.samples_per_epoch` (e.g. `200000`) to draw a fresh random subset per epoch from a multi-million-patch cache so validation and checkpoints run at regular intervals.

> **Note on `n2n.streaming: true` (`N2NStreamingShardDataset`):** Streams raw stripes per epoch in `< 2 GB` RAM with 0 extra disk space, but does **not** apply ROI trimming, cross-correlation alignment, or QA filtering. Prefer `crop_cache_dir` for curated N2N training.

### 4.2 Noise2Noise on Duplicate Tiles (`n2n.enabled: true`, `n2n.source: "duplicates"`)

Reads repeated single-tile acquisitions matching `*_gXXXX_tXXXX_sXXXXX.tif` from `n2n.duplicates_dir` using `N2NDuplicateTileDataset`.
- Groups overlapping/connected tiles by `(grid, tile)` into connected components (`split_unit: "group"`) so duplicate acquisitions of the same physical region never leak across `train`/`val`/`test`.
- `pair_mode: "all"` forms all `(i, j)` pairs within a group; `"neighbours"` pairs adjacent sections only.

### 4.3 Noise2Void / StructN2V (`model.model_type: "n2v"`)

Self-supervised blind-spot training (`n2n.enabled` must be `false`). Wraps raw noisy patches with `N2VDatasetWrapper` and trains `N2VUNet` on masked-pixel MSE:
- `n2v.mask_mode: "random"` (N2V2), `"horizontal"` (StructN2V for SBF-SEM horizontal scan streaks), or `"vertical"`.
- `n2v.blind_spot_strict: true` severs the top encoder-to-decoder skip connection to preserve the blind-spot invariant.

### 4.4 Supervised / Synthetic-Noise & COSDD Workflows

1. **Extract clean patches** (`dataset_prep.py`) or convert an existing TIFF folder (`convert_folder_to_lmdb.py`) into `patch_store.lmdb`.
2. *(Optional)* Calibrate an empirical noise model (`calibration.py`) if `noise.synthesizer_type: "empirical"`.
3. Train with `n2n.enabled: false` using `model_type` in `["connectomics", "residual_unet", "modern_unet", "cosdd"]`.

---

## 5. CLI Reference

### 5.1 Main Training & Inference (`train_and_infer.py`)

```bash
python -m em_denoise_refactor.execution.train_and_infer [OPTIONS]
```

| Flag | Description |
|---|---|
| `-c`, `--config <path>` | Path to YAML configuration file (default: `../config.yaml`). |
| `-o`, `--output-dir <dir>` | Override `experiment.output_dir`. |
| `-n`, `--name <str>` | Override `experiment.name` (run subdirectory under `output_dir`). |
| `--epochs <int>` | Override `training.epochs`. |
| `--batch-size <int>` | Override `training.batch_size`. |
| `--lr <float>` | Override `training.learning_rate`. |
| `-m`, `--model`, `--model-type <str>` | Override `model.model_type` (`connectomics`, `residual_unet`, `modern_unet`, `cosdd`, `n2v`). |
| `--device <str>` | Override `training.device` (`mps`, `cuda`, `cpu`). |
| `--eval-only` | Skip training & train/val dataset loading; run evaluation/inference directly. |
| `--checkpoint <path>` | Specific `.pt` checkpoint for `--eval-only` (default: `<run_dir>/checkpoints/checkpoint_best.pt`). |
| `--resume <path>` | Resume training from a `.pt` checkpoint (restores model, optimizer, scheduler, and epoch). |
| `-i`, `--input-dir <dir>` | Directory of `.tif`/`.tiff` images for custom tiled inference (overrides `inference.custom_input_dir`). |
| `-d`, `--data-dir <dir>` | Legacy/custom folder containing `train/` and `val/` clean/noisy TIFF subdirectories. |
| `--log-level <str>` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` (default: `INFO`). |

### 5.2 Classical Baselines (`run_classical_baselines.py`)

Runs Gaussian (`sigma=0.8`, `1.2`), Bilateral, and Non-Local Means (`nlm`):

```bash
python -m em_denoise_refactor.execution.run_classical_baselines \
  --output-dir /path/to/baselines_out \
  --input-dir /path/to/raw_tifs \
  --config src/em_denoise_refactor/config_n2n_dupl_baseline.yaml
```

- `--input-dir` saves denoised TIFFs and 3-panel comparison PNGs per method.
- `--config` evaluates each method on the held-out N2N duplicate-tile test split and writes `baselines_summary.json`.

### 5.3 Auxiliary ETL & Calibration CLIs

- `python -m em_denoise_refactor.execution.dataset_prep --config <yaml> [--lmdb / --no-lmdb]` — Extracts SBEMimage patches into TIFFs and/or `patch_store.lmdb`.
- `python -m em_denoise_refactor.execution.convert_folder_to_lmdb --input-dir <dir> --output-lmdb <path>` — Packs an existing TIFF directory into LMDB.
- `python -m em_denoise_refactor.execution.calibration --config <yaml>` — Calibrates `EmpiricalNoiseSynthesizer` (`noise_model.npy`).

---

## 6. Common Operational Workflows

### 6.1 Inference on Any `.tif` File(s) with a Specific Checkpoint

```bash
.venv/bin/python -m em_denoise_refactor.execution.train_and_infer \
  --config /path/to/matching_config.yaml \
  --eval-only \
  --checkpoint /path/to/run/checkpoints/checkpoint_best.pt \
  --input-dir /path/to/raw_tifs \
  -o /path/to/inference_outputs \
  -n infer_exp01
```

- Denoised images are written to `/path/to/inference_outputs/infer_exp01/custom_inference/denoised/` using 50%-overlapping `patch_size × patch_size` tiles with 2D Hann window blending.
- Ensure `--config` matches the checkpoint's `model.model_type`, `model.base_dim`, and `extraction.patch_size`.
- Use `-o` / `-n` to direct outputs into a separate directory so existing training summaries are not overwritten.

### 6.2 Resuming an Interrupted Training Run

```bash
.venv/bin/python -m em_denoise_refactor.execution.train_and_infer \
  --config /path/to/same_config.yaml \
  --resume /path/to/run/checkpoints/checkpoint_last.pt
```

- When `training.save_all_epochs: false`, resume from `checkpoints/checkpoint_last.pt`.
- When `training.save_all_epochs: true`, resume from `checkpoints/checkpoint_epoch_XXX.pt`.
- Ensure `experiment.name` and `experiment.output_dir` match the interrupted run so checkpoints and logs continue in the same folder.

---

## 7. Configuration Reference

Schemas are defined in `src/em_denoise_refactor/config.py`.

<details>
<summary><strong>experiment / data / splitting / extraction</strong></summary>

| Section & Key | Default | Description |
|---|---|---|
| `experiment.name` | `"N2V_run_32"` | Run subdirectory under `output_dir`. |
| `experiment.output_dir` | `Path(...)` | Root directory for experiment runs. |
| `data.sbem_root` | `Path(...)` | SBEMimage acquisition root (used by `dataset_prep.py` / `calibration.py`). |
| `data.grid_num` | `0` | SBEMimage grid index. |
| `data.first_sec`, `last_sec` | `1250`, `3000` | Inclusive section range filter (supports `null` for unbounded in N2N). |
| `splitting.strategy` | `"block"` | `"block"` (contiguous Z-blocks + guard bands) or `"mod10"`. |
| `splitting.block_size` | `100` | Sections per block when `strategy: "block"`. |
| `splitting.train_ratio`, `val_ratio`, `test_ratio` | `0.8`, `0.1`, `0.1` | Split fractions. |
| `splitting.guard_band_size` | `5` | Excluded boundary sections at each block edge. |
| `extraction.patch_size` | `512` | Square training/tiling patch size (`128` recommended for N2N). |

</details>

<details>
<summary><strong>n2n (Noise2Noise)</strong></summary>

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable Noise2Noise mode (bypasses synthetic noise). |
| `source` | `"overlaps"` | `"overlaps"` or `"duplicates"`. |
| `overlaps_dir` | `""` | Root `sbem-n2n-overlaps` directory (`manifest.parquet` + `shards/`). |
| `duplicates_dir` | `""` | Directory of duplicate-tile acquisitions. |
| `pair_mode` | `"all"` | `"all"` or `"neighbours"` (duplicates mode). |
| `split_unit` | `"group"` | `"group"`, `"image"`, or `"patch"` (duplicates mode). |
| `pair_direction` | `"random_swap"` | `"first_to_second"` (recommended with `align_subpixel`), `"second_to_first"`, or `"random_swap"`. |
| `orientation` | `"all"` | `"all"`, `"horizontal"`, or `"vertical"`. |
| `max_train_pairs`, `max_val_pairs`, `max_test_pairs` | `15000`, `1500`, `500` | Patch caps per split (`null` = keep all). |
| `augment_rot90` | `false` | Add 90° rotations (`false` preserves fast/slow scan noise anisotropy). |
| `align_patches` | `false` | Per-patch cross-correlation registration. |
| `align_subpixel` | `false` | Two-stage sub-pixel phase correlation + cubic spline shift. |
| `align_max_shift_px` | `16` | Max shift search radius (px) per axis (`6`–`8` recommended). |
| `align_blur_sigma` | `1.0` | Gaussian smoothing sigma (px) for shift estimation only. |
| `patch_sampling` | `"random"` | `"tile"` (deterministic full-stripe grid) or `"random"`. |
| `tile_stride` | `null` | Tile grid stride (`null` = `patch_size`). |
| `trim_enabled` | `false` | Trim distorted stripe borders before tiling. |
| `v_trim_top_px`, `v_trim_left_px`, `v_trim_bottom_px`, `v_trim_right_px` | `50, 500, 0, 0` | Margin trims for `_v` overlap stripes. |
| `h_trim_top_px`, `h_trim_bottom_px`, `h_trim_left_px`, `h_trim_right_px` | `0, 0, 32, 0` | Margin trims for `_h` overlap stripes. |
| `quality_filter_enabled` | `false` | Filter out elastically sheared/corrupt patches. |
| `quality_metric` | `"ncc"` | `"ssim"` or `"ncc"`. |
| `min_pair_quality` | `0.60` | Minimum smoothed quality score to keep a patch. |
| `quality_blur_sigma` | `1.5` | Gaussian smoothing sigma before quality/flatness scoring. |
| `min_patch_std` | `0.0` | Minimum smoothed intensity std (0–255) to reject flat/empty patches. |
| `crop_cache_dir` | `null` | Persistent `.npy` `mmap` crop cache directory (tile mode). |
| `max_tiles_per_stripe` | `null` | Max kept patches/stripe (`null` = auto-scaled across all shards). |
| `prep_workers` | `1` | Worker threads for alignment & QA during crop materialization. |
| `streaming` | `false` | Raw shard streaming without trim/alignment/QA. |

</details>

<details>
<summary><strong>model / training / loss / inference</strong></summary>

| Section & Key | Default | Description |
|---|---|---|
| `model.model_type` | `"connectomics"` | `"connectomics"`, `"residual_unet"`, `"modern_unet"`, `"cosdd"`, or `"n2v"`. |
| `model.base_dim` | `32` | Base channel width (`32`, `48`, `64`). |
| `training.batch_size` | `64` | Mini-batch size. |
| `training.epochs` | `100` | Total training epochs. |
| `training.learning_rate` | `0.0001` | Initial/peak AdamW learning rate (`0.0003`–`0.0005` typical for N2N). |
| `training.lr_scheduler` | `"none"` | `"none"` or `"cosine"`. |
| `training.lr_min` | `1e-6` | Minimum cosine LR. |
| `training.warmup_epochs` | `0` | Linear LR warmup epochs (`2` recommended with cosine). |
| `training.cache_in_ram` | `false` | Cache patches in RAM (in tile mode, materializes compact aligned uint8 crops). |
| `training.samples_per_epoch` | `null` | Optional random sample cap per epoch via `RandomSampler`. |
| `training.device` | `"mps"` | `"cuda"`, `"mps"`, or `"cpu"`. |
| `training.num_workers` | `0` | PyTorch `DataLoader` workers (`0` on macOS MPS, `4` on Linux CUDA). |
| `training.save_all_epochs` | `true` | `true` saves `checkpoint_epoch_XXX.pt`; `false` saves `checkpoint_last.pt` only. |
| `training.compile_model` | `false` | Enable `torch.compile()` on CUDA. |
| `loss.loss_type` | `"composite"` | `"composite"`, `"enhanced"`, or `"dynamic"`. |
| `loss.mse_weight`, `edge_weight`, `ssim_weight` | `1.0, 0.2, 0.1` | Weights for `"composite"` (`1.0, 0.0, 0.0` = pure MSE, recommended for N2N). |
| `inference.custom_input_dir` | `null` | Folder of raw `.tif` images evaluated after training. |
| `inference.preview_image` | `null` | Single `.tif` path used to render per-epoch preview PNGs. |
| `inference.preview_crop_size` | `512` | Center/custom crop size for per-epoch previews. |

</details>

---

## 8. Outputs, Metrics & Diagnostics

Each run writes to `<experiment.output_dir>/<experiment.name>/`:

- `train_and_infer.log` — Full timestamped log (dataset counts, QA stats, per-epoch metrics).
- `checkpoints/` — `checkpoint_best.pt` and `checkpoint_last.pt` (or `checkpoint_epoch_XXX.pt`). Each checkpoint stores `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `config`, and validation metrics.
- `logs/` — TensorBoard event files (`Loss/Train`, `Loss/Validation`, `Metrics/Val_PSNR`, `Metrics/Val_EstClean_PSNR`, `Train/LR`, preview images).
- `previews/` — Per-epoch 3-panel PNGs (`Raw | Denoised | Residual`) annotated with residual mean, std, and lag-1 spatial autocorrelations `ac_x` / `ac_y` (ideally near `0.0` so no membrane structure leaks into the residual).
- `custom_inference/denoised/` or `test/denoised/` — Denoised 8-bit TIFF outputs, comparison PNGs, and `evaluation_summary.txt`.
- `summary.json` — Machine-readable final metrics.

### Interpreting N2N Validation Metrics
- **Raw `Val` MSE & `PSNR`:** Measured against the noisy N2N target (`y`). Even a perfect denoiser cannot achieve zero error against noisy `y`, so raw N2N `PSNR` plateaus around ~17–21 dB. Always track **monotonic decrease in `Val` loss** within a run.
- **`EstPSNR` (`est_clean_psnr`):** Estimates clean MSE via $\widehat{\text{MSE}}_{\text{clean}} = \text{MSE}(\hat{x}, y) - \frac{1}{2}\text{MSE}(x, y)$ assuming equal independent noise variance in `x` and `y`.
  - **Caveat with `align_subpixel: true`:** Cubic spline resampling slightly smooths noise in `crop1` ($\sigma_y^2 < \sigma_x^2$), which can make `EstPSNR` clamp to `100.00 dB`. Use `Val` loss and visual/residual inspection of `previews/` when comparing sub-pixel runs.

---

## 9. Best Practices & Troubleshooting

- **Recommended N2N Starting Point:** `model_type: "residual_unet"`, `loss_type: "composite"` with pure MSE (`mse_weight: 1.0, edge_weight: 0.0, ssim_weight: 0.0`), `lr_scheduler: "cosine"`, `warmup_epochs: 2`, `pair_direction: "first_to_second"`, `augment_rot90: false`, `save_all_epochs: false`.
- **Scaling to Large Stacks:** Use `n2n.crop_cache_dir` + `prep_workers: 4`–`8`. Build one cache (`max_train_pairs: 1000000` ≈ 33 GB) and reuse it across all architecture/hyperparameter sweeps or `first_sec..last_sec` sub-range experiments.
- **SLURM Timeouts:** Checkpoints are saved at `checkpoints/checkpoint_last.pt`. Bump `#SBATCH --time` and resume with `--resume`.
- **Network Filesystems (Lustre / GPFS / SMB):** Keep `save_all_epochs: false` and `num_workers: 4` with `crop_cache_dir` so compute nodes do not repeatedly read `.tar` shards over the network.

---

## 10. Development & Testing

Run formatting, linting, and unit tests (all tests use synthetic fixtures):

```bash
.venv/bin/ruff format --check src/em_denoise_refactor/
.venv/bin/ruff check src/em_denoise_refactor/
.venv/bin/pytest -v src/em_denoise_refactor/tests/test_n2n_improvements.py src/em_denoise_refactor/tests/test_n2n_duplicate_dataset.py
```
