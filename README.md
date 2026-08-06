# em_denoise

Denoising of SBEM (serial block-face electron microscopy) micrographs.

Two approaches live in this repo:

- **Supervised** (`backend/` + `execution/`) — clean EM patches are degraded with a
  simulated Poisson–Gaussian noise model and a residual U-Net learns the inverse mapping.
- **Self-supervised** (`n2v/`) — Noise2Void, which needs only noisy images.

Recent changes and known issues are documented in [`docs/CHANGES.md`](docs/CHANGES.md).

## Setup

Local development uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

For the cluster, see [Running on the cluster](#running-on-the-cluster) — it uses pixi,
because compute nodes have no outbound network access.

## Supervised pipeline

Everything is driven by a YAML config; `src/em_denoise/execution/config.yaml` is the
template. It defines two roots and derives the rest by convention:

- `sandbox_root/{train,val,test}/{clean,noisy}/*.tif` — the expected dataset layout.
- `experiment_dir/` — receives `checkpoints/`, `logs/`, `predictions/`, `benchmarks/`.

Run scripts **by path, not with `python -m`**:

```bash
# Train, then run inference on the raw test tiles. Writes checkpoints/, logs/ under experiment_dir.
uv run src/em_denoise/execution/run_experiment.py --config src/em_denoise/execution/config.yaml

# Test-set metrics (PSNR/SSIM) + side-by-side visual panels under experiment_dir/test_visualizations/
uv run src/em_denoise/execution/evaluate_test.py --config src/em_denoise/execution/config.yaml

# Denoise a directory of images with a trained checkpoint
uv run src/em_denoise/execution/run_inference.py \
    --config src/em_denoise/execution/config.yaml \
    --test-dir /path/to/noisy --checkpoint /path/to/best_model.pt

# Compare several checkpoints on the same test set -> benchmarks/metric_comparisons.png
uv run src/em_denoise/execution/compare_model_performance.py \
    --config src/em_denoise/execution/config.yaml --ckpt-dir /path/to/checkpoints

# Watch training curves
tensorboard --logdir <experiment_dir>/logs --reload_interval=5 --port=6006
```

### Data preparation

These three scripts run in order and are configured by **editing the config object in
their `__main__` block** — they take no CLI arguments and contain absolute paths:

```bash
uv run src/em_denoise/data_ingestion/tiles_per_section.py   # -> section_tile_counts.csv
uv run src/em_denoise/data_ingestion/sbem_data_copier.py    # -> raw patches, split by section
uv run src/em_denoise/data_ingestion/create_nn_datasets.py  # -> paired clean/ + noisy/ trees
```

## Noise2Void

Needs only noisy images — no clean targets, no synthetic noise model.

```bash
# Train. Writes <out_dir>/checkpoints/best_model.pt and <out_dir>/logs/
uv run python -m em_denoise.n2v.train_n2v train /path/to/noisy /path/to/experiment \
    --epochs 50 --patch-size 128 --batch-size 32 --mask-ratio 0.02

# Denoise a directory using a trained checkpoint (tiled, so full-size tiles are fine)
uv run python -m em_denoise.n2v.train_n2v denoise /path/in /path/out \
    /path/to/experiment/checkpoints/best_model.pt

uv run python -m em_denoise.n2v.train_n2v --help
```

Expect the training loss to drop quickly for a few epochs and then flatten. Because the
loss is evaluated only at masked pixels, its absolute value is not comparable to the
supervised L1/SSIM losses — compare runs against each other, not across pipelines.

## Running on the cluster

Compute nodes have no outbound network, so the environment must be solved in advance.

**Once, on a login node:**

```bash
./install.sh     # installs pixi into infrastructure/ and solves the environment
```

**Then submit jobs from the repo root:**

```bash
sbatch slurm/train_n2v.sh --noisy-dir /path/to/noisy --out-dir /path/to/experiment \
    --epochs 50 --patch-size 128

sbatch slurm/denoise_n2v.sh -i /path/in -o /path/out \
    -c /path/to/experiment/checkpoints/best_model.pt
```

Logs land in `logs/<job-name>_<job-id>.out`. The training job prints the GPU it acquired
before starting, so a job that silently fell back to CPU is visible at the top of the log.

`--account`, `--partition` and `--constraint` are set at the top of each SLURM script —
adjust them to your allocation.

## Tests

```bash
uv run pytest src/em_denoise
```

## Repository layout

| Path | Contents |
| --- | --- |
| `src/em_denoise/backend/` | Library: model, dataset, losses, metrics, trainer |
| `src/em_denoise/execution/` | CLI entry points for the supervised pipeline |
| `src/em_denoise/data_ingestion/` | Patch extraction and dataset construction |
| `src/em_denoise/n2v/` | Noise2Void (`train_n2v.py` is the maintained entry point) |
| `slurm/` | SLURM submission scripts |
| `docs/` | Change log and design notes |
