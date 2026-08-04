# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Denoising of SBEM (serial block-face electron microscopy) micrographs. Two parallel approaches live
side by side:

- **Supervised** (`backend/` + `execution/`) — the main line. Clean EM patches are synthetically
  degraded with a Poisson–Gaussian + multiplicative-gain noise model, and a residual U-Net learns the
  inverse mapping.
- **Self-supervised** (`n2v/`) — Noise2Void variants, standalone scripts that share nothing with
  `backend/`. Four numbered `n2v_gemini*.py` files are iterative drafts of the same idea.

Data on disk is 8-bit TIFF patches; everything downstream normalizes to float32 in [0, 1].

## Environment & commands

`uv` project (`uv_build` backend, `src/` layout, Python ≥3.12). Note the global Pixi preference does
not apply here — this repo is uv/`uv.lock`-based.

```bash
uv sync
uv run ruff check --fix && uv run ruff format
uv run pytest src/em_denoise/data_ingestion/test_tile_per_section.py   # the only test file
pre-commit install                                                     # ruff + ruff-format
```

Run scripts **by path, not `-m`** — `run_experiment.py` does a flat `import run_inference`, which
only resolves because running a script by path puts its directory on `sys.path`:

```bash
uv run src/em_denoise/execution/run_experiment.py --config src/em_denoise/execution/config.yaml
uv run src/em_denoise/execution/compare_model_performance.py --config <cfg> \
    [--checkpoints a.pt b.pt | --ckpt-dir DIR]
uv run src/em_denoise/execution/evaluate_test.py --config <cfg>
tensorboard --logdir <experiment_dir>/logs --reload_interval=5 --port=6006
```

`run_experiment.py` trains **and then runs inference** on `sandbox_root/../raw_patches/test_raw`.

Data-ingestion scripts are run directly and are configured by **editing the config object in their
`__main__` block** — they take no CLI args and contain hard-coded absolute macOS/`/Volumes` paths:
`tiles_per_section.py` (builds `section_tile_counts.csv`) → `sbem_data_copier.py` (extracts patches,
splits) → `create_nn_datasets.py` (writes `clean/`+`noisy/` pairs).

## Architecture

1. **`data_ingestion/`** — dataset construction, ordered pipeline.
   - `tiles_per_section.py` — `ProcessPoolExecutor` scan of SOFIMA `tile_id_map.json` files, emits
     the section→tile-count CSV checked in at `src/em_denoise/section_tile_counts.csv`.
   - `sbem_data_copier.py` — OO pipeline (`ExtractionConfig` dataclass + `SectionQuotaCalculator` /
     `DataSplitter` / `PatchExtractor` / `DatasetWriter` orchestrated by `SBEMPatchPipeline`).
     Splits are assigned **by section** (`DataSplitter.get_split`), so patches from one section never
     straddle train/val/test. Also copies `n_test_raw` untouched full tiles for inference smoke tests.
   - `create_nn_datasets.py` — applies `add_em_noise` (Poisson shot → gain jitter → Gaussian read
     noise, clipped to 8-bit) to produce the paired tree.

2. **`backend/`** — importable library, no CLI. `config_schema.py` (pydantic `PipelineConfig`, single
   source of truth for a run), `dataset.py` (`SupervisedEMDataset`, caches the whole split in RAM by
   default), `architecture.py` (`ResidualUNet` + its own `UNetConfig`), `metrics.py` (torch PSNR/SSIM,
   kept on-device — no skimage in the loop), `loss.py`, `trainer.py`.

3. **`execution/`** — thin argparse wiring: config → dataset → model → trainer/evaluator. Logic
   belongs in `backend/`.

4. **`n2v/`** — self-contained Noise2Void scripts with their own `UNet`/`ModernUNet`, blind-spot
   masking datasets, and a `DenoisingStrategy` ABC. Not importable as a package (no `__init__.py`)
   and not wired into the supervised config. `run_n2v.py` is the entry point.

### Two conventions that pervade the supervised path

**Residual prediction.** The model predicts *noise*, not signal: `denoised = noisy - model_output`.
Controlled by `SupervisedTrainer(predict_residual=True)`, persisted into the checkpoint dict, and
re-read by both `run_inference.py` and `compare_model_performance.py`. Any new consumer of a
checkpoint must honor the flag or results are silently wrong.

**Compound checkpoint score.** `best_model.pt` is saved on improvement of
`_compute_compound_score` — `0.2·(PSNR/40) + 0.3·SSIM + 0.25·ESPI + 0.25·AUFRC`, hard-coded in
`trainer.py`, deliberately decoupled from the dynamic loss weights. Not plain val loss or PSNR.

### Loss design (`backend/loss.py`)

Two generations coexist. `ComprehensiveEMLoss` uses static YAML weights (`em_loss:` block).
`DynamicEMLoss` — what `run_experiment.py` actually uses — combines Charbonnier + focal-frequency +
generalized-Anscombe VST + Laplacian, balanced by **learnable log-variance parameters**. Two
consequences:

- The criterion holds trainable parameters and must be passed to the optimizer:
  `Adam(list(model.parameters()) + list(criterion.parameters()))`.
- Its `forward` returns `(loss, metrics_dict)`, not a scalar. `SupervisedTrainer` unpacks a 2-tuple,
  so **any criterion swapped in must return that tuple** — plain `nn.L1Loss` will break the loop.
  `criterion.state_dict()` is bundled into the checkpoint to preserve learned variances.

### Config contract

`PipelineConfig.paths` has two roots; everything else is derived by convention:

- `sandbox_root/{train,val,test}/{clean,noisy}/*.tif` — the layout `SupervisedEMDataset` requires
  (matching filenames in both subdirs, else it raises).
- `experiment_dir/` — `checkpoints/best_model.pt`, `logs/` (TensorBoard), `predictions/`,
  `test_visualizations/`, `benchmarks/`, plus `resolved_config.yaml` copied at train start.

`checkpoints:` is a `{tag: path}` map read only by `compare_model_performance.py`, whose resolution
cascade is `--checkpoints` → config block → `--ckpt-dir` glob → `experiment_dir/checkpoints/*.pt`.

There are **two U-Net config models**: `architecture.UNetConfig` (module-level, has `use_attention`)
and `config_schema.UNetConfigSchema` (the YAML surface). Execution scripts bridge them via
`UNetConfig(**cfg.unet.model_dump())`, so a new field must be added in both places.

### Cluster execution

Local dev is uv; the cluster path is **pixi** (`pixi.toml`, `install.sh`, `init.sh`, `slurm/`),
because FMI compute nodes have no outbound network and an in-job `uv sync` cannot work. `pixi.toml`
is `linux-64` only and puts `src/` on `PYTHONPATH` rather than installing the project, to avoid
re-resolving `pyproject.toml` against PyPI on top of the conda `pytorch-gpu` build. SLURM scripts
use `pixi run --frozen` so a job fails loudly instead of attempting a network solve.

### Known rough edges

- `TrainingConfig.min_lr` is declared but unused; `eta_min` is what reaches `CosineAnnealingLR`.
- `n2v_gemini.py` and `n2v_gemini_0.{1,2,3}.py` are four drafts of the same idea. `train_n2v.py` is
  the maintained N2V entry point; the drafts are kept only because they are the author's.
- `compare_model_performance.py` uses `torch.load(..., weights_only=False)` where the other scripts
  use `True` — it will execute arbitrary pickled code.
- `config.yaml` paths are the original author's macOS absolute paths — placeholders, not invariants.
  The current `experiment_dir` value contains literal backticks.
- `[project.scripts] em-denoise` points at a stub that only prints.
- Supervised scripts are run **by file path, not `python -m`** (see the commands above).

A set of correctness bugs (residual mode ignored in `evaluate_test.py`, a dead
`use_conv_downsample` compat path, config values bypassed in `run_experiment.py`, argv leakage in
`run_inference.py`, undeclared deps) was fixed on the `Niklas-clusterize-scripts` branch — see
`docs/CHANGES.md` for the before/after of each.

## Conventions

- Metrics and losses assume tensors clamped to [0, 1] with `data_range=1.0`.
- Device selection is duplicated per script as CUDA → MPS → CPU.
- Scripts use `print()` or stdlib `logging`; the repo predates the loguru preference — match the file
  you are editing rather than introducing a third style mid-module.
