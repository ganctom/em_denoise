# Changes on branch `Niklas-clusterize-scripts`

Written for Tomas. Two separate things happened on this branch, and you can take them
independently:

1. **Bug fixes** to the existing supervised pipeline (section 1). Each one is small and
   listed with a before/after so you can review or revert it individually.
2. **New Noise2Void + SLURM support** for running on the FMI cluster (section 2). Purely
   additive — new files only, nothing existing depends on them.

Section 3 lists things I deliberately did *not* touch, and section 4 is an honest account
of what was and was not actually executed.

Branch was fast-forwarded from `45f9800` to `dev-31` (`5eb866e`) before any of this, so
it sits on top of your latest work.

---

## 1. Bug fixes

### 1.1 `evaluate_test.py` ignored residual mode — reported metrics were meaningless

This is the most important one. Your trainer sets `predict_residual=True`, so the network
learns to output **noise**, and the denoised image is `noisy - output`. `run_inference.py`
and `compare_model_performance.py` both honour that flag, but `evaluate_test.py` was never
updated and treated the raw output as the final image.

```python
# before
denoised = torch.clamp(model(noisy_dev), 0.0, 1.0)

# after
model_output = model(noisy_dev)
denoised_signal = noisy_dev - model_output if predict_residual else model_output
denoised = torch.clamp(denoised_signal, 0.0, 1.0)
```

It also now reads the checkpoint the same way the other two scripts do (bare `state_dict`
or full bundle) and prints which mode it loaded.

**Impact:** any PSNR/SSIM number you got from `evaluate_test.py` on a residual-mode
checkpoint was measuring the wrong image and should be re-run. Numbers from
`compare_model_performance.py` were fine.

### 1.2 The old-checkpoint compatibility path in `compare_model_performance.py` did nothing

The script detects whether a checkpoint predates the strided-conv encoder:

```python
has_downsample_weights = any("down_samples" in k and "conv" in k for k in state_dict)
cfg_dict["use_conv_downsample"] = has_downsample_weights
```

but `UNetConfig` had no `use_conv_downsample` field. Pydantic ignores unknown keys by
default, so the flag was silently dropped, `ResidualUNet` always built the strided-conv
encoder, and loading a pre-change checkpoint failed on `load_state_dict(..., strict=True)`.

Rather than delete the dead code I made the field real, since the intent was right.
`UNetConfig.use_conv_downsample` (default `True`) now selects between learned strided
convs and the original `F.max_pool2d`, and the `Downsample` modules are only registered
when it is on — so the `state_dict` shape matches what old checkpoints expect.

**Impact:** benchmarking checkpoints from before the encoder change works again. New
checkpoints are unaffected (the default preserves current behaviour).

### 1.3 `run_experiment.py` ignored most of the config

```python
# before — config values silently unused
criterion = DynamicEMLoss(channels=1)
optimizer = torch.optim.Adam(..., lr=1e-4)
warmup_epochs = getattr(cfg.training, "warmup_epochs", 5)

# after
criterion = DynamicEMLoss(channels=unet_config.in_channels)
optimizer = torch.optim.AdamW(..., lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
warmup_epochs = cfg.training.warmup_epochs
```

Three things were going on: the learning rate was hard-coded to `1e-4` while
`config.yaml` said `0.0002`, `weight_decay` was configured but never reached the optimizer
(plain `Adam`, not `AdamW`), and the `getattr` fallback was defensive code for a field
that does exist on `TrainingConfig`.

**Impact:** your last runs trained at `lr=1e-4` with no weight decay, whatever the YAML
said. Runs are still valid — just not the hyperparameters recorded in
`resolved_config.yaml`. Worth knowing before you compare them against future runs.

### 1.4 Gradient clipping was hard-coded and missed the loss parameters

`TrainingConfig.max_grad_norm` existed but `trainer.py` used a literal `1.0`. It now takes
`max_grad_norm` as a constructor argument, passed from the config.

The clip also only covered `model.parameters()`. `DynamicEMLoss` holds trainable
log-variances that are in the same optimizer, so they were being updated unclipped. Both
parameter groups are now clipped together.

### 1.5 `run_inference.main()` re-parsed `sys.argv` when called from another script

`run_experiment.py` calls into `run_inference` after training. The old `main(test_dir,
ckpt_path)` accepted those arguments but then called `parser.parse_args()` internally,
picking up the *parent* process's flags and reading `--config` from whatever
`run_experiment.py` was given. It happened to work only because both take `--config`.

Split into:

- `run_from_config(config_path, test_dir=None, ckpt_path=None)` — no argument parsing,
  safe to call programmatically.
- `main()` — the CLI wrapper, now with `--test-dir` and `--checkpoint` flags.

The hard-coded `/Users/ganctoma/...` paths in the `__main__` block are gone; they are CLI
arguments now. `run_experiment.py` also skips the post-training inference with a warning
instead of crashing when the raw test tiles are missing.

### 1.6 Missing runtime dependencies

`tifffile`, `pyyaml` and `pillow` were imported throughout but never declared, so a fresh
`uv sync` produced an environment that crashed on import. Added, along with the three new
ones the N2V CLI needs (`cyclopts`, `loguru`, `natsort`). Also removed duplicate
`scikit-image` / `scipy` / `seaborn` / `torch` entries.

---

## 2. New: Noise2Void on the cluster

### 2.1 `src/em_denoise/n2v/train_n2v.py`

One self-contained script with two subcommands, `train` and `denoise`. It exists alongside
your `n2v_gemini*.py` drafts rather than replacing them — I did not want to throw away
work I do not have the context for.

Design choices worth flagging:

- **Reuses `backend.architecture.ResidualUNet`** instead of defining a new U-Net. The repo
  already had three separate U-Net definitions; this adds none. Anything you learn about
  the supervised model's capacity transfers directly.
- **Normalizes by dtype maximum to [0, 1]**, matching `backend/dataset.py`, so metrics and
  checkpoints stay comparable across both pipelines.
- **Stores `unet_config` in the checkpoint**, so `denoise` rebuilds the exact architecture
  without being handed the same flags again.
- **Sets `predict_residual: False`** in the checkpoint. N2V predicts the signal directly,
  unlike your supervised trainer. Anything consuming these checkpoints must not subtract.
- **Loss normalized by `mask.sum()`**, not by element count, so changing `--mask-ratio`
  does not silently rescale the effective learning rate.
- Uses `cyclopts` rather than `argparse` — Niklas's house standard for new code. Your
  existing scripts were left on `argparse`; mixing within a file would be worse.

Two subtleties in the masking implementation that are easy to get wrong, and that a test
caught while writing this:

1. A sampled offset of `(0, 0)` would copy a pixel onto itself, defeating the blind spot.
   Remapped to a real neighbour.
2. Out-of-bounds offsets are **reflected, not clamped**. Clamping maps a border pixel back
   onto itself (`y=0` with `dy=-1` clamps to `y=0`), reintroducing the same leak along
   every image edge. My first version had this bug; `test_masked_pixels_never_keep_their_own_value`
   is the regression test.

### 2.2 Cluster setup

The repo is a `uv` project, but FMI compute nodes have no outbound network, so an in-job
`uv sync` cannot work. The cluster path therefore uses `pixi`, matching the pattern already
working in Niklas's `napari_sam` and `vlad_segmentation` repos.

| File | Purpose |
| --- | --- |
| `pixi.toml` | conda-forge environment, `linux-64` only (`pytorch-gpu` has no Windows build) |
| `install.sh` | Run **once on a login node**: installs pixi into `infrastructure/` and solves the env |
| `init.sh` | Sourced to set pixi PATH/cache/TMPDIR for interactive use |
| `slurm/train_n2v.sh` | Training job |
| `slurm/denoise_n2v.sh` | Inference job |

`pixi.toml` puts `src/` on `PYTHONPATH` instead of installing the project as an editable
pypi dependency. An editable install would re-resolve `pyproject.toml` against PyPI and
risks pulling a second, CPU-only torch on top of the conda `pytorch-gpu` build. The
tradeoff is that it is less "proper" than a real install; the one-line switch is commented
in the file if you prefer that.

`uv` remains the local development path on Windows/macOS. Two dependency manifests is not
ideal, but they serve different platforms and the alternative is a job that cannot start.

Both SLURM scripts use `--account=dlthings --partition=main
--constraint="infiniband&gpuram32gb"`, copied from Niklas's working setup. **Change these
if your allocation differs** — they are at the top of each file. `pixi run --frozen` is
used deliberately so a job fails loudly rather than attempting a network solve.

---

## 3. Noticed but deliberately not changed

- **`TrainingConfig.min_lr` is unused.** It duplicates `eta_min`, which is what actually
  reaches `CosineAnnealingLR`. Removing one requires deciding which is authoritative —
  your call.
- **`n2v_gemini.py`, `n2v_gemini_0.{1,2,3}.py`** look like four iterations of the same
  script. If one is the keeper, deleting the rest would make the package a lot clearer.
- **`compare_model_performance.py` uses `torch.load(..., weights_only=False)`** while the
  other scripts use `True`. Fine for your own checkpoints, but it will execute arbitrary
  pickled code, so do not point it at a checkpoint you did not produce.
- **`config.yaml`'s `experiment_dir` contains literal backticks** in the path:
  ``.../`04_DynamicEMLoss_..._lr00002` ``. It works, but it will bite you in any shell command.
- **`README.md` was empty** — I filled it with run instructions rather than leaving it.
- The supervised scripts still must be run **by file path, not `python -m`**. Fixing
  `run_experiment.py`'s flat import (1.5) removed the reason that mattered, but I did not
  re-verify every entry point, so the documented commands stay as they were.

---

## 4. What was actually verified

Being explicit, because the environment here was constrained — no `.venv`, no torch, and a
slow connection that made installing a ~2 GB CUDA torch build impractical.

**Ran and passed:**

- `python -m py_compile` on every modified and new Python file.
- TOML validity of `pyproject.toml` and `pixi.toml`.
- A standalone NumPy reimplementation of the tiling and masking maths, over 10 image
  shapes and 3 mask ratios. This is what caught the clamp-vs-reflect bug in 2.1. Its
  assertions now live in `src/em_denoise/n2v/test_train_n2v.py`.

**Not run:**

- `pytest` — not installed locally. The test file compiles but has never been executed.
- Any actual training, inference or CUDA code path.
- `pixi install` and both SLURM scripts — these need the cluster.

So: the logic was checked where it could be checked without torch, and the bug fixes in
section 1 are small and reviewable by eye, but **nothing here has been run end-to-end.**
Treat the first cluster submission as a smoke test — start with `--epochs 1` on a handful
of images before committing to a long job.
