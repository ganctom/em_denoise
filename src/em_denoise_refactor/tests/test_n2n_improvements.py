# src/em_denoise_refactor/tests/test_n2n_improvements.py

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from scipy.ndimage import shift as ndi_shift
import tifffile
import torch

from em_denoise_refactor.config import PipelineConfig
from em_denoise_refactor.data.n2n_duplicate_dataset import N2NDuplicateTileDataset
from em_denoise_refactor.execution import run_classical_baselines, train_and_infer
from em_denoise_refactor.models.residual_unet import ModernUNet
from em_denoise_refactor.tests.test_n2n_duplicate_dataset import (
    _create_duplicate_fixture,
    _make_structured_base,
)


def test_cosine_scheduler_with_warmup():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-3)
    sched = train_and_infer._build_lr_scheduler(
        optimizer=opt,
        epochs=10,
        lr_scheduler="cosine",
        lr_min=1e-6,
        warmup_epochs=2,
    )
    assert sched is not None

    lrs = []
    for _ in range(10):
        lrs.append(float(opt.param_groups[0]["lr"]))
        opt.step()
        sched.step()

    # Warmup ramps up across epochs 1-2 to peak at epoch 3, then decays smoothly
    assert lrs[0] < lrs[1] < lrs[2]
    assert np.isclose(lrs[2], 1e-3, rtol=1e-4)
    assert lrs[-1] < lrs[2]
    assert float(opt.param_groups[0]["lr"]) == pytest.approx(1e-6, rel=1e-2)


def test_noise_corrected_mse_estimation():
    rng = np.random.default_rng(7)
    clean = np.linspace(0.2, 0.8, 128 * 128, dtype=np.float32).reshape(128, 128)
    sigma = 0.12
    n1 = rng.normal(0.0, sigma, size=(128, 128)).astype(np.float32)
    n2 = rng.normal(0.0, sigma, size=(128, 128)).astype(np.float32)
    x = clean + n1
    y = clean + n2
    # Predictor has known clean signal error delta = 0.04
    pred = clean + 0.04

    raw_pair_mse = float(np.mean((x - y) ** 2))
    pred_mse = float(np.mean((pred - y) ** 2))
    true_clean_mse = float(np.mean((pred - clean) ** 2))

    est_clean_mse, est_clean_psnr = train_and_infer.compute_noise_corrected_metrics(
        pred_mse, raw_pair_mse
    )
    assert abs(est_clean_mse - true_clean_mse) < 0.002
    assert abs(est_clean_psnr - (-10.0 * np.log10(true_clean_mse))) < 3.0


def test_subpixel_alignment_beats_integer_shift(tmp_path: Path):
    dupl_dir = tmp_path / "dupl"
    dupl_dir.mkdir()
    for t in range(3):
        base = _make_structured_base(384, 384, seed=t + 10)
        shifted = np.clip(
            ndi_shift(base, shift=(2.4, -1.6), order=3, mode="nearest"), 0, 255
        )
        tifffile.imwrite(
            dupl_dir / f"20260201_RoLi_F1_g0000_t{t:04d}_s00100.tif",
            base.astype(np.uint8),
        )
        tifffile.imwrite(
            dupl_dir / f"20260201_RoLi_F1_g0000_t{t:04d}_s00101.tif",
            shifted.astype(np.uint8),
        )

    ds_int = N2NDuplicateTileDataset(
        dupl_dir,
        split="train",
        patch_size=128,
        align_patches=True,
        align_subpixel=False,
        align_max_shift_px=16,
        pair_direction="first_to_second",
        patch_sampling="tile",
        tile_stride=64,
        seed=11,
    )
    ds_sub = N2NDuplicateTileDataset(
        dupl_dir,
        split="train",
        patch_size=128,
        align_patches=True,
        align_subpixel=True,
        align_max_shift_px=16,
        pair_direction="first_to_second",
        patch_sampling="tile",
        tile_stride=64,
        seed=11,
    )

    # First tile at (0, 0) triggers OOB rejection without clamping when shifted
    x_sub, y_sub, _ = ds_sub[0]
    assert ds_sub.rejected_oob_count > 0

    # Compare interior aligned patches (center tile)
    mses_int = []
    mses_sub = []
    for idx in range(len(ds_int)):
        xi, yi, pid_i = ds_int[idx]
        if "_y64_x64" in pid_i or "_y128_x128" in pid_i:
            mses_int.append(float(((xi - yi) ** 2).mean()))
    for idx in range(len(ds_sub)):
        xs, ys, pid_s = ds_sub[idx]
        if "_y64_x64" in pid_s or "_y128_x128" in pid_s:
            mses_sub.append(float(((xs - ys) ** 2).mean()))

    assert len(mses_int) > 0 and len(mses_sub) > 0
    assert float(np.mean(mses_sub)) < float(np.mean(mses_int))


def test_checkpoint_stores_config_and_preview_has_residual(tmp_path: Path):
    model = ModernUNet(in_channels=1, out_channels=1, base_dim=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = train_and_infer._build_lr_scheduler(
        opt, epochs=5, lr_scheduler="cosine", warmup_epochs=1
    )
    cfg = PipelineConfig()
    cfg_dict = train_and_infer._serialize_config(cfg)

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    trainer = train_and_infer.DenoisingTrainer(
        model=model,
        optimizer=opt,
        criterion=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        checkpoints_dir=ckpt_dir,
    )
    ckpt_path = trainer.save_checkpoint(
        epoch=1,
        val_loss=0.1,
        val_psnr=20.0,
        val_ssim=0.5,
        is_best=True,
        scheduler=sched,
        config=cfg_dict,
    )
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["config"] is not None
    assert loaded["config"]["model"]["model_type"] == cfg.model.model_type
    assert loaded["scheduler_state_dict"] is not None

    raw_crop = (_make_structured_base(128, 128, seed=3) / 255.0).astype(np.float32)
    previews_dir = tmp_path / "previews"
    stats = train_and_infer._save_checkpoint_preview(
        model=model,
        raw_crop=raw_crop,
        previews_dir=previews_dir,
        epoch=1,
        device=torch.device("cpu"),
    )
    assert set(stats.keys()) == {"mean", "std", "ac_x", "ac_y"}
    png_path = previews_dir / "epoch_001_comparison.png"
    assert png_path.exists()
    with Image.open(png_path) as im:
        w, h = im.size
        # 3-panel figure has aspect ratio ~3:1 (width > 2.2 * height)
        assert w > h * 2.2


def test_classical_baselines_cli(tmp_path: Path, monkeypatch):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=5, clusters=([100, 101],), h=128, w=128
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    tifffile.imwrite(
        raw_dir / "sample.tif",
        _make_structured_base(128, 128, seed=42).astype(np.uint8),
    )

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
splitting:
  strategy: "random"
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
extraction:
  patch_size: 64
n2n:
  enabled: true
  source: "duplicates"
  duplicates_dir: "{dupl_dir}"
  max_test_pairs: 2
""",
        encoding="utf-8",
    )

    out_dir = tmp_path / "baselines_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_classical_baselines",
            "--input-dir",
            str(raw_dir),
            "--output-dir",
            str(out_dir),
            "--config",
            str(cfg_path),
        ],
    )
    run_classical_baselines.main()

    for method in ("gaussian_0.8", "gaussian_1.2", "bilateral", "nlm"):
        assert (out_dir / method / "sample.tif").exists()
        assert (out_dir / method / "comparisons" / "sample_comparison.png").exists()

    summary_path = out_dir / "classical_baselines_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert set(summary.keys()) == {"gaussian_0.8", "gaussian_1.2", "bilateral", "nlm"}
    assert "est_clean_psnr" in summary["nlm"]


def test_end_to_end_baseline_pipeline(tmp_path: Path, monkeypatch):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=10, clusters=([100, 101],), h=256, w=256
    )
    raw_dir = tmp_path / "test_raw"
    raw_dir.mkdir()
    preview_path = raw_dir / "sample_01.tif"
    tifffile.imwrite(
        preview_path, _make_structured_base(256, 256, seed=99).astype(np.uint8)
    )

    out_dir = tmp_path / "runs"
    cfg_path = tmp_path / "baseline.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: "e2e_baseline"
  output_dir: "{out_dir}"
splitting:
  strategy: "random"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
extraction:
  patch_size: 64
n2n:
  enabled: true
  source: "duplicates"
  duplicates_dir: "{dupl_dir}"
  pair_mode: "all"
  split_unit: "group"
  split_seed: 42
  pair_direction: "random_swap"
  align_patches: true
  align_subpixel: true
  quality_filter_enabled: false
model:
  model_type: "residual_unet"
  base_dim: 8
training:
  batch_size: 4
  epochs: 2
  learning_rate: 0.0003
  lr_scheduler: "cosine"
  lr_min: 0.000001
  warmup_epochs: 1
  seed: 42
  cache_in_ram: false
  device: "cpu"
  num_workers: 0
loss:
  loss_type: "composite"
  mse_weight: 1.0
  edge_weight: 0.0
  ssim_weight: 0.0
inference:
  custom_input_dir: "{raw_dir}"
  preview_image: "{preview_path}"
  preview_crop_size: 128
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["train_and_infer", "--config", str(cfg_path)])
    train_and_infer.main()

    run_dir = out_dir / "e2e_baseline"
    ckpt = torch.load(
        run_dir / "checkpoints" / "checkpoint_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ckpt["config"]["model"]["model_type"] == "residual_unet"
    assert ckpt["scheduler_state_dict"] is not None
    assert (run_dir / "duplicate_splits.json").exists()
    assert (run_dir / "previews" / "epoch_002_comparison.png").exists()
    assert (run_dir / "custom_inference" / "denoised" / "sample_01.tif").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "est_clean_psnr" in summary["metrics"]


def test_resume_training_from_checkpoint(tmp_path: Path, monkeypatch):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=10, clusters=([100, 101],), h=128, w=128
    )
    out_dir = tmp_path / "runs"
    cfg_path = tmp_path / "resume_cfg.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: "resume_test"
  output_dir: "{out_dir}"
splitting:
  strategy: "random"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
extraction:
  patch_size: 64
n2n:
  enabled: true
  source: "duplicates"
  duplicates_dir: "{dupl_dir}"
  align_patches: false
  quality_filter_enabled: false
model:
  model_type: "modern_unet"
  base_dim: 8
training:
  batch_size: 4
  epochs: 1
  learning_rate: 0.0003
  lr_scheduler: "cosine"
  lr_min: 0.000001
  warmup_epochs: 1
  seed: 42
  device: "cpu"
loss:
  loss_type: "composite"
  mse_weight: 1.0
  edge_weight: 0.0
  ssim_weight: 0.0
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["train_and_infer", "--config", str(cfg_path)])
    train_and_infer.main()

    run_dir = out_dir / "resume_test"
    ckpt1 = run_dir / "checkpoints" / "checkpoint_epoch_001.pt"
    assert ckpt1.exists()

    # Resume up to epoch 2
    cfg_path.write_text(
        cfg_path.read_text().replace("epochs: 1", "epochs: 2"), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_and_infer", "--config", str(cfg_path), "--resume", str(ckpt1)],
    )
    train_and_infer.main()

    ckpt2 = run_dir / "checkpoints" / "checkpoint_epoch_002.pt"
    assert ckpt2.exists()
    loaded2 = torch.load(ckpt2, map_location="cpu", weights_only=False)
    assert loaded2["epoch"] == 2


def test_overlap_dataset_materializes_compact_tile_crops(tmp_path: Path):
    import tarfile

    from em_denoise_refactor.data.n2n_dataset import N2NOverlapPairDataset

    ov_dir = tmp_path / "overlaps"
    shards_dir = ov_dir / "shards"
    shards_dir.mkdir(parents=True)

    # Create 20 sections (so mod10 gives 16 train, 2 val, 2 test) of (2, 192, 192) TIFFs
    tar_path = shards_dir / "shard_0000.tar"
    with tarfile.open(tar_path, "w") as tf:
        for sec in range(20):
            base = _make_structured_base(192, 192, seed=sec)
            pair = np.stack([base, base], axis=0).astype(np.uint8)
            tif_bytes_path = tmp_path / f"s{sec:05d}_g0000_t0000_t0001_h.tif"
            tifffile.imwrite(tif_bytes_path, pair)
            tf.add(tif_bytes_path, arcname=tif_bytes_path.name)

    ds = N2NOverlapPairDataset(
        overlaps_dir=ov_dir,
        split="train",
        patch_size=64,
        split_strategy="mod10",
        cache_in_ram=True,
        patch_sampling="tile",
        align_patches=True,
        quality_filter_enabled=True,
        quality_metric="ssim",
        min_pair_quality=0.50,
        max_pairs=10,
    )
    assert ds.cached_pairs is None
    assert ds.cached_tile_crops is not None
    assert ds.cached_tile_crops.shape == (10, 2, 64, 64)
    assert ds.cached_tile_crops.dtype == np.uint8
    assert len(ds) == 10
    inp, tgt, pid = ds[0]
    assert inp.shape == (1, 64, 64)
    assert tgt.shape == (1, 64, 64)


def test_save_all_epochs_false_writes_checkpoint_last(tmp_path: Path):
    model = ModernUNet(in_channels=1, out_channels=1, base_dim=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    trainer = train_and_infer.DenoisingTrainer(
        model=model,
        optimizer=opt,
        criterion=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        checkpoints_dir=ckpt_dir,
    )
    trainer.save_checkpoint(
        epoch=1,
        val_loss=0.2,
        val_psnr=20.0,
        val_ssim=0.5,
        is_best=True,
        save_all_epochs=False,
    )
    assert (ckpt_dir / "checkpoint_last.pt").exists()
    assert (ckpt_dir / "checkpoint_best.pt").exists()
    assert not (ckpt_dir / "checkpoint_epoch_001.pt").exists()
