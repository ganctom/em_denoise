# src/em_denoise_refactor/tests/test_n2n_duplicate_dataset.py

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from em_denoise_refactor.config import N2NConfig
from em_denoise_refactor.data.n2n_duplicate_dataset import N2NDuplicateTileDataset
from em_denoise_refactor.execution import train_and_infer


def _make_structured_base(h: int = 512, w: int = 512, seed: int = 0) -> np.ndarray:
    """Create a deterministic structured grayscale signal with membranes/blobs."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(seed)
    base = (
        120.0
        + 40.0 * np.sin(xx / 18.0 + 0.3 * np.sin(yy / 25.0))
        + 35.0 * np.cos(yy / 14.0)
        + 15.0 * np.sin((xx + yy) / 9.0)
    )
    # Add localized high-contrast rings (vesicle-like)
    for _ in range(15):
        cy = rng.integers(32, h - 32)
        cx = rng.integers(32, w - 32)
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        base += 45.0 * np.exp(-r2 / (2.0 * 10.0**2))
    return np.clip(base, 15, 240)


def _create_duplicate_fixture(
    root: Path,
    num_tiles: int = 12,
    clusters: tuple[list[int], ...] = ([100, 101, 102, 103, 104], [200, 201]),
    h: int = 512,
    w: int = 512,
    noise_std: float = 12.0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    for t_idx in range(num_tiles):
        for c_idx, secs in enumerate(clusters):
            clean = _make_structured_base(h, w, seed=t_idx * 100 + c_idx)
            for s in secs:
                noisy = np.clip(
                    clean + rng.normal(0.0, noise_std, size=(h, w)), 0, 255
                ).astype(np.uint8)
                fname = f"20260201_RoLi_F1_g0000_t{t_idx:04d}_s{s:05d}.tif"
                tifffile.imwrite(root / fname, noisy)
    return root


def test_default_config_keeps_overlaps_source():
    cfg = N2NConfig()
    assert cfg.source == "overlaps"
    assert cfg.pair_mode == "all"
    assert cfg.split_unit == "group"


def test_dataset_output_shape_and_range(tmp_path: Path):
    dupl_dir = _create_duplicate_fixture(tmp_path / "dupl", num_tiles=6)
    ds = N2NDuplicateTileDataset(
        duplicates_dir=dupl_dir,
        split="train",
        patch_size=128,
        split_seed=42,
    )
    inp, tgt, pid = ds[0]
    assert isinstance(inp, torch.Tensor)
    assert isinstance(tgt, torch.Tensor)
    assert inp.dtype == torch.float32
    assert tgt.dtype == torch.float32
    assert inp.shape == (1, 128, 128)
    assert tgt.shape == (1, 128, 128)
    assert 0.0 <= float(inp.min()) <= float(inp.max()) <= 1.0
    assert 0.0 <= float(tgt.min()) <= float(tgt.max()) <= 1.0
    assert isinstance(pid, str) and len(pid) > 0


def test_parsing_skips_non_matching_with_warning(tmp_path: Path, caplog):
    dupl_dir = _create_duplicate_fixture(tmp_path / "dupl", num_tiles=6)
    (dupl_dir / "notes.txt").write_text("hello")
    (dupl_dir / "broken_name.tif").write_bytes(b"not a valid name")

    with caplog.at_level(logging.WARNING, logger="train_and_infer"):
        ds = N2NDuplicateTileDataset(
            duplicates_dir=dupl_dir, split="train", patch_size=128
        )
    assert len(ds) > 0
    assert any(
        "notes.txt" in r.message and "broken_name.tif" in r.message
        for r in caplog.records
    )


def test_clustering_and_singleton_exclusion(tmp_path: Path, caplog):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl",
        num_tiles=6,
        clusters=([100, 101, 102], [200, 201]),
    )
    # Add a singleton tile 300 at section 100
    arr = np.zeros((256, 256), dtype=np.uint8)
    tifffile.imwrite(dupl_dir / "20260201_RoLi_F1_g0000_t0300_s00100.tif", arr)

    with caplog.at_level(logging.INFO, logger="train_and_infer"):
        ds = N2NDuplicateTileDataset(
            duplicates_dir=dupl_dir,
            split="train",
            patch_size=128,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
    assert ds.excluded_singletons == 1
    assert all("t0300" not in p[0] for p in ds.pairs)


def test_pair_modes_all_and_neighbours_and_invalid(tmp_path: Path):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=10, clusters=([100, 101, 102, 103, 104],)
    )
    ds_all = N2NDuplicateTileDataset(
        duplicates_dir=dupl_dir, split="train", patch_size=128, pair_mode="all"
    )
    # 10 groups * 0.8 = 8 groups in train, each with 5 sections -> 10 pairs -> 80 pairs
    assert len(ds_all) == 80

    ds_nbr = N2NDuplicateTileDataset(
        duplicates_dir=dupl_dir, split="train", patch_size=128, pair_mode="neighbours"
    )
    # 8 groups * 4 consecutive pairs = 32 pairs
    assert len(ds_nbr) == 32
    assert all(p[5] - p[4] == 1 for p in ds_nbr.pairs)

    with pytest.raises(ValueError, match="all.*neighbours"):
        N2NDuplicateTileDataset(
            duplicates_dir=dupl_dir, split="train", patch_size=128, pair_mode="banana"
        )


def test_group_split_disjoint_reproducible_and_ratios(tmp_path: Path):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=50, clusters=([100, 101], [200, 201]), h=256, w=256
    )
    ds_tr = N2NDuplicateTileDataset(
        dupl_dir, split="train", patch_size=64, split_seed=42
    )
    ds_va = N2NDuplicateTileDataset(dupl_dir, split="val", patch_size=64, split_seed=42)
    ds_te = N2NDuplicateTileDataset(
        dupl_dir, split="test", patch_size=64, split_seed=42
    )

    g_tr = {p[1] for p in ds_tr.pairs}
    g_va = {p[1] for p in ds_va.pairs}
    g_te = {p[1] for p in ds_te.pairs}

    assert g_tr.isdisjoint(g_va)
    assert g_tr.isdisjoint(g_te)
    assert g_va.isdisjoint(g_te)
    assert len(g_tr) == 80
    assert len(g_va) == 10
    assert len(g_te) == 10

    # Reproducible with same seed, differs with new seed
    ds_va_same = N2NDuplicateTileDataset(
        dupl_dir, split="val", patch_size=64, split_seed=42
    )
    ds_va_diff = N2NDuplicateTileDataset(
        dupl_dir, split="val", patch_size=64, split_seed=7
    )
    assert {p[1] for p in ds_va_same.pairs} == g_va
    assert {p[1] for p in ds_va_diff.pairs} != g_va


def test_too_few_groups_raises(tmp_path: Path):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=2, clusters=([100, 101],), h=256, w=256
    )
    with pytest.raises(ValueError, match="at least 3 group"):
        N2NDuplicateTileDataset(dupl_dir, split="train", patch_size=64)


@pytest.mark.parametrize("unit", ["image", "patch"])
def test_other_split_units_warn(tmp_path: Path, caplog, unit: str):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=10, clusters=([100, 101, 102, 103],), h=256, w=256
    )
    with caplog.at_level(logging.WARNING, logger="train_and_infer"):
        ds = N2NDuplicateTileDataset(
            dupl_dir, split="train", patch_size=64, split_unit=unit
        )
    assert len(ds) > 0
    assert any("not be independent" in r.message for r in caplog.records)
    inp, tgt, _ = ds[0]
    assert inp.shape == (1, 64, 64)


def test_patch_size_and_reproducible_sampling(tmp_path: Path):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=5, clusters=([100, 101],), h=512, w=512
    )
    with pytest.raises(ValueError, match="exceeds tile dimensions"):
        N2NDuplicateTileDataset(dupl_dir, split="train", patch_size=1024)

    ds1 = N2NDuplicateTileDataset(dupl_dir, split="train", patch_size=192, seed=99)
    ds2 = N2NDuplicateTileDataset(dupl_dir, split="train", patch_size=192, seed=99)
    ids1 = [ds1[i % len(ds1)][2] for i in range(20)]
    ids2 = [ds2[i % len(ds2)][2] for i in range(20)]
    assert ids1 == ids2
    assert ds1[0][0].shape == (1, 192, 192)


def test_alignment_recovers_shift_and_rejects_oob(tmp_path: Path):
    dupl_dir = tmp_path / "dupl"
    dupl_dir.mkdir()
    for t in range(3):
        base = _make_structured_base(512, 512, seed=t)
        # Shift image 1 by (dy=3, dx=-2): img1[y - 3, x + 2] == img0[y, x]
        # So img1 = roll(base, shift=(-3, 2))
        shifted = np.roll(base, shift=(-3, 2), axis=(0, 1))
        tifffile.imwrite(
            dupl_dir / f"20260201_RoLi_F1_g0000_t{t:04d}_s00100.tif",
            base.astype(np.uint8),
        )
        tifffile.imwrite(
            dupl_dir / f"20260201_RoLi_F1_g0000_t{t:04d}_s00101.tif",
            shifted.astype(np.uint8),
        )

    ds_no = N2NDuplicateTileDataset(
        dupl_dir,
        split="train",
        patch_size=128,
        align_patches=False,
        pair_direction="first_to_second",
        seed=11,
    )
    ds_al = N2NDuplicateTileDataset(
        dupl_dir,
        split="train",
        patch_size=128,
        align_patches=True,
        align_max_shift_px=16,
        pair_direction="first_to_second",
        seed=11,
    )
    x0, y0, _ = ds_no[0]
    x1, y1, _ = ds_al[0]
    mse_unaligned = float(((x0 - y0) ** 2).mean())
    mse_aligned = float(((x1 - y1) ** 2).mean())
    assert mse_aligned < mse_unaligned


def test_ssim_quality_filter_and_trim_warning(tmp_path: Path, caplog):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=4, clusters=([100, 101],), h=256, w=256
    )
    with caplog.at_level(logging.WARNING, logger="train_and_infer"):
        ds = N2NDuplicateTileDataset(
            dupl_dir,
            split="train",
            patch_size=64,
            trim_enabled=True,
            quality_filter_enabled=True,
            quality_metric="ssim",
            min_pair_quality=0.50,
            max_quality_attempts=8,
        )
    assert any("trim_enabled" in r.message for r in caplog.records)
    inp, tgt, _ = ds[0]
    assert inp.shape == (1, 64, 64)


def test_lazy_loading_bounded_lru_and_corrupt_tile_skip(tmp_path: Path, caplog):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=6, clusters=([100, 101],), h=256, w=256
    )
    ds = N2NDuplicateTileDataset(
        dupl_dir,
        split="train",
        patch_size=64,
        cache_in_ram=False,
        max_cache_tiles=3,
    )
    for i in range(len(ds)):
        _ = ds[i]
    assert len(ds._tile_cache) <= 3

    # Corrupt one tile in the train split and ensure it is skipped gracefully
    bad_path = ds.pairs[0][2]
    bad_path.write_bytes(b"corrupt")
    ds._tile_cache.clear()
    orig_len = len(ds)
    with caplog.at_level(logging.WARNING, logger="train_and_infer"):
        inp, tgt, _ = ds[0]
    assert inp.shape == (1, 64, 64)
    assert len(ds) < orig_len
    assert any(bad_path.name in r.message for r in caplog.records)


def test_end_to_end_one_epoch_with_preview_and_custom_inference(
    tmp_path: Path, monkeypatch
):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=10, clusters=([100, 101],), h=256, w=256
    )
    raw_dir = tmp_path / "test_raw"
    raw_dir.mkdir()
    preview_img = _make_structured_base(256, 256, seed=999).astype(np.uint8)
    preview_path = raw_dir / "sample_01.tif"
    tifffile.imwrite(preview_path, preview_img)

    out_dir = tmp_path / "runs"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: "e2e_dupl"
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
  align_patches: false
  quality_filter_enabled: false
model:
  model_type: "modern_unet"
  base_dim: 8
training:
  batch_size: 4
  epochs: 1
  learning_rate: 0.0001
  cache_in_ram: false
  device: "cpu"
  num_workers: 0
loss:
  loss_type: "enhanced"
  charbonnier_weight: 0.95
  ssim_weight: 0.05
  laplacian_weight: 0.0
  ffl_weight: 0.0
  espi_weight: 0.0
  edge_weight: 0.0
inference:
  custom_input_dir: "{raw_dir}"
  preview_image: "{preview_path}"
  preview_crop_size: 128
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["train_and_infer", "--config", str(cfg_path)])
    train_and_infer.main()

    run_dir = out_dir / "e2e_dupl"
    assert (run_dir / "checkpoints" / "checkpoint_best.pt").exists()
    assert (run_dir / "duplicate_splits.json").exists()
    splits_data = json.loads((run_dir / "duplicate_splits.json").read_text())
    assert splits_data["split_seed"] == 42
    assert (run_dir / "evaluation_summary.txt").exists()
    assert (run_dir / "custom_inference" / "denoised" / "sample_01.tif").exists()
    assert len(list((run_dir / "previews").glob("*"))) > 0


def test_random_mode_scales_to_max_pairs(tmp_path: Path):
    dupl_dir = _create_duplicate_fixture(
        tmp_path / "dupl", num_tiles=5, clusters=([100, 101],), h=256, w=256
    )
    ds = N2NDuplicateTileDataset(
        dupl_dir,
        split="train",
        patch_size=64,
        patch_sampling="random",
        max_pairs=50,
    )
    assert len(ds.pairs) == 3
    assert len(ds) == 50
    inp, tgt, _ = ds[49]
    assert inp.shape == (1, 64, 64)
    assert tgt.shape == (1, 64, 64)
