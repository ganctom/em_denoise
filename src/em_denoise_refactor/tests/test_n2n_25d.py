"""Unit and integration tests for Tier 3 True 2.5D Noise2Noise."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
import tifffile
import torch

from em_denoise_refactor.config import PipelineConfig
from em_denoise_refactor.data.inference_25d_dataset import Custom25DTIFFStackDataset
from em_denoise_refactor.data.n2n_25d_dataset import (
    N2N25DOverlapDataset,
    assert_disjoint_z_supports,
)
from em_denoise_refactor.data.z_registration import (
    check_tile_consistency,
    estimate_prior_centered_shift,
    filter_section_consensus,
    is_confident_lock,
    masked_dog_ncc_shift,
)
from em_denoise_refactor.execution import diagnostics_25d, train_and_infer
from em_denoise_refactor.models.factory import build_model
from em_denoise_refactor.models.residual_unet import (
    ResidualUNet,
    UNetConfig,
)


def _make_em_texture(h: int, w: int, seed: int) -> np.ndarray:
    """Create deterministic rich multi-scale EM-like uint8 texture."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    tex = (
        0.55 * gaussian_filter(raw, sigma=2.5)
        + 0.35 * gaussian_filter(raw, sigma=7.0)
        + 0.10 * raw
    )
    tex = (tex - tex.mean()) / (tex.std() + 1e-6)
    img = np.clip(128.0 + 36.0 * tex, 10.0, 245.0)
    return img.astype(np.uint8)


def _create_synthetic_25d_overlaps(
    root: Path,
    sections: List[int],
    stripe_h: int = 160,
    stripe_w: int = 320,
    shard_split_sec: int | None = None,
    phys_shifts_by_sec: Dict[int, Tuple[int, int]] | None = None,
    inject_duplicate_row: bool = False,
    inject_second_grid: bool = False,
    inject_tile_flip_sec: int | None = None,
    inject_missed_cut_sec: int | None = None,
    inject_top_margin_sentinel: bool = False,
    inject_corrupt_stripe_shift: Tuple[int, int, int] | None = None,
) -> Path:
    """Build synthetic multi-shard .tar + manifest.parquet with known Z-shifts.

    All sections share a smoothly evolving base volume with known physical shifts
    so exact pixel alignment and geometric detectors can be verified deterministically.
    """
    root.mkdir(parents=True, exist_ok=True)
    shards_dir = root / "shards"
    shards_dir.mkdir(exist_ok=True)

    # Create a large synthetic canvas per tile_pair so shifted crops stay strictly valid
    pad_y, pad_x = 40, 40
    base_v_1 = _make_em_texture(stripe_h + 2 * pad_y, stripe_w + 2 * pad_x, seed=101)
    base_v_2 = _make_em_texture(stripe_h + 2 * pad_y, stripe_w + 2 * pad_x, seed=202)
    base_h_1 = _make_em_texture(stripe_w + 2 * pad_x, stripe_h + 2 * pad_y, seed=303)

    phys_shifts = phys_shifts_by_sec or {}
    # Assign sections to 1 or 2 shards to test cross-shard boundaries
    shard_groups: Dict[str, List[int]] = {}
    for s in sections:
        if shard_split_sec is not None and s >= shard_split_sec:
            sh = f"shards/shard_g0000_s{shard_split_sec:05d}_s{max(sections):05d}.tar"
        else:
            sh = f"shards/shard_g0000_s{min(sections):05d}_s{(shard_split_sec - 1) if shard_split_sec else max(sections):05d}.tar"
        shard_groups.setdefault(sh, []).append(s)

    manifest_rows = []
    rng_noise = np.random.default_rng(777)

    for shard_rel, sec_list in sorted(shard_groups.items()):
        tar_path = root / shard_rel
        with tarfile.open(tar_path, "w") as tf:
            for s in sec_list:
                # Cumulative mild physical drift per section (exact integer)
                pdy, pdx = phys_shifts.get(s, ((s % 5) - 2, ((s // 2) % 5) - 2))
                # Distinct synthetic intensity evolution per section (unless missed cut!)
                z_evol_sec = (
                    (s - 1)
                    if (
                        inject_missed_cut_sec is not None and s == inject_missed_cut_sec
                    )
                    else s
                )

                tile_specs = [
                    # (grid, anc, nb, orient, ftid, base_canvas, h_s, w_s, fy0_base, fx0_base)
                    (
                        0,
                        10,
                        35,
                        "vertical",
                        10,
                        base_v_1,
                        stripe_h,
                        stripe_w,
                        2850,
                        100,
                    ),
                    (
                        0,
                        10,
                        11,
                        "horizontal",
                        10,
                        base_h_1,
                        stripe_w,
                        stripe_h,
                        100,
                        3900,
                    ),
                    (
                        0,
                        12,
                        37,
                        "vertical",
                        12,
                        base_v_2,
                        stripe_h,
                        stripe_w,
                        2850,
                        100,
                    ),
                ]
                if inject_second_grid:
                    tile_specs.append(
                        (
                            1,
                            10,
                            35,
                            "vertical",
                            10,
                            base_v_2,
                            stripe_h,
                            stripe_w,
                            2850,
                            100,
                        )
                    )

                for (
                    grid,
                    anc,
                    nb,
                    orient,
                    ftid,
                    canvas,
                    hs,
                    ws,
                    fy_b,
                    fx_b,
                ) in tile_specs:
                    # Vary fy0/fx0 slightly across sections to test prior-centered shift
                    dfy0 = (s % 3) - 1
                    dfx0 = ((s + 1) % 3) - 1
                    fy0 = fy_b + dfy0
                    fx0 = fx_b + dfx0

                    eff_pdy, eff_pdx = pdy, pdx
                    if (
                        inject_corrupt_stripe_shift is not None
                        and s == inject_corrupt_stripe_shift[0]
                        and anc == 10
                        and orient == "vertical"
                        and grid == 0
                    ):
                        eff_pdy += inject_corrupt_stripe_shift[1]
                        eff_pdx += inject_corrupt_stripe_shift[2]

                    # Crop from canvas at (pad_y + dfy0 - eff_pdy, pad_x + dfx0 - eff_pdx)
                    # Wait: check exact relation!
                    # Let true world pixel Y_world = y + fy0 + eff_pdy_world.
                    # If canvas[pad_y + y + dfy0 - eff_pdy, pad_x + x + dfx0 - eff_pdx] is sampled:
                    # Between s0 and s1, same canvas pixel satisfies:
                    # y1 + dfy0_1 - eff_pdy_1 = y0 + dfy0_0 - eff_pdy_0
                    # => y1 = y0 - [(dfy0_1 - dfy0_0) - (eff_pdy_1 - eff_pdy_0)]
                    # => dy = (fy0_1 - fy0_0) - (eff_pdy_1 - eff_pdy_0), so phys_dy = -(eff_pdy_1 - eff_pdy_0).
                    cy0 = pad_y + dfy0 - eff_pdy
                    cx0 = pad_x + dfx0 - eff_pdx
                    clean = canvas[cy0 : cy0 + hs, cx0 : cx0 + ws].astype(np.float32)

                    # Realistic Z-evolution: normal cuts have DoG-NCC ~ 0.60;
                    # missed-cut section duplicates s-1's exact surface (DoG-NCC ~ 0.96).
                    mod_rng = np.random.default_rng(9000 + z_evol_sec * 31 + anc)
                    z_tex = gaussian_filter(
                        mod_rng.normal(0.0, 1.0, size=(hs, ws)).astype(np.float32),
                        sigma=2.5,
                    )
                    z_tex = 36.0 * (z_tex - z_tex.mean()) / (z_tex.std() + 1e-6)
                    clean_z = 128.0 + 0.78 * (clean - 128.0) + 0.62 * z_tex
                    n0 = rng_noise.normal(0.0, 6.0, size=(hs, ws)).astype(np.float32)
                    n1 = rng_noise.normal(0.0, 6.0, size=(hs, ws)).astype(np.float32)

                    p0 = np.clip(clean_z + n0, 0.0, 255.0).astype(np.uint8)
                    p1 = np.clip(clean_z + n1, 0.0, 255.0).astype(np.uint8)

                    if inject_top_margin_sentinel and orient == "vertical":
                        # Bright sentinel in top 20 px of page 1 (must NEVER leak into trimmed target!)
                        p1[:20, :] = 255

                    pair_arr = np.stack([p0, p1], axis=0)
                    buf = io.BytesIO()
                    tifffile.imwrite(buf, pair_arr, compression=None)
                    raw_bytes = buf.getvalue()

                    orient_code = "v" if orient == "vertical" else "h"
                    pair_id = f"s{s:05d}_g{grid:04d}_t{anc:04d}_t{nb:04d}_{orient_code}"
                    member_name = f"{pair_id}.tif"

                    ti = tarfile.TarInfo(name=member_name)
                    ti.size = len(raw_bytes)
                    # Record exact data byte offset (current tar pos + 512 header)
                    tar_file_pos = tf.fileobj.tell()
                    data_offset = tar_file_pos + 512
                    tf.addfile(ti, io.BytesIO(raw_bytes))

                    eff_ftid = (
                        nb
                        if (
                            inject_tile_flip_sec is not None
                            and s == inject_tile_flip_sec
                            and anc == 12
                        )
                        else ftid
                    )
                    row = (
                        pair_id,
                        s,
                        grid,
                        orient,
                        eff_ftid,
                        nb,
                        anc,
                        nb,
                        True,
                        0,
                        0,
                        0,
                        0,
                        hs,
                        ws,
                        fy0,
                        fx0,
                        0,
                        0,
                        shard_rel,
                        member_name,
                        data_offset,
                        len(raw_bytes),
                        "none",
                    )
                    manifest_rows.append(row)
                    if (
                        inject_duplicate_row
                        and s == sections[len(sections) // 2]
                        and anc == 10
                        and orient == "vertical"
                        and grid == 0
                    ):
                        manifest_rows.append(row)

    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE m (
            pair_id VARCHAR, section_num INTEGER, grid_num SMALLINT, orientation VARCHAR,
            first_exposed_tile_id INTEGER, second_exposed_tile_id INTEGER,
            anchor_tile_id INTEGER, neighbor_tile_id INTEGER, anchor_exposed_first BOOLEAN,
            grid_row_anchor SMALLINT, grid_col_anchor SMALLINT,
            coarse_dx INTEGER, coarse_dy INTEGER, height INTEGER, width INTEGER,
            first_exposed_crop_y0 INTEGER, first_exposed_crop_x0 INTEGER,
            second_exposed_crop_y0 INTEGER, second_exposed_crop_x0 INTEGER,
            shard_relpath VARCHAR, member_name VARCHAR,
            tif_byte_offset BIGINT, tif_byte_length BIGINT, compression VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO m VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        manifest_rows,
    )
    con.execute(f"COPY m TO '{root / 'manifest.parquet'}' (FORMAT PARQUET)")
    con.close()
    return root


# -------------------------------------------------------------------------
# 1. Registration & Geometric Detector Tests
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "orient,h,w", [("vertical", 180, 640), ("horizontal", 640, 180)]
)
def test_prior_centered_registration_exact_recovery_and_nulls(
    orient: str, h: int, w: int
):
    canvas = _make_em_texture(h + 80, w + 80, seed=11 if orient == "vertical" else 22)
    ref = canvas[40 : 40 + h, 40 : 40 + w].copy()
    # True shift: mov[y - dy, x - dx] == ref[y, x] -> sample mov at (40 + dy, 40 + dx)
    true_dy, true_dx = 11, -14
    mov = canvas[
        40 + true_dy : 40 + true_dy + h, 40 + true_dx : 40 + true_dx + w
    ].copy()

    # Add bright charging artifact into masked left margin (M7): must not break lock!
    mask_l = 80 if orient == "vertical" else 24
    ref[:, :mask_l] = 255
    mov[:, :mask_l] = 0

    res = estimate_prior_centered_shift(
        ref, mov, prior_dy=8, prior_dx=-10, search_half_px=32, mask_left_px=mask_l
    )
    assert res.valid
    assert (res.dy, res.dx) == (true_dy, true_dx)
    assert is_confident_lock(res)

    # C2: Verify ShiftResult.psr equals coarse-stage PSR exactly
    coarse = masked_dog_ncc_shift(
        ref,
        mov,
        center_dy=8,
        center_dx=-10,
        half_dy=32,
        half_dx=32,
        mask_left_px=mask_l,
        downsample=2,
    )
    assert res.psr == pytest.approx(coarse.psr, rel=1e-6)

    # Negative control: uncorrelated stripe (both _v and _h!)
    uncorr = _make_em_texture(h, w, seed=999)
    res_null = estimate_prior_centered_shift(
        ref, uncorr, prior_dy=0, prior_dx=0, search_half_px=32, mask_left_px=mask_l
    )
    assert not is_confident_lock(res_null)

    # Negative control: displaced beyond search box (+-32)
    res_disp = estimate_prior_centered_shift(
        ref,
        mov,
        prior_dy=true_dy + 60,
        prior_dx=true_dx + 60,
        search_half_px=32,
        mask_left_px=mask_l,
    )
    assert not is_confident_lock(res_disp)


def test_geometric_tile_consistency_and_consensus_reject_5_to_10px_mislock():
    # A 7 px false shift passes loose 12 px section consensus, but MUST fail <=2 px tile consistency (C3)
    tile_shifts = {
        ("v", 10): (0, 10, 2, -1),
        ("h", 10): (
            0,
            10,
            9,
            -1,
        ),  # 7 px disagreement on same first_exposed_tile_id=10!
        ("v", 11): (0, 11, 2, -1),
        ("h", 11): (0, 11, 3, -1),  # 1 px agreement -> kept
    }
    kept_tile = check_tile_consistency(tile_shifts, tol_px=2)
    assert ("v", 10) not in kept_tile and ("h", 10) not in kept_tile
    assert ("v", 11) in kept_tile and ("h", 11) in kept_tile

    # Section consensus rejects +20 px outlier while keeping 6 coherent stripes
    sec_shifts = {f"s{i}": (2, -1) for i in range(6)}
    sec_shifts["outlier"] = (22, -1)
    kept_cons = filter_section_consensus(sec_shifts, tol_px=12.0, min_samples=5)
    assert "outlier" not in kept_cons
    assert len(kept_cons) == 6


# -------------------------------------------------------------------------
# 2. Model Center-Channel Residual Skip, bf16 & Tiled Evaluator Tests
# -------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["residual_unet", "connectomics", "modern_unet"])
@pytest.mark.parametrize("in_ch", [1, 3, 5])
def test_models_center_channel_skip_and_gradients(model_type: str, in_ch: int):
    model = build_model(
        model_type=model_type, in_channels=in_ch, out_channels=1, base_dim=8
    )
    # ResidualUNet zero-initializes final_conv for step-0 identity; perturb slightly so
    # backward gradients through the conv trunk are non-zero on step 0 as well.
    if hasattr(model, "final_conv"):
        torch.nn.init.normal_(model.final_conv.weight, std=0.02)
    x = torch.randn(2, in_ch, 64, 64, requires_grad=True)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
    y.sum().backward()
    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    # Every input channel must receive non-zero gradient
    for ch in range(in_ch):
        assert x.grad[:, ch].abs().sum().item() > 0.0


def test_residual_unet_step0_identity_equals_center_channel():
    model = ResidualUNet(UNetConfig(in_channels=5, out_channels=1, base_channels=8))
    x = torch.rand(2, 5, 64, 64)
    y = model(x)
    assert torch.allclose(y, x[:, 2:3], atol=1e-6)


@pytest.mark.parametrize("model_type", ["residual_unet", "connectomics", "modern_unet"])
def test_models_reject_even_in_channels(model_type: str):
    with pytest.raises(ValueError, match="Global residual skip"):
        build_model(model_type=model_type, in_channels=2, out_channels=1, base_dim=8)


def test_r2_bfloat16_smoke_and_tiled_output_single_channel():
    model = build_model("residual_unet", in_channels=5, out_channels=1, base_dim=8)
    x = torch.rand(1, 5, 64, 64)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = model(x)
        loss = y.pow(2).mean()
    loss.backward()
    assert y.shape == (1, 1, 64, 64)
    assert torch.isfinite(loss)

    # _denoise_tiled must output (1, 1, H, W), never broadcast to (1, 5, H, W)
    evaluator = train_and_infer.DenoisingEvaluator(
        model=model, device=torch.device("cpu"), patch_size=64, tiled_inference=True
    )
    big_x = torch.rand(1, 5, 96, 112)
    out = evaluator._denoise_tiled(big_x)
    assert out.shape == (1, 1, 96, 112)


# -------------------------------------------------------------------------
# 3. Config & Z-Leakage Guard Tests (C4, M5)
# -------------------------------------------------------------------------


def test_config_validation_guards():
    base = {
        "splitting": {
            "strategy": "block",
            "origin": 0,
            "block_size": 20,
            "guard_band_size": 2,
        },
        "extraction": {"patch_size": 64},
        "n2n": {
            "enabled": True,
            "source": "overlaps",
            "pair_direction": "first_to_second",
            "patch_sampling": "tile",
            "z_context_radius": 1,
        },
        "model": {"model_type": "residual_unet", "base_dim": 8},
    }
    cfg = PipelineConfig.from_dict(base)
    assert cfg.n2n.z_context_radius == 1

    # mod10 rejected
    bad = json.loads(json.dumps(base))
    bad["splitting"]["strategy"] = "mod10"
    with pytest.raises(ValueError, match="splitting.strategy='block'"):
        PipelineConfig.from_dict(bad)

    # guard_band_size < R rejected
    bad = json.loads(json.dumps(base))
    bad["n2n"]["z_context_radius"] = 2
    bad["splitting"]["guard_band_size"] = 1
    with pytest.raises(ValueError, match="guard_band_size"):
        PipelineConfig.from_dict(bad)

    # Missing splitting.origin rejected
    bad = json.loads(json.dumps(base))
    bad["splitting"]["origin"] = None
    with pytest.raises(ValueError, match="splitting.origin"):
        PipelineConfig.from_dict(bad)

    # random_swap rejected
    bad = json.loads(json.dumps(base))
    bad["n2n"]["pair_direction"] = "random_swap"
    with pytest.raises(ValueError, match="first_to_second"):
        PipelineConfig.from_dict(bad)

    # streaming=True rejected
    bad = json.loads(json.dumps(base))
    bad["n2n"]["streaming"] = True
    with pytest.raises(ValueError, match="streaming"):
        PipelineConfig.from_dict(bad)

    # duplicates rejected
    bad = json.loads(json.dumps(base))
    bad["n2n"]["source"] = "duplicates"
    with pytest.raises(ValueError, match="overlaps"):
        PipelineConfig.from_dict(bad)

    # n2v rejected
    bad = json.loads(json.dumps(base))
    bad["model"]["model_type"] = "n2v"
    with pytest.raises(ValueError, match="incompatible"):
        PipelineConfig.from_dict(bad)

    # z_context_radius=3 rejected
    bad = json.loads(json.dumps(base))
    bad["n2n"]["z_context_radius"] = 3
    with pytest.raises(ValueError, match="z_context_radius"):
        PipelineConfig.from_dict(bad)


def test_assert_disjoint_z_supports():
    # Train sections [2..7], val sections [12..17], R=2 -> supports [0..9] and [10..19] -> disjoint!
    assert_disjoint_z_supports({"train": [2, 5, 7], "val": [12, 15, 17]}, radius=2)
    # Train max 8 + 2 = 10 overlaps val min 12 - 2 = 10 -> MUST raise!
    with pytest.raises(RuntimeError, match="Z-support leakage"):
        assert_disjoint_z_supports({"train": [2, 8], "val": [12, 15]}, radius=2)


# -------------------------------------------------------------------------
# 4. 2.5D Dataset, Cache, H1/H2/H3/H4/M4 & Determinism Tests
# -------------------------------------------------------------------------


def test_25d_dataset_r1_r2_determinism_views_and_invariants(tmp_path: Path):
    # Sections 0..99 with block_size=10, origin=0, guard_band_size=2:
    # train blocks 0..7 (secs 2..7, 12..17, ...), val block 8 (secs 82..87), test block 9 (secs 92..97)
    # Split shards at section 15 so shard boundary is crossed inside train block 1!
    secs = list(range(0, 22)) + list(range(78, 100))
    overlaps_dir = _create_synthetic_25d_overlaps(
        tmp_path / "ov",
        sections=secs,
        stripe_h=140,
        stripe_w=240,
        shard_split_sec=15,
        inject_duplicate_row=True,
        inject_second_grid=True,
        inject_tile_flip_sec=14,
        inject_top_margin_sentinel=True,
        inject_corrupt_stripe_shift=(
            5,
            7,
            0,
        ),  # 7 px mislock on anc=10 _v at sec 5 -> rejected by closure & tile consistency!
    )

    cache_w1 = tmp_path / "cache_w1"
    cache_w4 = tmp_path / "cache_w4"

    common_kw = dict(
        overlaps_dir=overlaps_dir,
        split="train",
        patch_size=64,
        z_context_radius=1,
        split_strategy="block",
        block_size=10,
        train_ratio=0.8,
        val_ratio=0.1,
        guard_band_size=2,
        split_origin=0,
        orientation="all",
        max_pairs=40,
        trim_enabled=True,
        v_trim_top_px=24,
        v_trim_left_px=20,
        h_trim_left_px=16,
        align_patches=True,
        align_subpixel=False,
        quality_filter_enabled=True,
        min_pair_quality=0.40,
        seed=42,
    )

    ds_w1 = N2N25DOverlapDataset(**common_kw, crop_cache_dir=cache_w1, prep_workers=1)
    ds_w4 = N2N25DOverlapDataset(**common_kw, crop_cache_dir=cache_w4, prep_workers=4)
    assert len(ds_w1) == len(ds_w4) > 0

    # H4 Determinism: prep_workers=1 vs 4 must produce byte-identical crops.npy & ids.txt
    dir_w1 = next(cache_w1.glob("train_25d_r1_*"))
    dir_w4 = next(cache_w4.glob("train_25d_r1_*"))
    assert (dir_w1 / "crops.npy").read_bytes() == (dir_w4 / "crops.npy").read_bytes()
    assert (dir_w1 / "ids.txt").read_bytes() == (dir_w4 / "ids.txt").read_bytes()

    # H1 Target trim-box invariant: bright 255 sentinel in top 20 px of page 1 must NEVER leak into target!
    crops_arr = np.load(dir_w1 / "crops.npy")
    assert crops_arr.ndim == 4 and crops_arr.shape[1:] == (4, 64, 64)
    # No row of target channel should be saturated at 255 across the whole row
    for i in range(len(crops_arr)):
        assert not np.any(crops_arr[i, -1].mean(axis=1) > 250.0)

    # M2 exposure bias recorded in metadata.json
    meta = json.loads((dir_w1 / "metadata.json").read_text())
    assert meta["mean_target_minus_input_255"] is not None
    assert meta["std_ratio_target_over_input"] is not None

    # H3 uniform section coverage: multiple sections represented (including across shard boundary >= 15)
    unique_secs = set(ds_w1.cached_tile_sections.tolist())
    assert len(unique_secs) >= 4
    assert any(s >= 15 for s in unique_secs)

    # C1 views on the exact same cache: center_only -> (1, 64, 64), center_filled -> (3, 64, 64) equal channels
    ds_center = N2N25DOverlapDataset(
        **common_kw, crop_cache_dir=cache_w1, z_context_view="center_only"
    )
    ds_filled = N2N25DOverlapDataset(
        **common_kw, crop_cache_dir=cache_w1, z_context_view="center_filled"
    )
    inp_full, tgt_full, id_full = ds_w1[0]
    inp_c, tgt_c, id_c = ds_center[0]
    inp_f, tgt_f, id_f = ds_filled[0]
    assert id_full == id_c == id_f
    assert inp_full.shape == (3, 64, 64)
    assert inp_c.shape == (1, 64, 64)
    assert inp_f.shape == (3, 64, 64)
    assert torch.allclose(inp_c[0], inp_full[1])
    assert torch.allclose(inp_f[0], inp_full[1]) and torch.allclose(
        inp_f[2], inp_full[1]
    )
    assert torch.allclose(tgt_full, tgt_c)

    # C4 Superset cache slicing with different first_sec + val split stays strictly Z-disjoint!
    ds_sub = N2N25DOverlapDataset(
        **{**common_kw, "first_sec": 10, "max_pairs": 10},
        crop_cache_dir=cache_w1,
    )
    assert int(ds_sub.cached_tile_sections.min()) >= 10
    ds_val = N2N25DOverlapDataset(
        **{**common_kw, "split": "val", "max_pairs": 20},
        crop_cache_dir=cache_w1,
    )
    assert_disjoint_z_supports(
        {"train": ds_sub.cached_tile_sections, "val": ds_val.cached_tile_sections},
        radius=1,
    )

    # R=2 (5-slice) + z_eval_radius=1 inner-channel slicing
    ds_r2 = N2N25DOverlapDataset(
        **{**common_kw, "z_context_radius": 2, "max_pairs": 20},
        crop_cache_dir=cache_w1,
    )
    x_r2, y_r2, _ = ds_r2[0]
    assert x_r2.shape == (5, 64, 64) and y_r2.shape == (1, 64, 64)
    ds_r2_eval1 = N2N25DOverlapDataset(
        **{**common_kw, "z_context_radius": 2, "z_eval_radius": 1, "max_pairs": 20},
        crop_cache_dir=cache_w1,
    )
    x_r2_e1, _, _ = ds_r2_eval1[0]
    assert x_r2_e1.shape == (3, 64, 64)
    assert torch.allclose(x_r2_e1, x_r2[1:4])


def test_missed_cut_hop_rejected(tmp_path: Path):
    # Inject missed cut at section 5 (s4 and s5 are identical clean surface -> DoG-NCC ~ 0.90+)
    overlaps_dir = _create_synthetic_25d_overlaps(
        tmp_path / "ov_mc",
        sections=list(range(0, 12)),
        stripe_h=140,
        stripe_w=240,
        inject_missed_cut_sec=5,
    )
    ds = N2N25DOverlapDataset(
        overlaps_dir=overlaps_dir,
        split="train",
        patch_size=64,
        z_context_radius=1,
        split_strategy="block",
        block_size=12,
        train_ratio=0.8,
        val_ratio=0.1,
        guard_band_size=2,
        split_origin=0,
        trim_enabled=False,
        quality_filter_enabled=False,
        z_align_missed_cut_max_ncc=0.75,
        seed=42,
    )
    # Hop (4->5) must be flagged as missed cut, so sections 4 and 5 cannot appear as centers
    assert ds.rejection_counts["rejected_missed_cut"] > 0
    kept_secs = set(ds.cached_tile_sections.tolist())
    assert 4 not in kept_secs and 5 not in kept_secs


# -------------------------------------------------------------------------
# 5. End-to-End Training, Checkpoint Mismatch & H6 Diagnostics Tests
# -------------------------------------------------------------------------


def test_e2e_25d_train_eval_preview_mismatch_and_diagnostics(
    tmp_path: Path, monkeypatch
):
    secs = list(range(0, 12)) + list(range(78, 100))
    overlaps_dir = _create_synthetic_25d_overlaps(
        tmp_path / "ov_e2e", sections=secs, stripe_h=140, stripe_w=240
    )

    # Preview directory with sibling s00004, s00005, s00006
    prev_dir = tmp_path / "prev"
    prev_dir.mkdir()
    for s in (4, 5, 6):
        tifffile.imwrite(
            prev_dir / f"20260201_RoLi_F1_g0000_t0010_s{s:05d}.tif",
            _make_em_texture(128, 128, seed=50 + s),
        )

    out_dir = tmp_path / "runs"
    cache_dir = tmp_path / "cache"
    cfg_path = tmp_path / "cfg_25d.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: "e2e_25d_r1"
  output_dir: "{out_dir}"
data:
  first_sec: 0
  last_sec: 100
splitting:
  strategy: "block"
  origin: 0
  block_size: 10
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  guard_band_size: 2
extraction:
  patch_size: 64
n2n:
  enabled: true
  source: "overlaps"
  overlaps_dir: "{overlaps_dir}"
  pair_direction: "first_to_second"
  orientation: "all"
  max_train_pairs: 16
  max_val_pairs: 8
  max_test_pairs: 8
  patch_sampling: "tile"
  trim_enabled: true
  v_trim_top_px: 16
  v_trim_left_px: 16
  h_trim_left_px: 16
  align_patches: true
  quality_filter_enabled: false
  crop_cache_dir: "{cache_dir}"
  z_context_radius: 1
  z_neighbor_dropout_prob: 0.15
  z_flip_augment: false
model:
  model_type: "residual_unet"
  base_dim: 8
training:
  batch_size: 4
  epochs: 2
  learning_rate: 0.0005
  lr_scheduler: "cosine"
  warmup_epochs: 1
  seed: 42
  cache_in_ram: true
  device: "cpu"
  num_workers: 0
  save_all_epochs: false
loss:
  loss_type: "composite"
  mse_weight: 1.0
  edge_weight: 0.0
  ssim_weight: 0.0
inference:
  preview_image: "{prev_dir / "20260201_RoLi_F1_g0000_t0010_s00005.tif"}"
  preview_crop_size: 64
  tiled: true
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["train_and_infer", "--config", str(cfg_path)])
    train_and_infer.main()

    run_dir = out_dir / "e2e_25d_r1"
    ckpt_path = run_dir / "checkpoints" / "checkpoint_best.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["in_channels"] == 3
    assert ckpt["z_context_radius"] == 1
    assert (run_dir / "previews" / "epoch_001_comparison.png").exists()
    assert (run_dir / "test" / "denoised" / "per_sample_denoised_mse.npy").exists()

    # Review §5 #7: --eval-only with 2D checkpoint on R=1 config MUST raise ValueError
    # (both normal AND legacy norm_transitions=False checkpoint!)
    for legacy in (False, True):
        m2d = build_model(
            "residual_unet",
            in_channels=1,
            out_channels=1,
            base_dim=8,
            norm_transitions=not legacy,
        )
        bad_ckpt = tmp_path / f"ckpt_2d_legacy_{legacy}.pt"
        torch.save({"model_state_dict": m2d.state_dict()}, bad_ckpt)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_and_infer",
                "--config",
                str(cfg_path),
                "--eval-only",
                "--checkpoint",
                str(bad_ckpt),
            ],
        )
        with pytest.raises(ValueError, match="in_channels=1"):
            train_and_infer.main()

    # H6 Diagnostics + C1 Paired Bootstrap CI
    model_r1 = build_model("residual_unet", in_channels=3, out_channels=1, base_dim=8)
    model_r1.load_state_dict(ckpt["model_state_dict"])
    test_ds = N2N25DOverlapDataset(
        overlaps_dir=overlaps_dir,
        split="test",
        patch_size=64,
        z_context_radius=1,
        split_strategy="block",
        block_size=10,
        guard_band_size=2,
        split_origin=0,
        max_pairs=8,
        trim_enabled=True,
        v_trim_top_px=16,
        v_trim_left_px=16,
        h_trim_left_px=16,
        align_patches=True,
        quality_filter_enabled=False,
        crop_cache_dir=cache_dir,
    )
    diag = diagnostics_25d.run_25d_diagnostics(
        model_r1,
        test_ds,
        device=torch.device("cpu"),
        shift_px_list=(2, 4),
        max_samples=8,
    )
    assert "context_gain_est_psnr_db" in diag
    assert set(diag["misregistration_sensitivity"].keys()) == {"shift_2px", "shift_4px"}
    assert diag["z_ghosting_probe"]["n_patches"] == len(test_ds)

    mse_a = np.load(run_dir / "test" / "denoised" / "per_sample_denoised_mse.npy")
    raw_a = np.load(run_dir / "test" / "denoised" / "per_sample_noisy_mse.npy")
    ci = diagnostics_25d.paired_bootstrap_mse_ci(mse_a, mse_a * 0.95, raw_a, n_boot=500)
    assert ci["delta_mse"] < 0.0 and ci["delta_est_psnr"] > 0.0


def test_custom_25d_raw_tile_inference_dataset(tmp_path: Path):
    raw_dir = tmp_path / "raw_stack"
    raw_dir.mkdir()
    canvas = _make_em_texture(320, 320, seed=404)
    # 4 consecutive sections of tile t0313 with +-5 px drift, plus 1 isolated tile t0999
    shifts = {10: (0, 0), 11: (4, -3), 12: (-2, 5), 13: (1, 1)}
    for sec, (dy, dx) in shifts.items():
        tile = canvas[30 + dy : 30 + dy + 256, 30 + dx : 30 + dx + 256]
        tifffile.imwrite(raw_dir / f"20260201_RoLi_F1_g0000_t0313_s{sec:05d}.tif", tile)
    tifffile.imwrite(
        raw_dir / "20260201_RoLi_F1_g0000_t0999_s00500.tif",
        _make_em_texture(256, 256, seed=505),
    )

    ds = Custom25DTIFFStackDataset(
        raw_dir, z_context_radius=1, z_register=True, z_register_max_shift_px=24
    )
    assert len(ds) == 5

    # Boundary slice s00010 (first of t0313): ch 0 is center-filled, ch 2 is registered s00011
    x0, y0, fn0 = ds[0]
    assert fn0 == "20260201_RoLi_F1_g0000_t0313_s00010.tif"
    assert x0.shape == (3, 256, 256) and y0.shape == (1, 256, 256)
    assert torch.allclose(x0[0], x0[1])

    # Interior slice s00011: both neighbors registered -> interior matches center within 0 (identical base canvas)
    x1, _, fn1 = ds[1]
    assert fn1 == "20260201_RoLi_F1_g0000_t0313_s00011.tif"
    assert torch.allclose(x1[0, 16:-16, 16:-16], x1[1, 16:-16, 16:-16], atol=1e-4)
    assert torch.allclose(x1[2, 16:-16, 16:-16], x1[1, 16:-16, 16:-16], atol=1e-4)

    # Isolated tile t0999: all channels center-filled
    x_iso, _, fn_iso = ds[4]
    assert fn_iso == "20260201_RoLi_F1_g0000_t0999_s00500.tif"
    assert torch.allclose(x_iso[0], x_iso[1]) and torch.allclose(x_iso[2], x_iso[1])

    # M6: num_workers > 0 must raise RuntimeError
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1)
    with pytest.raises(RuntimeError, match="num_workers=0"):
        next(iter(loader))
