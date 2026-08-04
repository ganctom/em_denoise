# src/em_denoise_refactor/tests/validate_step6_extraction.py

import sys
from pathlib import Path
import tifffile
from tqdm import tqdm
import torch
import random

from em_denoise_refactor.config import PipelineConfig
from em_denoise_refactor.etl.lmdb_store import (
    SMBStorageClient,
    ExtractionConfig,
    VarianceAwareROISampler,
)
from em_denoise_refactor.data.dataset import SBFSEMNoiseSynthesizer


def setup_directories(base_dir: Path):
    dirs = {}
    for split in ["train", "val", "test"]:
        clean_dir = base_dir / split / "clean"
        noisy_dir = base_dir / split / "noisy"
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)
        dirs[split] = (clean_dir, noisy_dir)

    return dirs


def main(config_path: Path | str = "config.yaml"):
    print(f"Loading pipeline settings from: {config_path}")
    cfg = PipelineConfig.from_yaml(config_path)

    base_output_dir = cfg.experiment.output_dir
    split_dirs = setup_directories(base_output_dir)

    extr_config = ExtractionConfig(
        sbem_root=cfg.data.sbem_root,
        first_sec=cfg.data.first_sec,
        last_sec=cfg.data.last_sec,
        grid_num=cfg.data.grid_num,
    )

    # Store the SQLite database locally on fast SSD inside output_dir
    local_db_path = base_output_dir / f"dataset_metadata_g{cfg.data.grid_num:04d}.db"

    print("Initializing Storage Client & Local SQLite Metadata Index...")
    client = SMBStorageClient(extr_config, db_path=local_db_path)

    # Ensure splits are assigned using config strategy and guard band parameters
    client.metadata_store.assign_splits(
        strategy=cfg.splitting.strategy,
        block_size=cfg.splitting.block_size,
        train_ratio=cfg.splitting.train_ratio,
        val_ratio=cfg.splitting.val_ratio,
        guard_band_size=cfg.splitting.guard_band_size,
    )

    # Fetch separated metadata
    train_metadata = client.fetch_metadata(split="train")
    val_metadata = client.fetch_metadata(split="val")
    test_metadata = client.fetch_metadata(split="test")

    print(
        f"Train Dataset: {train_metadata.total_sections} max section, {len(train_metadata.tiles_per_section)} active train sections"
    )
    print(
        f"Val Dataset  : {val_metadata.total_sections} max section, {len(val_metadata.tiles_per_section)} active val sections"
    )
    print(
        f"Test Dataset : {test_metadata.total_sections} max section, {len(test_metadata.tiles_per_section)} active test sections"
    )

    # Density-weighted ROI sampler driven by config
    sampler = VarianceAwareROISampler(
        z_depth_per_roi=cfg.extraction.z_depth_per_roi,
        tiles_per_section=cfg.extraction.tiles_per_section,
    )

    noise_injector = SBFSEMNoiseSynthesizer(
        poisson_alpha=cfg.noise.poisson_alpha,
        gaussian_sigma=cfg.noise.gaussian_sigma,
        streak_probability=cfg.noise.streak_probability,
    )

    patch_h = cfg.extraction.patch_size
    patch_w = cfg.extraction.patch_size
    max_patches_per_tile = cfg.extraction.patches_per_tile

    def extract_and_save(metadata, target_count, split_name, clean_dir, noisy_dir):
        print(f"\nExtracting {target_count} patches for {split_name}...")

        rois = sampler.generate_rois(metadata, num_regions=max(target_count * 2, 50))

        patch_count = 0
        section_coverage = {}

        pbar = tqdm(total=target_count)

        for roi in rois:
            if patch_count >= target_count:
                break

            for section_id, tiles in roi.section_tile_map.items():
                if patch_count >= target_count:
                    break

                if section_id not in section_coverage:
                    section_coverage[section_id] = 0

                for tile_id in tiles:
                    if patch_count >= target_count:
                        break

                    try:
                        tile_array = client.download_tile(section_id, tile_id)
                    except FileNotFoundError:
                        continue

                    img_h, img_w = tile_array.shape
                    if img_h < patch_h or img_w < patch_w:
                        continue

                    t_id_clean = tile_id[1:] if tile_id.startswith("t") else tile_id

                    for p_idx in range(max_patches_per_tile):
                        if patch_count >= target_count:
                            break

                        y_start = random.randint(0, img_h - patch_h)
                        x_start = random.randint(0, img_w - patch_w)
                        clean_patch = tile_array[
                            y_start : y_start + patch_h, x_start : x_start + patch_w
                        ]

                        # Apply noise
                        clean_tensor = (
                            torch.from_numpy(clean_patch).float() / 255.0
                        )  # normalize
                        clean_tensor = clean_tensor.view(1, 1, patch_h, patch_w)

                        noisy_tensor = noise_injector.inject(clean_tensor)
                        noisy_patch = (
                            noisy_tensor.view(patch_h, patch_w).numpy() * 255
                        ).astype("uint8")

                        # [split]_s[section_id]_t[tile_id]_p[patch_index].tif
                        filename = (
                            f"{split_name}_s{section_id:05d}_t{t_id_clean}_p{p_idx}.tif"
                        )

                        tifffile.imwrite(clean_dir / filename, clean_patch)
                        tifffile.imwrite(noisy_dir / filename, noisy_patch)

                        patch_count += 1
                        section_coverage[section_id] += 1
                        pbar.update(1)
        pbar.close()

        print(f"Coverage Report for {split_name}:")
        for s_id, count in sorted(section_coverage.items()):
            print(f"  Section {s_id}: {count} patches")

    extract_and_save(
        train_metadata,
        cfg.extraction.target_train_patches,
        "train",
        *split_dirs["train"],
    )
    extract_and_save(
        val_metadata, cfg.extraction.target_val_patches, "val", *split_dirs["val"]
    )
    extract_and_save(
        test_metadata, cfg.extraction.target_test_patches, "test", *split_dirs["test"]
    )


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg_file)
