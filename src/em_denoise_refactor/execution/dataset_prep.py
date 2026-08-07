# src/em_denoise_refactor/execution/dataset_prep.py

"""
Production-grade dataset extraction and preparation pipeline for EM denoising neural networks.

This script parses raw SBEM image tile acquisitions from remote storage, assigns spatial split metadata,
samples high-variance ROIs, synthesizes procedural noise for paired training, and stages patches into
a high-throughput LMDB binary store (and optional disk-backed TIFF images).
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Tuple, Optional

import tifffile
import torch
from tqdm import tqdm

from em_denoise_refactor.config import PipelineConfig
from em_denoise_refactor.etl.lmdb_store import (
    SMBStorageClient,
    ExtractionConfig,
    VarianceAwareROISampler,
    LMDBStoreWriter,
)
from em_denoise_refactor.data.dataset import (
    SBFSEMNoiseSynthesizer,
    EmpiricalNoiseSynthesizer,
)
from em_denoise_refactor.execution.calibration import run_calibration

logger = logging.getLogger("dataset_prep")


class DatasetPreparer:
    """Orchestrates remote-to-local patch extraction and LMDB dataset generation."""

    def __init__(
        self,
        config: PipelineConfig,
        output_dir_override: Optional[Path] = None,
        save_tiffs: bool = True,
    ) -> None:
        self.cfg = config
        self.save_tiffs = save_tiffs

        if output_dir_override:
            self.base_output_dir = Path(output_dir_override)
        else:
            self.base_output_dir = Path(self.cfg.experiment.output_dir)

        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.split_dirs: Dict[str, Tuple[Path, Path]] = {}

        if self.save_tiffs:
            self.split_dirs = self._setup_directories(self.base_output_dir)

    def _setup_directories(self, base_dir: Path) -> Dict[str, Tuple[Path, Path]]:
        dirs: Dict[str, Tuple[Path, Path]] = {}
        for split in ["train", "val", "test"]:
            clean_dir = base_dir / split / "clean"
            noisy_dir = base_dir / split / "noisy"
            clean_dir.mkdir(parents=True, exist_ok=True)
            noisy_dir.mkdir(parents=True, exist_ok=True)
            dirs[split] = (clean_dir, noisy_dir)
        return dirs

    def run(self) -> None:
        """Executes the dataset extraction pipeline."""
        logger.info(f"Base Output Directory: {self.base_output_dir}")

        extr_config = ExtractionConfig(
            sbem_root=self.cfg.data.sbem_root,
            first_sec=self.cfg.data.first_sec,
            last_sec=self.cfg.data.last_sec,
            grid_num=self.cfg.data.grid_num,
        )

        local_db_path = (
            self.base_output_dir / f"dataset_metadata_g{self.cfg.data.grid_num:04d}.db"
        )
        logger.info(f"Initializing Storage Client & Metadata Index at: {local_db_path}")
        client = SMBStorageClient(extr_config, db_path=local_db_path)

        logger.info("Assigning volumetric dataset splits...")
        client.metadata_store.assign_splits(
            strategy=self.cfg.splitting.strategy,
            block_size=self.cfg.splitting.block_size,
            train_ratio=self.cfg.splitting.train_ratio,
            val_ratio=self.cfg.splitting.val_ratio,
            guard_band_size=self.cfg.splitting.guard_band_size,
        )

        train_metadata = client.fetch_metadata(split="train")
        val_metadata = client.fetch_metadata(split="val")
        test_metadata = client.fetch_metadata(split="test")

        logger.info(
            f"Train split: {len(train_metadata.tiles_per_section)} active sections (max section {train_metadata.total_sections})"
        )
        logger.info(
            f"Val split  : {len(val_metadata.tiles_per_section)} active sections (max section {val_metadata.total_sections})"
        )
        logger.info(
            f"Test split : {len(test_metadata.tiles_per_section)} active sections (max section {test_metadata.total_sections})"
        )

        sampler = VarianceAwareROISampler(
            z_depth_per_roi=self.cfg.extraction.z_depth_per_roi,
            tiles_per_section=self.cfg.extraction.tiles_per_section,
        )

        if getattr(self.cfg.noise, "synthesizer_type", "parametric") == "empirical":
            model_path = Path(
                getattr(self.cfg.noise, "empirical_model_path", "noise_model.npy")
            )
            if not model_path.is_absolute():
                model_path = self.cfg.experiment.output_dir / model_path

            if (
                getattr(self.cfg.noise, "calibrate_on_the_fly", False)
                and not model_path.exists()
            ):
                logger.info(
                    "Empirical noise model not found. Running calibration on-the-fly..."
                )
                run_calibration(self.cfg)
                logger.info(
                    f"Empirical noise model calibrated and saved to: {model_path}"
                )

            noise_injector = EmpiricalNoiseSynthesizer(
                model_path=model_path,
                noise_amplification=getattr(self.cfg.noise, "noise_amplification", 1.0),
            )
        else:
            noise_injector = SBFSEMNoiseSynthesizer(
                photon_scale=self.cfg.noise.photon_scale,
                read_std=self.cfg.noise.read_std,
                gain_variance=self.cfg.noise.gain_variance,
                streak_probability=self.cfg.noise.streak_probability,
                inverted_bse=getattr(self.cfg.noise, "inverted_bse", True),
            )

        lmdb_path = self.base_output_dir / "patch_store.lmdb"
        lmdb_writer = LMDBStoreWriter(map_size=100 * 1024 * 1024 * 1024)
        lmdb_writer.initialize(lmdb_path)
        logger.info(f"LMDB Patch Store initialized at: {lmdb_path}")

        try:
            self._extract_split(
                client=client,
                metadata=train_metadata,
                target_count=self.cfg.extraction.target_train_patches,
                split_name="train",
                dirs=self.split_dirs.get("train"),
                sampler=sampler,
                noise_injector=noise_injector,
                lmdb_writer=lmdb_writer,
            )

            self._extract_split(
                client=client,
                metadata=val_metadata,
                target_count=self.cfg.extraction.target_val_patches,
                split_name="val",
                dirs=self.split_dirs.get("val"),
                sampler=sampler,
                noise_injector=noise_injector,
                lmdb_writer=lmdb_writer,
            )

            self._extract_split(
                client=client,
                metadata=test_metadata,
                target_count=self.cfg.extraction.target_test_patches,
                split_name="test",
                dirs=self.split_dirs.get("test"),
                sampler=sampler,
                noise_injector=noise_injector,
                lmdb_writer=lmdb_writer,
            )
            logger.info("Committing LMDB transactions...")
            lmdb_writer.commit_transaction()
            logger.info(f"✓ LMDB patch store successfully written to {lmdb_path}")
        except Exception as e:
            logger.error(f"Extraction failed due to error: {e}")
            raise
        finally:
            lmdb_writer.close()

    def _extract_split(
        self,
        client: SMBStorageClient,
        metadata,
        target_count: int,
        split_name: str,
        dirs: Optional[Tuple[Path, Path]],
        sampler: VarianceAwareROISampler,
        noise_injector: SBFSEMNoiseSynthesizer,
        lmdb_writer: LMDBStoreWriter,
    ) -> None:
        if target_count <= 0:
            logger.info(f"Skipping extraction for '{split_name}' (target count is 0).")
            return

        logger.info(f"Extracting {target_count} patches for '{split_name}' split...")

        patch_h = self.cfg.extraction.patch_size
        patch_w = self.cfg.extraction.patch_size
        max_patches_per_tile = self.cfg.extraction.patches_per_tile

        rois = sampler.generate_rois(metadata, num_regions=max(target_count * 2, 50))
        patch_count = 0
        section_coverage: Dict[int, int] = {}

        clean_dir, noisy_dir = dirs if dirs else (None, None)

        with tqdm(total=target_count, desc=f"Split [{split_name}]") as pbar:
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
                            logger.warning(
                                f"Tile missing: section {section_id}, tile {tile_id}"
                            )
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
                                y_start : y_start + patch_h,
                                x_start : x_start + patch_w,
                            ]

                            # Apply noise synthesis
                            clean_tensor = torch.from_numpy(clean_patch).float() / 255.0
                            clean_tensor = clean_tensor.view(1, 1, patch_h, patch_w)
                            noisy_tensor = noise_injector.inject(clean_tensor)
                            noisy_patch = (
                                noisy_tensor.view(patch_h, patch_w).numpy() * 255
                            ).astype("uint8")

                            filename = f"{split_name}_s{section_id:05d}_t{t_id_clean}_p{p_idx}.tif"
                            if self.save_tiffs and clean_dir and noisy_dir:
                                tifffile.imwrite(clean_dir / filename, clean_patch)
                                tifffile.imwrite(noisy_dir / filename, noisy_patch)

                            patch_key = (
                                f"{split_name}_s{section_id:05d}_t{t_id_clean}_p{p_idx}"
                            )
                            lmdb_writer.write_patch(patch_key, clean_patch)

                            patch_count += 1
                            section_coverage[section_id] += 1
                            pbar.update(1)

        logger.info(
            f"Coverage Report for '{split_name}' ({patch_count} patches extracted across {len(section_coverage)} sections):"
        )
        for s_id, count in sorted(section_coverage.items()):
            logger.debug(f"  Section {s_id:05d}: {count} patches")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production SBF-SEM Dataset Extraction and Patch Preparation Pipeline."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="../config.yaml",
        help="Path to pipeline configuration YAML file (default: src/em_denoise_refactor/config.yaml).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override for staged patches and metadata index.",
    )
    parser.add_argument(
        "--save-tiffs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save individual TIFF patch files to disk in addition to LMDB (default: True).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = PipelineConfig.from_yaml(args.config)
    loaded_path = getattr(cfg, "_loaded_from", args.config)
    logger.info(f"Loading pipeline configuration from: {loaded_path}")

    preparer = DatasetPreparer(
        config=cfg,
        output_dir_override=Path(args.output_dir) if args.output_dir else None,
        save_tiffs=args.save_tiffs,
    )
    preparer.run()


if __name__ == "__main__":
    main()
