from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import random
import shutil
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import tifffile
from tqdm import tqdm
import yaml


# =============================================================================
# Configuration Management
# =============================================================================


@dataclass
class ExtractionConfig:
    sbem_root: Path
    dest_root: Path
    grid_num: int
    tile_ids_file: Path
    section_db_file: Path
    target_total_patches: int = 50000
    patch_size: int = 128
    section_start: int = 1250
    section_end: int = 22815  # Upper bound inclusive of section 22814
    random_seed: Optional[int] = 43
    n_test_raw: int = 20

    @property
    def section_range(self) -> range:
        return range(self.section_start, self.section_end)

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionConfig":
        data = data.copy()
        data["sbem_root"] = Path(data["sbem_root"])
        data["dest_root"] = Path(data["dest_root"])
        data["tile_ids_file"] = Path(data["tile_ids_file"])
        data["section_db_file"] = Path(data["section_db_file"])
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExtractionConfig":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "ExtractionConfig":
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# Parsers & Input Loaders
# =============================================================================


def load_tile_ids(txt_path: Path) -> Set[int]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Tile ID file not found: {txt_path}")

    tile_ids = set()
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    tile_ids.add(int(line))
                except ValueError:
                    continue
    return tile_ids


def load_section_tile_db(db_path: Path) -> Dict[int, int]:
    if not db_path.exists():
        raise FileNotFoundError(f"Section DB file not found: {db_path}")

    section_db = {}
    with open(db_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sec_str, count_str = line.split(",")
                section_db[int(sec_str)] = int(count_str)
    return section_db


# =============================================================================
# Processing Components
# =============================================================================


class SectionQuotaCalculator:
    """Calculates cross-sectional patch quotas proportional to dataset ROI area."""

    @staticmethod
    def compute(
        section_db: Dict[int, int],
        section_range: range,
        target_total_patches: int,
    ) -> Dict[int, int]:
        active_sections = {s: c for s, c in section_db.items() if s in section_range}
        total_volume_tiles = sum(active_sections.values())

        if total_volume_tiles == 0:
            raise ValueError("No matching sections found in dataset database.")

        global_density = target_total_patches / total_volume_tiles

        return {
            section: int(math.ceil(tile_count * global_density))
            for section, tile_count in active_sections.items()
        }


class DataSplitter:
    """Handles Z-axis striding rules to prevent spatial leakage."""

    @staticmethod
    def get_split(section: int) -> str:
        mod = section % 10
        if mod < 8:
            return "train"
        elif mod == 8:
            return "val"
        else:
            return "test"


class PatchExtractor:
    """Handles spatial grid partitioning and slice extraction from array structures."""

    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def get_candidate_coordinates(
        self, image_shape: Tuple[int, ...]
    ) -> List[Tuple[int, int]]:
        h, w = image_shape[-2:]
        if h < self.patch_size or w < self.patch_size:
            return []

        y_bins = h // self.patch_size
        x_bins = w // self.patch_size
        coords = list(itertools.product(range(y_bins), range(x_bins)))
        random.shuffle(coords)
        return coords

    def extract_patch(self, img: np.ndarray, grid_y: int, grid_x: int) -> np.ndarray:
        y = grid_y * self.patch_size
        x = grid_x * self.patch_size
        return img[y : y + self.patch_size, x : x + self.patch_size]


class DatasetWriter:
    """Encapsulates target filesystem structure and write IO."""

    def __init__(self, dest_root: Path):
        self.dest_root = dest_root
        self.split_dirs = {
            "train": dest_root / "train",
            "val": dest_root / "val",
            "test": dest_root / "test",
            "test_raw": dest_root / "test_raw",
        }
        self._init_directories()

    def _init_directories(self) -> None:
        for path in self.split_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def save_patch(self, patch: np.ndarray, split: str, filename: str) -> Path:
        target_path = self.split_dirs[split] / filename
        tifffile.imwrite(str(target_path), patch)
        return target_path

    def save_raw(self, src_path: Path, filename: str) -> Path:
        target_path = self.split_dirs["test_raw"] / filename
        shutil.copy2(src_path, target_path)
        return target_path


# =============================================================================
# Pipeline Orchestration
# =============================================================================


class SBEMPatchPipeline:
    """Orchestrates density-aware patch sampling without patch quality filtering."""

    def __init__(self, config: ExtractionConfig):
        self.cfg = config
        self.writer = DatasetWriter(config.dest_root)
        self.extractor = PatchExtractor(config.patch_size)

        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

    def get_tile_path(self, tile_id: int, section_num: int) -> Tuple[str, str]:
        tile_str = f"{tile_id:04d}"
        slice_str = f"{section_num:05d}"
        exp_name = f"{self.cfg.sbem_root.name}"
        grid_num = self.cfg.grid_num
        grid_str = "g" + (str(grid_num).zfill(4))
        filename = f"{exp_name}_{grid_str}_t{tile_str}_s{slice_str}.tif"
        filepath = str(
            self.cfg.sbem_root / "tiles" / grid_str / f"t{tile_str}" / filename
        )
        return filepath, filename

    def get_patch_name(self, tile_id: int, section_num: int, section_counts) -> str:
        slice_str = f"{section_num:05d}"
        tile_str = f"{tile_id:04d}"
        return f"s{slice_str}_t{tile_str}_p{section_counts[section_num]:04d}.tif"

    def run(self) -> Dict[str, int]:
        tile_ids = sorted(load_tile_ids(self.cfg.tile_ids_file))
        section_db = load_section_tile_db(self.cfg.section_db_file)

        section_quotas = SectionQuotaCalculator.compute(
            section_db, self.cfg.section_range, self.cfg.target_total_patches
        )

        section_counts = {s: 0 for s in section_quotas}
        total_saved_split = {"train": 0, "val": 0, "test": 0}
        raw_saved_count = 0

        image_search_space = list(itertools.product(tile_ids, section_quotas.keys()))
        random.shuffle(image_search_space)

        pbar = tqdm(
            total=self.cfg.target_total_patches,
            desc="Extracting SBEM Patches",
            unit="ptch",
        )

        for tile_id, section in image_search_space:
            if section_counts[section] >= section_quotas[section]:
                continue

            split_name = DataSplitter.get_split(section)

            src_path, filename = self.get_tile_path(tile_id, section)
            if not src_path.exists():
                continue

            if split_name == "test" and raw_saved_count < self.cfg.n_test_raw:
                self.writer.save_raw(Path(src_path), filename)
                raw_saved_count += 1

            try:
                img = tifffile.imread(str(src_path))
                candidate_coords = self.extractor.get_candidate_coordinates(img.shape)
                if not candidate_coords:
                    continue

                rem_quota = section_quotas[section] - section_counts[section]
                avg_per_tile = math.ceil(section_quotas[section] / section_db[section])
                tile_target = min(rem_quota, avg_per_tile, len(candidate_coords))

                saved_from_tile = 0
                for by, bx in candidate_coords:
                    if saved_from_tile >= tile_target:
                        break

                    patch = self.extractor.extract_patch(img, by, bx)

                    patch_name = self.get_patch_name(tile_id, section, section_counts)
                    self.writer.save_patch(patch, split_name, patch_name)

                    total_saved_split[split_name] += 1
                    section_counts[section] += 1
                    saved_from_tile += 1
                    pbar.update(1)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

            if sum(section_counts.values()) >= self.cfg.target_total_patches:
                break

        pbar.close()
        self._print_summary(section_counts, total_saved_split, raw_saved_count)
        return total_saved_split

    def _print_summary(
        self,
        section_counts: Dict[int, int],
        total_saved_split: Dict[str, int],
        raw_saved_count: int,
    ) -> None:
        print("\nExtraction Summary:")
        print(f"  Target Volume Patches   : {self.cfg.target_total_patches:,}")
        print(f"  Total Extracted Patches : {sum(section_counts.values()):,}")
        print(f"  Saved Train Patches     : {total_saved_split['train']:,}")
        print(f"  Saved Val Patches       : {total_saved_split['val']:,}")
        print(f"  Saved Test Patches      : {total_saved_split['test']:,}")
        print(f"  Saved Test Raw TIFs     : {raw_saved_count:,}")


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    config = ExtractionConfig(
        sbem_root=Path(
            "/Volumes/tachyon/groups/scratch/gmicro_sem/gfriedri/_EM_acquisitions/20260201_RoLi_F1"
        ),
        dest_root=Path("/Users/ganctoma/SW/sort/N2V/run_30/raw_patches"),
        tile_ids_file=Path("/Users/ganctoma/SW/sort/N2V/all_tile_ids.txt"),
        section_db_file=Path("/Users/ganctoma/SW/sort/N2V/section_tile_counts.csv"),
        target_total_patches=5000,
        patch_size=128,
        section_start=16500,
        section_end=17000,
        random_seed=42,
        n_test_raw=40,
        grid_num=0,
    )

    pipeline = SBEMPatchPipeline(config)
    pipeline.run()
