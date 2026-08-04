# src/em_denoise_refactor/etl/lmdb_store.py
from dataclasses import dataclass

import lmdb
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Iterator, List, Dict, Set
import random
import itertools
import json
import tifffile
from tqdm import tqdm

from em_denoise_refactor.etl.interfaces import (
    DatasetMetadata,
    RemoteStorageClient,
    RegionOfInterest,
    PatchExtractor,
    ROISampler,
)


@dataclass
class ExtractionConfig:
    sbem_root: Path | str
    grid_num: int = 0


class SMBStorageClient(RemoteStorageClient):
    """
    Implements RemoteStorageClient to pull raw SBF-SEM tiles directly from
    company SMB mount points using local filesystem paths and directory layout.
    """

    def __init__(
        self,
        extraction_config: ExtractionConfig,
        metadata_cache_path: Optional[Path | str] = None,
    ):
        self.sbem_root = Path(extraction_config.sbem_root)
        self.grid_num = extraction_config.grid_num
        self.grid_str = f"g{extraction_config.grid_num:04d}"
        self.exp_name = self.sbem_root.name
        self.metadata_cache_path = (
            Path(metadata_cache_path)
            if metadata_cache_path
            else self.sbem_root / f"dataset_metadata_{self.grid_str}.json"
        )

    def save_metadata(self, metadata: DatasetMetadata) -> None:
        """Serializes DatasetMetadata to a local JSON file to avoid expensive SMB rescans."""
        data = {
            "total_sections": metadata.total_sections,
            "z_resolutions_nm": {
                str(k): v for k, v in metadata.z_resolutions_nm.items()
            },
            "tiles_per_section": {
                str(k): v for k, v in metadata.tiles_per_section.items()
            },
            "snr_estimates": {str(k): v for k, v in metadata.snr_estimates.items()},
        }
        with open(self.metadata_cache_path, "w") as f:
            json.dump(data, f, indent=4)

    def load_cached_metadata(self) -> Optional[DatasetMetadata]:
        """Loads DatasetMetadata from local disk cache if available."""
        if not self.metadata_cache_path.exists():
            return None

        try:
            with open(self.metadata_cache_path, "r") as f:
                data = json.load(f)

            return DatasetMetadata(
                total_sections=data["total_sections"],
                z_resolutions_nm={
                    int(k): v for k, v in data["z_resolutions_nm"].items()
                },
                tiles_per_section={
                    int(k): v for k, v in data["tiles_per_section"].items()
                },
                snr_estimates={int(k): v for k, v in data["snr_estimates"].items()},
            )
        except Exception:
            return None

    def fetch_metadata(self, force_rescan: bool = False) -> DatasetMetadata:
        if not force_rescan:
            cached = self.load_cached_metadata()
            if cached is not None:
                print(
                    f"Loaded dataset metadata from local cache: {self.metadata_cache_path}"
                )
                return cached

        print("Cache not found or rescan forced. Scanning SMB path structure...")
        tiles_root = self.sbem_root / "tiles" / self.grid_str
        if not tiles_root.exists():
            raise FileNotFoundError(f"SMB tiles root not found: {tiles_root}")

        # Scan ALL acquired tile directories. (Previously this was hardcoded to the
        # first 4 directories via a stray `[:4]` slice, which silently truncated
        # total_sections / tiles_per_section for any real dataset. Any deliberate
        # "mini extraction" subsampling belongs in ROISampler.generate_rois via
        # num_regions, not in metadata scanning.)
        tile_dirs = [
            d for d in tiles_root.iterdir() if d.is_dir() and d.name.startswith("t")
        ]
        tiles_per_section: Dict[int, List[int]] = {}
        z_resolutions_nm: Dict[int, float] = {}
        snr_estimates: Dict[int, float] = {}

        all_sections: Set[int] = set()

        for tile_dir in tqdm(
            tile_dirs, desc="Scanning SMB Tile Directories", unit="dir"
        ):
            try:
                tile_id = int(tile_dir.name[1:])
            except ValueError:
                continue

            tif_files = list(tile_dir.glob("*.tif"))
            for f_path in tif_files:
                parts = f_path.stem.split("_")
                try:
                    sec_part = [p for p in parts if p.startswith("s")][-1]
                    sec_num = int(sec_part[1:])
                    all_sections.add(sec_num)

                    if sec_num not in tiles_per_section:
                        tiles_per_section[sec_num] = []
                    if tile_id not in tiles_per_section[sec_num]:
                        tiles_per_section[sec_num].append(tile_id)
                except (IndexError, ValueError):
                    continue

        for sec_num in tiles_per_section:
            tiles_per_section[sec_num].sort()

        total_sections = max(all_sections) if all_sections else 0

        for sec in all_sections:
            z_resolutions_nm[sec] = 30.0
            snr_estimates[sec] = 1.0

        metadata = DatasetMetadata(
            total_sections=total_sections,
            z_resolutions_nm=z_resolutions_nm,
            tiles_per_section=tiles_per_section,
            snr_estimates=snr_estimates,
        )

        self.save_metadata(metadata)
        print(f"Metadata cached successfully to {self.metadata_cache_path}")
        return metadata

    def download_tile(self, section_id: int, tile_id: str) -> np.ndarray:
        tile_str = f"{tile_id}" if tile_id.startswith("t") else f"t{int(tile_id):04d}"
        slice_str = f"{section_id:05d}"
        filename = f"{self.exp_name}_{self.grid_str}_{tile_str}_s{slice_str}.tif"
        filepath = self.sbem_root / "tiles" / self.grid_str / tile_str / filename

        if not filepath.exists():
            raise FileNotFoundError(f"SMB Tile path does not exist: {filepath}")

        return tifffile.imread(str(filepath))


class SMBHighFreqPatchExtractor(PatchExtractor):
    """
    Extracts high-frequency structural sub-regions from SMB-streamed tiles
    using grid coordinate permutations.
    """

    def __init__(self, patch_size: int = 128, patches_per_tile: int = 10):
        self.patch_size = patch_size
        self.patches_per_tile = patches_per_tile

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

    def extract_from_roi(
        self,
        client: RemoteStorageClient,
        roi: RegionOfInterest,
        patch_size: Tuple[int, int],
    ) -> Iterator[np.ndarray]:
        patch_h, patch_w = patch_size

        for section_id, tiles in roi.section_tile_map.items():
            for tile_id in tiles:
                try:
                    tile_array = client.download_tile(section_id, tile_id)
                except FileNotFoundError:
                    continue

                img_h, img_w = tile_array.shape
                if img_h < patch_h or img_w < patch_w:
                    continue

                for _ in range(self.patches_per_tile):
                    y_start = random.randint(0, img_h - patch_h)
                    x_start = random.randint(0, img_w - patch_w)
                    yield tile_array[
                        y_start : y_start + patch_h, x_start : x_start + patch_w
                    ]


class VarianceAwareROISampler(ROISampler):
    """
    Generates balanced Regions of Interest by sampling from the actual
    acquired tile IDs per section available in the dataset metadata.
    """

    def __init__(self, z_depth_per_roi: int = 5, tiles_per_section: int = 4):
        self.z_depth_per_roi = z_depth_per_roi
        self.tiles_per_section = tiles_per_section

    def generate_rois(
        self, metadata: DatasetMetadata, num_regions: int
    ) -> List[RegionOfInterest]:
        rois: List[RegionOfInterest] = []
        valid_sections = [
            s for s, t_list in metadata.tiles_per_section.items() if len(t_list) > 0
        ]

        if not valid_sections:
            raise ValueError("No sections with valid acquired tiles found in metadata.")

        for _ in range(num_regions):
            z_start = random.choice(valid_sections)
            z_end = min(z_start + self.z_depth_per_roi, max(valid_sections))

            section_tile_map: Dict[int, List[str]] = {}

            for z in range(z_start, z_end + 1):
                if z in metadata.tiles_per_section and metadata.tiles_per_section[z]:
                    available_tile_ids = metadata.tiles_per_section[z]

                    # Sample actual acquired tile IDs from this section's list
                    k = min(self.tiles_per_section, len(available_tile_ids))
                    sampled_ids = random.sample(available_tile_ids, k)

                    # Format back to company tXXXX convention
                    section_tile_map[z] = [f"t{tid:04d}" for tid in sampled_ids]

            if section_tile_map:
                rois.append(
                    RegionOfInterest(
                        z_start=z_start, z_end=z_end, section_tile_map=section_tile_map
                    )
                )

        return rois


class LMDBStoreWriter:
    """
    ETL stage writer for staging SMB SBF-SEM patches into a local LMDB database
    optimized for Apple M4 Pro SSD storage and zero-copy read access.
    """

    def __init__(
        self, map_size: int = 100 * 1024 * 1024 * 1024
    ):  # Default 100GB ceiling
        self.map_size = map_size
        self.env: Optional[lmdb.Environment] = None
        self.txn: Optional[lmdb.Transaction] = None

    def initialize(self, path: Path | str) -> None:
        db_path = Path(path)
        db_path.mkdir(parents=True, exist_ok=True)

        self.env = lmdb.open(
            str(db_path),
            subdir=True,
            map_size=self.map_size,
            sync=False,
            metasync=False,
            writemap=True,
        )
        self.txn = self.env.begin(write=True)

    def write_patch(self, patch_id: str, data: np.ndarray) -> None:
        if self.txn is None:
            raise RuntimeError(
                "LMDB transaction not initialized. Call initialize() first."
            )

        payload = np.ascontiguousarray(data, dtype=np.uint8).tobytes()
        self.txn.put(patch_id.encode("utf-8"), payload)

    def commit_transaction(self) -> None:
        if self.txn is not None:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self) -> None:
        if self.txn is not None:
            try:
                self.txn.commit()
            except Exception:
                pass
            self.txn = None
        if self.env is not None:
            self.env.sync()
            self.env.close()
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self) -> None:
        self.close()
