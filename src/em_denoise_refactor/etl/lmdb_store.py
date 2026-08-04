# src/em_denoise_refactor/etl/lmdb_store.py
from dataclasses import dataclass

import lmdb
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Iterator, List, Dict
import random
import itertools
import tifffile
from tqdm import tqdm
import sqlite3
import ast

from em_denoise_refactor.etl.interfaces import (
    DatasetMetadata,
    RemoteStorageClient,
    RegionOfInterest,
    PatchExtractor,
    ROISampler,
    FileMetadataStore,
)


@dataclass
class ExtractionConfig:
    sbem_root: Path | str
    first_sec: int
    last_sec: int
    grid_num: int = 0
    patch_size: int = 512


class SQLiteMetadataStore(FileMetadataStore):
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    filepath TEXT NOT NULL,
                    section_id INTEGER NOT NULL,
                    tile_id INTEGER NOT NULL,
                    split TEXT DEFAULT 'unassigned'
                )
            """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_section_id ON tiles(section_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tile_id ON tiles(tile_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_filename ON tiles(filename)"
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_split ON tiles(split)")

    def update_db(
        self,
        meta_logs_dir: Path,
        first_sec: int,
        last_sec: int,
        nr_of_files=None,
    ) -> None:
        if not meta_logs_dir.exists() or not meta_logs_dir.is_dir():
            raise FileNotFoundError(f"Meta logs dir not found: {meta_logs_dir}")

        log_files = list(meta_logs_dir.glob("metadata_*.txt"))
        if nr_of_files is not None:
            log_files = log_files[:nr_of_files]
        if not log_files:
            print(f"No metadata_*.txt files found in {meta_logs_dir}")
            return

        print(f"Parsing {len(log_files)} metadata files for updates...")
        inserts = []
        for log_file in tqdm(log_files, desc="Parsing logs"):
            with open(log_file, "r") as f:
                for line in f:
                    if line.startswith("TILE: "):
                        try:
                            # Parse the dictionary string after "TILE: "
                            dict_str = line[len("TILE: ") :].strip()
                            data = ast.literal_eval(dict_str)

                            filename = data.get("filename")
                            if not filename:
                                continue

                            # Extract tile_id and section_id from tileid e.g. "0000.0411.00146"
                            # or just use slice_counter and extract tile_id from filename or tileid string
                            tileid_str = data.get("tileid", "")
                            parts = tileid_str.split(".")
                            if len(parts) == 3:
                                tile_id = int(parts[1])
                                section_id = int(parts[2])
                            else:
                                section_id = int(data.get("slice_counter", 0))
                                # Fallback extract from filename e.g. "tiles/g0000/t0411/20260201_RoLi_F1_g0000_t0411_s00146.tif"
                                name_parts = filename.split("_")
                                t_part = [p for p in name_parts if p.startswith("t")][
                                    -1
                                ]
                                tile_id = int(t_part[1:])
                            if first_sec <= section_id <= last_sec:
                                inserts.append(
                                    (filename, filename, section_id, tile_id)
                                )
                        except Exception as e:
                            print(f"Error parsing line: {line}. Error: {e}")

        if inserts:
            with self.conn:
                self.conn.executemany(
                    """
                    INSERT OR IGNORE INTO tiles (filename, filepath, section_id, tile_id)
                    VALUES (?, ?, ?, ?)
                """,
                    inserts,
                )
            print(f"Inserted up to {len(inserts)} records into SQLite.")

    def assign_splits(
        self,
        strategy: str = "block",
        block_size: int = 100,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        guard_band_size: int = 5,
    ) -> None:
        """
        Assigns splits dynamically using either:
        - strategy="block": Volumetric block splitting with guard bands (buffer sections) between splits.
        - strategy="mod10": Legacy modulo-10 section splitting (0..7 train, 8 val, 9 test).
        """
        with self.conn:
            self.conn.execute("UPDATE tiles SET split = 'unassigned'")

            # Fetch all distinct sections
            cursor = self.conn.execute(
                "SELECT DISTINCT section_id FROM tiles ORDER BY section_id"
            )
            sections = [row[0] for row in cursor.fetchall()]

            if not sections:
                return

            counts = {"train": 0, "val": 0, "test": 0, "guard_band": 0}

            if strategy == "mod10":
                for s in sections:
                    mod = s % 10
                    if mod < 8:
                        split = "train"
                    elif mod == 8:
                        split = "val"
                    else:
                        split = "test"
                    counts[split] += 1
                    self.conn.execute(
                        "UPDATE tiles SET split = ? WHERE section_id = ?", (split, s)
                    )
                print(
                    f"Assigned splits via Modulo-10: {counts['train']} train, {counts['val']} val, {counts['test']} test sections."
                )
            else:
                # Volumetric Block Splitting with Guard Bands across full section sequence
                num_sections = len(sections)
                raw_splits = []
                for i in range(0, num_sections, block_size):
                    b_len = min(block_size, num_sections - i)
                    n_train = max(1, int(round(b_len * train_ratio)))
                    n_val = max(1, int(round(b_len * val_ratio)))

                    for idx in range(b_len):
                        if idx < n_train:
                            raw_splits.append("train")
                        elif idx < n_train + n_val:
                            raw_splits.append("val")
                        else:
                            raw_splits.append("test")

                # Apply guard bands around all split transition boundaries (including block boundaries)
                left_guard = (guard_band_size + 1) // 2
                right_guard = guard_band_size // 2

                guard_mask = [False] * num_sections
                for idx in range(1, num_sections):
                    if raw_splits[idx] != raw_splits[idx - 1]:
                        start_g = max(0, idx - left_guard)
                        end_g = min(num_sections, idx + right_guard)
                        for g in range(start_g, end_g):
                            guard_mask[g] = True

                for idx, s in enumerate(sections):
                    split = "guard_band" if guard_mask[idx] else raw_splits[idx]
                    counts[split] += 1
                    self.conn.execute(
                        "UPDATE tiles SET split = ? WHERE section_id = ?", (split, s)
                    )

                print(
                    f"Assigned splits via Volumetric Block (size={block_size}, guard={guard_band_size}): "
                    f"{counts['train']} train, {counts['val']} val, {counts['test']} test, {counts['guard_band']} guard_band sections."
                )

    def get_filepath(self, section_id: int, tile_id: int) -> str:
        cursor = self.conn.execute(
            "SELECT filepath FROM tiles WHERE section_id = ? AND tile_id = ?",
            (section_id, tile_id),
        )
        row = cursor.fetchone()
        if not row:
            raise FileNotFoundError(
                f"File for section {section_id}, tile {tile_id} not found in DB."
            )
        return row[0]

    def get_dataset_metadata(
        self,
        split: Optional[str] = None,
        first_sec: Optional[int] = None,
        last_sec: Optional[int] = None,
    ) -> DatasetMetadata:
        query = "SELECT section_id, tile_id FROM tiles WHERE 1=1"
        params = []
        if split:
            query += " AND split = ?"
            params.append(split)
        if first_sec is not None:
            query += " AND section_id >= ?"
            params.append(first_sec)
        if last_sec is not None:
            query += " AND section_id <= ?"
            params.append(last_sec)

        cursor = self.conn.execute(query, params)

        tiles_per_section: Dict[int, List[int]] = {}
        all_sections = set()

        for section_id, tile_id in cursor.fetchall():
            all_sections.add(section_id)
            if section_id not in tiles_per_section:
                tiles_per_section[section_id] = []
            tiles_per_section[section_id].append(tile_id)

        for sec_num in tiles_per_section:
            tiles_per_section[sec_num].sort()

        total_sections = max(all_sections) if all_sections else 0

        return DatasetMetadata(
            total_sections=total_sections, tiles_per_section=tiles_per_section
        )


class SMBStorageClient(RemoteStorageClient):
    """
    Implements RemoteStorageClient to pull raw SBF-SEM tiles directly from
    company SMB mount points using local filesystem paths and directory layout,
    now backed by a fast SQLite metadata store.
    """

    def __init__(
        self,
        extraction_config: ExtractionConfig,
        db_path: Optional[Path | str] = None,
    ):
        self.sbem_root = Path(extraction_config.sbem_root)
        self.first_sec = extraction_config.first_sec
        self.last_sec = extraction_config.last_sec
        self.grid_num = extraction_config.grid_num
        self.grid_str = f"g{extraction_config.grid_num:04d}"
        self.exp_name = self.sbem_root.name
        self.db_path = (
            Path(db_path)
            if db_path
            else self.sbem_root / f"dataset_metadata_{self.grid_str}.db"
        )
        self.metadata_store = SQLiteMetadataStore(self.db_path)

    def fetch_metadata(
        self, force_rescan: bool = False, split: Optional[str] = None
    ) -> DatasetMetadata:
        # Check if DB has data
        cursor = self.metadata_store.conn.execute("SELECT COUNT(*) FROM tiles")
        count = cursor.fetchone()[0]

        if force_rescan or count == 0:
            meta_logs_dir = self.sbem_root / "meta" / "logs"
            print(f"Scanning metadata logs from {meta_logs_dir}...")
            self.metadata_store.update_db(
                meta_logs_dir=meta_logs_dir,
                nr_of_files=None,
                first_sec=self.first_sec,
                last_sec=self.last_sec,
            )
            self.metadata_store.assign_splits()

        return self.metadata_store.get_dataset_metadata(
            split=split, first_sec=self.first_sec, last_sec=self.last_sec
        )

    def download_tile(self, section_id: int, tile_id: str) -> np.ndarray:
        t_id_int = int(tile_id[1:]) if tile_id.startswith("t") else int(tile_id)
        rel_filepath = self.metadata_store.get_filepath(section_id, t_id_int)

        # Determine the absolute filepath. The filename stored in 'filepath' from the logs
        # is relative to the root (e.g. 'tiles/g0000/t0411/...tif').
        filepath = self.sbem_root / rel_filepath

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
    Generates balanced 3D Regions of Interest (ROIs) across the dataset.

    Sampling is weighted by section density: sections containing more acquired tiles
    are sampled proportionally more often, guaranteeing homogeneous volume coverage.

    Attributes:
        z_depth_per_roi (int): The number of consecutive Z-sections (slices) included in a single 3D ROI.
            - Set z_depth_per_roi = 1 for 2D patch extraction (e.g. 2D Noise2Void / NAFNet models).
              In this case, each ROI spans a single slice [z_start, z_start].
            - Set z_depth_per_roi > 1 (e.g., 5 or 10) for 3D volumetric models (e.g. 3D UNet / pseudo-3D models).
              In this case, each ROI spans a 3D sub-volume across z_depth_per_roi consecutive Z-sections [z_start, z_start + z_depth_per_roi - 1].
        tiles_per_section (int): Maximum number of unstitched raw tile IDs to sample per Z-section within each ROI.
    """

    def __init__(self, z_depth_per_roi: int = 1, tiles_per_section: int = 4) -> None:
        """
        Args:
            z_depth_per_roi: Number of consecutive Z-slices spanned by each 3D Region of Interest (default: 1 for 2D models).
            tiles_per_section: Maximum number of raw tile IDs to sample per Z-section (default: 4).
        """
        self.z_depth_per_roi: int = z_depth_per_roi
        self.tiles_per_section: int = tiles_per_section

    def generate_rois(
        self, metadata: DatasetMetadata, num_regions: int
    ) -> List[RegionOfInterest]:
        """
        Generates a list of RegionOfInterest objects.

        Args:
            metadata: DatasetMetadata containing valid section numbers and available tile IDs.
            num_regions: Total number of 3D ROIs to generate.

        Returns:
            List[RegionOfInterest]: Generated volumetric sampling targets.
        """
        rois: List[RegionOfInterest] = []
        valid_sections: List[int] = [
            s for s, t_list in metadata.tiles_per_section.items() if len(t_list) > 0
        ]

        if not valid_sections:
            raise ValueError("No sections with valid acquired tiles found in metadata.")

        # Compute sampling weights proportional to the number of acquired tiles in each section
        weights: List[int] = [
            len(metadata.tiles_per_section[s]) for s in valid_sections
        ]
        total_weight: float = float(sum(weights))
        probabilities: List[float] = [w / total_weight for w in weights]

        for _ in range(num_regions):
            # 1. Weighted random selection of the starting Z-section (z_start)
            z_start: int = random.choices(valid_sections, weights=probabilities, k=1)[0]

            # 2. Compute ending Z-section (z_end) based on z_depth_per_roi
            # Example: if z_start=1250 and z_depth_per_roi=5, then z_end=1254 (sections 1250, 1251, 1252, 1253, 1254)
            z_end: int = min(z_start + self.z_depth_per_roi - 1, max(valid_sections))

            section_tile_map: Dict[int, List[str]] = {}

            # 3. For each Z-section in the [z_start, z_end] volumetric span, sample up to tiles_per_section tile IDs
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
