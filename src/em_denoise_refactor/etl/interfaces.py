import numpy as np
from typing import Protocol, List, Tuple, Dict, Iterator, runtime_checkable, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetMetadata:
    """
    Captures variance across the remote acquisition system.

    Attributes:
        total_sections: Global slice count in the z-stack.
        tiles_per_section: Maps section index to raw tile yield.
    """

    total_sections: int
    tiles_per_section: Dict[int, list[int]]


@dataclass
class RegionOfInterest:
    """
    Defines a volumetric target boundary operating strictly on unstitched acquisition data.

    Instead of continuous Cartesian bounds (x, y), regions map explicit section
    identifiers to lists of unaligned raw tile keys acquired by the scope.
    """

    z_start: int
    z_end: int
    section_tile_map: Dict[int, List[str]]


@runtime_checkable
class FileMetadataStore(Protocol):
    """
    Interface for querying file metadata (section_id, tile_id, filepath, split)
    from an indexed database (e.g., SQLite).
    """

    def update_db(
        self,
        meta_logs_dir: Path,
        first_sec: int,
        last_sec: int,
        nr_of_files: int | None,
    ) -> None: ...

    def assign_splits(
        self,
        strategy: str = "block",
        block_size: int = 100,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        guard_band_size: int = 5,
    ) -> None: ...

    def get_filepath(self, section_id: int, tile_id: int) -> str: ...

    def get_dataset_metadata(
        self,
        split: Optional[str] = None,
        first_sec: Optional[int] = None,
        last_sec: Optional[int] = None,
    ) -> DatasetMetadata: ...


@runtime_checkable
class RemoteStorageClient(Protocol):
    """Interface for streaming unstitched raw .tif tiles from the remote acquisition host."""

    def fetch_metadata(
        self, force_rescan: bool = False, split: Optional[str] = None
    ) -> DatasetMetadata: ...

    def download_tile(self, section_id: int, tile_id: str) -> np.ndarray: ...


@runtime_checkable
class ROISampler(Protocol):
    """Generates balanced sampling targets across anisotropic sections and variable SNR zones."""

    def generate_rois(
        self, metadata: DatasetMetadata, num_regions: int
    ) -> List[RegionOfInterest]: ...


@runtime_checkable
class PatchExtractor(Protocol):
    """Extracts sub-crops (patches) directly from unstitched raw tile arrays."""

    def extract_from_roi(
        self,
        client: RemoteStorageClient,
        roi: RegionOfInterest,
        patch_size: Tuple[int, int],
    ) -> Iterator[np.ndarray]: ...


@runtime_checkable
class BinaryStoreWriter(Protocol):
    """Stages extracted patches into local zero-copy binary formats (LMDB) optimized for M4 Pro unified memory."""

    def initialize(self, path: Path, expected_capacity: int) -> None: ...

    def write_patch(self, patch_id: str, data: np.ndarray) -> None: ...

    def commit_transaction(self) -> None: ...

    def close(self) -> None: ...
