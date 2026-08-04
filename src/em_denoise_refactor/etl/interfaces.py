import numpy as np
from typing import Protocol, List, Tuple, Dict, Iterator, runtime_checkable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetMetadata:
    """
    Captures variance across the remote acquisition system.

    Attributes:
        total_sections: Global slice count in the z-stack.
        z_resolutions_nm: Maps section index to physical cutting thickness (nm).
            Physical z-drift caused by microtome thermal variations alters feature
            proportions. Downstream models (especially 3D or pseudo-3D) use this to
            scale anisotropic spatial loss weights or kernel dilation steps.
        tiles_per_section: Maps section index to raw tile yield.
        snr_estimates: Empirical signal-to-noise ratio per section.
            Captures beam current degradation, detector noise, and specimen charging.
            Used directly by the on-the-fly NoiseSynthesizer to bound synthetic noise
            injection parameters, matching real degradation profiles during training.
    """

    total_sections: int
    z_resolutions_nm: Dict[int, float]
    tiles_per_section: Dict[int, list[int]]
    snr_estimates: Dict[int, float]


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
class RemoteStorageClient(Protocol):
    """Interface for streaming unstitched raw .tif tiles from the remote acquisition host."""

    def fetch_metadata(self) -> DatasetMetadata: ...

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
