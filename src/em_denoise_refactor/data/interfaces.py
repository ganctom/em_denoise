# src/em_denoise_refactor/data/interfaces.py

import torch
from torch.utils.data import Dataset
from typing import Protocol, Tuple, runtime_checkable
from pathlib import Path


@runtime_checkable
class LocalPatchStore(Protocol):
    """Contract for reading from the local LMDB/HDF5 store initialized in Phase 1."""

    def open(self, path: Path) -> None: ...

    def __len__(self) -> int: ...

    def get_patch(self, index: int) -> torch.Tensor: ...

    def close(self) -> None: ...


@runtime_checkable
class NoiseSynthesizer(Protocol):
    """
    Contract for procedural noise injection.
    Must simulate SBF-SEM specific artifacts (e.g., dynamic Poisson/Gaussian, streak artifacts)
    without causing RAM cache bloat.
    """

    def inject(self, clean_tensor: torch.Tensor) -> torch.Tensor: ...


class SBFSEMDataset(Dataset):
    """
    PyTorch Dataset implementation.
    Yields (noisy_input, clean_target) pairs for supervised training.
    Executes entirely offline using the local store.
    """

    def __init__(
        self, store: LocalPatchStore, noise_injector: NoiseSynthesizer
    ) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reads clean target from store, applies noise injector to generate input."""
        ...
