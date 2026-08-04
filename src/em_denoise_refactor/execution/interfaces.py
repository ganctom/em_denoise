# src/execution/interfaces.pyi
import torch
import torch.nn as nn
from typing import Protocol, Dict, Any, Callable, runtime_checkable
from pathlib import Path
from torch.utils.data import DataLoader


@runtime_checkable
class MPSDeviceManager(Protocol):
    """
    Contract for managing Apple Silicon (M4 Pro) specific optimizations.
    Handles device targeting and unified memory lifecycle (cache clearing).
    """

    def get_device(self) -> torch.device: ...
    def optimize_memory(self) -> None: ...
    def log_unified_memory_usage(self) -> Dict[str, float]: ...


@runtime_checkable
class ModelTrainer(Protocol):
    """
    Contract for executing a single training run with a specific hyperparameter configuration[cite: 13].
    """

    def train_epoch(self, dataloader: DataLoader, device: torch.device) -> float: ...
    def validate(
        self, dataloader: DataLoader, device: torch.device
    ) -> Dict[str, float]: ...
    def save_checkpoint(self, path: Path) -> None: ...


@runtime_checkable
class GridSearchEngine(Protocol):
    """
    Contract for orchestrating offline hyperparameter tuning over the MPS backend[cite: 13].
    """

    def __init__(
        self,
        device_manager: MPSDeviceManager,
        trainer_factory: Callable[[Dict[str, Any]], ModelTrainer],
    ) -> None: ...

    def execute_search(
        self,
        param_grid: Dict[str, list[Any]],
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Iterates through param_grid, trains models, logs metrics, and returns the optimal config.
        """
        ...


@runtime_checkable
class InferenceEngine(Protocol):
    """
    Executes trained denoising inference across staged datasets or raw single-frame targets.
    """

    def __init__(self, model: nn.Module, device_manager: MPSDeviceManager) -> None: ...

    def process_volume(
        self, input_store_path: Path, output_store_path: Path, batch_size: int = 32
    ) -> None:
        """
        High-throughput inference pass reading from and writing to local LMDB binary stores.
        """
        ...

    def process_raw_tif(
        self, input_tif_path: Path, output_tif_path: Path, tile_overlap: int = 64
    ) -> None:
        """
        Direct-to-disk inference on a single unstitched 8-bit .tif file.
        Applies windowed sliding-crop inference with spatial overlap blending on the mps backend,
        bypassing the binary store ETL for fast visual validation[cite: 15].
        """
        ...
