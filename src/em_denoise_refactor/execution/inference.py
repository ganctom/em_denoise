import torch
import torch.nn as nn
import numpy as np
import tifffile
import itertools
from pathlib import Path
from typing import Dict, Any, Callable, List
from torch.utils.data import DataLoader
from em_denoise_refactor.execution.interfaces import MPSDeviceManager, ModelTrainer


from tqdm import tqdm


class DenoisingModelTrainer:
    """Executes a single training run with a specific hyperparameter configuration."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device_manager: MPSDeviceManager,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device_manager = device_manager

    def train_epoch(
        self, dataloader: DataLoader, device: torch.device, epoch: int = 1
    ) -> float:
        self.model.train()
        self.model.to(device)
        total_loss = 0.0

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch:03d} [Train]",
            leave=False,
        )

        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            # Prevent cache bloat during large batch processing
            if batch_idx % 10 == 0:
                self.device_manager.optimize_memory()

            postfix = {"loss": f"{batch_loss:.5f}"}
            if device.type == "mps":
                allocated_mb = torch.mps.current_allocated_memory() / (1024**2)
                postfix["mps_mem"] = f"{allocated_mb:.0f}MB"
            pbar.set_postfix(postfix)

        return total_loss / len(dataloader)

    def validate(
        self, dataloader: DataLoader, device: torch.device, epoch: int = 1
    ) -> Dict[str, float]:
        self.model.eval()
        self.model.to(device)
        val_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [Val]", leave=False)

        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

        self.device_manager.optimize_memory()
        return {"val_loss": val_loss / len(dataloader)}

    def save_checkpoint(self, path: Path) -> None:
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()}, path)


class MPSGridSearchEngine:
    """Orchestrates offline hyperparameter tuning over the MPS backend."""

    def __init__(
        self,
        device_manager: MPSDeviceManager,
        trainer_factory: Callable[[Dict[str, Any]], ModelTrainer],
    ) -> None:
        self.device_manager = device_manager
        self.trainer_factory = trainer_factory

    def execute_search(
        self,
        param_grid: Dict[str, List[Any]],
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        device = self.device_manager.get_device()
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        best_loss = float("inf")
        best_config = {}

        for combo in combinations:
            config = dict(zip(keys, combo))
            trainer = self.trainer_factory(config)

            # E.g., execute a limited epoch test for the grid search param combination
            trainer.train_epoch(train_loader, device)
            metrics = trainer.validate(val_loader, device)

            if metrics["val_loss"] < best_loss:
                best_loss = metrics["val_loss"]
                best_config = config

            # Purge memory between grid search trials
            self.device_manager.optimize_memory()

        return best_config


class StandaloneInferenceEngine:
    """Executes trained denoising inference across staged datasets or raw single-frame targets."""

    def __init__(self, model: nn.Module, device_manager: MPSDeviceManager) -> None:
        self.model = model
        self.device_manager = device_manager
        self.device = device_manager.get_device()
        self.model.to(self.device)

    def process_volume(
        self, input_store_path: Path, output_store_path: Path, batch_size: int = 32
    ) -> None:
        """High-throughput inference pass reading from and writing to local LMDB binary stores."""
        # LMDB batch inference integration
        pass

    def _create_2d_hann_window(self, window_size: int) -> torch.Tensor:
        """
        Generates a 2D partition-of-unity window for 50% spatial overlap blending.
        Uses a sine-window (sqrt-Hann) to guarantee sum(W^2) == 1.0 across overlap steps.
        """
        # Linear space from 0.5/N to (N-0.5)/N to prevent exact boundary zeros
        n = torch.linspace(0.5, window_size - 0.5, window_size, device=self.device)
        window_1d = torch.sin(torch.pi * n / window_size)
        window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
        return window_2d

    def process_raw_tif(
        self, input_tif_path: Path, output_tif_path: Path, tile_overlap: int = 256
    ) -> None:
        """
        Applies windowed sliding-crop inference with spatial overlap blending on the MPS backend.
        Assumes 50% overlap where tile_overlap = 256 yields a 512x512 window.
        """
        window_size = tile_overlap * 2
        step = window_size - tile_overlap

        # 1. Zero-Copy Memory Mapping
        in_mmap = tifffile.memmap(input_tif_path, mode="r")
        h, w = in_mmap.shape

        # 2. Local Accumulator Buffers (CPU RAM)
        out_accum = np.zeros((h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)

        # 3. Blending Window Setup
        hann_2d = self._create_2d_hann_window(window_size)
        hann_2d_cpu = hann_2d.cpu().numpy()  # Cached for CPU accumulation

        self.model.eval()
        with torch.no_grad():
            for y in range(0, h - window_size + 1, step):
                for x in range(0, w - window_size + 1, step):
                    # Pull 512x512 tile from SSD into RAM
                    patch = in_mmap[y : y + window_size, x : x + window_size]

                    # Cast & Transfer to Acceleration Device
                    patch_tensor = torch.from_numpy(patch.astype(np.float32) / 255.0)
                    patch_tensor = (
                        patch_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                    )

                    # Network Forward Pass
                    pred = self.model(patch_tensor).squeeze()

                    # Apply Hann Window Blending
                    blended_pred = pred * hann_2d

                    # Accumulate in CPU memory map
                    out_accum[y : y + window_size, x : x + window_size] += (
                        blended_pred.cpu().numpy()
                    )
                    weight_accum[y : y + window_size, x : x + window_size] += (
                        hann_2d_cpu
                    )

                # M4 Pro Unified Memory Protection (Row-level cache purge)
                self.device_manager.optimize_memory()

        # 4. Normalize overlapping weights & finalize TIF reconstruction
        normalized_out = out_accum / (weight_accum + 1e-8)
        final_out = np.clip(normalized_out * 255.0, 0, 255).astype(np.uint8)
        tifffile.imwrite(output_tif_path, final_out)
