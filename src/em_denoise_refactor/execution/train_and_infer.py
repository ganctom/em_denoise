# src/em_denoise_refactor/execution/train_and_infer.py

"""
Production-grade training and evaluation pipeline for Connectomics Denoising neural networks.

Executes offline model training reading from high-speed LMDB patch stores (or disk TIFF patch folders),
monitors convergence via TensorBoard, purges Apple Silicon MPS memory buffers, saves model checkpoints,
and evaluates performance metrics (MSE, PSNR, SSIM) on test sets.
"""

import argparse
from dataclasses import asdict
import gc
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from em_denoise_refactor.config import PipelineConfig
from em_denoise_refactor.data.dataset import (
    SBFSEMNoiseSynthesizer,
    EmpiricalNoiseSynthesizer,
)
from em_denoise_refactor.data.n2v_dataset import N2VDatasetWrapper
from em_denoise_refactor.data.n2n_dataset import (
    N2NOverlapPairDataset,
    N2NStreamingShardDataset,
)
from em_denoise_refactor.data.n2n_duplicate_dataset import N2NDuplicateTileDataset
from em_denoise_refactor.execution.calibration import run_calibration
from em_denoise_refactor.models.factory import build_model
from em_denoise_refactor.models.loss import (
    CompositeDenoisingLoss,
    EnhancedCompositeLoss,
    DynamicEMLoss,
    CharbonnierLoss,
    LaplacianLoss,
    FocalFrequencyLoss,
    ESPILoss,
    SobelEdgeLoss,
    SSIMLoss,
)
from em_denoise_refactor.data.interfaces import NoiseSynthesizer

logger = logging.getLogger("train_and_infer")


def _set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_plain_data(obj: Any) -> Any:
    """Recursively convert Path objects to strings for checkpoint/JSON safety."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_data(v) for v in obj]
    return obj


def _serialize_config(cfg: PipelineConfig) -> Dict[str, Any]:
    """Serialize a PipelineConfig dataclass to a plain JSON-safe dictionary."""
    return _to_plain_data(asdict(cfg))


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    lr_scheduler: str = "none",
    lr_min: float = 1e-6,
    warmup_epochs: int = 0,
) -> Optional[Any]:
    """Construct an epoch-stepped LR scheduler ('none' or 'cosine' with optional warmup)."""
    mode = lr_scheduler.lower().strip()
    if mode not in ("none", "cosine"):
        raise ValueError(
            f"lr_scheduler must be 'none' or 'cosine', got '{lr_scheduler}'"
        )
    if mode == "none":
        return None

    total_epochs = max(1, int(epochs))
    warmup = max(0, int(warmup_epochs))
    if warmup > 0 and total_epochs > warmup:
        start_factor = 0.1 if warmup == 1 else max(0.05, 1.0 / warmup)
        warmup_sched = LinearLR(
            optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup
        )
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup), eta_min=float(lr_min)
        )
        return SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
        )
    if warmup > 0:
        return LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=total_epochs
        )
    return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=float(lr_min))


def compute_noise_corrected_metrics(
    pred_mse: float, raw_pair_mse: float
) -> Tuple[float, float]:
    """
    Estimate noise-free MSE and PSNR on Noise2Noise pairs in [0, 1] range.

    Under independent zero-mean noise with variance sigma^2 on both images:
      MSE(x, y) = 2 * sigma^2
      MSE(pred, y) = MSE(pred, clean) + sigma^2
    Therefore:
      est_clean_mse ≈ MSE(pred, y) - 0.5 * MSE(x, y).
    """
    est_clean_mse = max(float(pred_mse) - 0.5 * float(raw_pair_mse), 1e-10)
    est_clean_psnr = float(-10.0 * math.log10(est_clean_mse))
    return est_clean_mse, est_clean_psnr


def _compute_residual_stats(
    raw_255: np.ndarray, denoised_255: np.ndarray
) -> Dict[str, float]:
    """Compute mean, std, and lag-1 X/Y autocorrelations of (raw - denoised)."""
    res = raw_255.astype(np.float32) - denoised_255.astype(np.float32)
    mean_val = float(res.mean())
    std_val = float(res.std())
    centered = res - mean_val
    var = float(np.mean(centered * centered))
    if var < 1e-12:
        ac_x = 0.0
        ac_y = 0.0
    else:
        ac_x = (
            float(np.mean(centered[:, :-1] * centered[:, 1:]) / var)
            if res.shape[1] > 1
            else 0.0
        )
        ac_y = (
            float(np.mean(centered[:-1, :] * centered[1:, :]) / var)
            if res.shape[0] > 1
            else 0.0
        )
    return {"mean": mean_val, "std": std_val, "ac_x": ac_x, "ac_y": ac_y}


# -------------------------------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------------------------------


class EvaluationMetrics:
    """Provides spatial image quality assessment metrics."""

    @staticmethod
    def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """Computes Mean Squared Error between two images."""
        return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))

    @classmethod
    def compute_psnr(
        cls, img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0
    ) -> float:
        """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
        mse = cls.compute_mse(img1, img2)
        if mse < 1e-10:
            return 100.0
        return float(10 * math.log10((data_range**2) / mse))

    @classmethod
    def compute_ssim(
        cls, img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0
    ) -> float:
        """Computes Structural Similarity Index (SSIM)."""
        try:
            from skimage.metrics import structural_similarity as ssim

            return float(ssim(img1, img2, data_range=data_range))
        except ImportError:
            c1 = (0.01 * data_range) ** 2
            c2 = (0.03 * data_range) ** 2
            mu1 = float(np.mean(img1))
            mu2 = float(np.mean(img2))
            sigma1_sq = float(np.var(img1))
            sigma2_sq = float(np.var(img2))
            sigma12 = float(np.cov(img1.flat, img2.flat)[0, 1])

            num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
            den = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
            return float(num / den)


# -------------------------------------------------------------------------
# Datasets (TIFF Folder & LMDB Store)
# -------------------------------------------------------------------------


class PatchFolderDataset(Dataset):
    """Dataset reading clean patch pairs from TIFF directories with optional on-the-fly noise synthesis."""

    def __init__(
        self,
        clean_dir: Path | str,
        noisy_dir: Optional[Path | str] = None,
        noise_injector: Optional[NoiseSynthesizer] = None,
        cache_in_ram: bool = True,
        augment: bool = False,
    ) -> None:
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir) if noisy_dir else None
        self.noise_injector = noise_injector
        self.cache_in_ram = cache_in_ram
        self.augment = augment

        self.filenames = sorted([f.name for f in self.clean_dir.glob("*.tif")])
        if not self.filenames:
            raise FileNotFoundError(f"No .tif patches found in {self.clean_dir}")

        self.is_val_or_test = any(
            x in str(self.clean_dir).lower() for x in ["val", "test"]
        )
        self.cached_clean_samples: List[Tuple[torch.Tensor, str]] = []
        self.fixed_noisy_samples: List[torch.Tensor] = []

        if self.cache_in_ram or self.is_val_or_test:
            logger.info(
                f"Caching {len(self.filenames)} TIFF patches from {self.clean_dir.parent.name} into RAM..."
            )
            for fname in tqdm(
                self.filenames, desc=f"RAM Caching ({self.clean_dir.parent.name})"
            ):
                clean_path = self.clean_dir / fname
                clean_img = tifffile.imread(str(clean_path)).astype(np.float32) / 255.0
                clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
                self.cached_clean_samples.append((clean_tensor, fname))

                if self.is_val_or_test:
                    if self.noise_injector is not None:
                        noisy_tensor = self.noise_injector(
                            clean_tensor.unsqueeze(0)
                        ).squeeze(0)
                    elif self.noisy_dir is not None:
                        noisy_path = self.noisy_dir / fname
                        noisy_img = (
                            tifffile.imread(str(noisy_path)).astype(np.float32) / 255.0
                        )
                        noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
                    else:
                        noisy_tensor = clean_tensor.clone()
                    self.fixed_noisy_samples.append(noisy_tensor)

    def __len__(self) -> int:
        return len(self.filenames)

    def _apply_augmentation(
        self, noisy: torch.Tensor, clean: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > 0.5:
            noisy = torch.flip(noisy, dims=[-1])
            clean = torch.flip(clean, dims=[-1])
        if torch.rand(1).item() > 0.5:
            noisy = torch.flip(noisy, dims=[-2])
            clean = torch.flip(clean, dims=[-2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            noisy = torch.rot90(noisy, k, dims=[-2, -1])
            clean = torch.rot90(clean, k, dims=[-2, -1])
        return noisy, clean

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.is_val_or_test:
            clean_tensor, fname = self.cached_clean_samples[idx]
            noisy_tensor = self.fixed_noisy_samples[idx].clone()
            clean_tensor = clean_tensor.clone()
        elif self.cache_in_ram:
            clean_tensor, fname = self.cached_clean_samples[idx]
            clean_tensor = clean_tensor.clone()
            if self.noise_injector is not None:
                noisy_tensor = self.noise_injector(clean_tensor.unsqueeze(0)).squeeze(0)
            else:
                noisy_tensor = clean_tensor.clone()
        else:
            fname = self.filenames[idx]
            clean_img = (
                tifffile.imread(str(self.clean_dir / fname)).astype(np.float32) / 255.0
            )
            clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)

            if self.noise_injector is not None:
                noisy_tensor = self.noise_injector(clean_tensor.unsqueeze(0)).squeeze(0)
            elif self.noisy_dir is not None:
                noisy_img = (
                    tifffile.imread(str(self.noisy_dir / fname)).astype(np.float32)
                    / 255.0
                )
                noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
            else:
                noisy_tensor = clean_tensor.clone()

        if self.augment:
            noisy_tensor, clean_tensor = self._apply_augmentation(
                noisy_tensor, clean_tensor
            )
        return noisy_tensor, clean_tensor, fname


class CustomTIFFDataset(Dataset):
    """Dataset for running inference on arbitrary .tif / .tiff images from a custom folder."""

    def __init__(
        self,
        image_dir: Path | str,
        cache_in_ram: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Custom image directory not found: {self.image_dir}"
            )

        self.filenames = sorted(
            [f.name for f in self.image_dir.glob("*.tif")]
            + [f.name for f in self.image_dir.glob("*.tiff")]
        )
        if not self.filenames:
            raise FileNotFoundError(
                f"No .tif or .tiff images found in custom directory: {self.image_dir}"
            )

        self.cache_in_ram = cache_in_ram
        self.cached_samples: List[Tuple[torch.Tensor, str]] = []
        if self.cache_in_ram:
            logger.info(
                f"Caching {len(self.filenames)} custom TIFF images from {self.image_dir.name} into RAM..."
            )
            for fname in tqdm(self.filenames, desc="RAM Caching (custom)"):
                img_path = self.image_dir / fname
                img = tifffile.imread(str(img_path)).astype(np.float32) / 255.0
                tensor = torch.from_numpy(img).unsqueeze(0)
                self.cached_samples.append((tensor, fname))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.cache_in_ram:
            img_tensor, fname = self.cached_samples[idx]
            img_tensor = img_tensor.clone()
        else:
            fname = self.filenames[idx]
            img = (
                tifffile.imread(str(self.image_dir / fname)).astype(np.float32) / 255.0
            )
            img_tensor = torch.from_numpy(img).unsqueeze(0)

        return img_tensor, img_tensor, fname


class LMDBPatchDataset(Dataset):
    """Dataset reading memory-mapped clean patches from LMDB store with on-the-fly noise synthesis."""

    def __init__(
        self,
        lmdb_path: Path | str,
        split: str = "train",
        noise_injector: Optional[NoiseSynthesizer] = None,
        cache_in_ram: bool = True,
        augment: bool = False,
    ) -> None:
        import lmdb

        self.lmdb_path = Path(lmdb_path)
        self.split = split
        self.noise_injector = noise_injector
        self.cache_in_ram = cache_in_ram
        self.augment = augment
        self.is_val_or_test = split in ["val", "test"]

        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB store not found at {self.lmdb_path}")

        env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False)
        self.keys: List[bytes] = []
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            prefix = f"{self.split}_".encode("utf-8")
            for key, _ in cursor:
                if key.startswith(prefix):
                    self.keys.append(key)
        env.close()

        if not self.keys:
            raise ValueError(
                f"No patches found for split '{split}' in {self.lmdb_path}"
            )

        # Map LMDB sequential keys (e.g. test_00000000) to original TIFF filenames if disk patches exist
        tiff_clean_dir = self.lmdb_path.parent / split / "clean"
        self.filename_map: Dict[str, str] = {}
        if tiff_clean_dir.exists():
            tiff_files = sorted([f.name for f in tiff_clean_dir.glob("*.tif")])
            if len(tiff_files) == len(self.keys):
                for key_bytes, fname in zip(self.keys, tiff_files):
                    self.filename_map[key_bytes.decode("utf-8")] = fname

        self.cached_clean_samples: List[Tuple[torch.Tensor, str]] = []
        self.fixed_val_samples: List[Tuple[torch.Tensor, torch.Tensor, str]] = []

        if self.cache_in_ram or self.is_val_or_test:
            logger.info(f"Caching {len(self.keys)} LMDB '{split}' patches into RAM...")
            env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False)
            with env.begin(write=False) as txn:
                for key in tqdm(self.keys, desc=f"RAM Caching ({split})"):
                    buf = txn.get(key)
                    side_len = int(np.sqrt(len(buf)))
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape(side_len, side_len)
                    clean_tensor = torch.from_numpy(
                        arr.astype(np.float32) / 255.0
                    ).unsqueeze(0)

                    key_str = key.decode("utf-8")
                    fname_str = self.filename_map.get(key_str, key_str)

                    if self.is_val_or_test:
                        if self.noise_injector is not None:
                            noisy_tensor = self.noise_injector(
                                clean_tensor.unsqueeze(0)
                            ).squeeze(0)
                        else:
                            noisy_tensor = clean_tensor.clone()
                        self.fixed_val_samples.append(
                            (noisy_tensor, clean_tensor, fname_str)
                        )
                    else:
                        self.cached_clean_samples.append((clean_tensor, fname_str))
            env.close()

    def __len__(self) -> int:
        return len(self.keys)

    def _apply_augmentation(
        self, noisy: torch.Tensor, clean: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > 0.5:
            noisy = torch.flip(noisy, dims=[-1])
            clean = torch.flip(clean, dims=[-1])
        if torch.rand(1).item() > 0.5:
            noisy = torch.flip(noisy, dims=[-2])
            clean = torch.flip(clean, dims=[-2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            noisy = torch.rot90(noisy, k, dims=[-2, -1])
            clean = torch.rot90(clean, k, dims=[-2, -1])
        return noisy, clean

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.is_val_or_test:
            noisy_tensor, clean_tensor, key_str = self.fixed_val_samples[idx]
            noisy_tensor = noisy_tensor.clone()
            clean_tensor = clean_tensor.clone()
        elif self.cache_in_ram:
            clean_tensor, key_str = self.cached_clean_samples[idx]
            clean_tensor = clean_tensor.clone()
            if self.noise_injector is not None:
                noisy_tensor = self.noise_injector(clean_tensor.unsqueeze(0)).squeeze(0)
            else:
                noisy_tensor = clean_tensor.clone()
        else:
            import lmdb

            key = self.keys[idx]
            env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False)
            with env.begin(write=False) as txn:
                buf = txn.get(key)
                side_len = int(np.sqrt(len(buf)))
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(side_len, side_len)
            env.close()

            clean_tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
            if self.noise_injector is not None:
                noisy_tensor = self.noise_injector(clean_tensor.unsqueeze(0)).squeeze(0)
            else:
                noisy_tensor = clean_tensor.clone()
            key_str = self.filename_map.get(key.decode("utf-8"), key.decode("utf-8"))

        if self.augment:
            noisy_tensor, clean_tensor = self._apply_augmentation(
                noisy_tensor, clean_tensor
            )
        return noisy_tensor, clean_tensor, key_str


# -------------------------------------------------------------------------
# Denoising Trainer & Evaluator
# -------------------------------------------------------------------------


class DenoisingTrainer:
    """Manages model training, validation metrics calculation, and hardware cache purging."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoints_dir: Path,
        use_amp: bool = False,
        writer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoints_dir = checkpoints_dir
        self.use_amp = use_amp
        self.writer = writer
        self.non_blocking = device.type == "cuda"
        self.val_ssim_fn = SSIMLoss(window_size=11, in_channels=1).to(device)
        self.amp_dtype = (
            torch.bfloat16
            if (
                use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            )
            else torch.float16
        )
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if (use_amp and self.amp_dtype == torch.float16)
            else None
        )

        self.mps_manager = None
        if device.type == "mps":
            try:
                from em_denoise_refactor.execution.mps_manager import (
                    AppleSiliconManager,
                )

                self.mps_manager = AppleSiliconManager()
            except Exception:
                self.mps_manager = None

    def _purge_memory(self) -> None:
        if self.mps_manager is not None:
            self.mps_manager.optimize_memory()
        elif self.device.type == "mps":
            gc.collect()
            torch.mps.empty_cache()

    def train_epoch(
        self, dataloader: DataLoader, epoch: int, total_epochs: int
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        total_metrics: Dict[str, float] = {}
        is_n2v = getattr(self.model, "is_n2v", False)

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Train",
            leave=False,
            bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        for batch_idx, batch in pbar:
            if is_n2v:
                # N2V batch: (masked_input, original_noisy, mask, fname)
                masked_b, original_b, mask_b, _ = batch
                masked_b = masked_b.to(self.device, non_blocking=self.non_blocking)
                original_b = original_b.to(self.device, non_blocking=self.non_blocking)
                mask_b = mask_b.to(self.device, non_blocking=self.non_blocking).float()

                self.optimizer.zero_grad()
                pred_b = self.model(masked_b)
                # Blind-spot loss: MSE only at masked positions
                loss = ((pred_b - original_b) ** 2 * mask_b).sum() / torch.clamp(
                    mask_b.sum(), min=1.0
                )
                metrics = {"loss_masked_mse": float(loss.item())}
                batch_size = masked_b.size(0)
            else:
                # Supervised batch: (noisy, clean, fname)
                noisy_b, clean_b, _ = batch
                noisy_b = noisy_b.to(self.device, non_blocking=self.non_blocking)
                clean_b = clean_b.to(self.device, non_blocking=self.non_blocking)

                self.optimizer.zero_grad()
                if hasattr(self.model, "ar_decoder"):
                    pred_b, loss_out = self.model(noisy_b)
                elif self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        pred_b = self.model(noisy_b)
                        loss_out = self.criterion(pred_b, clean_b)
                else:
                    pred_b = self.model(noisy_b)
                    loss_out = self.criterion(pred_b, clean_b)

                if isinstance(loss_out, tuple):
                    loss, metrics = loss_out
                else:
                    loss = loss_out
                    metrics = {"loss_total": float(loss.item())}
                batch_size = noisy_b.size(0)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            batch_loss = float(loss.item())
            total_loss += batch_loss * batch_size
            total_samples += batch_size

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + float(v) * batch_size

            if batch_idx % 10 == 0:
                self._purge_memory()

            postfix = {"loss": f"{batch_loss:.5f}"}
            if is_n2v:
                postfix["masked_mse"] = f"{batch_loss:.5f}"
            elif "loss_charbonnier" in metrics:
                postfix["chrb"] = f"{metrics['loss_charbonnier']:.4f}"
            elif "loss_edge" in metrics and metrics["loss_edge"] > 0:
                postfix["edge"] = f"{metrics['loss_edge']:.4f}"
            pbar.set_postfix(postfix)

        denom = total_samples if total_samples > 0 else len(dataloader.dataset)
        avg_metrics = {k: v / denom for k, v in total_metrics.items()}
        return total_loss / denom, avg_metrics

    def validate(
        self, dataloader: DataLoader, epoch: int, total_epochs: int
    ) -> Tuple[float, float, float, Dict[str, float]]:
        self.model.eval()
        total_val_loss = 0.0
        total_samples = 0
        total_raw_pair_mse = 0.0
        total_pred_mse = 0.0
        total_val_psnr_sum = 0.0
        total_val_ssim_sum = 0.0
        total_val_metrics: Dict[str, float] = {}
        is_n2v = getattr(self.model, "is_n2v", False)

        pbar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Val",
            leave=False,
            bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        with torch.no_grad():
            for batch in pbar:
                if is_n2v:
                    # N2V val batch: (masked_input, original_noisy, mask, fname)
                    masked_b, original_b, mask_b, _ = batch
                    masked_b = masked_b.to(self.device, non_blocking=self.non_blocking)
                    original_b = original_b.to(
                        self.device, non_blocking=self.non_blocking
                    )
                    mask_b = mask_b.to(
                        self.device, non_blocking=self.non_blocking
                    ).float()

                    pred_b = self.model(masked_b)
                    # Blind-spot loss at masked positions only
                    loss = ((pred_b - original_b) ** 2 * mask_b).sum() / torch.clamp(
                        mask_b.sum(), min=1.0
                    )
                    metrics = {"loss_masked_mse": float(loss.item())}
                    batch_size = masked_b.size(0)
                    # N2V has no clean reference for PSNR/SSIM during training —
                    # skip to avoid misleading metrics (val PSNR will show 0.0)
                else:
                    # Supervised / N2N val batch: (noisy, target, fname)
                    noisy_b, clean_b, _ = batch
                    noisy_b = noisy_b.to(self.device, non_blocking=self.non_blocking)
                    clean_b = clean_b.to(self.device, non_blocking=self.non_blocking)

                    if hasattr(self.model, "ar_decoder"):
                        pred_b, loss_out = self.model(noisy_b)
                    elif self.use_amp:
                        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                            pred_b = self.model(noisy_b)
                            loss_out = self.criterion(pred_b, clean_b)
                    else:
                        pred_b = self.model(noisy_b)
                        loss_out = self.criterion(pred_b, clean_b)

                    if isinstance(loss_out, tuple):
                        loss, metrics = loss_out
                    else:
                        loss = loss_out
                        metrics = {"loss_total": float(loss.item())}

                    batch_size = noisy_b.size(0)
                    pred_clamp = torch.clamp(pred_b.float(), 0.0, 1.0)
                    clean_f = clean_b.float()
                    raw_pair_mse = float(
                        torch.mean((noisy_b.float() - clean_f) ** 2).item()
                    )
                    mse_per_sample = torch.mean(
                        (pred_clamp - clean_f) ** 2, dim=(1, 2, 3)
                    )
                    pred_mse = float(mse_per_sample.mean().item())
                    total_raw_pair_mse += raw_pair_mse * batch_size
                    total_pred_mse += pred_mse * batch_size

                    psnr_per_sample = torch.where(
                        mse_per_sample < 1e-10,
                        torch.full_like(mse_per_sample, 100.0),
                        -10.0 * torch.log10(mse_per_sample),
                    )
                    batch_ssim = float(
                        1.0 - self.val_ssim_fn(pred_clamp, clean_f).item()
                    )
                    total_val_psnr_sum += float(psnr_per_sample.sum().item())
                    total_val_ssim_sum += batch_ssim * batch_size

                total_val_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                for k, v in metrics.items():
                    total_val_metrics[k] = (
                        total_val_metrics.get(k, 0.0) + float(v) * batch_size
                    )

        self._purge_memory()
        denom = total_samples if total_samples > 0 else len(dataloader.dataset)
        mean_val_loss = total_val_loss / denom
        mean_val_psnr = (
            total_val_psnr_sum / denom if (not is_n2v and denom > 0) else 0.0
        )
        mean_val_ssim = (
            total_val_ssim_sum / denom if (not is_n2v and denom > 0) else 0.0
        )
        avg_val_metrics = {k: v / denom for k, v in total_val_metrics.items()}
        if not is_n2v and denom > 0:
            mean_raw_pair_mse = total_raw_pair_mse / denom
            mean_pred_mse = total_pred_mse / denom
            est_clean_mse, est_clean_psnr = compute_noise_corrected_metrics(
                mean_pred_mse, mean_raw_pair_mse
            )
            avg_val_metrics["raw_pair_mse"] = mean_raw_pair_mse
            avg_val_metrics["pred_mse"] = mean_pred_mse
            avg_val_metrics["est_clean_mse"] = est_clean_mse
            avg_val_metrics["est_clean_psnr"] = est_clean_psnr

        return mean_val_loss, mean_val_psnr, mean_val_ssim, avg_val_metrics

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_psnr: float,
        val_ssim: float,
        is_best: bool = False,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        save_all_epochs: bool = True,
    ) -> Path:
        raw_model = getattr(self.model, "_orig_mod", self.model)
        state = {
            "epoch": epoch,
            "model_state_dict": {k: v.cpu() for k, v in raw_model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "config": config,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
        }

        if save_all_epochs:
            ckpt_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        else:
            ckpt_path = self.checkpoints_dir / "checkpoint_last.pt"
        torch.save(state, ckpt_path)

        if is_best:
            best_path = self.checkpoints_dir / "checkpoint_best.pt"
            torch.save(state, best_path)
            return best_path
        return ckpt_path


class DenoisingEvaluator:
    """Executes final evaluation and TIFF output generation using trained checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_amp: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.amp_dtype = (
            torch.bfloat16
            if (
                use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            )
            else torch.float16
        )

        # Loss metric calculators for detailed test set evaluation
        self.charbonnier_fn = CharbonnierLoss().to(device)
        self.laplacian_fn = LaplacianLoss(channels=1).to(device)
        self.ffl_fn = FocalFrequencyLoss().to(device)
        self.espi_fn = ESPILoss(channels=1).to(device)
        self.sobel_fn = SobelEdgeLoss().to(device)

    def _denoise_tiled(
        self, x: torch.Tensor, patch_size: int = 128, num_samples: int = 5
    ) -> torch.Tensor:
        if hasattr(self.model, "crop_size"):
            patch_size = self.model.crop_size[0]

        # Use 50% overlap for smooth blending
        stride = patch_size // 2
        B, C, H, W = x.shape
        output = torch.zeros_like(x)
        weight_sum = torch.zeros_like(x)

        # 2D Hann window for smooth cosine blending across tile boundaries
        w1d = torch.hann_window(patch_size, periodic=False, device=x.device)
        window = torch.outer(w1d, w1d).clamp(min=1e-4).unsqueeze(0).unsqueeze(0)

        for y in range(0, H, stride):
            for x_coord in range(0, W, stride):
                y1 = min(y, H - patch_size)
                x1 = min(x_coord, W - patch_size)
                y2 = y1 + patch_size
                x2 = x1 + patch_size
                patch = x[:, :, y1:y2, x1:x2]

                if hasattr(self.model, "denoise"):
                    sample_preds = [
                        self.model.denoise(patch) for _ in range(num_samples)
                    ]
                    pred_patch = torch.stack(sample_preds, dim=0).mean(dim=0)
                else:
                    pred_patch = self.model(patch)

                output[:, :, y1:y2, x1:x2] += pred_patch * window
                weight_sum[:, :, y1:y2, x1:x2] += window

        return output / weight_sum

    def evaluate_test_set(
        self, test_loader: DataLoader, output_denoised_dir: Path
    ) -> Dict[str, float]:
        output_denoised_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()

        noisy_mses, denoised_mses = [], []
        noisy_psnrs, denoised_psnrs = [], []
        noisy_ssims, denoised_ssims = [], []
        noisy_charbs, denoised_charbs = [], []
        noisy_laps, denoised_laps = [], []
        noisy_ffls, denoised_ffls = [], []
        noisy_espis, denoised_espis = [], []
        noisy_edges, denoised_edges = [], []

        total_test_samples = len(test_loader.dataset)
        logger.info(f"Running inference over {total_test_samples} test samples...")

        pbar = tqdm(
            test_loader,
            desc="Testing",
            bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        with torch.no_grad():
            for noisy_b, clean_b, fname_tuple in pbar:
                fname = fname_tuple[0]
                noisy_b, clean_b = noisy_b.to(self.device), clean_b.to(self.device)

                if hasattr(self.model, "ar_decoder"):
                    pred_b = self._denoise_tiled(noisy_b)
                elif self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        pred_b = self.model(noisy_b)
                else:
                    pred_b = self.model(noisy_b)

                pred_b = torch.clamp(pred_b.float(), 0.0, 1.0)

                clean_np = clean_b.squeeze().cpu().numpy()
                noisy_np = noisy_b.squeeze().cpu().numpy()
                denoised_np = pred_b.squeeze().cpu().numpy()
                denoised_uint8 = np.clip(denoised_np * 255.0, 0, 255).astype(np.uint8)

                # Standard spatial metrics
                noisy_mses.append(EvaluationMetrics.compute_mse(noisy_np, clean_np))
                denoised_mses.append(
                    EvaluationMetrics.compute_mse(denoised_np, clean_np)
                )

                noisy_psnrs.append(EvaluationMetrics.compute_psnr(noisy_np, clean_np))
                denoised_psnrs.append(
                    EvaluationMetrics.compute_psnr(denoised_np, clean_np)
                )

                noisy_ssims.append(EvaluationMetrics.compute_ssim(noisy_np, clean_np))
                denoised_ssims.append(
                    EvaluationMetrics.compute_ssim(denoised_np, clean_np)
                )

                # Specialized loss metrics
                noisy_charbs.append(float(self.charbonnier_fn(noisy_b, clean_b).item()))
                denoised_charbs.append(
                    float(self.charbonnier_fn(pred_b, clean_b).item())
                )

                noisy_laps.append(float(self.laplacian_fn(noisy_b, clean_b).item()))
                denoised_laps.append(float(self.laplacian_fn(pred_b, clean_b).item()))

                noisy_ffls.append(float(self.ffl_fn(noisy_b, clean_b).item()))
                denoised_ffls.append(float(self.ffl_fn(pred_b, clean_b).item()))

                noisy_espis.append(float(self.espi_fn(noisy_b, clean_b).item()))
                denoised_espis.append(float(self.espi_fn(pred_b, clean_b).item()))

                noisy_edges.append(float(self.sobel_fn(noisy_b, clean_b).item()))
                denoised_edges.append(float(self.sobel_fn(pred_b, clean_b).item()))

                out_fname = (
                    fname
                    if (fname.endswith(".tif") or fname.endswith(".tiff"))
                    else f"{fname}.tif"
                )
                tifffile.imwrite(output_denoised_dir / out_fname, denoised_uint8)

        mean_noisy_mse = float(np.mean(noisy_mses))
        mean_denoised_mse = float(np.mean(denoised_mses))
        est_clean_mse, est_clean_psnr = compute_noise_corrected_metrics(
            mean_denoised_mse, mean_noisy_mse
        )
        results = {
            "noisy_mse": mean_noisy_mse,
            "denoised_mse": mean_denoised_mse,
            "est_clean_mse": est_clean_mse,
            "est_clean_psnr": est_clean_psnr,
            "noisy_psnr": float(np.mean(noisy_psnrs)),
            "denoised_psnr": float(np.mean(denoised_psnrs)),
            "noisy_ssim": float(np.mean(noisy_ssims)),
            "denoised_ssim": float(np.mean(denoised_ssims)),
            "noisy_charbonnier": float(np.mean(noisy_charbs)),
            "denoised_charbonnier": float(np.mean(denoised_charbs)),
            "noisy_laplacian": float(np.mean(noisy_laps)),
            "denoised_laplacian": float(np.mean(denoised_laps)),
            "noisy_ffl": float(np.mean(noisy_ffls)),
            "denoised_ffl": float(np.mean(denoised_ffls)),
            "noisy_espi": float(np.mean(noisy_espis)),
            "denoised_espi": float(np.mean(denoised_espis)),
            "noisy_edge": float(np.mean(noisy_edges)),
            "denoised_edge": float(np.mean(denoised_edges)),
        }
        return results


# -------------------------------------------------------------------------
# Checkpoint Preview Helper
# -------------------------------------------------------------------------


def _load_preview_crop(
    image_path: Path,
    crop_size: int = 512,
    crop_y: Optional[int] = None,
    crop_x: Optional[int] = None,
) -> np.ndarray:
    """Load a fixed 2D float32 [0, 1] crop from a raw TIFF for checkpoint previews."""
    arr = tifffile.imread(str(image_path))
    if arr.ndim > 2:
        arr = arr[0]
    h, w = arr.shape
    cs = min(crop_size, h, w)
    y0 = (h - cs) // 2 if crop_y is None else min(max(0, crop_y), h - cs)
    x0 = (w - cs) // 2 if crop_x is None else min(max(0, crop_x), w - cs)
    crop = arr[y0 : y0 + cs, x0 : x0 + cs]
    if crop.dtype == np.uint8:
        return crop.astype(np.float32) / 255.0
    crop_f = crop.astype(np.float32)
    if crop_f.max() > 1.0:
        crop_f = crop_f / 255.0
    return np.clip(crop_f, 0.0, 1.0)


def save_comparison_panel(
    raw_255: np.ndarray,
    denoised_255: np.ndarray,
    out_path: Path,
    denoised_title: str = "Denoised",
) -> Dict[str, float]:
    """Save a 3-panel PNG (Raw | Denoised | Residual) and return residual stats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_f = raw_255.astype(np.float32)
    den_f = denoised_255.astype(np.float32)
    residual = raw_f - den_f
    stats = _compute_residual_stats(raw_f, den_f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    axes[0].imshow(raw_f, cmap="gray", vmin=0.0, vmax=255.0)
    axes[0].set_title("Raw Input")
    axes[0].axis("off")

    axes[1].imshow(den_f, cmap="gray", vmin=0.0, vmax=255.0)
    axes[1].set_title(denoised_title)
    axes[1].axis("off")

    im_res = axes[2].imshow(residual, cmap="RdBu_r", vmin=-60.0, vmax=60.0)
    axes[2].set_title(
        f"Residual (Raw - Denoised)\n"
        f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
        f"ac_x={stats['ac_x']:.3f}, ac_y={stats['ac_y']:.3f}"
    )
    axes[2].axis("off")
    fig.colorbar(im_res, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)
    return stats


def _save_checkpoint_preview(
    model: nn.Module,
    raw_crop: np.ndarray,
    previews_dir: Path,
    epoch: int,
    device: torch.device,
    writer: Optional[Any] = None,
) -> Dict[str, float]:
    """Denoise a fixed raw crop and save TIFF + 3-panel PNG + TensorBoard images."""
    previews_dir.mkdir(parents=True, exist_ok=True)
    if epoch == 1:
        raw_u8 = np.clip(raw_crop * 255.0, 0, 255).astype(np.uint8)
        tifffile.imwrite(str(previews_dir / "raw_crop.tif"), raw_u8)
        if writer is not None:
            writer.add_image("Preview/Raw", raw_crop[None, ...], global_step=0)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(raw_crop).unsqueeze(0).unsqueeze(0).to(device)
        if hasattr(model, "ar_decoder"):
            pred = model.denoise(inp)
        else:
            pred = model(inp)
        denoised = torch.clamp(pred, 0.0, 1.0)[0, 0].cpu().numpy()
    if was_training:
        model.train()

    denoised_u8 = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    tifffile.imwrite(str(previews_dir / f"epoch_{epoch:03d}.tif"), denoised_u8)

    stats = save_comparison_panel(
        raw_255=raw_crop * 255.0,
        denoised_255=denoised * 255.0,
        out_path=previews_dir / f"epoch_{epoch:03d}_comparison.png",
        denoised_title=f"Denoised (Epoch {epoch:02d})",
    )

    if writer is not None:
        writer.add_image("Preview/Denoised", denoised[None, ...], global_step=epoch)
        writer.add_scalar("Preview/Residual_Std", stats["std"], epoch)
        writer.add_scalar("Preview/Residual_AC_X", stats["ac_x"], epoch)
        writer.add_scalar("Preview/Residual_AC_Y", stats["ac_y"], epoch)
    return stats


# -------------------------------------------------------------------------
# CLI & Execution Entry Point
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production Connectomics Neural Network Training and Evaluation Engine."
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
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epoch count.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "-m",
        "--model",
        "--model-type",
        type=str,
        default=None,
        choices=["connectomics", "residual_unet", "modern_unet", "cosdd", "n2v"],
        help="Override neural network model architecture.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Override hardware execution device.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip model training and run evaluation/inference directly using existing trained checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint file for evaluation (default: checkpoint_best.pt in checkpoints_dir).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume training from (restores model, optimizer, scheduler, and epoch).",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=None,
        help="Path to custom directory of .tif images to run inference on.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default=None,
        help="Path to legacy/custom dataset directory containing train/val/test folders with clean patches.",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Override experiment run name (results subdirectory inside output_dir).",
    )
    return parser.parse_args()


def _build_n2n_overlap_dataset(
    cfg: PipelineConfig,
    split: str,
    max_pairs: Optional[int],
    augment: bool,
) -> N2NOverlapPairDataset:
    """Build an N2NOverlapPairDataset from PipelineConfig for a given split."""
    return N2NOverlapPairDataset(
        overlaps_dir=Path(cfg.n2n.overlaps_dir),
        split=split,
        patch_size=cfg.extraction.patch_size,
        first_sec=cfg.data.first_sec,
        last_sec=cfg.data.last_sec,
        split_strategy=cfg.splitting.strategy,
        block_size=cfg.splitting.block_size,
        train_ratio=cfg.splitting.train_ratio,
        val_ratio=cfg.splitting.val_ratio,
        guard_band_size=cfg.splitting.guard_band_size,
        orientation=cfg.n2n.orientation,
        max_pairs=max_pairs,
        pair_direction=cfg.n2n.pair_direction,
        cache_in_ram=cfg.training.cache_in_ram,
        augment=augment,
        augment_rot90=cfg.n2n.augment_rot90 if augment else False,
        seed=cfg.n2n.seed,
        align_patches=cfg.n2n.align_patches,
        align_subpixel=getattr(cfg.n2n, "align_subpixel", False),
        align_max_shift_px=cfg.n2n.align_max_shift_px,
        align_blur_sigma=cfg.n2n.align_blur_sigma,
        patch_sampling=cfg.n2n.patch_sampling,
        tile_stride=cfg.n2n.tile_stride,
        trim_enabled=cfg.n2n.trim_enabled,
        v_trim_top_px=cfg.n2n.v_trim_top_px,
        v_trim_left_px=cfg.n2n.v_trim_left_px,
        v_trim_bottom_px=cfg.n2n.v_trim_bottom_px,
        v_trim_right_px=cfg.n2n.v_trim_right_px,
        h_trim_left_px=cfg.n2n.h_trim_left_px,
        h_trim_top_px=cfg.n2n.h_trim_top_px,
        h_trim_bottom_px=cfg.n2n.h_trim_bottom_px,
        h_trim_right_px=cfg.n2n.h_trim_right_px,
        quality_filter_enabled=cfg.n2n.quality_filter_enabled,
        quality_metric=cfg.n2n.quality_metric,
        min_pair_quality=cfg.n2n.min_pair_quality,
        quality_blur_sigma=cfg.n2n.quality_blur_sigma,
        min_patch_std=getattr(cfg.n2n, "min_patch_std", 0.0),
    )


def _build_n2n_duplicate_dataset(
    cfg: PipelineConfig,
    split: str,
    max_pairs: Optional[int],
    augment: bool,
) -> N2NDuplicateTileDataset:
    """Build an N2NDuplicateTileDataset from PipelineConfig for a given split."""
    return N2NDuplicateTileDataset(
        duplicates_dir=Path(cfg.n2n.duplicates_dir),
        split=split,
        patch_size=cfg.extraction.patch_size,
        train_ratio=cfg.splitting.train_ratio,
        val_ratio=cfg.splitting.val_ratio,
        test_ratio=cfg.splitting.test_ratio,
        pair_mode=getattr(cfg.n2n, "pair_mode", "all"),
        split_unit=getattr(cfg.n2n, "split_unit", "group"),
        split_seed=getattr(cfg.n2n, "split_seed", 42),
        pair_direction=cfg.n2n.pair_direction,
        max_pairs=max_pairs,
        cache_in_ram=cfg.training.cache_in_ram,
        max_cache_tiles=getattr(cfg.n2n, "max_cache_tiles", 32),
        augment=augment,
        augment_rot90=cfg.n2n.augment_rot90,
        seed=cfg.n2n.seed,
        align_patches=cfg.n2n.align_patches,
        align_subpixel=getattr(cfg.n2n, "align_subpixel", False),
        align_max_shift_px=cfg.n2n.align_max_shift_px,
        align_blur_sigma=cfg.n2n.align_blur_sigma,
        patch_sampling=cfg.n2n.patch_sampling,
        tile_stride=cfg.n2n.tile_stride,
        trim_enabled=cfg.n2n.trim_enabled,
        quality_filter_enabled=cfg.n2n.quality_filter_enabled,
        quality_metric=cfg.n2n.quality_metric,
        min_pair_quality=cfg.n2n.min_pair_quality,
        quality_blur_sigma=cfg.n2n.quality_blur_sigma,
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    clean_config_path = str(args.config).lstrip("=")
    cfg = PipelineConfig.from_yaml(clean_config_path)
    loaded_path = getattr(cfg, "_loaded_from", clean_config_path)
    logger.info(f"Loaded pipeline configuration from: {loaded_path}")

    seed = getattr(cfg.training, "seed", None)
    if seed is not None:
        _set_global_seed(int(seed))
        logger.info(f"Global RNG seed set to: {seed}")

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(cfg.experiment.output_dir)
    )
    exp_name = (
        args.name
        if args.name is not None
        else (cfg.experiment.name if cfg.experiment and cfg.experiment.name else None)
    )

    epochs = args.epochs if args.epochs is not None else cfg.training.epochs
    batch_size = (
        args.batch_size if args.batch_size is not None else cfg.training.batch_size
    )
    learning_rate = args.lr if args.lr is not None else cfg.training.learning_rate
    device_str = args.device if args.device is not None else cfg.training.device

    # Resolve model_type early so N2V mode can influence dataset construction below
    model_type = (
        args.model
        if args.model
        else (
            cfg.model.model_type
            if hasattr(cfg, "model") and cfg.model
            else "connectomics"
        )
    )
    is_n2v_mode = model_type.lower() in ("n2v", "noise2void", "n2v2")
    is_n2n_mode = hasattr(cfg, "n2n") and bool(getattr(cfg.n2n, "enabled", False))
    n2n_source = (
        getattr(cfg.n2n, "source", "overlaps").lower().strip()
        if is_n2n_mode
        else "overlaps"
    )
    if is_n2n_mode and n2n_source not in ("overlaps", "duplicates"):
        raise ValueError(
            f"n2n.source must be 'overlaps' or 'duplicates', got '{n2n_source}'"
        )
    if is_n2n_mode and is_n2v_mode:
        raise ValueError(
            "n2n.enabled=True cannot be combined with model_type='n2v'. "
            "Use a non-blind-spot backbone such as 'connectomics', 'residual_unet', or 'modern_unet'."
        )

    run_dir = output_dir / exp_name if exp_name else output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = run_dir / "train_and_infer.log"
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging training and validation outputs to file: {log_file_path}")

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    lmdb_store_path = output_dir / "patch_store.lmdb"

    if is_n2v_mode:
        # N2V is self-supervised: training uses raw noisy patches directly.
        # The noise synthesizer must NOT be applied — LMDB patches ARE the noisy inputs.
        noise_injector = None
        logger.info(
            "N2V self-supervised mode: noise synthesizer BYPASSED. "
            "LMDB patches will be used as raw noisy inputs."
        )
    elif is_n2n_mode and n2n_source == "duplicates":
        noise_injector = None
        logger.info(
            "N2N duplicate-tile mode: noise synthesizer BYPASSED. "
            f"Reading paired duplicate tiles from: {cfg.n2n.duplicates_dir}"
        )
    elif is_n2n_mode:
        # N2N uses real paired noisy overlap crops from sbem-n2n-overlaps.
        noise_injector = None
        logger.info(
            "N2N overlap-pair mode: noise synthesizer BYPASSED. "
            f"Reading paired noisy overlaps from: {cfg.n2n.overlaps_dir}"
        )
    elif getattr(cfg.noise, "synthesizer_type", "parametric") == "empirical":
        model_path = Path(getattr(cfg.noise, "empirical_model_path", "noise_model.npy"))
        if not model_path.is_absolute():
            model_path = output_dir / model_path

        if (
            getattr(cfg.noise, "calibrate_on_the_fly", False)
            and not model_path.exists()
        ):
            logger.info(
                "Empirical noise model not found. Running calibration on-the-fly..."
            )
            run_calibration(cfg)

        noise_injector = EmpiricalNoiseSynthesizer(
            model_path=model_path,
            noise_amplification=getattr(cfg.noise, "noise_amplification", 1.0),
        )
    else:
        noise_injector = SBFSEMNoiseSynthesizer(
            photon_scale=cfg.noise.photon_scale,
            read_std=cfg.noise.read_std,
            gain_variance=cfg.noise.gain_variance,
            streak_probability=cfg.noise.streak_probability,
            inverted_bse=getattr(cfg.noise, "inverted_bse", True),
        )

    data_dir = Path(args.data_dir) if args.data_dir else None

    if not args.eval_only:
        logger.info("Initializing Train & Validation Datasets...")
        if is_n2n_mode and n2n_source == "duplicates":
            train_dataset = _build_n2n_duplicate_dataset(
                cfg=cfg,
                split="train",
                max_pairs=cfg.n2n.max_train_pairs,
                augment=True,
            )
            val_dataset = _build_n2n_duplicate_dataset(
                cfg=cfg,
                split="val",
                max_pairs=cfg.n2n.max_val_pairs,
                augment=False,
            )
            split_summary_path = run_dir / "duplicate_splits.json"
            train_dataset.save_split_summary(split_summary_path)
            logger.info(
                f"Saved duplicate-tile split table (seed={cfg.n2n.split_seed}) to: {split_summary_path}"
            )
        elif is_n2n_mode and getattr(cfg.n2n, "streaming", False):
            overlaps_dir = Path(cfg.n2n.overlaps_dir)
            train_dataset = N2NStreamingShardDataset(
                overlaps_dir=overlaps_dir,
                split="train",
                patch_size=cfg.extraction.patch_size,
                first_sec=cfg.data.first_sec,
                last_sec=cfg.data.last_sec,
                split_strategy=cfg.splitting.strategy,
                block_size=cfg.splitting.block_size,
                train_ratio=cfg.splitting.train_ratio,
                val_ratio=cfg.splitting.val_ratio,
                guard_band_size=cfg.splitting.guard_band_size,
                orientation=cfg.n2n.orientation,
                max_pairs=cfg.n2n.max_train_pairs,
                pair_direction=cfg.n2n.pair_direction,
                augment=True,
                augment_rot90=cfg.n2n.augment_rot90,
                seed=cfg.n2n.seed,
                shuffle_buffer_size=getattr(cfg.n2n, "shuffle_buffer_size", 2048),
            )
            val_dataset = N2NStreamingShardDataset(
                overlaps_dir=overlaps_dir,
                split="val",
                patch_size=cfg.extraction.patch_size,
                first_sec=cfg.data.first_sec,
                last_sec=cfg.data.last_sec,
                split_strategy=cfg.splitting.strategy,
                block_size=cfg.splitting.block_size,
                train_ratio=cfg.splitting.train_ratio,
                val_ratio=cfg.splitting.val_ratio,
                guard_band_size=cfg.splitting.guard_band_size,
                orientation=cfg.n2n.orientation,
                max_pairs=cfg.n2n.max_val_pairs,
                pair_direction=cfg.n2n.pair_direction,
                augment=False,
                seed=cfg.n2n.seed,
                shuffle_buffer_size=0,
            )
        elif is_n2n_mode:
            train_dataset = _build_n2n_overlap_dataset(
                cfg=cfg,
                split="train",
                max_pairs=cfg.n2n.max_train_pairs,
                augment=True,
            )
            val_dataset = _build_n2n_overlap_dataset(
                cfg=cfg,
                split="val",
                max_pairs=cfg.n2n.max_val_pairs,
                augment=False,
            )
        elif data_dir is not None:
            logger.info(f"Loading custom dataset from directory: {data_dir}")
            root_dir = (
                data_dir.parent
                if data_dir.name in ["train", "val", "test", "clean"]
                else data_dir
            )
            if root_dir.name in ["clean", "noisy"]:
                root_dir = root_dir.parent.parent

            train_clean = (
                root_dir / "train" / "clean"
                if (root_dir / "train" / "clean").exists()
                else (root_dir / "clean" if (root_dir / "clean").exists() else root_dir)
            )
            val_clean = (
                root_dir / "val" / "clean"
                if (root_dir / "val" / "clean").exists()
                else (root_dir / "clean" if (root_dir / "clean").exists() else root_dir)
            )
            train_noisy = (
                root_dir / "train" / "noisy"
                if (root_dir / "train" / "noisy").exists()
                else None
            )
            val_noisy = (
                root_dir / "val" / "noisy"
                if (root_dir / "val" / "noisy").exists()
                else None
            )

            train_dataset = PatchFolderDataset(
                train_clean,
                noisy_dir=train_noisy if noise_injector is None else None,
                noise_injector=noise_injector,
                cache_in_ram=cfg.training.cache_in_ram,
                augment=True,
            )
            val_dataset = PatchFolderDataset(
                val_clean,
                noisy_dir=val_noisy if noise_injector is None else None,
                noise_injector=noise_injector,
                cache_in_ram=cfg.training.cache_in_ram,
                augment=False,
            )
        elif lmdb_store_path.exists():
            logger.info(f"Found LMDB store at: {lmdb_store_path}")
            train_dataset = LMDBPatchDataset(
                lmdb_store_path,
                split="train",
                noise_injector=noise_injector,
                cache_in_ram=cfg.training.cache_in_ram,
                augment=True,
            )
            val_dataset = LMDBPatchDataset(
                lmdb_store_path,
                split="val",
                noise_injector=noise_injector,
                cache_in_ram=cfg.training.cache_in_ram,
                augment=False,
            )
        else:
            logger.info(
                f"LMDB store not found. Reading TIFF patch directories under: {output_dir}"
            )
            train_dataset = PatchFolderDataset(
                output_dir / "train" / "clean",
                noisy_dir=output_dir / "train" / "noisy"
                if noise_injector is None
                else None,
                noise_injector=noise_injector,
                cache_in_ram=cfg.training.cache_in_ram,
                augment=True,
            )
            val_dataset = PatchFolderDataset(
                output_dir / "val" / "clean",
                noisy_dir=output_dir / "val" / "noisy"
                if noise_injector is None
                else None,
                noise_injector=noise_injector,
                cache_in_ram=cfg.training.cache_in_ram,
                augment=False,
            )

        num_workers = int(getattr(cfg.training, "num_workers", 0))
        use_pin_memory = device_str == "cuda"
        is_iterable = isinstance(train_dataset, IterableDataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, len(train_dataset)),
            shuffle=(not is_iterable),
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(num_workers > 0),
        )

        if is_n2v_mode:
            # Wrap both splits with N2V blind-spot masking.
            # The inner datasets already have noise_injector=None so noisy == clean == raw EM.
            n2v_mask_ratio = (
                getattr(cfg.n2v, "mask_ratio", 0.015) if hasattr(cfg, "n2v") else 0.015
            )
            n2v_mask_mode = (
                getattr(cfg.n2v, "mask_mode", "random")
                if hasattr(cfg, "n2v")
                else "random"
            )
            train_loader = DataLoader(
                N2VDatasetWrapper(
                    train_dataset, mask_ratio=n2v_mask_ratio, mask_mode=n2v_mask_mode
                ),
                batch_size=min(batch_size, len(train_dataset)),
                shuffle=True,
            )
            val_loader = DataLoader(
                N2VDatasetWrapper(
                    val_dataset, mask_ratio=n2v_mask_ratio, mask_mode=n2v_mask_mode
                ),
                batch_size=min(batch_size, len(val_dataset)),
                shuffle=False,
            )
            logger.info(
                f"N2V masking: mode='{n2v_mask_mode}', ratio={n2v_mask_ratio:.4f}. "
                f"Datasets wrapped with N2VDatasetWrapper."
            )

        logger.info(
            f"Dataset Size: {len(train_dataset)} Train patches | {len(val_dataset)} Val patches"
        )

    if device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using target device: {device}")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # model_type resolved early (before dataset construction) as `model_type` and `is_n2v_mode`
    base_dim = cfg.model.base_dim if hasattr(cfg, "model") and cfg.model else 32
    cosdd_opts = {}
    if hasattr(cfg, "cosdd") and cfg.cosdd:
        cosdd_opts = {
            "s_code_channels": getattr(cfg.cosdd, "s_code_channels", 64),
            "number_layers": getattr(cfg.cosdd, "number_layers", 14),
            "number_components": getattr(cfg.cosdd, "number_components", 3),
            "scale_initialisation": getattr(cfg.cosdd, "scale_initialisation", True),
            "noise_direction": getattr(cfg.cosdd, "noise_direction", "x"),
            "use_direct_denoiser": getattr(cfg.cosdd, "use_direct_denoiser", False),
        }
    n2v_opts = {}
    if hasattr(cfg, "n2v") and cfg.n2v:
        n2v_opts = {
            "blind_spot_strict": getattr(cfg.n2v, "blind_spot_strict", True),
        }
    model = build_model(
        model_type=model_type,
        in_channels=1,
        out_channels=1,
        base_dim=base_dim,
        **cosdd_opts,
        **n2v_opts,
    ).to(device)
    logger.info(
        f"Initialized neural network architecture: '{model_type}' (base_dim={base_dim})"
    )
    if (
        device.type == "cuda"
        and getattr(cfg.training, "compile_model", False)
        and hasattr(torch, "compile")
    ):
        model = torch.compile(model)
        logger.info("Compiled model with torch.compile() for CUDA acceleration.")
    loss_type = getattr(cfg.loss, "loss_type", "composite").lower()
    if loss_type == "dynamic":
        criterion = DynamicEMLoss(
            channels=1, inverted_bse=getattr(cfg.noise, "inverted_bse", True)
        ).to(device)
        logger.info(
            "Initialized DynamicEMLoss (learnable homoscedastic uncertainty weighting)"
        )
    elif loss_type == "enhanced":
        criterion = EnhancedCompositeLoss(
            charbonnier_weight=getattr(cfg.loss, "charbonnier_weight", 0.3),
            ssim_weight=getattr(cfg.loss, "ssim_weight", 0.3),
            laplacian_weight=getattr(cfg.loss, "laplacian_weight", 0.15),
            ffl_weight=getattr(cfg.loss, "ffl_weight", 0.15),
            espi_weight=getattr(cfg.loss, "espi_weight", 0.1),
            edge_weight=getattr(cfg.loss, "edge_weight", 0.0),
        ).to(device)
        logger.info(
            f"Initialized EnhancedCompositeLoss (chrb={criterion.charbonnier_weight}, "
            f"ssim={criterion.ssim_weight}, lap={criterion.laplacian_weight}, "
            f"ffl={criterion.ffl_weight}, espi={criterion.espi_weight}, edge={criterion.edge_weight})"
        )
    else:
        criterion = CompositeDenoisingLoss(
            mse_weight=cfg.loss.mse_weight if cfg.loss else 1.0,
            edge_weight=cfg.loss.edge_weight if cfg.loss else 0.2,
            ssim_weight=cfg.loss.ssim_weight if cfg.loss else 0.1,
        ).to(device)
        logger.info("Initialized CompositeDenoisingLoss (MSE + Sobel + SSIM)")

    trainable_params = list(model.parameters())
    if hasattr(criterion, "parameters"):
        crit_params = list(criterion.parameters())
        if crit_params:
            trainable_params += crit_params

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    lr_scheduler_name = getattr(cfg.training, "lr_scheduler", "none")
    lr_min = float(getattr(cfg.training, "lr_min", 1e-6))
    warmup_epochs = int(getattr(cfg.training, "warmup_epochs", 0))
    scheduler = _build_lr_scheduler(
        optimizer=optimizer,
        epochs=epochs,
        lr_scheduler=lr_scheduler_name,
        lr_min=lr_min,
        warmup_epochs=warmup_epochs,
    )
    if scheduler is not None:
        logger.info(
            f"Initialized LR scheduler: '{lr_scheduler_name}' "
            f"(lr_min={lr_min}, warmup_epochs={warmup_epochs})"
        )
    serialized_cfg = _serialize_config(cfg)

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard logging initialized at: {log_dir}")
    except Exception:
        logger.warning("TensorBoard unavailable. Continuing with standard logging.")

    best_checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else checkpoints_dir / "checkpoint_best.pt"
    )

    if not args.eval_only:
        trainer = DenoisingTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoints_dir=checkpoints_dir,
            use_amp=use_amp,
            writer=writer,
        )

        preview_crop: Optional[np.ndarray] = None
        previews_dir = run_dir / "previews"
        if (
            hasattr(cfg, "inference")
            and cfg.inference
            and getattr(cfg.inference, "preview_image", None) is not None
        ):
            preview_path = Path(cfg.inference.preview_image)
            if preview_path.exists():
                preview_crop = _load_preview_crop(
                    preview_path,
                    crop_size=getattr(cfg.inference, "preview_crop_size", 512),
                    crop_y=getattr(cfg.inference, "preview_crop_y", None),
                    crop_x=getattr(cfg.inference, "preview_crop_x", None),
                )
                logger.info(
                    f"Checkpoint preview enabled on {preview_crop.shape} crop from: {preview_path}"
                )
            else:
                logger.warning(
                    f"preview_image not found, skipping previews: {preview_path}"
                )

        logger.info(f"Starting Network Training ({epochs} epochs)...")
        start_time = time.time()
        start_epoch = 1
        best_val_loss = float("inf")

        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"Resume checkpoint not found at: {resume_path}"
                )
            resume_ckpt = torch.load(
                resume_path, map_location=device, weights_only=False
            )
            getattr(model, "_orig_mod", model).load_state_dict(
                resume_ckpt["model_state_dict"]
            )
            if (
                "optimizer_state_dict" in resume_ckpt
                and resume_ckpt["optimizer_state_dict"] is not None
            ):
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            if (
                scheduler is not None
                and resume_ckpt.get("scheduler_state_dict") is not None
            ):
                scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
            best_val_loss = float(resume_ckpt.get("val_loss", float("inf")))
            existing_best = checkpoints_dir / "checkpoint_best.pt"
            if existing_best.exists():
                try:
                    best_meta = torch.load(
                        existing_best, map_location="cpu", weights_only=False
                    )
                    best_val_loss = min(
                        best_val_loss, float(best_meta.get("val_loss", float("inf")))
                    )
                except Exception:
                    pass
            logger.info(
                f"Resuming training from {resume_path} at epoch {start_epoch}/{epochs} "
                f"(best_val_loss={best_val_loss:.6f})"
            )
            if start_epoch > epochs:
                logger.info(
                    f"Checkpoint epoch ({start_epoch - 1}) >= target epochs ({epochs}); skipping training loop."
                )

        for epoch in range(start_epoch, epochs + 1):
            cur_lr = float(optimizer.param_groups[0]["lr"])
            train_loss, train_metrics = trainer.train_epoch(train_loader, epoch, epochs)
            val_loss, val_psnr, val_ssim, val_metrics = trainer.validate(
                val_loader, epoch, epochs
            )
            if scheduler is not None:
                scheduler.step()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            best_ckpt = trainer.save_checkpoint(
                epoch,
                val_loss,
                val_psnr,
                val_ssim,
                is_best=is_best,
                scheduler=scheduler,
                config=serialized_cfg,
                save_all_epochs=getattr(cfg.training, "save_all_epochs", True),
            )
            if is_best:
                best_checkpoint_path = best_ckpt

            if preview_crop is not None:
                _save_checkpoint_preview(
                    model=model,
                    raw_crop=preview_crop,
                    previews_dir=previews_dir,
                    epoch=epoch,
                    device=device,
                    writer=writer,
                )

            if writer is not None:
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Validation", val_loss, epoch)
                writer.add_scalar("Metrics/Val_PSNR", val_psnr, epoch)
                writer.add_scalar("Metrics/Val_SSIM", val_ssim, epoch)
                writer.add_scalar("Train/LR", cur_lr, epoch)
                if is_n2n_mode and "est_clean_psnr" in val_metrics:
                    writer.add_scalar(
                        "Metrics/Val_EstClean_PSNR",
                        val_metrics["est_clean_psnr"],
                        epoch,
                    )
                    writer.add_scalar(
                        "Metrics/Val_EstClean_MSE",
                        val_metrics["est_clean_mse"],
                        epoch,
                    )
                for k, v in train_metrics.items():
                    writer.add_scalar(f"Train_Metrics/{k}", v, epoch)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"Val_Metrics/{k}", v, epoch)

            best_str = " [BEST]" if is_best else ""
            est_str = (
                f" | EstPSNR: {val_metrics['est_clean_psnr']:.2f}dB"
                if is_n2n_mode and "est_clean_psnr" in val_metrics
                else ""
            )
            lr_str = f" | LR: {cur_lr:.2e}"
            if loss_type == "enhanced":
                chrb = train_metrics.get("loss_charbonnier", 0.0)
                lap = train_metrics.get("loss_laplacian", 0.0)
                ffl = train_metrics.get("loss_ffl", 0.0)
                espi = train_metrics.get("loss_espi", 0.0)
                logger.info(
                    f"Ep [{epoch:02d}/{epochs:02d}] | Tr: {train_loss:.5f} | Val: {val_loss:.5f} | "
                    f"Chrb: {chrb:.4f} | Lap: {lap:.4f} | FFL: {ffl:.4f} | ESPI: {espi:.4f} | "
                    f"PSNR: {val_psnr:.2f}dB{est_str} | SSIM: {val_ssim:.4f}{lr_str}{best_str}"
                )
            elif loss_type == "dynamic":
                chrb = train_metrics.get("loss_charbonnier", 0.0)
                ffl = train_metrics.get("loss_ffl", 0.0)
                vst = train_metrics.get("loss_vst", 0.0)
                lap = train_metrics.get("loss_laplacian", 0.0)
                logger.info(
                    f"Ep [{epoch:02d}/{epochs:02d}] | Tr: {train_loss:.5f} | Val: {val_loss:.5f} | "
                    f"Chrb: {chrb:.4f} | FFL: {ffl:.4f} | VST: {vst:.4f} | Lap: {lap:.4f} | "
                    f"PSNR: {val_psnr:.2f}dB{est_str} | SSIM: {val_ssim:.4f}{lr_str}{best_str}"
                )
            else:
                edge = train_metrics.get("loss_edge", 0.0)
                logger.info(
                    f"Ep [{epoch:02d}/{epochs:02d}] | Tr: {train_loss:.5f} | Val: {val_loss:.5f} | "
                    f"Edge: {edge:.4f} | PSNR: {val_psnr:.2f}dB{est_str} | SSIM: {val_ssim:.4f}{lr_str}{best_str}"
                )

            file_handler.flush()

        logger.info(
            f"✓ Training completed in {time.time() - start_time:.2f}s. Best Val Loss: {best_val_loss:.6f}"
        )
        if writer is not None:
            writer.close()
    else:
        logger.info("Bypassing model training as requested (--eval-only).")

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {best_checkpoint_path}")

    logger.info(f"Loading checkpoint for testing: {best_checkpoint_path.name}")
    checkpoint = torch.load(
        best_checkpoint_path, map_location=device, weights_only=False
    )
    getattr(model, "_orig_mod", model).load_state_dict(checkpoint["model_state_dict"])

    custom_input_dir = (
        Path(args.input_dir)
        if args.input_dir
        else (
            cfg.inference.custom_input_dir
            if cfg.inference and cfg.inference.custom_input_dir
            else None
        )
    )

    test_clean_dir = output_dir / "test" / "clean"
    test_noisy_dir = output_dir / "test" / "noisy"

    if custom_input_dir is not None:
        logger.info(
            f"Loading custom inference images from directory: {custom_input_dir}"
        )
        test_dataset = CustomTIFFDataset(custom_input_dir, cache_in_ram=True)
        test_denoised_dir = run_dir / "custom_inference" / "denoised"
    elif is_n2n_mode and n2n_source == "duplicates":
        logger.info(f"Loading N2N duplicate test pairs from: {cfg.n2n.duplicates_dir}")
        test_denoised_dir = run_dir / "test" / "denoised"
        test_dataset = _build_n2n_duplicate_dataset(
            cfg=cfg,
            split="test",
            max_pairs=cfg.n2n.max_test_pairs,
            augment=False,
        )
    elif is_n2n_mode and getattr(cfg.n2n, "streaming", False):
        overlaps_dir = Path(cfg.n2n.overlaps_dir)
        logger.info(
            f"Loading N2N streaming test pairs from overlaps store: {overlaps_dir}"
        )
        test_denoised_dir = run_dir / "test" / "denoised"
        test_dataset = N2NStreamingShardDataset(
            overlaps_dir=overlaps_dir,
            split="test",
            patch_size=cfg.extraction.patch_size,
            first_sec=cfg.data.first_sec,
            last_sec=cfg.data.last_sec,
            split_strategy=cfg.splitting.strategy,
            block_size=cfg.splitting.block_size,
            train_ratio=cfg.splitting.train_ratio,
            val_ratio=cfg.splitting.val_ratio,
            guard_band_size=cfg.splitting.guard_band_size,
            orientation=cfg.n2n.orientation,
            max_pairs=cfg.n2n.max_test_pairs,
            pair_direction=cfg.n2n.pair_direction,
            augment=False,
            seed=cfg.n2n.seed,
            shuffle_buffer_size=0,
        )
    elif is_n2n_mode:
        overlaps_dir = Path(cfg.n2n.overlaps_dir)
        logger.info(f"Loading N2N test pairs from overlaps store: {overlaps_dir}")
        test_denoised_dir = run_dir / "test" / "denoised"
        test_dataset = _build_n2n_overlap_dataset(
            cfg=cfg,
            split="test",
            max_pairs=cfg.n2n.max_test_pairs,
            augment=False,
        )
    elif test_clean_dir.exists() and len(list(test_clean_dir.glob("*.tif"))) > 0:
        logger.info(f"Loading test set from TIFF directory: {test_clean_dir}")
        test_denoised_dir = run_dir / "test" / "denoised"
        test_dataset = PatchFolderDataset(
            test_clean_dir, test_noisy_dir, cache_in_ram=True, augment=False
        )
    elif lmdb_store_path.exists():
        logger.info(f"Loading test set from LMDB store: {lmdb_store_path}")
        test_denoised_dir = run_dir / "test" / "denoised"
        test_dataset = LMDBPatchDataset(
            lmdb_store_path,
            split="test",
            noise_injector=noise_injector,
            cache_in_ram=cfg.training.cache_in_ram,
            augment=False,
        )
    else:
        raise FileNotFoundError(
            f"No test set found in {test_clean_dir} or {lmdb_store_path}"
        )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    evaluator = DenoisingEvaluator(model=model, device=device, use_amp=use_amp)
    results = evaluator.evaluate_test_set(test_loader, test_denoised_dir)

    summary_lines = [
        "==================================================",
        f" INFERENCE & DENOISING PERFORMANCE SUMMARY (Loss Mode: {loss_type.upper()})",
        "==================================================",
        f" Model Architecture : {model_type}",
        f" Run Name           : {exp_name}",
        "--------------------------------------------------",
        f" Mean MSE         : Noisy = {results['noisy_mse']:.6f}  ==>  Denoised = {results['denoised_mse']:.6f} (Lower is better)",
        f" Mean PSNR        : Noisy = {results['noisy_psnr']:.2f} dB ==>  Denoised = {results['denoised_psnr']:.2f} dB (Higher is better)",
        f" Mean SSIM        : Noisy = {results['noisy_ssim']:.4f}    ==>  Denoised = {results['denoised_ssim']:.4f}    (Higher is better)",
    ]
    if is_n2n_mode and results.get("noisy_mse", 0.0) > 0.0:
        summary_lines.append(
            f" Est. Clean (N2N) : MSE = {results['est_clean_mse']:.6f} | PSNR = {results['est_clean_psnr']:.2f} dB"
        )
    if "noisy_charbonnier" in results:
        summary_lines.append(
            f" Mean Charbonnier : Noisy = {results['noisy_charbonnier']:.6f}  ==>  Denoised = {results['denoised_charbonnier']:.6f} (Lower is better)"
        )
    if "noisy_laplacian" in results:
        summary_lines.append(
            f" Mean Laplacian   : Noisy = {results['noisy_laplacian']:.6f}  ==>  Denoised = {results['denoised_laplacian']:.6f} (Lower is better)"
        )
    if "noisy_ffl" in results:
        summary_lines.append(
            f" Mean FFL         : Noisy = {results['noisy_ffl']:.6f}  ==>  Denoised = {results['denoised_ffl']:.6f} (Lower is better)"
        )
    if "noisy_espi" in results:
        summary_lines.append(
            f" Mean ESPI        : Noisy = {results['noisy_espi']:.6f}  ==>  Denoised = {results['denoised_espi']:.6f} (Lower is better)"
        )
    if "noisy_edge" in results and loss_type == "composite":
        summary_lines.append(
            f" Mean Sobel Edge  : Noisy = {results['noisy_edge']:.6f}  ==>  Denoised = {results['denoised_edge']:.6f} (Lower is better)"
        )
    summary_lines.append("==================================================")

    for line in summary_lines:
        logger.info(line)

    summary_txt = "\n".join(summary_lines) + "\n"

    # Save evaluation_summary.txt to run_dir and test_denoised_dir
    (run_dir / "evaluation_summary.txt").write_text(summary_txt, encoding="utf-8")
    (test_denoised_dir / "evaluation_summary.txt").write_text(
        summary_txt, encoding="utf-8"
    )

    # Save summary.json to run_dir
    summary_dict = {
        "experiment_name": exp_name,
        "model_type": model_type,
        "loss_type": loss_type,
        "metrics": results,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    logger.info(f"✓ Denoised test images saved to: {test_denoised_dir}")
    logger.info(
        f"✓ Performance summary saved to: {run_dir / 'evaluation_summary.txt'} and {run_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
