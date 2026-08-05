# src/em_denoise_refactor/execution/train_and_infer.py

"""
Production-grade training and evaluation pipeline for Connectomics Denoising neural networks.

Executes offline model training reading from high-speed LMDB patch stores (or disk TIFF patch folders),
monitors convergence via TensorBoard, purges Apple Silicon MPS memory buffers, saves model checkpoints,
and evaluates performance metrics (MSE, PSNR, SSIM) on test sets.
"""

import argparse
import gc
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from em_denoise_refactor.config import PipelineConfig
from em_denoise_refactor.data.dataset import (
    SBFSEMNoiseSynthesizer,
    EmpiricalNoiseSynthesizer,
)
from em_denoise_refactor.execution.calibration import run_calibration
from em_denoise_refactor.models.connectomics import ConnectomicsDenoisingNet
from em_denoise_refactor.models.loss import CompositeDenoisingLoss

logger = logging.getLogger("train_and_infer")


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
    """Dataset reading clean/noisy patch pairs directly from TIFF directories."""

    def __init__(
        self,
        clean_dir: Path | str,
        noisy_dir: Path | str,
        cache_in_ram: bool = True,
        augment: bool = False,
    ) -> None:
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.cache_in_ram = cache_in_ram
        self.augment = augment

        self.filenames = sorted([f.name for f in self.clean_dir.glob("*.tif")])
        if not self.filenames:
            raise FileNotFoundError(f"No .tif patches found in {self.clean_dir}")

        self.cached_samples: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
        if self.cache_in_ram:
            logger.info(
                f"Caching {len(self.filenames)} TIFF patches from {self.clean_dir.parent.name} into RAM..."
            )
            for fname in tqdm(
                self.filenames, desc=f"RAM Caching ({self.clean_dir.parent.name})"
            ):
                clean_path = self.clean_dir / fname
                noisy_path = self.noisy_dir / fname

                clean_img = tifffile.imread(str(clean_path)).astype(np.float32) / 255.0
                noisy_img = tifffile.imread(str(noisy_path)).astype(np.float32) / 255.0

                clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
                noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
                self.cached_samples.append((noisy_tensor, clean_tensor, fname))

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
        if self.cache_in_ram:
            noisy_tensor, clean_tensor, fname = self.cached_samples[idx]
            noisy_tensor = noisy_tensor.clone()
            clean_tensor = clean_tensor.clone()
        else:
            fname = self.filenames[idx]
            clean_img = (
                tifffile.imread(str(self.clean_dir / fname)).astype(np.float32) / 255.0
            )
            noisy_img = (
                tifffile.imread(str(self.noisy_dir / fname)).astype(np.float32) / 255.0
            )
            clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
            noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)

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
        noise_injector: Optional[SBFSEMNoiseSynthesizer] = None,
        cache_in_ram: bool = True,
        augment: bool = False,
    ) -> None:
        import lmdb

        self.lmdb_path = Path(lmdb_path)
        self.split = split
        self.noise_injector = noise_injector
        self.cache_in_ram = cache_in_ram
        self.augment = augment

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

        self.cached_samples: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
        if self.cache_in_ram:
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

                    if self.noise_injector is not None:
                        noisy_tensor = self.noise_injector(
                            clean_tensor.unsqueeze(0)
                        ).squeeze(0)
                    else:
                        noisy_tensor = clean_tensor.clone()

                    key_str = key.decode("utf-8")
                    fname_str = self.filename_map.get(key_str, key_str)
                    self.cached_samples.append((noisy_tensor, clean_tensor, fname_str))
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
        if self.cache_in_ram:
            noisy_tensor, clean_tensor, key_str = self.cached_samples[idx]
            noisy_tensor = noisy_tensor.clone()
            clean_tensor = clean_tensor.clone()
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
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_edge_loss = 0.0

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Train",
            leave=False,
        )

        for batch_idx, (noisy_b, clean_b, _) in pbar:
            noisy_b, clean_b = noisy_b.to(self.device), clean_b.to(self.device)

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_b = self.model(noisy_b)
                    loss_out = self.criterion(pred_b, clean_b)
            else:
                pred_b = self.model(noisy_b)
                loss_out = self.criterion(pred_b, clean_b)

            if isinstance(loss_out, tuple):
                loss, metrics = loss_out
            else:
                loss = loss_out
                metrics = {"loss_total": loss.item()}

            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            batch_edge = metrics.get("loss_edge", 0.0)

            total_loss += batch_loss * noisy_b.size(0)
            total_edge_loss += batch_edge * noisy_b.size(0)

            if batch_idx % 10 == 0:
                self._purge_memory()

            postfix = {"loss": f"{batch_loss:.5f}"}
            if batch_edge > 0:
                postfix["edge"] = f"{batch_edge:.4f}"
            pbar.set_postfix(postfix)

        num_samples = len(dataloader.dataset)
        return total_loss / num_samples, total_edge_loss / num_samples

    def validate(
        self, dataloader: DataLoader, epoch: int, total_epochs: int
    ) -> Tuple[float, float, float]:
        self.model.eval()
        total_val_loss = 0.0
        val_psnrs: List[float] = []
        val_ssims: List[float] = []

        pbar = tqdm(
            dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Val", leave=False
        )

        with torch.no_grad():
            for noisy_b, clean_b, _ in pbar:
                noisy_b, clean_b = noisy_b.to(self.device), clean_b.to(self.device)

                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_b = self.model(noisy_b)
                        loss_out = self.criterion(pred_b, clean_b)
                else:
                    pred_b = self.model(noisy_b)
                    loss_out = self.criterion(pred_b, clean_b)

                if isinstance(loss_out, tuple):
                    loss = loss_out[0]
                else:
                    loss = loss_out

                total_val_loss += loss.item() * noisy_b.size(0)
                pred_clamp = torch.clamp(pred_b, 0.0, 1.0)

                for i in range(pred_b.size(0)):
                    c_np = clean_b[i, 0].cpu().numpy()
                    p_np = pred_clamp[i, 0].cpu().numpy()
                    val_psnrs.append(EvaluationMetrics.compute_psnr(p_np, c_np))
                    val_ssims.append(EvaluationMetrics.compute_ssim(p_np, c_np))

        self._purge_memory()
        mean_val_loss = total_val_loss / len(dataloader.dataset)
        mean_val_psnr = float(np.mean(val_psnrs)) if val_psnrs else 0.0
        mean_val_ssim = float(np.mean(val_ssims)) if val_ssims else 0.0

        return mean_val_loss, mean_val_psnr, mean_val_ssim

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_psnr: float,
        val_ssim: float,
        is_best: bool = False,
    ) -> Path:
        state = {
            "epoch": epoch,
            "model_state_dict": {
                k: v.cpu() for k, v in self.model.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
        }

        epoch_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(state, epoch_path)

        if is_best:
            best_path = self.checkpoints_dir / "checkpoint_best.pt"
            torch.save(state, best_path)
            return best_path
        return epoch_path


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

    def evaluate_test_set(
        self, test_loader: DataLoader, output_denoised_dir: Path
    ) -> Dict[str, float]:
        output_denoised_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()

        noisy_mses, denoised_mses = [], []
        noisy_psnrs, denoised_psnrs = [], []
        noisy_ssims, denoised_ssims = [], []

        logger.info(
            f"Running inference over {len(test_loader.dataset)} test samples..."
        )
        with torch.no_grad():
            for noisy_b, clean_b, fname_tuple in tqdm(test_loader, desc="Testing"):
                fname = fname_tuple[0]
                noisy_b, clean_b = noisy_b.to(self.device), clean_b.to(self.device)

                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_b = self.model(noisy_b)
                else:
                    pred_b = self.model(noisy_b)

                pred_b = torch.clamp(pred_b.float(), 0.0, 1.0)

                clean_np = clean_b.squeeze().cpu().numpy()
                noisy_np = noisy_b.squeeze().cpu().numpy()
                denoised_np = pred_b.squeeze().cpu().numpy()
                denoised_uint8 = np.clip(denoised_np * 255.0, 0, 255).astype(np.uint8)

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

                out_fname = (
                    fname
                    if (fname.endswith(".tif") or fname.endswith(".tiff"))
                    else f"{fname}.tif"
                )
                tifffile.imwrite(output_denoised_dir / out_fname, denoised_uint8)

        results = {
            "noisy_mse": float(np.mean(noisy_mses)),
            "denoised_mse": float(np.mean(denoised_mses)),
            "noisy_psnr": float(np.mean(noisy_psnrs)),
            "denoised_psnr": float(np.mean(denoised_psnrs)),
            "noisy_ssim": float(np.mean(noisy_ssims)),
            "denoised_ssim": float(np.mean(denoised_ssims)),
        }
        return results


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
        "-i",
        "--input-dir",
        type=str,
        default=None,
        help="Path to custom directory of .tif images to run inference on.",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Override experiment run name (results subdirectory inside output_dir).",
    )
    return parser.parse_args()


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

    run_dir = output_dir / exp_name if exp_name else output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    lmdb_store_path = output_dir / "patch_store.lmdb"
    if getattr(cfg.noise, "synthesizer_type", "parametric") == "empirical":
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

        noise_injector = EmpiricalNoiseSynthesizer(model_path=model_path)
    else:
        noise_injector = SBFSEMNoiseSynthesizer(
            photon_scale=cfg.noise.photon_scale,
            read_std=cfg.noise.read_std,
            gain_variance=cfg.noise.gain_variance,
            streak_probability=cfg.noise.streak_probability,
        )

    if not args.eval_only:
        logger.info("Initializing Train & Validation Datasets...")
        if lmdb_store_path.exists():
            logger.info(f"Found LMDB store at: {lmdb_store_path}")
            train_dataset = LMDBPatchDataset(
                lmdb_store_path,
                split="train",
                noise_injector=noise_injector,
                cache_in_ram=True,
                augment=True,
            )
            val_dataset = LMDBPatchDataset(
                lmdb_store_path,
                split="val",
                noise_injector=noise_injector,
                cache_in_ram=True,
                augment=False,
            )
        else:
            logger.info(
                f"LMDB store not found. Reading TIFF patch directories under: {output_dir}"
            )
            train_dataset = PatchFolderDataset(
                output_dir / "train" / "clean",
                output_dir / "train" / "noisy",
                cache_in_ram=True,
                augment=True,
            )
            val_dataset = PatchFolderDataset(
                output_dir / "val" / "clean",
                output_dir / "val" / "noisy",
                cache_in_ram=True,
                augment=False,
            )

        train_loader = DataLoader(
            train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False
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

    model = ConnectomicsDenoisingNet(in_channels=1, out_channels=1, base_dim=32).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = CompositeDenoisingLoss(
        mse_weight=cfg.loss.mse_weight if cfg.loss else 1.0,
        edge_weight=cfg.loss.edge_weight if cfg.loss else 0.2,
        ssim_weight=cfg.loss.ssim_weight if cfg.loss else 0.1,
    ).to(device)

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

        logger.info(f"Starting Network Training ({epochs} epochs)...")
        start_time = time.time()
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss, train_edge_loss = trainer.train_epoch(
                train_loader, epoch, epochs
            )
            val_loss, val_psnr, val_ssim = trainer.validate(val_loader, epoch, epochs)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            best_ckpt = trainer.save_checkpoint(
                epoch, val_loss, val_psnr, val_ssim, is_best=is_best
            )
            if is_best:
                best_checkpoint_path = best_ckpt

            if writer is not None:
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Train_Edge", train_edge_loss, epoch)
                writer.add_scalar("Loss/Validation", val_loss, epoch)
                writer.add_scalar("Metrics/Val_PSNR", val_psnr, epoch)
                writer.add_scalar("Metrics/Val_SSIM", val_ssim, epoch)

            best_str = " [BEST]" if is_best else ""
            logger.info(
                f"Ep [{epoch:02d}/{epochs:02d}] | Tr: {train_loss:.5f} | "
                f"Val: {val_loss:.5f} | Edge: {train_edge_loss:.4f} | "
                f"PSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.4f}{best_str}"
            )

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
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

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
            cache_in_ram=True,
            augment=False,
        )
    else:
        raise FileNotFoundError(
            f"No test set found in {test_clean_dir} or {lmdb_store_path}"
        )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    evaluator = DenoisingEvaluator(model=model, device=device, use_amp=use_amp)
    results = evaluator.evaluate_test_set(test_loader, test_denoised_dir)

    logger.info("==================================================")
    logger.info(" INFERENCE & DENOISING PERFORMANCE SUMMARY")
    logger.info("==================================================")
    logger.info(
        f" Mean MSE  : Noisy = {results['noisy_mse']:.6f}  ==>  Denoised = {results['denoised_mse']:.6f} (Lower is better)"
    )
    logger.info(
        f" Mean PSNR : Noisy = {results['noisy_psnr']:.2f} dB ==>  Denoised = {results['denoised_psnr']:.2f} dB (Higher is better)"
    )
    logger.info(
        f" Mean SSIM : Noisy = {results['noisy_ssim']:.4f}    ==>  Denoised = {results['denoised_ssim']:.4f}    (Higher is better)"
    )
    logger.info("==================================================")
    logger.info(f"✓ Denoised test images saved to: {test_denoised_dir}")


if __name__ == "__main__":
    main()
