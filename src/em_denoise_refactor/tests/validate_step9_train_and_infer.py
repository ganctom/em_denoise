# src/em_denoise_refactor/tests/validate_step9_train_and_infer.py

import sys
from typing import Tuple

import math
import time
from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from em_denoise_refactor.config import PipelineConfig
from em_denoise_refactor.models.connectomics import ConnectomicsDenoisingNet


# -------------------------------------------------------------------------
# Evaluation Metrics (MSE, PSNR, SSIM)
# -------------------------------------------------------------------------


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))


def compute_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    mse = compute_mse(img1, img2)
    if mse < 1e-10:
        return 100.0
    return float(10 * math.log10((data_range**2) / mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim

        return float(ssim(img1, img2, data_range=data_range))
    except ImportError:
        # Fallback SSIM calculation for 2D float images
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        mu1 = float(np.mean(img1))
        mu2 = float(np.mean(img2))
        sigma1_sq = float(np.var(img1))
        sigma2_sq = float(np.var(img2))
        sigma12 = float(np.cov(img1.flat, img2.flat)[0, 1])

        num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        return float(num / den)


# -------------------------------------------------------------------------
# PyTorch Dataset for Clean/Noisy Patch Pairs
# -------------------------------------------------------------------------


class PatchFolderDataset(Dataset):
    def __init__(
        self,
        clean_dir: Path,
        noisy_dir: Path,
        cache_in_ram: bool = True,
        augment: bool = False,
    ):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.augment = augment
        self.cache_in_ram = cache_in_ram

        self.filenames = sorted([f.name for f in self.clean_dir.glob("*.tif")])
        if not self.filenames:
            raise FileNotFoundError(f"No .tif patches found in {self.clean_dir}")

        self.cached_samples = []
        if self.cache_in_ram:
            print(
                f"Caching {len(self.filenames)} patches from {self.clean_dir.parent.name} into RAM for fast training..."
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
        # D4 Dihedral Group Symmetry: random flips + 90-degree rotations
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

    def __getitem__(self, idx: int):
        if self.cache_in_ram:
            noisy_tensor, clean_tensor, fname = self.cached_samples[idx]
            noisy_tensor = noisy_tensor.clone()
            clean_tensor = clean_tensor.clone()
        else:
            fname = self.filenames[idx]
            clean_path = self.clean_dir / fname
            noisy_path = self.noisy_dir / fname

            clean_img = tifffile.imread(str(clean_path)).astype(np.float32) / 255.0
            noisy_img = tifffile.imread(str(noisy_path)).astype(np.float32) / 255.0

            clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
            noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)

        if self.augment:
            noisy_tensor, clean_tensor = self._apply_augmentation(
                noisy_tensor, clean_tensor
            )

        return noisy_tensor, clean_tensor, fname


# -------------------------------------------------------------------------
# Main Training & Inference Workflow
# -------------------------------------------------------------------------


def main(config_path: Path | str = "config.yaml"):
    print(f"Loading pipeline configuration from: {config_path}")
    cfg = PipelineConfig.from_yaml(config_path)

    output_dir = cfg.experiment.output_dir
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_clean_dir = output_dir / "train" / "clean"
    train_noisy_dir = output_dir / "train" / "noisy"
    val_clean_dir = output_dir / "val" / "clean"
    val_noisy_dir = output_dir / "val" / "noisy"
    test_clean_dir = output_dir / "test" / "clean"
    test_noisy_dir = output_dir / "test" / "noisy"
    test_denoised_dir = output_dir / "test" / "denoised"
    test_denoised_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- 1. Initializing Datasets & Dataloaders ---")
    train_dataset = PatchFolderDataset(
        train_clean_dir, train_noisy_dir, cache_in_ram=True, augment=True
    )
    val_dataset = PatchFolderDataset(
        val_clean_dir, val_noisy_dir, cache_in_ram=True, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(cfg.training.batch_size, len(train_dataset)),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(cfg.training.batch_size, len(val_dataset)),
        shuffle=False,
    )

    print(f"Train Patches: {len(train_dataset)} | Val Patches: {len(val_dataset)}")

    # Target Hardware Device
    if cfg.training.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        try:
            from em_denoise_refactor.execution.mps_manager import AppleSiliconManager

            mps_manager = AppleSiliconManager()
        except Exception:
            mps_manager = None
    elif cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        mps_manager = None
    else:
        device = torch.device("cpu")
        mps_manager = None

    print(f"Using execution device: {device}")

    print("\n--- 2. Instantiating ConnectomicsDenoisingNet & Monitoring ---")
    model = ConnectomicsDenoisingNet(in_channels=1, out_channels=1, base_dim=32).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.MSELoss()

    # TensorBoard Logger
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logging initialized at: {log_dir}")
    except Exception:
        writer = None
        print("TensorBoard logger unavailable. Continuing with console progress bars.")

    best_val_loss = float("inf")
    best_checkpoint_path = checkpoints_dir / "checkpoint_best.pt"

    # For validation test script, default to e.g. 5 epochs unless specified in config
    num_epochs = min(cfg.training.epochs, 10)
    print(f"\n--- 3. Starting Network Training ({num_epochs} epochs) ---")

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch [{epoch:02d}/{num_epochs:02d}] Train",
            leave=False,
        )

        for batch_idx, (noisy_b, clean_b, _) in train_pbar:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)

            optimizer.zero_grad()
            pred_b = model(noisy_b)
            loss = criterion(pred_b, clean_b)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss * noisy_b.size(0)

            # Periodically purge MPS cache every 10 batches to prevent memory leaks on Unified Memory
            if mps_manager is not None and batch_idx % 10 == 0:
                mps_manager.optimize_memory()
            elif device.type == "mps" and batch_idx % 10 == 0:
                import gc

                gc.collect()
                torch.mps.empty_cache()

            postfix = {"loss": f"{batch_loss:.5f}"}
            if device.type == "mps":
                allocated_mb = torch.mps.current_allocated_memory() / (1024**2)
                postfix["mps_mem"] = f"{allocated_mb:.0f}MB"
            train_pbar.set_postfix(postfix)

        train_loss /= len(train_dataset)

        # Validation Pass with Metrics Calculation
        model.eval()
        val_loss = 0.0
        val_psnrs = []
        val_ssims = []

        val_pbar = tqdm(
            val_loader, desc=f"Epoch [{epoch:02d}/{num_epochs:02d}] Val", leave=False
        )

        with torch.no_grad():
            for noisy_b, clean_b, _ in val_pbar:
                noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
                pred_b = model(noisy_b)
                loss = criterion(pred_b, clean_b)
                val_loss += loss.item() * noisy_b.size(0)

                # Compute batch validation metrics
                pred_clamp = torch.clamp(pred_b, 0.0, 1.0)
                for i in range(pred_b.size(0)):
                    c_np = clean_b[i, 0].cpu().numpy()
                    p_np = pred_clamp[i, 0].cpu().numpy()
                    val_psnrs.append(compute_psnr(p_np, c_np))
                    val_ssims.append(compute_ssim(p_np, c_np))

        val_loss /= len(val_dataset)
        mean_val_psnr = float(np.mean(val_psnrs)) if val_psnrs else 0.0
        mean_val_ssim = float(np.mean(val_ssims)) if val_ssims else 0.0

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Metrics/Val_PSNR", mean_val_psnr, epoch)
            writer.add_scalar("Metrics/Val_SSIM", mean_val_ssim, epoch)
            if device.type == "mps":
                writer.add_scalar(
                    "Memory/MPS_Allocated_MB",
                    torch.mps.current_allocated_memory() / (1024**2),
                    epoch,
                )

        # Purge memory after validation
        if mps_manager is not None:
            mps_manager.optimize_memory()
        elif device.type == "mps":
            import gc

            gc.collect()
            torch.mps.empty_cache()

        # Save every epoch checkpoint with CPU state dict to avoid holding MPS memory references
        epoch_ckpt = checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_psnr": mean_val_psnr,
                "val_ssim": mean_val_ssim,
            },
            epoch_ckpt,
        )

        # Track and save best checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": {
                        k: v.cpu() for k, v in model.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_psnr": mean_val_psnr,
                    "val_ssim": mean_val_ssim,
                },
                best_checkpoint_path,
            )

        best_str = " [BEST MODEL SAVED]" if is_best else ""
        mem_str = (
            f" | MPS Mem: {torch.mps.current_allocated_memory() / (1024**2):.0f}MB"
            if device.type == "mps"
            else ""
        )
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] | Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val PSNR: {mean_val_psnr:.2f} dB | "
            f"Val SSIM: {mean_val_ssim:.4f}{mem_str}{best_str}"
        )

    training_time = time.time() - start_time
    if writer is not None:
        writer.close()
    print(
        f"\n✓ Training completed in {training_time:.2f} seconds. Best Val Loss: {best_val_loss:.6f}"
    )

    print("\n--- 4. Executing Inference with BEST Checkpoint ---")
    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {best_checkpoint_path}")

    print(f"Loading weights from: {best_checkpoint_path.name}")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = PatchFolderDataset(test_clean_dir, test_noisy_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    noisy_psnrs, denoised_psnrs = [], []
    noisy_mses, denoised_mses = [], []
    noisy_ssims, denoised_ssims = [], []

    print(f"Processing {len(test_dataset)} test noisy patches...")
    with torch.no_grad():
        for noisy_b, clean_b, fname_tuple in tqdm(
            test_loader, desc="Running Inference"
        ):
            fname = fname_tuple[0]
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)

            # Predict denoised tensor
            pred_b = model(noisy_b)
            pred_b = torch.clamp(pred_b, 0.0, 1.0)

            # Convert to numpy arrays (H, W) in range [0.0, 1.0]
            clean_np = clean_b.squeeze().cpu().numpy()
            noisy_np = noisy_b.squeeze().cpu().numpy()
            denoised_np = pred_b.squeeze().cpu().numpy()

            # Compute metrics
            mse_noisy = compute_mse(noisy_np, clean_np)
            mse_denoised = compute_mse(denoised_np, clean_np)

            psnr_noisy = compute_psnr(noisy_np, clean_np)
            psnr_denoised = compute_psnr(denoised_np, clean_np)

            ssim_noisy = compute_ssim(noisy_np, clean_np)
            ssim_denoised = compute_ssim(denoised_np, clean_np)

            noisy_mses.append(mse_noisy)
            denoised_mses.append(mse_denoised)
            noisy_psnrs.append(psnr_noisy)
            denoised_psnrs.append(psnr_denoised)
            noisy_ssims.append(ssim_noisy)
            denoised_ssims.append(ssim_denoised)

            # Save denoised .tif image
            denoised_uint8 = (denoised_np * 255.0).astype(np.uint8)
            tifffile.imwrite(test_denoised_dir / fname, denoised_uint8)

    print("\n==================================================")
    print(" INFERENCE & DENOISING PERFORMANCE SUMMARY")
    print("==================================================")
    print(
        f" Mean MSE  : Noisy = {np.mean(noisy_mses):.6f}  ==>  Denoised = {np.mean(denoised_mses):.6f} (Lower is better)"
    )
    print(
        f" Mean PSNR : Noisy = {np.mean(noisy_psnrs):.2f} dB ==>  Denoised = {np.mean(denoised_psnrs):.2f} dB (Higher is better)"
    )
    print(
        f" Mean SSIM : Noisy = {np.mean(noisy_ssims):.4f}    ==>  Denoised = {np.mean(denoised_ssims):.4f}    (Higher is better)"
    )
    print("==================================================")
    print(f"✓ Denoised test images saved to: {test_denoised_dir}")


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg_file)
