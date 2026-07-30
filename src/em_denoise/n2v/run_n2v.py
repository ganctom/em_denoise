from abc import ABC, abstractmethod
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tifffile
from tqdm import tqdm


# =====================================================================
# 0. Normalization Utilities
# =====================================================================
def normalize_percentile(
    image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0
) -> tuple[np.ndarray, float, float]:
    img_float = image.astype(np.float32)
    p1, p99 = np.percentile(img_float, [p_low, p_high])
    denom = (p99 - p1) if (p99 - p1) > 1e-6 else 1.0
    normalized = np.clip((img_float - p1) / denom, 0.0, 1.0).astype(np.float32)
    return normalized, p1, denom


# =====================================================================
# Seed Utility
# =====================================================================
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# =====================================================================
# 1. Abstraction Layer
# =====================================================================
class DenoisingStrategy(ABC):
    @abstractmethod
    def load_model(self, model_path: Path | None) -> None:
        pass

    @abstractmethod
    def train_on_loader(
        self,
        dataloader: DataLoader,
        epochs: int,
        lr: float,
        checkpoint_path: Path | None = None,
        base_seed: int = 42,
    ) -> None:
        pass

    @abstractmethod
    def save_weights(self, path: Path) -> None:
        pass

    @abstractmethod
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        pass


# =====================================================================
# 2. Updated Dataset - Per-Epoch File Randomization
# =====================================================================
class CachedPatchDataset(Dataset):
    """
    In-memory patch dataset with direct control over number of patches per epoch.
    """

    def __init__(
        self,
        input_dir: Path,
        patch_size: int = 128,
        stride: int = 128,
        augment: bool = True,
        patches_per_epoch: int = 2000,
        base_seed: int = 42,
    ) -> None:
        self.patch_size = patch_size
        self.stride = stride
        self.input_dir = Path(input_dir)
        self.augment = augment
        self.patches_per_epoch = patches_per_epoch
        self.base_seed = base_seed

        # Get all files
        self.all_file_paths: list[Path] = sorted(
            [
                p
                for p in self.input_dir.iterdir()
                if p.suffix.lower() in (".tif", ".tiff")
            ]
        )
        if not self.all_file_paths:
            raise FileNotFoundError(f"No TIFF files found in {input_dir}")

        print(f"[*] Found {len(self.all_file_paths)} TIFF files.")

        self.current_file_paths: list[Path] = []
        self.images: list[np.ndarray] = []
        self.patch_coordinates: list[tuple[int, int, int]] = []

        # Initial sampling
        self._resample_files(epoch=0)

    def _resample_files(self, epoch: int = 0) -> None:
        """Randomly select files and extract approximately patches_per_epoch patches."""
        set_seed(self.base_seed + epoch)

        file_list = self.all_file_paths.copy()
        random.shuffle(file_list)

        self.images.clear()
        self.patch_coordinates.clear()
        self.current_file_paths.clear()

        patches_needed = self.patches_per_epoch

        for fp in file_list:
            if patches_needed <= 0:
                break

            # Load and normalize image
            img = tifffile.imread(str(fp))
            norm_img, _, _ = normalize_percentile(img)
            self.images.append(norm_img)
            self.current_file_paths.append(fp)

            height, width = norm_img.shape[-2:]

            # Calculate patches in this image
            y_steps = (height - self.patch_size) // self.stride + 1
            x_steps = (width - self.patch_size) // self.stride + 1
            patches_in_image = y_steps * x_steps

            patches_to_take = min(patches_in_image, patches_needed)

            # Add patch coordinates
            y_coords = range(0, height - self.patch_size + 1, self.stride)
            x_coords = range(0, width - self.patch_size + 1, self.stride)

            count = 0
            for y in y_coords:
                for x in x_coords:
                    self.patch_coordinates.append((len(self.images) - 1, y, x))
                    count += 1
                    if count >= patches_to_take:
                        break
                if count >= patches_to_take:
                    break

            patches_needed -= patches_to_take

        print(
            f"[*] Epoch {epoch} → Using {len(self.patch_coordinates)} patches "
            f"from {len(self.current_file_paths)} files"
        )

    def __len__(self) -> int:
        return len(self.patch_coordinates)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_idx, y, x = self.patch_coordinates[idx]
        normalized: np.ndarray = self.images[img_idx][
            y : y + self.patch_size, x : x + self.patch_size
        ].copy()

        if self.augment:
            k = np.random.randint(0, 4)
            if k > 0:
                normalized = np.rot90(normalized, k)
            if np.random.random() > 0.5:
                normalized = np.fliplr(normalized)
            if np.random.random() > 0.5:
                normalized = np.flipud(normalized)
            normalized = np.ascontiguousarray(normalized)

        tensor_in = torch.from_numpy(normalized).unsqueeze(0).float()
        masked_tensor, target_mask = self._apply_blind_spot_mask(tensor_in)

        return masked_tensor.float(), tensor_in, target_mask.float()

    @staticmethod
    def _apply_blind_spot_mask(
        tensor: torch.Tensor, mask_ratio: float = 0.015
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C, H, W = tensor.shape
        mask = torch.rand((C, H, W)) < mask_ratio
        masked_tensor = tensor.clone()

        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        shifted_tensors = [torch.roll(tensor, shifts=s, dims=(-2, -1)) for s in shifts]
        stacked_shifts = torch.stack(shifted_tensors, dim=0)
        median_neighbors = torch.median(stacked_shifts, dim=0).values

        masked_tensor[mask] = median_neighbors[mask]
        return masked_tensor, mask


# =====================================================================
# 3. U-Net Architecture
# =====================================================================
class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._conv_block(64, 128)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=1),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=1),
        )
        self.dec2 = self._conv_block(128, 64)
        self.dec1 = self._conv_block(64, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def _conv_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        b = self.bottleneck(self.pool(x2))
        u2 = self.up2(b)
        dec2 = self.dec2(torch.cat([u2, x2], dim=1))
        u1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([u1, x1], dim=1))
        return self.final(dec1)


# =====================================================================
# 4. Noise2Void Strategy
# =====================================================================
class Noise2VoidStrategy(DenoisingStrategy):
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device(
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)
        self.model = None

    def load_model(self, model_path: Path | None) -> None:
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        if model_path and model_path.exists():
            print(f"[*] Loading weights from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("[!] Initializing fresh model.")

    def train_on_loader(
        self,
        dataloader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-3,
        checkpoint_path: Path | None = None,
        base_seed: int = 42,
    ) -> None:
        if self.model is None:
            self.model = UNet(in_channels=1, out_channels=1).to(self.device)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_loss = float("inf")

        print(f"[*] Starting training on {self.device}...")

        for epoch in range(epochs):
            set_seed(base_seed + epoch)
            dataloader.dataset._resample_files(epoch=epoch)

            epoch_loss = 0.0
            for masked_batch, target_batch, mask_batch in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
            ):
                masked_batch = masked_batch.to(self.device, non_blocking=True)
                target_batch = target_batch.to(self.device, non_blocking=True)
                mask_batch = mask_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(masked_batch)

                mask_float = mask_batch.float()
                loss = (
                    (predictions - target_batch) ** 2 * mask_float
                ).sum() / torch.clamp(mask_float.sum(), min=1.0)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"    Epoch {epoch + 1:02d} | Loss: {avg_loss:.6f}")

            if checkpoint_path:
                epoch_dir = checkpoint_path.parent / "epochs"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                self.save_weights(
                    epoch_dir / f"{checkpoint_path.stem}_epoch_{epoch + 1:02d}.pt"
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_weights(
                        checkpoint_path.parent / f"{checkpoint_path.stem}_best.pt"
                    )

        self.model.eval()

    def save_weights(self, path: Path) -> None:
        if self.model is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), path)
            print(f"    [*] Saved checkpoint: {path.name}")

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        self.model.eval()
        orig_dtype = image.dtype

        normalized, p1, denom = normalize_percentile(image)
        tensor_in = (
            torch.from_numpy(normalized)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            tensor_out = self.model(tensor_in)

        denoised = tensor_out.squeeze().cpu().numpy()
        denoised_rescaled = (denoised * denom) + p1

        if np.issubdtype(orig_dtype, np.integer):
            info = np.iinfo(orig_dtype)
            denoised_rescaled = np.clip(denoised_rescaled, info.min, info.max)

        return denoised_rescaled.astype(orig_dtype)


# =====================================================================
# 5. Inference Strategy (Cleaned up)
# =====================================================================
class InferenceStrategy:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device(
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)
        self.model = None

    def load_model(self, model_path: Path) -> None:
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[*] Inference model loaded on {self.device}")

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        return Noise2VoidStrategy.denoise_image(self, image)  # Reuse logic


# =====================================================================
# 6. Pipeline & Batch Inference
# =====================================================================
class TifBatchProcessor:
    def __init__(self, strategy: DenoisingStrategy):
        self.strategy = strategy

    def run_pipeline(
        self,
        input_dir: Path,
        output_dir: Path,
        model_path: Path,
        train_epochs: int = 50,
        batch_size: int = 32,
        files_per_epoch: int | float | None = None,
        base_seed: int = 42,
    ) -> None:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.strategy.load_model(model_path)

        if not model_path.exists():
            dataset = CachedPatchDataset(
                input_dir=input_dir,
                patch_size=32,
                stride=128,
                augment=True,
                patches_per_epoch=files_per_epoch,
                base_seed=base_seed,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
            )

            self.strategy.train_on_loader(
                dataloader=dataloader,
                epochs=train_epochs,
                lr=1.0e-3,
                checkpoint_path=model_path,
                base_seed=base_seed,
            )
        else:
            print("[*] Checkpoint found. Skipping training.")


def run_batch_inference(
    input_dir: Path, output_dir: Path, checkpoint_path: Path, device: str = None
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy = InferenceStrategy(device=device)
    strategy.load_model(checkpoint_path)

    tif_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")]
    )

    print(f"[*] Starting inference on {len(tif_files)} images...")

    for file_path in tqdm(tif_files, desc="Denoising"):
        raw_img = tifffile.imread(str(file_path))
        denoised_img = strategy.denoise_image(raw_img)

        out_file = output_dir / file_path.name
        tifffile.imwrite(str(out_file), denoised_img, metadata={"axes": "YX"})

    print(f"[+] Inference completed. Results saved to: {output_dir}")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    root = Path("/Users/ganctoma/SW/sort/N2V/run_28")
    src_path = Path("/Users/ganctoma/SW/sort/N2V/run_28/raw_patches/train")
    dst_path = root / "denoised"
    checkpoint_file = root / "checkpoints/weights/n2v_2d_em.pt"

    n2v_strategy = Noise2VoidStrategy()
    pipeline = TifBatchProcessor(strategy=n2v_strategy)

    pipeline.run_pipeline(
        input_dir=src_path,
        output_dir=dst_path,
        model_path=checkpoint_file,
        train_epochs=50,
        batch_size=48,
        files_per_epoch=5000,  # ← This is clean and direct
        base_seed=42,
    )

    # Inference
    best_checkpoint = checkpoint_file.parent / f"{checkpoint_file.stem}_best.pt"
    active_checkpoint = best_checkpoint if best_checkpoint.exists() else checkpoint_file

    test_path = Path("/Users/ganctoma/SW/sort/N2V/run_26/raw_patches/test_raw")
    run_batch_inference(test_path, dst_path, active_checkpoint, device="mps")
