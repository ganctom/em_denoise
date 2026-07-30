import argparse
from abc import ABC
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tifffile
from tqdm import tqdm


# =====================================================================
# 1. Normalization & Utilities
# =====================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_percentile(
    image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0
) -> tuple[np.ndarray, float, float]:
    img_float = image.astype(np.float32)
    p1, p99 = np.percentile(img_float, [p_low, p_high])
    denom = (p99 - p1) if (p99 - p1) > 1e-6 else 1.0
    normalized = np.clip((img_float - p1) / denom, 0.0, 1.0).astype(np.float32)
    return normalized, p1, denom


class StandardConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        # Roll back to a non-residual architecture while keeping stable normalizations
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class N2V2UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.enc1 = StandardConvBlock(in_channels, 32)
        self.pool1 = MaxBlurPool2d(32)

        self.enc2 = StandardConvBlock(32, 64)
        self.pool2 = MaxBlurPool2d(64)

        self.bottleneck = StandardConvBlock(64, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.dec2 = StandardConvBlock(128, 64)  # Takes concatenated [u2, x2]

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv1 = nn.Conv2d(64, 32, kernel_size=1)

        # Takes ONLY u1; topmost skip connection is removed per N2V2 spec
        self.dec1 = StandardConvBlock(32, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        b = self.bottleneck(self.pool2(x2))

        u2 = self.up_conv2(self.up2(b))
        dec2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = self.up_conv1(self.up1(dec2))

        # N2V2 Architectural Bypass: Skip concatenation with x1
        dec1 = self.dec1(u1)

        return self.final(dec1)


# =====================================================================
# 3. Stochastic Dataset Pipeline & Vectorized Masking
# =====================================================================
class StochasticPatchDataset(Dataset):
    """Yields true stochastic crops per step, resolving grid-scanning bias."""

    def __init__(
        self,
        input_dir: Path,
        patch_size: int = 128,
        patches_per_epoch: int = 2000,
        augment: bool = True,
    ) -> None:
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment

        file_paths = sorted(
            [
                p
                for p in Path(input_dir).iterdir()
                if p.suffix.lower() in (".tif", ".tiff")
            ]
        )
        if not file_paths:
            raise FileNotFoundError(f"No TIFF files found in {input_dir}")

        # Eager load into memory (acceptable for standard EM directories; use memmap for VRAM-exceeding datasets)
        self.images = [
            normalize_percentile(tifffile.imread(str(p)))[0] for p in file_paths
        ]

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_idx = torch.randint(0, len(self.images), (1,)).item()
        img = self.images[img_idx]
        h, w = img.shape

        y = torch.randint(0, h - self.patch_size + 1, (1,)).item()
        x = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        patch = img[y : y + self.patch_size, x : x + self.patch_size].copy()

        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            patch = np.rot90(patch, k)
            if torch.rand(1).item() > 0.5:
                patch = np.fliplr(patch)
            if torch.rand(1).item() > 0.5:
                patch = np.flipud(patch)

        tensor_in = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).float()
        masked_tensor, target_mask = self._apply_n2v2_mask(tensor_in)
        return masked_tensor, tensor_in, target_mask

    @staticmethod
    def _apply_n2v2_mask(
        tensor: torch.Tensor, mask_ratio: float = 0.015
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """N2V2 masking: replaces blind-spot with the median of the 3x3 neighborhood (excluding center)."""
        C, H, W = tensor.shape
        mask = torch.rand((H, W)) < mask_ratio
        masked_tensor = tensor.clone()

        y, x = torch.where(mask)
        if len(y) == 0:
            return masked_tensor, mask.unsqueeze(0).float()

        # Pad tensor to handle masked pixels on the boundaries
        padded = torch.nn.functional.pad(tensor, (1, 1, 1, 1), mode="reflect").squeeze(
            0
        )
        y_p, x_p = y + 1, x + 1

        # Extract the 8 local neighbors dynamically
        neighbors = torch.stack(
            [
                padded[y_p - 1, x_p - 1],
                padded[y_p - 1, x_p],
                padded[y_p - 1, x_p + 1],
                padded[y_p, x_p - 1],
                padded[y_p, x_p + 1],
                padded[y_p + 1, x_p - 1],
                padded[y_p + 1, x_p],
                padded[y_p + 1, x_p + 1],
            ],
            dim=1,
        )

        # Compute exact median (excluding the center/target pixel)
        medians, _ = torch.median(neighbors, dim=1)
        masked_tensor[:, y, x] = medians

        return masked_tensor, mask.unsqueeze(0).float()


class MaxBlurPool2d(nn.Module):
    """Anti-aliased downsampling layer utilizing a 3x3 binomial filter."""

    def __init__(self, channels: int):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1)

        # 3x3 Binomial filter: [1, 2, 1] outer product
        filt = torch.tensor([1.0, 2.0, 1.0])
        filt2d = filt[:, None] * filt[None, :]
        filt2d = filt2d / filt2d.sum()

        # Register as an immutable buffer, grouped by channel
        self.register_buffer(
            "blur_filter", filt2d.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate padding for MaxPool bounds
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode="replicate")
        x = self.max_pool(x)

        # Replicate padding for Blur filter bounds
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
        return torch.nn.functional.conv2d(
            x, self.blur_filter, stride=2, groups=x.shape[1]
        )


# =====================================================================
# 4. Object-Oriented Orchestration
# =====================================================================
class BaseDenoisingStrategy(ABC):
    def __init__(self, device: torch.device):
        self.device = device
        self.model = N2V2UNet(in_channels=1, out_channels=1).to(self.device)

    def load_weights(self, path: Path) -> None:
        if path.exists():
            self.model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            print(f"[*] Loaded checkpoint from {path}")
        else:
            print(f"[!] Checkpoint {path} not found. Initialization skipped.")

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        self.model.eval()
        # Strictly enforce 8-bit pipeline bounds per ledger requirements
        if image.dtype != np.uint8:
            raise TypeError(
                "Pipeline explicitly configured for 8-bit unsigned integer data."
            )

        normalized, p1, denom = normalize_percentile(image)
        tensor_in = (
            torch.from_numpy(normalized)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type in ["cuda", "mps"],
            ):
                tensor_out = self.model(tensor_in)

        denoised = tensor_out.squeeze().cpu().numpy()
        denoised_rescaled = (denoised * denom) + p1

        # Output bounds mapping
        return np.clip(denoised_rescaled, 0, 255).astype(np.uint8)


class Noise2VoidTrainer(BaseDenoisingStrategy):
    def train(
        self, dataloader: DataLoader, epochs: int, lr: float, checkpoint_path: Path
    ) -> None:
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # AMP Initialization - PyTorch limits GradScaler to CUDA backend
        use_scaler = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if use_scaler else None

        best_loss = float("inf")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for masked_batch, target_batch, mask_batch in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
            ):
                masked_batch = masked_batch.to(self.device, non_blocking=True)
                target_batch = target_batch.to(self.device, non_blocking=True)
                mask_batch = mask_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.device.type in ["cuda", "mps"],
                ):
                    predictions = self.model(masked_batch)
                    loss = (
                        (predictions - target_batch) ** 2 * mask_batch
                    ).sum() / torch.clamp(mask_batch.sum(), min=1.0)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"    Epoch {epoch + 1:02d} | Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    self.model.state_dict(),
                    checkpoint_path.with_name(f"{checkpoint_path.stem}_best.pt"),
                )


class InferenceEngine(BaseDenoisingStrategy):
    def run_directory(self, input_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tif_files = sorted(
            [p for p in input_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")]
        )

        for file_path in tqdm(tif_files, desc="Batch Inference"):
            raw_img = tifffile.imread(str(file_path))
            denoised_img = self.denoise_image(raw_img)
            tifffile.imwrite(
                str(output_dir / file_path.name), denoised_img, metadata={"axes": "YX"}
            )


# =====================================================================
# 5. Pipeline Orchestration
# =====================================================================
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise2Void Connectomics EM Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        required=True,
        help="Pipeline execution mode",
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Input directory containing 8-bit .tif images",
    )
    parser.add_argument("--dst", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to model weights (.pt)"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--patches_per_epoch",
        type=int,
        default=1000,
        help="Patches extracted per epoch",
    )
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    if args.mode == "train":
        dataset = StochasticPatchDataset(
            args.src, patch_size=64, patches_per_epoch=args.patches_per_epoch
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        trainer = Noise2VoidTrainer(device)
        if args.weights.exists():
            trainer.load_weights(args.weights)

        trainer.train(
            dataloader, epochs=args.epochs, lr=1e-3, checkpoint_path=args.weights
        )

    elif args.mode == "infer":
        engine = InferenceEngine(device)
        engine.load_weights(args.weights)
        engine.run_directory(args.src, args.dst)
