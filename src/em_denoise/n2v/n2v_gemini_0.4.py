# HDN pipeline
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
from scipy.ndimage import median_filter


def calibrate_noise_model(
    tiff_paths: list[Path], patch_size: int = 3, epsilon: float = 1e-6
) -> np.ndarray:
    hist_2d = np.zeros((256, 256), dtype=np.float64)

    for path in tqdm(tiff_paths, desc="Calibrating Noise Model"):
        noisy_img = tifffile.imread(path).astype(np.float32)
        pseudo_clean = median_filter(noisy_img, size=patch_size)
        # pseudo_clean = gaussian_filter(noisy_img, sigma=1)

        pseudo_clean = np.clip(np.round(pseudo_clean), 0, 255).astype(np.int32)
        noisy_img = np.clip(np.round(noisy_img), 0, 255).astype(np.int32)

        np.add.at(hist_2d, (pseudo_clean, noisy_img), 1)

    hist_2d += epsilon
    row_sums = hist_2d.sum(axis=1, keepdims=True)
    p_x_given_s = hist_2d / row_sums

    return p_x_given_s


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


# =====================================================================
# 2. Sensor Noise Modeling
# =====================================================================
class HistogramNoiseModel(nn.Module):
    def __init__(self, p_x_given_s: np.ndarray):
        super().__init__()
        # Precompute the negative log probabilities
        nll_matrix = -np.log(p_x_given_s)
        # Register as an un-updatable buffer on the model's device
        self.register_buffer("nll_lut", torch.tensor(nll_matrix, dtype=torch.float32))

    def compute_nll(
        self, noisy: torch.Tensor, clean_prior: torch.Tensor
    ) -> torch.Tensor:
        # Un-normalize [0, 1] tensors to [0, 255] 8-bit range
        noisy_scaled = noisy * 255.0
        clean_scaled = clean_prior * 255.0

        noisy_idx = torch.clamp(torch.round(noisy_scaled), 0, 255).long()
        clean_idx = torch.clamp(torch.round(clean_scaled), 0, 255).long()

        b, c, h, w = noisy.shape
        clean_idx_flat = clean_idx.view(-1)
        noisy_idx_flat = noisy_idx.view(-1)

        # Direct lookup in the 2D NLL matrix: NLL = -log P(x=noisy_idx | s=clean_idx)
        nll_flat = self.nll_lut[clean_idx_flat, noisy_idx_flat]
        return nll_flat.view(b, c, h, w)


# =====================================================================
# 3. Dataset (Unmasked)
# =====================================================================
class HDNPatchDataset(Dataset):
    """
    Yields stochastic crops of noisy data directly. Blind-spot masking is discarded.
    """

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

        # Note: We retain normalizations here to bound tensor to [0, 1]
        self.images = [tifffile.imread(str(p)) for p in file_paths]

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
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

        return torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).float()


# =====================================================================
# 4. Multi-Scale Hierarchical VAE Architecture
# =====================================================================
class StandardConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
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


class HierarchicalVAE(nn.Module):
    """
    Two-level Hierarchical VAE mapping bottom-up deterministic features
    to stochastic top-down latent variables.
    """

    def __init__(
        self, in_channels: int = 1, base_filters: int = 64, latent_dim: int = 32
    ):
        super().__init__()

        # Bottom-up Deterministic Path
        self.enc1 = StandardConvBlock(in_channels, base_filters)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = StandardConvBlock(base_filters, base_filters * 2)

        # Top-down Stochastic Path - Level 2 (Global Context)
        self.prior2_mu = nn.Conv2d(base_filters * 2, latent_dim, 1)
        self.prior2_logvar = nn.Conv2d(base_filters * 2, latent_dim, 1)
        self.post2_mu = nn.Conv2d(base_filters * 2, latent_dim, 1)
        self.post2_logvar = nn.Conv2d(base_filters * 2, latent_dim, 1)

        self.dec2 = StandardConvBlock(latent_dim, base_filters * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Top-down Stochastic Path - Level 1 (Local Detail)
        self.prior1_mu = nn.Conv2d(base_filters * 2, latent_dim, 1)
        self.prior1_logvar = nn.Conv2d(base_filters * 2, latent_dim, 1)
        self.post1_mu = nn.Conv2d(base_filters * 2 + base_filters, latent_dim, 1)
        self.post1_logvar = nn.Conv2d(base_filters * 2 + base_filters, latent_dim, 1)

        self.dec1 = StandardConvBlock(latent_dim, base_filters)
        self.final = nn.Conv2d(base_filters, in_channels, 1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(
        self,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> torch.Tensor:
        # Computes analytically bounded KL between two Gaussians
        return 0.5 * (
            logvar_p
            - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1.0
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Bottom-up evaluation
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))

        kl_divs = {}

        # Level 2 Latent Compute
        p2_mu, p2_logvar = self.prior2_mu(e2), self.prior2_logvar(e2)
        if self.training:
            q2_mu, q2_logvar = self.post2_mu(e2), self.post2_logvar(e2)
            z2 = self.reparameterize(q2_mu, q2_logvar)
            kl_divs["z2"] = self.kl_divergence(q2_mu, q2_logvar, p2_mu, p2_logvar)
        else:
            z2 = self.reparameterize(p2_mu, p2_logvar)

        d2 = self.dec2(z2)
        d2_up = self.up1(d2)

        # Level 1 Latent Compute
        p1_mu, p1_logvar = self.prior1_mu(d2_up), self.prior1_logvar(d2_up)
        if self.training:
            cat1 = torch.cat([d2_up, e1], dim=1)
            q1_mu, q1_logvar = self.post1_mu(cat1), self.post1_logvar(cat1)
            z1 = self.reparameterize(q1_mu, q1_logvar)
            kl_divs["z1"] = self.kl_divergence(q1_mu, q1_logvar, p1_mu, p1_logvar)
        else:
            z1 = self.reparameterize(p1_mu, p1_logvar)

        d1 = self.dec1(z1)
        # Sigmoid explicitly constraints the outputs to [0,1] for Histogram Noise Model NLL mapping
        out = torch.sigmoid(self.final(d1))

        return out, kl_divs


# =====================================================================
# 5. Object-Oriented Orchestration
# =====================================================================
class BaseDenoisingStrategy(ABC):
    def __init__(self, device: torch.device):
        self.device = device
        self.model = HierarchicalVAE(in_channels=1).to(self.device)

    def load_weights(self, path: Path) -> None:
        if path.exists():
            self.model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            print(f"[*] Loaded checkpoint from {path}")
        else:
            print(f"[!] Checkpoint {path} not found. Initialization skipped.")

    def denoise_image(
        self, image: np.ndarray, mmse_samples: int = 100, batch_size: int = 4
    ) -> np.ndarray:
        """
        Executes batched stochastic sampling from the prior to generate an MMSE estimation.
        """
        self.model.eval()
        if image.dtype != np.uint8:
            raise TypeError(
                "Pipeline explicitly configured for 8-bit unsigned integer data."
            )

        # Apply the same fixed scaling used in the corrected dataset
        img_float = image.astype(np.float32) / 255.0
        tensor_in = (
            torch.from_numpy(img_float)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        accumulated_preds = torch.zeros_like(tensor_in)

        # Note: Disable MPS autocast; it frequently causes silent regressions in custom distributions
        use_autocast = self.device.type == "cuda"

        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, enabled=use_autocast),
        ):
            # Batch the MMSE samples to eliminate loop overhead and saturate compute
            for i in range(0, mmse_samples, batch_size):
                current_bs = min(batch_size, mmse_samples - i)
                batch_in = tensor_in.expand(current_bs, -1, -1, -1)

                batch_out, _ = self.model(batch_in)
                # Sum across the batch dimension and accumulate
                accumulated_preds += batch_out.sum(dim=0, keepdim=True)

        mean_pred = (accumulated_preds / mmse_samples).squeeze().cpu().numpy()

        # Scale back to 8-bit range
        return np.clip(mean_pred * 255.0, 0, 255).astype(np.uint8)


class HDNTrainer(BaseDenoisingStrategy):
    def __init__(
        self, device: torch.device, noise_model: nn.Module, beta_kl: float = 1.0
    ):
        super().__init__(device)
        self.noise_model = noise_model
        self.beta_kl = beta_kl

        # Freeze noise model parameters during VAE training
        self.noise_model.eval()
        for param in self.noise_model.parameters():
            param.requires_grad = False

    def compute_kl_loss(
        self, kl_divergences: dict, free_bits: float = 0.05
    ) -> torch.Tensor:
        """
        Prevents posterior collapse by guaranteeing a minimal capacity (free_bits) per latent layer.
        """
        kl_terms = []
        for kl_map in kl_divergences.values():
            # Average over spatial H, W dimensions
            mean_kl_per_dim = kl_map.mean(dim=[0, 2, 3])
            # Clamp lower bound to free_bits (e.g., 0.05 nats)
            bounded_kl = torch.clamp(mean_kl_per_dim, min=free_bits)
            kl_terms.append(bounded_kl.sum())

        return torch.stack(kl_terms).sum()

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        lr: float = 1e-4,
        checkpoint_path: Path = None,
        free_bits: float = 0.05,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Calculate total optimization steps for smooth linear scheduling
        # total_steps = epochs * len(dataloader)
        # warmup_steps = int(total_steps * 0.2)  # Warmup over first 20% of ALL steps

        # current_step = 0
        best_eval_elbo = float("inf")

        for epoch in range(epochs):
            epoch_loss, epoch_nll, epoch_kl = 0.0, 0.0, 0.0

            for noisy_batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                # Ensure explicit float32 casting on MPS/CPU to avoid NNPack mismatch
                noisy_batch = noisy_batch.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                optimizer.zero_grad(set_to_none=True)

                # Only enable autocast for CUDA backends
                use_autocast = self.device.type == "cuda"

                with torch.autocast(device_type="cuda", enabled=use_autocast):
                    clean_prior, kl_divergences = self.model(noisy_batch)

                    nll_loss = self.noise_model.compute_nll(
                        noisy_batch, clean_prior
                    ).mean()
                    kl_loss = torch.stack(
                        [kl.mean() for kl in kl_divergences.values()]
                    ).sum()
                    # loss = nll_loss + (current_beta * kl_loss)
                    loss = nll_loss + kl_loss

                loss.backward()

                # CRITICAL: Prevent high-NLL likelihood gradients from blowing up latents
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                epoch_nll += nll_loss.item()
                epoch_kl += kl_loss.item()

            batches = len(dataloader)
            avg_nll = epoch_nll / batches
            avg_kl = epoch_kl / batches
            # True ELBO (Unweighted beta = 1.0) for objective monitoring
            true_elbo = avg_nll + avg_kl

            print(
                f"Epoch {epoch + 1:02d} | True ELBO: {true_elbo:.4f} | "
                # f"NLL: {avg_nll:.4f} | KL: {avg_kl:.4f} | Beta: {current_beta:.3f}"
            )

            # Checkpoint based on TRUE ELBO (Beta = 1.0)
            if true_elbo < best_eval_elbo:
                best_eval_elbo = true_elbo
                torch.save(
                    self.model.state_dict(),
                    checkpoint_path.with_name(f"{checkpoint_path.stem}_best.pt"),
                )
                print(f"  --> Checkpoint saved (ELBO: {best_eval_elbo:.4f})")


class InferenceEngine(BaseDenoisingStrategy):
    def denoise_image_tiled(
        self,
        image: np.ndarray,
        tile_size: int = 256,
        overlap: int = 32,
        mmse_samples: int = 100,
        gpu_batch_size: int = 16,
    ) -> np.ndarray:
        self.model.eval()
        if image.dtype != np.uint8:
            raise TypeError(
                "Pipeline explicitly configured for 8-bit unsigned integer data."
            )

        # Fixed 8-bit dynamic range scaling matching corrected dataset
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape
        stride = tile_size - overlap

        # Construct 2D Hann window for smooth continuous overlap-add blending
        hann_1d = np.hanning(tile_size)
        window_2d = np.outer(hann_1d, hann_1d).astype(np.float32)

        # Extraction grids
        y_coords = list(range(0, h - tile_size + 1, stride))
        if y_coords[-1] + tile_size < h:
            y_coords.append(h - tile_size)

        x_coords = list(range(0, w - tile_size + 1, stride))
        if x_coords[-1] + tile_size < w:
            x_coords.append(w - tile_size)

        # Canvas for accumulation
        output_canvas = np.zeros((h, w), dtype=np.float32)
        weight_canvas = np.zeros((h, w), dtype=np.float32)

        # Collect all tile crop coordinates
        crop_coords = [(y, x) for y in y_coords for x in x_coords]

        use_autocast = self.device.type == "cuda"

        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, enabled=use_autocast),
        ):
            for y, x in crop_coords:
                patch = img_float[y : y + tile_size, x : x + tile_size]
                patch_tensor = (
                    torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                )

                patch_accum = torch.zeros_like(patch_tensor)

                # Batch MMSE stochastic draws over the single tile
                for i in range(0, mmse_samples, gpu_batch_size):
                    curr_bs = min(gpu_batch_size, mmse_samples - i)
                    batch_in = patch_tensor.expand(curr_bs, -1, -1, -1)

                    batch_out, _ = self.model(batch_in)
                    patch_accum += batch_out.sum(dim=0, keepdim=True)

                patch_mean = (patch_accum / mmse_samples).squeeze().cpu().numpy()

                # Blend into full canvas
                output_canvas[y : y + tile_size, x : x + tile_size] += (
                    patch_mean * window_2d
                )
                weight_canvas[y : y + tile_size, x : x + tile_size] += window_2d

        # Normalize overlap regions
        denoised_rescaled = output_canvas / np.maximum(weight_canvas, 1e-6)

        return np.clip(denoised_rescaled * 255.0, 0, 255).astype(np.uint8)

    def run_directory(
        self, input_dir: Path, output_dir: Path, mmse_samples: int = 100
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tif_files = sorted(
            [p for p in input_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")]
        )

        for file_path in tqdm(tif_files, desc="Batch Tiled Inference"):
            raw_img = tifffile.imread(str(file_path))
            denoised_img = self.denoise_image_tiled(
                raw_img,
                tile_size=256,
                overlap=32,
                mmse_samples=mmse_samples,
                gpu_batch_size=16,
            )
            tifffile.imwrite(
                str(output_dir / file_path.name), denoised_img, metadata={"axes": "YX"}
            )


# =====================================================================
# 6. Pipeline Orchestration
# =====================================================================
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDN Connectomics EM Pipeline")
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
        default=3000,
        help="Patches extracted per epoch",
    )
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    if args.mode == "train":
        # 1. Calibrate Sensor Noise Model
        tiff_files = sorted(
            list(args.src.glob("*.tif")) + list(args.src.glob("*.tiff"))
        )
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {args.src}")

        p_x_given_s = calibrate_noise_model(
            tiff_files[::100]
        )  # Calibrate on a subset for speed
        noise_model = HistogramNoiseModel(p_x_given_s).to(device)

        # 2. Instantiate Dataset
        dataset = HDNPatchDataset(
            args.src, patch_size=128, patches_per_epoch=args.patches_per_epoch
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),  # Conditional pinning
            drop_last=True,
        )

        # 3. Train
        trainer = HDNTrainer(device, noise_model=noise_model, beta_kl=1.0)
        if args.weights.exists():
            trainer.load_weights(args.weights)

        trainer.train(
            dataloader, epochs=args.epochs, lr=1e-4, checkpoint_path=args.weights
        )

    elif args.mode == "infer":
        engine = InferenceEngine(device)
        engine.load_weights(args.weights)
        engine.run_directory(args.src, args.dst, mmse_samples=100)
