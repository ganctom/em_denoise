"""
Noise2Void training and inference for SBEM micrographs.

Noise2Void needs only noisy images: no clean ground truth and no synthetic noise model.
It works by blind-spot masking. A random subset of pixels is replaced by a neighbour's
value before the image is shown to the network, and the loss is evaluated *only* at those
pixels against their original values. The network therefore cannot learn the identity
function -- it must predict each masked pixel from its surroundings. Because the noise is
assumed pixel-wise independent given the signal, the best achievable prediction is the
underlying clean signal.

This is a deliberately small, single-file implementation meant to be readable and to run
unattended on a SLURM node. It reuses `backend.architecture.ResidualUNet` instead of
defining yet another U-Net, so anything learned about the supervised model's capacity
transfers directly.

Run `python -m em_denoise.n2v.train_n2v --help` for the CLI.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
from cyclopts import App
from loguru import logger
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from em_denoise.backend.architecture import ResidualUNet, UNetConfig

app = App(
    name="train-n2v",
    help="Self-supervised Noise2Void denoising for EM images.",
)

IMAGE_SUFFIXES = {".tif", ".tiff"}


def select_device(requested: str = "auto") -> torch.device:
    """Resolve the compute device, matching the CUDA -> MPS -> CPU order used repo-wide."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_images(image_dir: Path) -> list[Path]:
    """Return naturally sorted TIFF paths in `image_dir`, failing loudly if empty."""
    paths = natsorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    )
    if not paths:
        raise FileNotFoundError(f"No TIFF images found in {image_dir}")
    return paths


def to_unit_float(image: np.ndarray) -> np.ndarray:
    """
    Scale an image to float32 in [0, 1] using its dtype range.

    The repo normalizes by dtype maximum everywhere (see backend/dataset.py); N2V follows
    the same convention so checkpoints and metrics stay comparable across both pipelines.
    """
    if np.issubdtype(image.dtype, np.integer):
        max_val = float(np.iinfo(image.dtype).max)
    else:
        max_val = 1.0
    return image.astype(np.float32) / max_val


def apply_blind_spot_mask(
    patch: torch.Tensor, mask_ratio: float, neighbourhood_radius: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replace a random subset of pixels with a neighbour's value (N2V "UPS" scheme).

    Returns the corrupted patch and a binary mask marking the replaced pixels, which is
    where the loss will be evaluated.

    Two details matter for correctness, and both are easy to get wrong:

    1. A zero offset is remapped to a real neighbour. Sampling (0, 0) would copy the pixel
       onto itself, leaking the value the network is supposed to infer.
    2. Out-of-bounds offsets are *reflected*, not clamped. Clamping at an edge maps the
       neighbour back onto the masked pixel itself (y=0 with dy=-1 clamps to y=0), which
       reintroduces exactly the leak that (1) removes -- silently, and only along image
       borders. Reflection keeps the offset non-zero.
    """
    _, height, width = patch.shape
    n_pixels = height * width
    n_masked = max(1, int(round(n_pixels * mask_ratio)))

    flat_indices = torch.randperm(n_pixels)[:n_masked]
    ys = flat_indices // width
    xs = flat_indices % width

    offsets = torch.randint(
        -neighbourhood_radius, neighbourhood_radius + 1, (2, n_masked)
    )
    both_zero = (offsets[0] == 0) & (offsets[1] == 0)
    offsets[1][both_zero] = 1

    neighbour_ys = ys + offsets[0]
    neighbour_ys = torch.where(
        (neighbour_ys < 0) | (neighbour_ys >= height), ys - offsets[0], neighbour_ys
    )
    neighbour_xs = xs + offsets[1]
    neighbour_xs = torch.where(
        (neighbour_xs < 0) | (neighbour_xs >= width), xs - offsets[1], neighbour_xs
    )

    corrupted = patch.clone()
    corrupted[0, ys, xs] = patch[0, neighbour_ys, neighbour_xs]

    mask = torch.zeros_like(patch)
    mask[0, ys, xs] = 1.0

    return corrupted, mask


class N2VPatchDataset(Dataset):
    """
    Serves blind-spot-masked crops from a directory of noisy TIFFs.

    Images are cached in RAM (EM patches are small and the cluster nodes have plenty), and
    each __getitem__ takes a fresh random crop plus a fresh mask, so one epoch never sees
    the same training example twice.
    """

    def __init__(
        self,
        image_paths: list[Path],
        patch_size: int = 128,
        mask_ratio: float = 0.02,
        samples_per_image: int = 4,
        augment: bool = True,
    ):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.samples_per_image = samples_per_image
        self.augment = augment

        self.images: list[np.ndarray] = []
        for path in tqdm(image_paths, desc="Caching images", leave=False):
            image = to_unit_float(tifffile.imread(str(path)))
            if image.ndim != 2:
                raise ValueError(
                    f"Expected a 2D grayscale image, got {image.shape}: {path}"
                )
            if min(image.shape) < patch_size:
                logger.warning(
                    f"Skipping {path.name}: {image.shape} smaller than patch size {patch_size}"
                )
                continue
            self.images.append(image)

        if not self.images:
            raise ValueError(
                f"No image was large enough for patch_size={patch_size}. "
                "Lower --patch-size or supply larger images."
            )

    def __len__(self) -> int:
        return len(self.images) * self.samples_per_image

    def _random_crop(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        return image[top : top + self.patch_size, left : left + self.patch_size]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.images[idx % len(self.images)]
        patch = torch.from_numpy(self._random_crop(image).copy()).unsqueeze(0)

        if self.augment:
            if random.random() > 0.5:
                patch = torch.flip(patch, dims=[2])
            k = random.randint(0, 3)
            if k:
                patch = torch.rot90(patch, k, dims=[1, 2])

        corrupted, mask = apply_blind_spot_mask(patch, self.mask_ratio)
        return corrupted, patch, mask


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Mean squared error restricted to the blind-spot pixels.

    Normalizing by the mask sum rather than by the number of elements keeps the loss
    magnitude independent of `mask_ratio`, so changing that knob does not silently
    rescale the effective learning rate.
    """
    squared_error = (prediction - target) ** 2 * mask
    return squared_error.sum() / mask.sum().clamp(min=1.0)


def split_paths(
    paths: list[Path], val_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Deterministically split file paths into train/val sets."""
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def build_model(patch_size: int, base_channels: int) -> tuple[ResidualUNet, UNetConfig]:
    """Instantiate the U-Net and return it together with the config needed to rebuild it."""
    config = UNetConfig(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        dropout_rate=0.0,
    )
    # Three downsampling levels: the crop must stay divisible by 8.
    if patch_size % 8:
        raise ValueError(f"patch_size must be a multiple of 8, got {patch_size}")
    return ResidualUNet(config), config


@torch.no_grad()
def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return mean masked-MSE over the validation loader."""
    model.eval()
    total, n_samples = 0.0, 0
    for corrupted, target, mask in loader:
        corrupted = corrupted.to(device)
        target = target.to(device)
        mask = mask.to(device)
        loss = masked_mse(model(corrupted), target, mask)
        total += loss.item() * corrupted.size(0)
        n_samples += corrupted.size(0)
    return total / max(n_samples, 1)


@app.command
def train(
    noisy_dir: Path,
    out_dir: Path,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    patch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    mask_ratio: float = 0.02,
    samples_per_image: int = 4,
    base_channels: int = 32,
    val_fraction: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    device: str = "auto",
) -> None:
    """
    Train a Noise2Void model on a directory of noisy TIFF images.

    Parameters
    ----------
    noisy_dir
        Directory of noisy TIFFs. No clean targets are needed.
    out_dir
        Destination for `checkpoints/best_model.pt` and TensorBoard `logs/`.
    mask_ratio
        Fraction of pixels blind-spotted per patch. 0.5-2% is the usual range; higher
        gives more supervision per patch but degrades the visible context.
    samples_per_image
        Random crops drawn per image per epoch; the effective epoch length is
        `len(images) * samples_per_image`.
    """
    set_seed(seed)
    torch_device = select_device(device)

    out_dir = Path(out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "logs"))

    paths = find_images(Path(noisy_dir))
    train_paths, val_paths = split_paths(paths, val_fraction, seed)
    logger.info(
        f"Found {len(paths)} images -> {len(train_paths)} train / {len(val_paths)} val"
    )

    train_dataset = N2VPatchDataset(
        train_paths, patch_size, mask_ratio, samples_per_image, augment=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(torch_device.type == "cuda"),
        drop_last=True,
    )

    val_loader = None
    if val_paths:
        val_dataset = N2VPatchDataset(
            val_paths, patch_size, mask_ratio, samples_per_image, augment=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
        )

    model, unet_config = build_model(patch_size, base_channels)
    model = model.to(torch_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(f"Training on {torch_device} for {epochs} epochs")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_samples = 0.0, 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs:03d}")
        for corrupted, target, mask in progress:
            corrupted = corrupted.to(torch_device, non_blocking=True)
            target = target.to(torch_device, non_blocking=True)
            mask = mask.to(torch_device, non_blocking=True)

            optimizer.zero_grad()
            loss = masked_mse(model(corrupted), target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += loss.item() * corrupted.size(0)
            n_samples += corrupted.size(0)
            progress.set_postfix({"loss": f"{loss.item():.6f}"})

        train_loss = running / max(n_samples, 1)
        scheduler.step()

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Params/LearningRate", optimizer.param_groups[0]["lr"], epoch)

        # Without a validation split, selection falls back to training loss.
        monitored = train_loss
        if val_loader is not None:
            val_loss = _validate(model, val_loader, torch_device)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            monitored = val_loss
            logger.info(
                f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}"
            )
        else:
            logger.info(f"Epoch {epoch:03d} | train {train_loss:.6f}")

        if monitored < best_loss:
            best_loss = monitored
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": monitored,
                    # Stored so `denoise` can rebuild the exact architecture without
                    # being handed the same CLI flags again.
                    "unet_config": unet_config.model_dump(),
                    "patch_size": patch_size,
                    "mask_ratio": mask_ratio,
                    # N2V predicts the signal directly, unlike the supervised trainer
                    # which predicts the residual noise. Consumers must not subtract.
                    "predict_residual": False,
                },
                checkpoint_path,
            )
            logger.info(
                f"Saved best checkpoint (loss {monitored:.6f}) -> {checkpoint_path}"
            )

    writer.close()
    logger.success(f"Training finished. Best loss: {best_loss:.6f}")


@torch.inference_mode()
def denoise_array(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """
    Denoise a single 2D image with overlapping tiles blended by a Hann window.

    Full SBEM tiles do not fit in GPU memory, and naive tiling leaves visible seams; the
    Hann weighting tapers each tile's contribution so overlapping predictions average
    smoothly.
    """
    tensor = torch.from_numpy(to_unit_float(image)).unsqueeze(0)
    _, height, width = tensor.shape

    # Small images need no tiling, but must still be divisible by 8 for the U-Net.
    tile_size = min(tile_size, (min(height, width) // 8) * 8)
    if tile_size < 8:
        raise ValueError(f"Image {image.shape} is too small to denoise")

    stride = max(8, tile_size - overlap)
    window = torch.from_numpy(np.outer(np.hanning(tile_size), np.hanning(tile_size)))
    window = window.float().clamp(min=1e-3).unsqueeze(0)

    canvas = torch.zeros((1, height, width), dtype=torch.float32)
    weights = torch.zeros((1, height, width), dtype=torch.float32)

    y_positions = list(range(0, max(1, height - tile_size + 1), stride))
    x_positions = list(range(0, max(1, width - tile_size + 1), stride))
    if y_positions[-1] + tile_size < height:
        y_positions.append(height - tile_size)
    if x_positions[-1] + tile_size < width:
        x_positions.append(width - tile_size)

    for top in y_positions:
        for left in x_positions:
            tile = tensor[:, top : top + tile_size, left : left + tile_size]
            prediction = model(tile.unsqueeze(0).to(device))[0].cpu()
            canvas[:, top : top + tile_size, left : left + tile_size] += (
                prediction * window
            )
            weights[:, top : top + tile_size, left : left + tile_size] += window

    denoised = (canvas / weights.clamp(min=1e-6)).clamp(0.0, 1.0)
    return (denoised[0].numpy() * 255.0).astype(np.uint8)


@app.command
def denoise(
    input_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    device: str = "auto",
) -> None:
    """
    Apply a trained Noise2Void checkpoint to every TIFF in `input_dir`.

    Outputs are written as 8-bit TIFFs prefixed with `denoised_`.
    """
    torch_device = select_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    if "unet_config" not in payload:
        raise KeyError(
            f"{checkpoint} has no 'unet_config'. It was probably not produced by "
            "train_n2v.py; supervised checkpoints belong to execution/run_inference.py."
        )

    model = ResidualUNet(UNetConfig(**payload["unet_config"])).to(torch_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    paths = find_images(Path(input_dir))
    logger.info(f"Denoising {len(paths)} images on {torch_device}")

    for path in tqdm(paths, desc="Denoising"):
        image = tifffile.imread(str(path))
        result = denoise_array(model, image, torch_device, tile_size, overlap)
        tifffile.imwrite(str(output_dir / f"denoised_{path.name}"), result)

    logger.success(f"Wrote {len(paths)} images to {output_dir}")


if __name__ == "__main__":
    app()
