import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from em_denoise.backend.architecture import ResidualUNet, UNetConfig
from em_denoise.backend.config_schema import PipelineConfig

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)


class SingleDirectoryImageDataset(Dataset):
    """
    Loads grayscale/single-channel images from a target directory.
    Supports .png, .tif, .tiff, .jpg, and .npy formats.
    """

    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}

    def __init__(self, image_dir: str | Path):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.suffix.lower() in self.VALID_EXTENSIONS
            ]
        )

        if not self.image_paths:
            raise FileNotFoundError(
                f"No valid image artifacts found in {self.image_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]

        if img_path.suffix.lower() == ".npy":
            arr = np.load(img_path).astype(np.float32)
            tensor = torch.from_numpy(arr)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
        else:
            with Image.open(img_path) as pil_img:
                tensor = TF.to_tensor(pil_img.convert("L"))

        return tensor, img_path.name


def _save_image_artifact(tensor: torch.Tensor, save_path: Path) -> None:
    """I/O Worker target for serializing output arrays to disk."""
    if save_path.suffix.lower() == ".npy":
        np.save(save_path, tensor.squeeze(0).numpy())
    else:
        out_np = (tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
        out_pil = Image.fromarray(out_np)
        out_pil.save(save_path)


def run_directory_inference(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 2,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = True,
    max_io_workers: int = 4,
    patch_size: int = 512,
    overlap: int = 64,
) -> None:
    """
    Executes sliding-window tile inference over large high-resolution micrographs
    to prevent MPS/CUDA out-of-memory errors.
    """
    test_path = Path(test_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logging.info("Deserializing checkpoint from: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        predict_residual = checkpoint.get("predict_residual", True)
    else:
        state_dict = checkpoint
        predict_residual = True

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = SingleDirectoryImageDataset(test_path)
    logging.info(
        "Executing tile-based sliding-window inference on %d samples via target device: %s | Residual Mode: %s",
        len(dataset),
        device,
        predict_residual,
    )

    amp_device = (
        "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    )
    stride = patch_size - overlap

    with (
        torch.inference_mode(),
        ThreadPoolExecutor(max_workers=max_io_workers) as io_executor,
    ):
        for idx in range(len(dataset)):
            tensor, filename = dataset[idx]  # Shape: (1, H, W)
            _, h, w = tensor.shape

            # Initialize empty output canvas and weight mask for blending
            canvas = torch.zeros((1, h, w), dtype=torch.float32)
            weight_mask = torch.zeros((1, h, w), dtype=torch.float32)

            # Create a 2D Hann/Cosine blending window for smooth tile stitching
            wy = np.hanning(patch_size)
            wx = np.hanning(patch_size)
            window_2d = torch.from_numpy(np.outer(wy, wx)).float()

            # Generate coordinate grid for sliding window
            y_coords = list(range(0, max(1, h - patch_size + stride), stride))
            x_coords = list(range(0, max(1, w - patch_size + stride), stride))

            # Ensure edge boundaries cover the full image dimensions
            if y_coords[-1] + patch_size < h:
                y_coords.append(h - patch_size)
            if x_coords[-1] + patch_size < w:
                x_coords.append(w - patch_size)

            patches_buffer = []
            coords_buffer = []

            for y in y_coords:
                for x in x_coords:
                    y_end = min(y + patch_size, h)
                    x_end = min(x + patch_size, w)

                    # Extract patch (handle boundary padding if necessary)
                    patch = tensor[:, y:y_end, x:x_end]
                    actual_h, actual_w = patch.shape[1], patch.shape[2]

                    if actual_h < patch_size or actual_w < patch_size:
                        # Pad patch to match standard patch_size for the network
                        padded_patch = torch.nn.functional.pad(
                            patch,
                            (0, patch_size - actual_w, 0, patch_size - actual_h),
                            mode="reflect",
                        )
                        patches_buffer.append(padded_patch)
                    else:
                        patches_buffer.append(patch)

                    coords_buffer.append((y, y_end, x, x_end, actual_h, actual_w))

                    # Process when batch is full or at the end of image tiles
                    if len(patches_buffer) == batch_size or (
                        y == y_coords[-1] and x == x_coords[-1]
                    ):
                        batch_tensor = torch.stack(patches_buffer).to(
                            device, non_blocking=True
                        )

                        with torch.autocast(
                            device_type=amp_device,
                            enabled=use_amp and device.type != "cpu",
                        ):
                            model_output = model(batch_tensor)
                            if predict_residual:
                                denoised_signal = batch_tensor - model_output
                            else:
                                denoised_signal = model_output
                            batch_out = torch.clamp(denoised_signal, 0.0, 1.0)

                        batch_out_cpu = batch_out.cpu()

                        for b_idx, (py, py_end, px, px_end, ah, aw) in enumerate(
                            coords_buffer
                        ):
                            p_out = batch_out_cpu[b_idx, :, :ah, :aw]
                            p_window = window_2d[:ah, :aw].unsqueeze(0)

                            canvas[:, py:py_end, px:px_end] += p_out * p_window
                            weight_mask[:, py:py_end, px:px_end] += p_window

                        patches_buffer.clear()
                        coords_buffer.clear()

            # Normalize by accumulated weights to remove seam lines
            weight_mask = torch.clamp(weight_mask, min=1e-5)
            final_tensor = canvas / weight_mask

            save_file_path = out_path / f"denoised_{filename}"
            io_executor.submit(_save_image_artifact, final_tensor, save_file_path)
            logging.info("Completed tiled inference for: %s", filename)

    logging.info("Inference complete. Exported artifacts to: %s", out_path.resolve())


def run_from_config(
    config_path: str | Path,
    test_dir: str | Path | None = None,
    ckpt_path: str | Path | None = None,
) -> None:
    """
    Resolve model, checkpoint and I/O paths from a YAML config and run inference.

    Kept free of argument parsing so it is safe to call from run_experiment.py; the
    previous version re-parsed sys.argv and inherited the caller's flags.
    """
    cfg = PipelineConfig.from_yaml(config_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    unet_config = UNetConfig(**cfg.unet.model_dump())
    model = ResidualUNet(unet_config)

    if ckpt_path is None:
        ckpt_path = cfg.paths.experiment_dir / "checkpoints" / "best_model.pt"
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint binary missing at {ckpt_path}")

    if test_dir is None:
        test_dir = cfg.paths.sandbox_root / "test" / "noisy"

    if not Path(test_dir).exists():
        raise FileNotFoundError(f"Test directory missing at {test_dir}")

    output_dir = cfg.paths.experiment_dir / "predictions"

    run_directory_inference(
        model=model,
        checkpoint_path=ckpt_path,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers_val,
        device=device,
        use_amp=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised Denoising Inference Engine"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Override input directory (default: <sandbox_root>/test/noisy)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint (default: <experiment_dir>/checkpoints/best_model.pt)",
    )
    args = parser.parse_args()
    run_from_config(
        config_path=args.config, test_dir=args.test_dir, ckpt_path=args.checkpoint
    )


if __name__ == "__main__":
    main()
