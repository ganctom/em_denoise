# import argparse
# import os
# import torch
# import numpy as np
# from pathlib import Path
# from PIL import Image
# import torchvision.transforms.functional as TF
# from torch.utils.data import Dataset, DataLoader
# from config_schema import PipelineConfig
# from architecture import ResidualUNet, UNetConfig
#
#
# # ==========================================
# # 1. Custom Image Dataset for Testing
# # ==========================================
# class SingleDirectoryImageDataset(Dataset):
#     """
#     Loads grayscale/single-channel images from a test directory.
#     Supports .png, .tif, .tiff, .jpg, and .npy arrays.
#     """
#
#     def __init__(self, image_dir: str | Path, transform=None):
#         self.image_dir = Path(image_dir)
#         self.transform = transform
#
#         # Collect supported image files
#         valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
#         self.image_paths = sorted([
#             p for p in self.image_dir.iterdir()
#             if p.suffix.lower() in valid_extensions
#         ])
#
#         if not self.image_paths:
#             raise FileNotFoundError(f"No valid image files found in {self.image_dir}")
#
#     def __len__(self) -> int:
#         return len(self.image_paths)
#
#     def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
#         img_path = self.image_paths[idx]
#
#         if img_path.suffix.lower() == ".npy":
#             # Direct NumPy load (preserves float32/float64 precision)
#             arr = np.load(img_path).astype(np.float32)
#             tensor = torch.from_numpy(arr)
#             if tensor.ndim == 2:
#                 tensor = tensor.unsqueeze(0)  # Shape: (1, H, W)
#         else:
#             # Standard image load via PIL
#             pil_img = Image.open(img_path).convert("L")  # Convert to 1-channel grayscale
#             tensor = TF.to_tensor(pil_img)  # Scales uint8 [0, 255] -> float32 [0.0, 1.0]
#
#         return tensor, img_path.name
#
#
# # ==========================================
# # 2. Batch Inference Function
# # ==========================================
# def run_directory_inference(
#         model: torch.nn.Module,
#         checkpoint_path: str | Path,
#         test_dir: str | Path,
#         output_dir: str | Path,
#         batch_size: int = 16,
#         num_workers: int = 2,
#         device: str = "cuda" if torch.cuda.is_available() else "cpu"
# ):
#     """
#     Runs batched inference on all images in test_dir and saves outputs to output_dir.
#     """
#     test_path = Path(test_dir)
#     out_path = Path(output_dir)
#     out_path.mkdir(parents=True, exist_ok=True)
#
#     # Load Model Weights
#     print(f"Loading checkpoint from: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     state_dict = checkpoint.get("model_state_dict", checkpoint)
#     model.load_state_dict(state_dict)
#
#     model.to(device)
#     model.eval()
#
#     # Instantiate DataLoader
#     dataset = SingleDirectoryImageDataset(test_path)
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=(device == "cuda")
#     )
#
#     print(f"Executing inference on {len(dataset)} images via {device}...")
#
#     with torch.no_grad():
#         for batch_tensors, filenames in loader:
#             batch_tensors = batch_tensors.to(device)
#
#             # UNet forward pass
#             outputs = model(batch_tensors)
#
#             # Post-processing: Bound output to valid intensity range [0, 1]
#             outputs = torch.clamp(outputs, 0.0, 1.0)
#
#             # Serialization to disk
#             for i in range(outputs.size(0)):
#                 out_tensor = outputs[i].cpu()
#                 filename = filenames[i]
#
#                 # Convert (1, H, W) tensor back to uint8 image
#                 out_np = (out_tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
#                 out_pil = Image.fromarray(out_np)
#
#                 # Save processed file
#                 save_file_path = out_path / f"denoised_{filename}"
#                 out_pil.save(save_file_path)
#
#     print(f"Inference complete. Exported results to: {out_path.resolve()}")
#
#
#
# if __name__ == "__main__":
#
#     # 1. Parse configuration
#     parser = argparse.ArgumentParser(description="Run Supervised Denoising Inference Pipeline")
#     parser.add_argument(
#         "--config",
#         type=Path,
#         default=Path("config.yaml"),
#         help="Path to YAML configuration file",
#     )
#     args = parser.parse_args()
#     cfg = PipelineConfig.from_yaml(args.config)
#
#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
#     )
#
#     unet_config = UNetConfig(**cfg.unet.model_dump())
#     model = ResidualUNet(unet_config).to(device)
#
#     # CHECKPOINT_PATH
#     ckpt_path = cfg.paths.experiment_dir / "checkpoints" / "best_model.pt"
#     if not ckpt_path.exists():
#         raise FileNotFoundError(f"Checkpoint binary missing at {ckpt_path}")
#
#     # SANDBOX_DIR
#     test_dir = cfg.paths.sandbox_root / "test" / "noisy"
#     if not test_dir.exists():
#         raise FileNotFoundError(f"Test dir missing at {test_dir}")
#
#     # OUTPUT_DIR
#     output_dir = cfg.paths.experiment_dir / "predictions"
#
#     # 3. Execute Pipeline
#     run_directory_inference(
#         model=model,
#         checkpoint_path=ckpt_path,
#         test_dir=test_dir,
#         output_dir=output_dir,
#         batch_size=cfg.dataset.batch_size,
#         num_workers=cfg.dataset.num_workers_val
#     )


import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
) -> None:
    """
    Executes batched neural network inference over input artifacts in test_dir.
    """
    test_path = Path(test_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logging.info("Deserializing checkpoint from: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    dataset = SingleDirectoryImageDataset(test_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    logging.info(
        "Executing inference on %d samples via target device: %s", len(dataset), device
    )

    amp_device = (
        "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    )

    with (
        torch.inference_mode(),
        ThreadPoolExecutor(max_workers=max_io_workers) as io_executor,
    ):
        for batch_tensors, filenames in loader:
            batch_tensors = batch_tensors.to(device, non_blocking=True)

            with torch.autocast(
                device_type=amp_device, enabled=use_amp and device.type != "cpu"
            ):
                outputs = model(batch_tensors)
                outputs = torch.clamp(outputs, 0.0, 1.0)

            outputs_cpu = outputs.cpu()

            for i in range(outputs_cpu.size(0)):
                out_tensor = outputs_cpu[i]
                filename = filenames[i]
                save_file_path = out_path / f"denoised_{filename}"

                io_executor.submit(_save_image_artifact, out_tensor, save_file_path)

    logging.info("Inference complete. Exported artifacts to: %s", out_path.resolve())


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
    args = parser.parse_args()
    cfg = PipelineConfig.from_yaml(args.config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    unet_config = UNetConfig(**cfg.unet.model_dump())
    model = ResidualUNet(unet_config)

    ckpt_path = cfg.paths.experiment_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint binary missing at {ckpt_path}")

    test_dir = cfg.paths.sandbox_root / "test" / "noisy"
    if not test_dir.exists():
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


if __name__ == "__main__":
    main()
