import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader

from em_denoise.backend.architecture import ResidualUNet, UNetConfig
from em_denoise.backend.config_schema import PipelineConfig
from em_denoise.backend.dataset import SupervisedEMDataset
from em_denoise.backend.metrics import compute_psnr, compute_ssim


@torch.inference_mode()
def evaluate_test_set(config_path: Path) -> None:
    cfg = PipelineConfig.from_yaml(config_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    test_dataset = SupervisedEMDataset(
        data_dir=cfg.paths.sandbox_root / "test", augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    unet_config = UNetConfig(**cfg.unet.model_dump())
    model = ResidualUNet(unet_config).to(device)

    ckpt_path = cfg.paths.experiment_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint binary missing at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    out_vis_dir = cfg.paths.experiment_dir / "test_visualizations"
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    psnr_scores, ssim_scores = [], []

    print(f"[*] Running test evaluation on {len(test_dataset)} samples...")
    for idx, (noisy, clean) in enumerate(test_loader):
        noisy_dev, clean_dev = (
            noisy.to(device, non_blocking=True),
            clean.to(device, non_blocking=True),
        )
        denoised = torch.clamp(model(noisy_dev), 0.0, 1.0)

        # Explicit scalar extraction to prevent device-bound tensor aggregation in NumPy
        psnr_val = compute_psnr(denoised, clean_dev)
        ssim_val = compute_ssim(denoised, clean_dev)

        psnr_scores.append(
            psnr_val.item() if isinstance(psnr_val, torch.Tensor) else psnr_val
        )
        ssim_scores.append(
            ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val
        )

        if idx < 10:
            panel = torch.cat([noisy_dev[0, 0], denoised[0, 0], clean_dev[0, 0]], dim=1)
            panel_np = (panel.cpu().numpy() * 255.0).astype(np.uint8)
            tifffile.imwrite(
                str(out_vis_dir / f"test_sample_{idx:03d}_panel.tif"), panel_np
            )

    print("\n" + "=" * 40)
    print("           TEST SET RESULTS           ")
    print("=" * 40)
    print(f"Mean PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")
    print(f"Mean SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"Visual panels exported to: {out_vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Supervised Test Performance")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    evaluate_test_set(args.config)
