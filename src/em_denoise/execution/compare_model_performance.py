import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import sobel

from em_denoise.backend.architecture import ResidualUNet, UNetConfig
from em_denoise.backend.config_schema import PipelineConfig
from em_denoise.backend.dataset import SupervisedEMDataset
from em_denoise.backend.metrics import compute_psnr, compute_ssim


def calculate_espi(raw: np.ndarray, denoised: np.ndarray) -> float:
    """Computes edge preservation index"""
    raw_f, den_f = raw.astype(np.float32), denoised.astype(np.float32)

    grad_raw = np.hypot(sobel(raw_f, axis=1), sobel(raw_f, axis=0))
    grad_den = np.hypot(sobel(den_f, axis=1), sobel(den_f, axis=0))

    threshold = np.percentile(grad_raw, 90)
    edge_mask = grad_raw > threshold
    if not np.any(edge_mask):
        return 0.0

    return float(np.mean(grad_den[edge_mask]) / (np.mean(grad_raw[edge_mask]) + 1e-8))


def compute_aufrc(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes Area Under the Fourier Ring Correlation (AUFRC) for spectral fidelity."""
    h, w = pred.shape
    f_pred = np.fft.fftshift(np.fft.fft2(pred))
    f_gt = np.fft.fftshift(np.fft.fft2(target))

    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r_grid = np.hypot(y - cy, x - cx).astype(int)

    r_max = min(cy, cx)
    frc_vals = []

    for r in range(1, r_max):
        mask = r_grid == r
        if not np.any(mask):
            continue
        p_ring = f_pred[mask]
        g_ring = f_gt[mask]

        num = np.real(np.sum(p_ring * np.conj(g_ring)))
        den = np.sqrt(np.sum(np.abs(p_ring) ** 2) * np.sum(np.abs(g_ring) ** 2))

        frc_r = num / den if den > 0 else 0.0
        frc_vals.append(np.clip(frc_r, 0.0, 1.0))

    return float(np.mean(frc_vals)) if frc_vals else 0.0


def resolve_checkpoints(
    cfg: PipelineConfig,
    explicit_paths: list[Path] | None = None,
    custom_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Ingestion cascade:
    1. CLI explicit file list (--checkpoints)
    2. Config schema `checkpoints` block
    3. Custom directory override (--ckpt-dir)
    4. Default experiment directory globbing (`cfg.paths.experiment_dir / "checkpoints"`)
    """
    ckpt_map: dict[str, Path] = {}

    # Priority 1: Direct CLI arguments
    if explicit_paths:
        for path in explicit_paths:
            if path.exists():
                ckpt_map[path.stem] = path
            else:
                print(f"[!] Warning: Specified CLI checkpoint missing: {path}")
        return ckpt_map

    # Priority 2: Config YAML `checkpoints` block
    cfg_checkpoints = getattr(cfg, "checkpoints", None)
    if cfg_checkpoints:
        for tag, path_str in cfg_checkpoints.items():
            path = Path(path_str)
            if path.exists():
                ckpt_map[tag] = path
            else:
                print(f"[!] Warning: Config checkpoint missing [{tag}]: {path}")
        if ckpt_map:
            return ckpt_map

    # Priority 3 & 4: Directory globbing
    search_dir = custom_dir if custom_dir else cfg.paths.experiment_dir / "checkpoints"
    if search_dir.exists():
        discovered = sorted(search_dir.glob("*.pt")) + sorted(search_dir.glob("*.pth"))
        for path in discovered:
            ckpt_map[path.stem] = path

    return ckpt_map


def _plot_metric_distributions(df_metrics: pd.DataFrame, save_path: Path) -> None:
    """Renders visual comparative distributions (PSNR, SSIM, AUFRC, ESPI) per model variant."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    metrics = [
        ("PSNR", "PSNR (dB)"),
        ("SSIM", "SSIM Index"),
        ("AUFRC", "AUFRC Score"),
        ("ESPI", "Edges Score"),
    ]

    for idx, (metric_key, label) in enumerate(metrics):
        sns.boxplot(
            data=df_metrics,
            x="Model",
            y=metric_key,
            hue="Model",
            legend=False,
            ax=axes[idx],
            palette="Set2",
            boxprops=dict(alpha=0.7),
        )
        sns.stripplot(
            data=df_metrics,
            x="Model",
            y=metric_key,
            ax=axes[idx],
            color="black",
            alpha=0.4,
            jitter=0.2,
            size=4,
        )
        axes[idx].set_title(f"{metric_key} Distribution")
        axes[idx].set_ylabel(label)
        axes[idx].set_xlabel("Model Tag")
        axes[idx].tick_params(axis="x", rotation=15)
        axes[idx].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


@torch.inference_mode()
def run_benchmark(
    config_path: Path,
    explicit_ckpts: list[Path] | None = None,
    ckpt_dir_override: Path | None = None,
) -> None:
    cfg = PipelineConfig.from_yaml(config_path)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[*] Target device: {device}")

    # Resolve targeted checkpoint paths from config or CLI overrides
    ckpt_map = resolve_checkpoints(
        cfg=cfg,
        explicit_paths=explicit_ckpts,
        custom_dir=ckpt_dir_override,
    )

    if not ckpt_map:
        raise RuntimeError("Zero valid checkpoint binaries resolved for benchmarking.")

    print(f"[*] Registered {len(ckpt_map)} model target(s):")
    for tag, path in ckpt_map.items():
        print(f"    - [{tag}]: {path}")

    # Shared Dataset and DataLoader
    test_dataset = SupervisedEMDataset(
        data_dir=cfg.paths.sandbox_root / "test", augment=False, cache_in_ram=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.dataset.num_workers_val,
        pin_memory=(device.type == "cuda"),
    )

    # Store tuple of (model, predict_residual_flag) to support mixed-architecture benchmarks
    loaded_models: dict[str, tuple[ResidualUNet, bool]] = {}

    for tag, path in ckpt_map.items():
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            predict_residual = checkpoint.get("predict_residual", True)
        else:
            state_dict = checkpoint
            predict_residual = True

        # Check if the state_dict actually contains parameterized downsampling weights
        has_downsample_weights = any(
            "down_samples" in k and "conv" in k for k in state_dict.keys()
        )

        # Override the UNet config dynamically for this specific checkpoint. The encoder
        # switched from max-pooling to strided convs, so the architecture is chosen from
        # what the checkpoint actually contains rather than from the YAML.
        cfg_dict = cfg.unet.model_dump()
        cfg_dict["use_conv_downsample"] = has_downsample_weights
        active_unet_cfg = UNetConfig(**cfg_dict)

        model = ResidualUNet(active_unet_cfg).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        loaded_models[tag] = (model, predict_residual)

    out_dir = cfg.paths.experiment_dir / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_records = []

    print(f"[*] Benchmarking on test set ({len(test_dataset)} samples)...")
    for idx, (noisy, clean) in enumerate(test_loader):
        noisy_dev = noisy.to(device=device, non_blocking=True)
        clean_dev = clean.to(device=device, non_blocking=True)

        clean_np = clean_dev[0, 0].cpu().numpy()
        panel_tensors = [noisy_dev[0, 0]]

        # Unpack both the model and its specific residual flag
        for tag, (model, predict_residual) in loaded_models.items():
            model_output = model(noisy_dev)

            # Rank 2: Signal Reconstruction
            if predict_residual:
                denoised_signal = noisy_dev - model_output
            else:
                denoised_signal = model_output

            denoised = torch.clamp(denoised_signal, 0.0, 1.0)

            psnr_val = compute_psnr(denoised, clean_dev)
            ssim_val = compute_ssim(denoised, clean_dev)

            psnr_scalar = (
                psnr_val.item() if isinstance(psnr_val, torch.Tensor) else psnr_val
            )
            ssim_scalar = (
                ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val
            )

            denoised_np = denoised[0, 0].cpu().numpy()
            aufrc_scalar = compute_aufrc(denoised_np, clean_np)
            espi_scalar = calculate_espi(clean_np, denoised_np)

            metrics_records.append(
                {
                    "Model": tag,
                    "Sample_ID": idx,
                    "PSNR": psnr_scalar,
                    "SSIM": ssim_scalar,
                    "AUFRC": aufrc_scalar,
                    "ESPI": espi_scalar,
                }
            )

            panel_tensors.append(denoised[0, 0])

        panel_tensors.append(clean_dev[0, 0])

        if idx < 10:
            panel = torch.cat(panel_tensors, dim=1)
            panel_np = (panel.cpu().numpy() * 255.0).astype(np.uint8)
            tifffile.imwrite(
                str(out_dir / f"sample_{idx:03d}_comparative_panel.tif"), panel_np
            )

    df_metrics = pd.DataFrame(metrics_records)

    # Plot metric distributions
    plot_path = out_dir / "metric_comparisons.png"
    _plot_metric_distributions(df_metrics, plot_path)
    print(f"[*] Benchmark distribution plot written: {plot_path}")

    # Console summary output
    summary_data = []
    for tag in loaded_models.keys():
        sub_df = df_metrics[df_metrics["Model"] == tag]
        summary_data.append(
            {
                "Model": tag,
                "PSNR (dB)": f"{sub_df['PSNR'].mean():.2f} ± {sub_df['PSNR'].std():.2f}",
                "SSIM": f"{sub_df['SSIM'].mean():.3f} ± {sub_df['SSIM'].std():.3f}",
                "AUFRC": f"{sub_df['AUFRC'].mean():.3f} ± {sub_df['AUFRC'].std():.3f}",
                "ESPI": f"{sub_df['ESPI'].mean():.3f} ± {sub_df['ESPI'].std():.3f}",
            }
        )

    summary_df = pd.DataFrame(summary_data).set_index("Model")

    print("\n" + "=" * 60)
    print("           IDENTICAL DATASET BENCHMARK SUMMARY           ")
    print("=" * 60)
    try:
        print(summary_df.to_markdown())
    except ImportError:
        print(summary_df.to_string())
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark explicit model checkpoints defined in config.yaml or CLI."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        help="CLI override: space-separated paths to specific checkpoint files (.pt/.pth)",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        help="CLI override: directory containing checkpoint binaries to discover automatically",
    )
    args = parser.parse_args()

    run_benchmark(
        config_path=args.config,
        explicit_ckpts=args.checkpoints,
        ckpt_dir_override=args.ckpt_dir,
    )
