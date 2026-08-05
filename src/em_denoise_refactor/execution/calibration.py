import argparse
from pathlib import Path
import random
import numpy as np
import tifffile
from tqdm import tqdm
from scipy.ndimage import median_filter

from em_denoise_refactor.config import PipelineConfig


def run_calibration(
    config: PipelineConfig,
    num_samples: int = 100,
    patch_size: int = 3,
    epsilon: float = 1e-6,
) -> Path:
    sbem_root = config.data.sbem_root

    # print(f"Searching for .tif files in {sbem_root}...")
    all_tiffs = list(sbem_root.rglob("*.tif"))
    if not all_tiffs:
        all_tiffs = list(sbem_root.rglob("*.tiff"))

    if not all_tiffs:
        raise ValueError(f"No .tif files found in {sbem_root}")

    # print(f"Found {len(all_tiffs)} files. Sampling {min(num_samples, len(all_tiffs))}...")
    sampled_tiffs = random.sample(all_tiffs, min(num_samples, len(all_tiffs)))

    hist_2d = np.zeros((256, 256), dtype=np.float64)

    for path in tqdm(sampled_tiffs, desc="Calibrating Noise Model"):
        try:
            noisy_img = tifffile.imread(path).astype(np.float32)
            pseudo_clean = median_filter(noisy_img, size=patch_size)

            pseudo_clean = np.clip(np.round(pseudo_clean), 0, 255).astype(np.int32)
            noisy_img = np.clip(np.round(noisy_img), 0, 255).astype(np.int32)

            np.add.at(hist_2d, (pseudo_clean, noisy_img), 1)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    hist_2d += epsilon
    row_sums = hist_2d.sum(axis=1, keepdims=True)
    p_x_given_s = hist_2d / row_sums

    out_path = Path(config.noise.empirical_model_path)
    if not out_path.is_absolute():
        out_path = config.experiment.output_dir / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, p_x_given_s.astype(np.float32))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate empirical noise model for SBF-SEM"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of files to sample for calibration",
    )
    parser.add_argument(
        "--patch_size", type=int, default=3, help="Median filter patch size"
    )
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    cfg = PipelineConfig.from_yaml(args.config)

    out = run_calibration(cfg, num_samples=args.samples, patch_size=args.patch_size)
    print(f"Calibration complete. Empirical model saved to {out}")
