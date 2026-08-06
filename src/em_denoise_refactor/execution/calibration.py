import argparse
from pathlib import Path
import random
from typing import Optional
import numpy as np
import tifffile
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, median_filter

from em_denoise_refactor.config import PipelineConfig


def run_calibration(
    config: PipelineConfig,
    num_samples: int = 100,
    patch_size: int = 3,
    filter_type: Optional[str] = None,
    gaussian_sigma: Optional[float] = None,
    num_workers: int = 8,
    epsilon: float = 1e-6,
) -> Path:
    sbem_root = config.data.sbem_root
    all_tiffs = []

    # Determine filter settings (CLI overrides config)
    f_type = filter_type or getattr(config.noise, "filter_type", "gaussian")
    g_sigma = (
        gaussian_sigma
        if gaussian_sigma is not None
        else getattr(config.noise, "gaussian_sigma", 0.7)
    )

    def apply_filter(img: np.ndarray) -> np.ndarray:
        if f_type == "gaussian":
            return gaussian_filter(img, sigma=g_sigma)
        else:
            return median_filter(img, size=patch_size)

    # 1. Try querying tile filepaths directly from SQLite metadata store if available
    grid_str = f"g{config.data.grid_num:04d}"
    sqlite_candidates = [
        config.experiment.output_dir / f"dataset_metadata_{grid_str}.db",
        sbem_root / f"dataset_metadata_{grid_str}.db",
    ]
    db_file = next((p for p in sqlite_candidates if p.exists()), None)
    if db_file:
        try:
            import sqlite3

            conn = sqlite3.connect(str(db_file))
            cur = conn.cursor()
            cur.execute(
                """
                SELECT filepath FROM tiles 
                WHERE section_id >= ? AND section_id <= ? 
                ORDER BY RANDOM() 
                LIMIT ?
                """,
                (config.data.first_sec, config.data.last_sec, num_samples * 3),
            )
            rows = cur.fetchall()
            conn.close()
            for r in rows:
                p = sbem_root / r[0]
                if p.exists():
                    all_tiffs.append(p)
        except Exception:
            all_tiffs = []

    # 2. Fallback to lazy filesystem search with early termination (avoids stalling SMB mounts)
    if not all_tiffs and sbem_root.exists():
        target_limit = num_samples * 3
        for p in sbem_root.rglob("*.tif"):
            all_tiffs.append(p)
            if len(all_tiffs) >= target_limit:
                break
        if not all_tiffs:
            for p in sbem_root.rglob("*.tiff"):
                all_tiffs.append(p)
                if len(all_tiffs) >= target_limit:
                    break

    hist_2d = np.zeros((256, 256), dtype=np.float64)

    lmdb_store_path = config.experiment.output_dir / "patch_store.lmdb"

    sources_used = []
    filter_info = (
        f"filter={f_type} (sigma={g_sigma})"
        if f_type == "gaussian"
        else f"filter={f_type} (size={patch_size})"
    )
    print(f"Calibrating noise model using {filter_info} (workers={num_workers})...")

    def _process_tiff_file(path: Path) -> Optional[np.ndarray]:
        try:
            noisy_img = tifffile.imread(path).astype(np.float32)
            pseudo_clean = apply_filter(noisy_img)

            pseudo_clean = np.clip(np.round(pseudo_clean), 0, 255).astype(np.int32)
            noisy_img = np.clip(np.round(noisy_img), 0, 255).astype(np.int32)

            sub_hist = np.zeros((256, 256), dtype=np.float64)
            np.add.at(sub_hist, (pseudo_clean, noisy_img), 1)
            return sub_hist
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None

    if all_tiffs:
        print(
            f"Found {len(all_tiffs)} candidate .tif tiles in sections {config.data.first_sec}–{config.data.last_sec}. Randomly sampling {min(num_samples, len(all_tiffs))}..."
        )
        sampled_tiffs = random.sample(all_tiffs, min(num_samples, len(all_tiffs)))
        sources_used = [str(p) for p in sampled_tiffs]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(_process_tiff_file, sampled_tiffs),
                    total=len(sampled_tiffs),
                    desc="Calibrating Noise Model from TIFFs",
                )
            )

        for res in results:
            if res is not None:
                hist_2d += res
    elif lmdb_store_path.exists():
        import lmdb

        print(
            f"No .tif files accessible. Found LMDB store at {lmdb_store_path}. Sampling from LMDB..."
        )
        env = lmdb.open(str(lmdb_store_path), readonly=True, lock=False)
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            all_keys = [k for k, _ in cursor if k != b"len"]

        if not all_keys:
            raise ValueError(f"LMDB store at {lmdb_store_path} is empty.")

        sampled_keys = random.sample(all_keys, min(num_samples, len(all_keys)))
        sources_used = [f"lmdb://{k.decode('utf-8')}" for k in sampled_keys]
        with env.begin(write=False) as txn:
            for key in tqdm(sampled_keys, desc="Calibrating Noise Model from LMDB"):
                buf = txn.get(key)
                side_len = int(np.sqrt(len(buf)))
                noisy_img = (
                    np.frombuffer(buf, dtype=np.uint8)
                    .reshape(side_len, side_len)
                    .astype(np.float32)
                )
                pseudo_clean = apply_filter(noisy_img)

                pseudo_clean = np.clip(np.round(pseudo_clean), 0, 255).astype(np.int32)
                noisy_img = np.clip(np.round(noisy_img), 0, 255).astype(np.int32)

                np.add.at(hist_2d, (pseudo_clean, noisy_img), 1)
        env.close()
    else:
        raise ValueError(
            f"No .tif files found in {sbem_root} and no LMDB patch store found at {lmdb_store_path}."
        )

    hist_2d += epsilon
    row_sums = hist_2d.sum(axis=1, keepdims=True)
    p_x_given_s = hist_2d / row_sums

    out_path = Path(config.noise.empirical_model_path)
    if not out_path.is_absolute():
        out_path = config.experiment.output_dir / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, p_x_given_s.astype(np.float32))

    sources_path = out_path.parent / f"{out_path.stem}_sources.txt"
    with open(sources_path, "w", encoding="utf-8") as f:
        f.write(f"# Filter: {filter_info}\n")
        f.write("\n".join(sources_used))
    print(f"Recorded {len(sources_used)} source items to {sources_path}")

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
        "--filter",
        type=str,
        default=None,
        choices=["gaussian", "median"],
        help="Filter type for pseudo-clean reference (default: from config or gaussian)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian sigma for pseudo-clean reference (default: 0.7)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=3,
        help="Median filter patch size (if filter=median)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for I/O and filtering (default: 8)",
    )
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    cfg = PipelineConfig.from_yaml(args.config)

    out = run_calibration(
        cfg,
        num_samples=args.samples,
        patch_size=args.patch_size,
        filter_type=args.filter,
        gaussian_sigma=args.sigma,
        num_workers=args.workers,
    )
    print(f"Calibration complete. Empirical model saved to {out}")
