# src/em_denoise_refactor/execution/run_classical_baselines.py

"""
Run classical denoising baselines (Gaussian, Bilateral, Non-Local Means) on raw
EM TIFFs and/or evaluate them on the held-out N2N duplicate-tile test split.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_bilateral, denoise_nl_means
import tifffile

from em_denoise_refactor.config import PipelineConfig
from em_denoise_refactor.execution.train_and_infer import (
    _build_n2n_duplicate_dataset,
    compute_noise_corrected_metrics,
    save_comparison_panel,
)

logger = logging.getLogger("classical_baselines")


def _estimate_noise_sigma(img: np.ndarray) -> float:
    """Wavelet-free 2D Haar diagonal MAD noise std estimator (Donoho & Johnstone)."""
    h, w = img.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    if h2 < 2 or w2 < 2:
        return max(float(np.std(img)), 1e-3)
    hh = 0.5 * (
        img[:h2:2, :w2:2]
        - img[:h2:2, 1:w2:2]
        - img[1:h2:2, :w2:2]
        + img[1:h2:2, 1:w2:2]
    )
    sigma = float(np.median(np.abs(hh)) / 0.67448975)
    return max(sigma, 1e-3)


def _denoise_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    return np.clip(
        gaussian_filter(img.astype(np.float32), sigma=sigma), 0.0, 1.0
    ).astype(np.float32)


def _denoise_bilateral(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    sigma_est = _estimate_noise_sigma(img_f)
    out = denoise_bilateral(
        img_f,
        sigma_color=sigma_est * 1.5,
        sigma_spatial=2.0,
        channel_axis=None,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _denoise_nlm(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    sigma_est = _estimate_noise_sigma(img_f)
    out = denoise_nl_means(
        img_f,
        h=1.15 * sigma_est,
        sigma=sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        channel_axis=None,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _get_baseline_methods() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "gaussian_0.8": lambda x: _denoise_gaussian(x, sigma=0.8),
        "gaussian_1.2": lambda x: _denoise_gaussian(x, sigma=1.2),
        "bilateral": _denoise_bilateral,
        "nlm": _denoise_nlm,
    }


def _load_tif_float01(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim > 2:
        arr = arr[0]
    arr_f = arr.astype(np.float32)
    if arr.dtype == np.uint8 or arr_f.max() > 1.0:
        arr_f = arr_f / 255.0
    return np.clip(arr_f, 0.0, 1.0)


def run_on_input_dir(
    input_dir: Path,
    output_dir: Path,
    methods: Dict[str, Callable[[np.ndarray], np.ndarray]],
) -> List[Path]:
    """Run all classical baselines on .tif files in input_dir and save TIFFs + 3-panel PNGs."""
    tif_paths = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))
    if not tif_paths:
        raise FileNotFoundError(f"No .tif/.tiff files found in {input_dir}")

    written: List[Path] = []
    for method_name, fn in methods.items():
        method_dir = output_dir / method_name
        comp_dir = method_dir / "comparisons"
        method_dir.mkdir(parents=True, exist_ok=True)
        comp_dir.mkdir(parents=True, exist_ok=True)

        for p in tif_paths:
            raw_f = _load_tif_float01(p)
            den_f = fn(raw_f)
            den_u8 = np.clip(den_f * 255.0, 0, 255).astype(np.uint8)
            out_tif = method_dir / p.name
            tifffile.imwrite(str(out_tif), den_u8)
            written.append(out_tif)

            save_comparison_panel(
                raw_255=raw_f * 255.0,
                denoised_255=den_f * 255.0,
                out_path=comp_dir / f"{p.stem}_comparison.png",
                denoised_title=f"{method_name}",
            )
    logger.info(
        f"Saved classical baseline outputs for {len(tif_paths)} images across "
        f"{len(methods)} methods to {output_dir}"
    )
    return written


def evaluate_on_duplicate_test_split(
    config_path: Path,
    output_dir: Path,
    methods: Dict[str, Callable[[np.ndarray], np.ndarray]],
) -> Dict[str, Dict[str, float]]:
    """Evaluate classical baselines on the held-out N2N duplicate test split."""
    cfg = PipelineConfig.from_yaml(config_path)
    if getattr(cfg.n2n, "source", "overlaps").lower().strip() != "duplicates":
        raise ValueError(
            "Classical baseline test-split evaluation requires n2n.source == 'duplicates'."
        )

    test_ds = _build_n2n_duplicate_dataset(
        cfg=cfg,
        split="test",
        max_pairs=cfg.n2n.max_test_pairs,
        augment=False,
    )

    raw_pair_mses: List[float] = []
    pred_mses: Dict[str, List[float]] = {name: [] for name in methods}

    for idx in range(len(test_ds)):
        inp_t, tgt_t, _ = test_ds[idx]
        inp_np = inp_t[0].numpy()
        tgt_np = tgt_t[0].numpy()
        raw_pair_mses.append(float(np.mean((inp_np - tgt_np) ** 2)))
        for name, fn in methods.items():
            den_np = fn(inp_np)
            pred_mses[name].append(float(np.mean((den_np - tgt_np) ** 2)))

    mean_raw_mse = float(np.mean(raw_pair_mses))
    summary: Dict[str, Dict[str, float]] = {}
    for name in methods:
        mean_pred_mse = float(np.mean(pred_mses[name]))
        est_clean_mse, est_clean_psnr = compute_noise_corrected_metrics(
            mean_pred_mse, mean_raw_mse
        )
        summary[name] = {
            "raw_pair_mse": mean_raw_mse,
            "pred_mse": mean_pred_mse,
            "est_clean_mse": est_clean_mse,
            "est_clean_psnr": est_clean_psnr,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "classical_baselines_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved classical baseline test summary to {summary_path}")
    return summary


def run_baselines(
    output_dir: Path,
    input_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Run requested classical baselines on input_dir and/or config test split."""
    if input_dir is None and config_path is None:
        raise ValueError("Provide at least one of --input-dir or --config.")

    methods = _get_baseline_methods()
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_dir is not None:
        run_on_input_dir(input_dir=input_dir, output_dir=output_dir, methods=methods)
    if config_path is not None:
        return evaluate_on_duplicate_test_split(
            config_path=config_path, output_dir=output_dir, methods=methods
        )
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classical EM denoising baselines (Gaussian, Bilateral, NLM)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of raw .tif images to denoise.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for baseline TIFFs, comparison PNGs, and summary JSON.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional duplicate-tile YAML config to evaluate on held-out N2N test pairs.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    summary = run_baselines(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        config_path=args.config,
    )
    if summary:
        for name, metrics in summary.items():
            logger.info(
                f"{name:14s} | pred_mse={metrics['pred_mse']:.6f} | "
                f"est_clean_mse={metrics['est_clean_mse']:.6f} | "
                f"est_clean_psnr={metrics['est_clean_psnr']:.2f} dB"
            )


if __name__ == "__main__":
    main()
