"""Phase 3 scientific diagnostics for True 2.5D Noise2Noise models (C1 & H6).

Provides:
  1. Paired per-patch bootstrap CI on ΔMSE and ΔEstPSNR between two models on
     the exact same frozen evaluation set (C1).
  2. Neighbor-ablation & deployment-fallback evaluation (`full` vs `center_filled`) (C1, H6).
  3. Misregistration sensitivity curve under synthetic neighbor shifts (2, 4, 8 px) (H6).
  4. Z-ghosting probe:
     - Synthetic 40 nm (~4 px) vesicle blob injected ONLY in z+1 -> peak/mean bleed into denoised(z).
     - Real data cross-slice correlation: corr(denoised_z, raw_{z+-1}) vs corr(raw_z, raw_{z+-1}).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from em_denoise_refactor.execution.train_and_infer import (
    compute_noise_corrected_metrics,
)


def paired_bootstrap_mse_ci(
    mse_baseline: np.ndarray,
    mse_candidate: np.ndarray,
    raw_pair_mse: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute paired bootstrap CI on (mse_candidate - mse_baseline) and ΔEstPSNR."""
    b = np.asarray(mse_baseline, dtype=np.float64)
    c = np.asarray(mse_candidate, dtype=np.float64)
    r = np.asarray(raw_pair_mse, dtype=np.float64)
    if not (b.shape == c.shape == r.shape and b.ndim == 1 and len(b) > 0):
        raise ValueError(
            f"Expected matching 1D non-empty arrays, got {b.shape}, {c.shape}, {r.shape}"
        )

    n = len(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    b_means = b[idx].mean(axis=1)
    c_means = c[idx].mean(axis=1)
    r_means = r[idx].mean(axis=1)

    delta_mse = c_means - b_means
    b_est_mse = np.maximum(b_means - 0.5 * r_means, 1e-10)
    c_est_mse = np.maximum(c_means - 0.5 * r_means, 1e-10)
    delta_est_psnr = (-10.0 * np.log10(c_est_mse)) - (-10.0 * np.log10(b_est_mse))

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    obs_b_est = compute_noise_corrected_metrics(float(b.mean()), float(r.mean()))[1]
    obs_c_est = compute_noise_corrected_metrics(float(c.mean()), float(r.mean()))[1]

    return {
        "n_samples": int(n),
        "baseline_mse": float(b.mean()),
        "candidate_mse": float(c.mean()),
        "delta_mse": float(c.mean() - b.mean()),
        "delta_mse_ci_lo": float(np.percentile(delta_mse, lo_q)),
        "delta_mse_ci_hi": float(np.percentile(delta_mse, hi_q)),
        "baseline_est_psnr": float(obs_b_est),
        "candidate_est_psnr": float(obs_c_est),
        "delta_est_psnr": float(obs_c_est - obs_b_est),
        "delta_est_psnr_ci_lo": float(np.percentile(delta_est_psnr, lo_q)),
        "delta_est_psnr_ci_hi": float(np.percentile(delta_est_psnr, hi_q)),
        "p_candidate_better": float(np.mean(delta_mse < 0.0)),
    }


def _eval_with_transform(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    transform_fn,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Run forward pass over dataset applying `transform_fn(inp_b)` and compute EstPSNR."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pred_mses: List[float] = []
    raw_mses: List[float] = []
    seen = 0

    with torch.no_grad():
        for inp_b, tgt_b, _ in loader:
            if max_samples is not None and seen >= max_samples:
                break
            if max_samples is not None and seen + inp_b.size(0) > max_samples:
                keep = max_samples - seen
                inp_b = inp_b[:keep]
                tgt_b = tgt_b[:keep]
            inp_b = transform_fn(inp_b).to(device)
            tgt_b = tgt_b.to(device)
            pred_b = torch.clamp(model(inp_b).float(), 0.0, 1.0)
            c_center = inp_b.shape[1] // 2
            center_b = inp_b[:, c_center : c_center + 1].float()
            tgt_f = tgt_b.float()

            pm = torch.mean((pred_b - tgt_f) ** 2, dim=(1, 2, 3)).cpu().numpy()
            rm = torch.mean((center_b - tgt_f) ** 2, dim=(1, 2, 3)).cpu().numpy()
            pred_mses.extend(pm.tolist())
            raw_mses.extend(rm.tolist())
            seen += inp_b.size(0)

    mean_pred = float(np.mean(pred_mses))
    mean_raw = float(np.mean(raw_mses))
    est_mse, est_psnr = compute_noise_corrected_metrics(mean_pred, mean_raw)
    return {
        "n_samples": len(pred_mses),
        "pred_mse": mean_pred,
        "raw_pair_mse": mean_raw,
        "est_clean_mse": est_mse,
        "est_clean_psnr": est_psnr,
    }


def run_25d_diagnostics(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    shift_px_list: Sequence[int] = (2, 4, 8),
    batch_size: int = 32,
    max_samples: Optional[int] = 256,
    ghost_blob_sigma: float = 1.5,
    ghost_blob_amp: float = -0.35,
) -> Dict[str, object]:
    """Run H6 scientific diagnostics on a trained 2.5D model and evaluation dataset."""
    model.eval()
    model.to(device)

    # 1. Full context vs neighbor-ablated (center_filled)
    full_metrics = _eval_with_transform(
        model,
        dataset,
        device,
        lambda x: x,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    def _center_fill(x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 2
        return x[:, c : c + 1].repeat(1, x.shape[1], 1, 1)

    ablated_metrics = _eval_with_transform(
        model,
        dataset,
        device,
        _center_fill,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    # 2. Misregistration sensitivity curve
    misreg: Dict[str, Dict[str, float]] = {}
    for shift_px in shift_px_list:

        def _shift_neighbors(x: torch.Tensor, s: int = int(shift_px)) -> torch.Tensor:
            out = x.clone()
            c = out.shape[1] // 2
            for ch in range(out.shape[1]):
                if ch != c:
                    out[:, ch] = torch.roll(out[:, ch], shifts=(s, s), dims=(-2, -1))
            return out

        misreg[f"shift_{shift_px}px"] = _eval_with_transform(
            model,
            dataset,
            device,
            _shift_neighbors,
            batch_size=batch_size,
            max_samples=max_samples,
        )

    # 3. Synthetic single-slice Z-ghosting probe & real cross-slice correlation
    n_probe = min(len(dataset), max_samples if max_samples is not None else 64, 64)
    ghost_peaks: List[float] = []
    ghost_ratios: List[float] = []
    corr_den_nb: List[float] = []
    corr_raw_nb: List[float] = []

    with torch.no_grad():
        for i in range(n_probe):
            inp, _, _ = dataset[i]
            if inp.shape[0] < 3:
                continue
            c = inp.shape[0] // 2
            _, h, w = inp.shape

            # Real-data cross-slice correlation
            x0 = inp.unsqueeze(0).to(device)
            pred0 = torch.clamp(model(x0).float(), 0.0, 1.0)[0, 0].cpu().numpy()
            raw_c = inp[c].numpy()
            for nb_ch in (c - 1, c + 1):
                raw_nb = inp[nb_ch].numpy()
                if raw_nb.std() > 1e-6 and pred0.std() > 1e-6 and raw_c.std() > 1e-6:
                    corr_den_nb.append(
                        float(np.corrcoef(pred0.ravel(), raw_nb.ravel())[0, 1])
                    )
                    corr_raw_nb.append(
                        float(np.corrcoef(raw_c.ravel(), raw_nb.ravel())[0, 1])
                    )

            # Synthetic 40 nm (~4 px) vesicle blob injected ONLY into z+1 (channel c+1)
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = h // 2, w // 2
            blob = ghost_blob_amp * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * ghost_blob_sigma**2)
            ).astype(np.float32)
            x_pert = inp.clone()
            x_pert[c + 1] = torch.clamp(
                x_pert[c + 1] + torch.from_numpy(blob), 0.0, 1.0
            )
            pred_pert = (
                torch.clamp(model(x_pert.unsqueeze(0).to(device)).float(), 0.0, 1.0)[
                    0, 0
                ]
                .cpu()
                .numpy()
            )

            diff_roi = (
                pred_pert[cy - 4 : cy + 5, cx - 4 : cx + 5]
                - pred0[cy - 4 : cy + 5, cx - 4 : cx + 5]
            )
            peak_bleed = float(np.max(np.abs(diff_roi)))
            ghost_peaks.append(peak_bleed)
            ghost_ratios.append(peak_bleed / max(abs(ghost_blob_amp), 1e-6))

    return {
        "full_context": full_metrics,
        "center_filled_ablation": ablated_metrics,
        "context_gain_est_psnr_db": float(
            full_metrics["est_clean_psnr"] - ablated_metrics["est_clean_psnr"]
        ),
        "misregistration_sensitivity": misreg,
        "z_ghosting_probe": {
            "n_patches": len(ghost_peaks),
            "injected_blob_amp": float(ghost_blob_amp),
            "mean_peak_bleed": float(np.mean(ghost_peaks)) if ghost_peaks else 0.0,
            "p95_peak_bleed": float(np.percentile(ghost_peaks, 95))
            if ghost_peaks
            else 0.0,
            "mean_bleed_ratio": float(np.mean(ghost_ratios)) if ghost_ratios else 0.0,
            "mean_corr_denoised_vs_raw_neighbor": (
                float(np.mean(corr_den_nb)) if corr_den_nb else 0.0
            ),
            "mean_corr_raw_center_vs_raw_neighbor": (
                float(np.mean(corr_raw_nb)) if corr_raw_nb else 0.0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap CI comparison on frozen 2.5D test per-sample MSEs."
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    mse_b = np.load(args.baseline_dir / "per_sample_denoised_mse.npy")
    mse_c = np.load(args.candidate_dir / "per_sample_denoised_mse.npy")
    raw_b = np.load(args.baseline_dir / "per_sample_noisy_mse.npy")
    res = paired_bootstrap_mse_ci(mse_b, mse_c, raw_b)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
