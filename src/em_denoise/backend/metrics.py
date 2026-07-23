import math
import torch
import torch.nn.functional as F


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generates a 2D Gaussian kernel replicated across channels."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size // 2)
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _prepare_tensors(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensures input tensors are floats to prevent uint8 underflow/overflow errors."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    if pred.dtype == torch.uint8:
        pred = pred.to(torch.float32)
    if target.dtype == torch.uint8:
        target = target.to(torch.float32)

    return pred, target


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) sample-wise over a batch [B, C, H, W].
    """
    pred, target = _prepare_tensors(pred, target)

    # Compute MSE per sample across [C, H, W] dimensions
    dim = tuple(range(1, pred.ndim))
    mse = torch.mean((pred - target) ** 2, dim=dim)

    # Prevent log10(0) for identical tensors
    eps = 1e-10
    psnr = 20.0 * math.log10(data_range) - 10.0 * torch.log10(mse + eps)

    if reduction == "mean":
        return psnr.mean()
    elif reduction == "sum":
        return psnr.sum()
    elif reduction == "none":
        return psnr
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Computes Structural Similarity Index (SSIM) using a Gaussian weighting window.
    """
    pred, target = _prepare_tensors(pred, target)

    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    pad = window_size // 2

    # Constants scaled dynamically to image dynamic range
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Local means via depthwise convolution
    mu_x = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_y = F.conv2d(target, window, padding=pad, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    # Local variances and covariance
    sigma_x_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = (
        F.conv2d(target * target, window, padding=pad, groups=channels) - mu_y_sq
    )
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_xy

    # Clamp variances to zero to avoid precision issues
    sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
    sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)

    # SSIM map calculation
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / denominator

    # Mean over spatial dims [C, H, W] per sample
    ssim_per_sample = ssim_map.mean(dim=(-3, -2, -1))

    if reduction == "mean":
        return ssim_per_sample.mean()
    elif reduction == "sum":
        return ssim_per_sample.sum()
    elif reduction == "none":
        return ssim_per_sample
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
