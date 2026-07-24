import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) Loss.
    Computes 1.0 - SSIM to serve as a minimization objective.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError(f"window_size must be odd, got {window_size}")

        self.window_size = window_size
        self.channels = channels
        self.data_range = data_range
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2

        # Precompute and register 2D Gaussian window buffer
        coords = torch.arange(window_size, dtype=torch.float32) - (window_size // 2)
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        window_1d = g.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window_4d = window_2d.expand(channels, 1, window_size, window_size).contiguous()

        self.register_buffer("window", window_4d)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        channels = pred.shape[1]

        # Dynamic expansion if batch channels differ from initialization
        if channels != self.channels:
            window = self.window.expand(channels, 1, self.window_size, self.window_size)
        else:
            window = self.window

        # Local means via depthwise conv
        mu_x = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_y = F.conv2d(target, window, padding=pad, groups=channels)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        # Local variances and covariance
        sigma_x_sq = (
            F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(target * target, window, padding=pad, groups=channels) - mu_y_sq
        )
        sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_xy

        sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
        sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)

        # SSIM computation
        numerator = (2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)
        denominator = (mu_x_sq + mu_y_sq + self.c1) * (
            sigma_x_sq + sigma_y_sq + self.c2
        )
        ssim_map = numerator / denominator

        # Loss minimization scalar
        return 1.0 - ssim_map.mean()


class CombinedL1SSIMLoss(nn.Module):
    """
    Hybrid Loss: L1 + SSIM
    Loss = alpha * L1 + (1 - alpha) * SSIM_Loss
    Recommended alpha = 0.84 for structural restoration.
    """

    def __init__(
        self,
        alpha: float = 0.84,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.ssim_loss = SSIMLoss(
            window_size=window_size,
            sigma=sigma,
            channels=channels,
            data_range=data_range,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.alpha * l1 + (1.0 - self.alpha) * ssim
