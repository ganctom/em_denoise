# src/em_denoise_refactor/models/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SobelEdgeLoss(nn.Module):
    """Computes L1 loss between spatial gradient magnitude maps using Sobel filters."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        # Expects shape (B, C, H, W)
        channels = x.shape[1]
        weight_x = self.sobel_x.repeat(channels, 1, 1, 1)
        weight_y = self.sobel_y.repeat(channels, 1, 1, 1)

        gx = F.conv2d(x, weight_x, padding=1, groups=channels)
        gy = F.conv2d(x, weight_y, padding=1, groups=channels)

        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_grad = self._gradient_magnitude(pred)
        target_grad = self._gradient_magnitude(target)
        return F.l1_loss(pred_grad, target_grad)


class SSIMLoss(nn.Module):
    """Differentiable 2D Structural Similarity Index (SSIM) Loss (1.0 - SSIM)."""

    def __init__(self, window_size: int = 11, in_channels: int = 1) -> None:
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels

        # 1D Gaussian kernel
        sigma = 1.5
        gauss = torch.exp(
            -torch.arange(window_size).float().sub(window_size // 2).pow(2)
            / (2 * sigma**2)
        )
        gauss = (gauss / gauss.sum()).unsqueeze(1)
        kernel_2d = gauss.mm(gauss.t()).unsqueeze(0).unsqueeze(0)

        window = kernel_2d.repeat(in_channels, 1, 1, 1)
        self.register_buffer("window", window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.shape[1]
        c1 = 0.01**2
        c2 = 0.03**2

        mu1 = F.conv2d(
            pred, self.window, padding=self.window_size // 2, groups=channels
        )
        mu2 = F.conv2d(
            target, self.window, padding=self.window_size // 2, groups=channels
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(
                pred * pred, self.window, padding=self.window_size // 2, groups=channels
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                target * target,
                self.window,
                padding=self.window_size // 2,
                groups=channels,
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                pred * target,
                self.window,
                padding=self.window_size // 2,
                groups=channels,
            )
            - mu1_mu2
        )

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - ssim_map.mean()


class CompositeDenoisingLoss(nn.Module):
    """Combines MSE Loss, Sobel Spatial Edge Loss, and SSIM Loss with configurable weights."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        edge_weight: float = 0.2,
        ssim_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight

        self.mse_loss = nn.MSELoss()
        self.edge_loss = SobelEdgeLoss()
        self.ssim_loss = SSIMLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_mse = self.mse_loss(pred, target)
        l_edge = (
            self.edge_loss(pred, target) if self.edge_weight > 0 else torch.tensor(0.0)
        )
        l_ssim = (
            self.ssim_loss(pred, target) if self.ssim_weight > 0 else torch.tensor(0.0)
        )

        total_loss = (
            self.mse_weight * l_mse
            + self.edge_weight * l_edge
            + self.ssim_weight * l_ssim
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_mse": float(l_mse.item()),
            "loss_edge": float(l_edge.item()) if self.edge_weight > 0 else 0.0,
            "loss_ssim": float(l_ssim.item()) if self.ssim_weight > 0 else 0.0,
        }

        return total_loss, metrics
