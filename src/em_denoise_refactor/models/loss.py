# src/em_denoise_refactor/models/loss.py

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Component Losses
# ---------------------------------------------------------------------------


class CharbonnierLoss(nn.Module):
    """
    Smooth L1 (Pseudo-Huber) Loss.
    Provides L2-like stability near zero error and L1-like robustness
    to heavy-tailed outliers. Produces sharper results than MSE.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps**2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class SobelEdgeLoss(nn.Module):
    """Computes L1 loss between spatial gradient magnitude maps using Sobel filters."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
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
                pred * pred,
                self.window,
                padding=self.window_size // 2,
                groups=channels,
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


class ESPILoss(nn.Module):
    """
    Differentiable Edge Preservation Index.
    Focuses on the top 10% strongest edges (cell membranes, vesicle walls)
    rather than penalizing all gradients uniformly.
    Minimizes 1.0 - ESPI.
    """

    def __init__(self, channels: int = 1) -> None:
        super().__init__()

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

        sobel_x = sobel_x.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        sobel_y = sobel_y.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.channels = channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode="replicate")
        target_pad = F.pad(target, (1, 1, 1, 1), mode="replicate")

        grad_x_pred = F.conv2d(pred_pad, self.sobel_x, groups=self.channels)
        grad_y_pred = F.conv2d(pred_pad, self.sobel_y, groups=self.channels)
        grad_pred = torch.hypot(grad_x_pred, grad_y_pred)

        grad_x_target = F.conv2d(target_pad, self.sobel_x, groups=self.channels)
        grad_y_target = F.conv2d(target_pad, self.sobel_y, groups=self.channels)
        grad_target = torch.hypot(grad_x_target, grad_y_target)

        # 90th percentile threshold per image in batch
        B = target.size(0)
        grad_target_flat = grad_target.view(B, -1)
        thresholds = torch.quantile(grad_target_flat, 0.9, dim=1).view(B, 1, 1, 1)

        # Edge mask (detached to prevent gradient flow through mask logic)
        edge_mask = (grad_target > thresholds).detach()

        espi_vals = []
        for b in range(B):
            mask_b = edge_mask[b]
            if not torch.any(mask_b):
                espi_vals.append(torch.tensor(0.0, device=pred.device))
                continue

            mean_pred = grad_pred[b][mask_b].mean()
            mean_target = grad_target[b][mask_b].mean()
            espi = mean_pred / (mean_target + 1e-8)
            espi_vals.append(torch.clamp(espi, 0.0, 1.0))

        espi_batch = torch.stack(espi_vals).mean()
        return 1.0 - espi_batch


class FocalFrequencyLoss(nn.Module):
    """
    2D Fourier space spatial-frequency synthesis enforcement.
    Dynamically up-weights high-frequency components that are hard to restore
    (fine textures, vesicle walls, membrane edges).
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f_pred = torch.fft.rfft2(pred, norm="ortho")
        f_target = torch.fft.rfft2(target, norm="ortho")

        freq_distance = torch.abs(f_pred - f_target) ** 2

        # Dynamic focal weights (normalized per batch)
        weight = freq_distance**self.alpha
        weight = weight / (weight.detach().max() + self.eps)

        loss = weight * freq_distance
        return torch.mean(loss)


class GeneralizedAnscombeLoss(nn.Module):
    """
    Evaluates L1 distance in a Variance-Stabilized Transformation (VST) space.
    Converts intensity-dependent Poisson-Gaussian noise to approximately
    homoscedastic Gaussian noise so dark and bright regions contribute equally.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        sigma: float = 0.0,
        gain: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.gain = gain
        self.eps = eps

    def _anscombe(self, x: torch.Tensor) -> torch.Tensor:
        val = (
            self.alpha * x
            + (3.0 / 8.0) * (self.alpha**2)
            + (self.sigma**2)
            - (self.alpha * self.gain)
        )
        return (2.0 / self.alpha) * torch.sqrt(torch.clamp(val, min=self.eps))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._anscombe(pred), self._anscombe(target))


class LaplacianLoss(nn.Module):
    """
    Enforces isotropic second-order derivative consistency
    to preserve fine structural boundaries (vesicle curvature, organelle edges).
    """

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.channels = channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lap_pred = F.conv2d(pred, self.kernel, padding=1, groups=self.channels)
        lap_target = F.conv2d(target, self.kernel, padding=1, groups=self.channels)
        return F.l1_loss(lap_pred, lap_target)


# ---------------------------------------------------------------------------
# Composite Loss (Manual Weights)
# ---------------------------------------------------------------------------


class CompositeDenoisingLoss(nn.Module):
    """
    Combines MSE Loss, Sobel Spatial Edge Loss, and SSIM Loss with configurable weights.
    (Legacy interface — kept for backward compatibility.)
    """

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


# ---------------------------------------------------------------------------
# Enhanced Composite Loss (Manual Weights, All Components)
# ---------------------------------------------------------------------------


class EnhancedCompositeLoss(nn.Module):
    """
    Full composite loss with all available components and manual weights.
    Combines: Charbonnier + SSIM + Laplacian + FFL + ESPI + Sobel Edge.
    """

    def __init__(
        self,
        charbonnier_weight: float = 0.3,
        ssim_weight: float = 0.3,
        laplacian_weight: float = 0.15,
        ffl_weight: float = 0.15,
        espi_weight: float = 0.1,
        edge_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.laplacian_weight = laplacian_weight
        self.ffl_weight = ffl_weight
        self.espi_weight = espi_weight
        self.edge_weight = edge_weight

        self.charbonnier = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.laplacian = LaplacianLoss()
        self.ffl = FocalFrequencyLoss()
        self.espi = ESPILoss()
        self.edge_loss = SobelEdgeLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_charb = (
            self.charbonnier(pred, target)
            if self.charbonnier_weight > 0
            else torch.tensor(0.0, device=pred.device)
        )
        l_ssim = (
            self.ssim_loss(pred, target)
            if self.ssim_weight > 0
            else torch.tensor(0.0, device=pred.device)
        )
        l_lap = (
            self.laplacian(pred, target)
            if self.laplacian_weight > 0
            else torch.tensor(0.0, device=pred.device)
        )
        l_ffl = (
            self.ffl(pred, target)
            if self.ffl_weight > 0
            else torch.tensor(0.0, device=pred.device)
        )
        l_espi = (
            self.espi(pred, target)
            if self.espi_weight > 0
            else torch.tensor(0.0, device=pred.device)
        )
        l_edge = (
            self.edge_loss(pred, target)
            if self.edge_weight > 0
            else torch.tensor(0.0, device=pred.device)
        )

        total = (
            self.charbonnier_weight * l_charb
            + self.ssim_weight * l_ssim
            + self.laplacian_weight * l_lap
            + self.ffl_weight * l_ffl
            + self.espi_weight * l_espi
            + self.edge_weight * l_edge
        )

        metrics = {
            "loss_total": float(total.item()),
            "loss_charbonnier": float(l_charb.item()),
            "loss_ssim": float(l_ssim.item()),
            "loss_laplacian": float(l_lap.item()),
            "loss_ffl": float(l_ffl.item()),
            "loss_espi": float(l_espi.item()),
            "loss_edge": float(l_edge.item()),
        }

        return total, metrics


# ---------------------------------------------------------------------------
# Dynamic Auto-Balanced Loss (Learnable Uncertainty Weighting)
# ---------------------------------------------------------------------------


class DynamicEMLoss(nn.Module):
    """
    Composite Loss with learnable homoscedastic uncertainty for dynamic
    gradient balancing. Combines: Charbonnier + FFL + VST + Laplacian.
    The network discovers the optimal weighting during training.
    """

    def __init__(
        self,
        channels: int = 1,
        charbonnier_eps: float = 1e-3,
        ffl_alpha: float = 1.0,
        vst_alpha: float = 1.0,
        vst_sigma: float = 0.0,
        vst_gain: float = 0.0,
    ) -> None:
        super().__init__()

        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.ffl = FocalFrequencyLoss(alpha=ffl_alpha)
        self.vst = GeneralizedAnscombeLoss(
            alpha=vst_alpha, sigma=vst_sigma, gain=vst_gain
        )
        self.laplacian = LaplacianLoss(channels=channels)

        # Learnable log-variance parameters for four loss components.
        # Initialized to zero (corresponding to a weight of 1.0).
        self.log_vars = nn.Parameter(torch.zeros(4, dtype=torch.float32))

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_charb = self.charbonnier(pred, target)
        l_ffl = self.ffl(pred, target)
        l_vst = self.vst(pred, target)
        l_lap = self.laplacian(pred, target)

        losses = torch.stack([l_charb, l_ffl, l_vst, l_lap])

        # precision = exp(-log_var)
        precision = torch.exp(-self.log_vars)

        # Composite: sum( L_i * precision_i + log_var_i )
        dynamic_loss = torch.sum(precision * losses + self.log_vars)

        with torch.no_grad():
            weights = precision.cpu().numpy()
            metrics = {
                "loss_total": dynamic_loss.item(),
                "loss_charbonnier": l_charb.item(),
                "loss_ffl": l_ffl.item(),
                "loss_vst": l_vst.item(),
                "loss_laplacian": l_lap.item(),
                "weight_charbonnier": float(weights[0]),
                "weight_ffl": float(weights[1]),
                "weight_vst": float(weights[2]),
                "weight_laplacian": float(weights[3]),
                "loss_edge": 0.0,
            }

        return dynamic_loss, metrics
