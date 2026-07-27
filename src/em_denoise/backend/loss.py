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


class ESPILoss(nn.Module):
    """
    Differentiable Edge Preservation Index.
    Minimizes 1.0 - ESPI.
    """

    def __init__(self, channels: int = 1):
        super().__init__()

        # Define Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

        # Reshape for F.conv2d (out_channels, in_channels/groups, kH, kW)
        sobel_x = sobel_x.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        sobel_y = sobel_y.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.channels = channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Pad to maintain spatial dimensions
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode="replicate")
        target_pad = F.pad(target, (1, 1, 1, 1), mode="replicate")

        # Compute spatial gradients
        grad_x_pred = F.conv2d(pred_pad, self.sobel_x, groups=self.channels)
        grad_y_pred = F.conv2d(pred_pad, self.sobel_y, groups=self.channels)
        grad_pred = torch.hypot(grad_x_pred, grad_y_pred)

        grad_x_target = F.conv2d(target_pad, self.sobel_x, groups=self.channels)
        grad_y_target = F.conv2d(target_pad, self.sobel_y, groups=self.channels)
        grad_target = torch.hypot(grad_x_target, grad_y_target)

        # Calculate 90th percentile threshold per image in batch
        # Flatten spatial dims to compute quantiles
        B = target.size(0)
        grad_target_flat = grad_target.view(B, -1)
        thresholds = torch.quantile(grad_target_flat, 0.9, dim=1).view(B, 1, 1, 1)

        # Create edge mask (detach to stop gradients through the mask logic)
        edge_mask = (grad_target > thresholds).detach()

        # Compute ESPI per image, then average over batch
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


class AUFRCLoss(nn.Module):
    """
    Differentiable Area Under the Fourier Ring Correlation.
    Minimizes 1.0 - AUFRC.
    """

    def __init__(self):
        super().__init__()
        self.masks = None
        self.r_max = None

    def _build_masks(self, h: int, w: int, device: torch.device):
        """Lazily builds and caches radial masks to avoid O(N^2) graph overhead."""
        cy, cx = h // 2, w // 2
        y = torch.arange(h, device=device).view(h, 1) - cy
        x = torch.arange(w, device=device).view(1, w) - cx

        r_grid = torch.hypot(y.float(), x.float()).long()
        self.r_max = min(cy, cx)

        # Precompute boolean masks for each ring
        # Shape: (r_max, h, w)
        self.masks = torch.stack([r_grid == r for r in range(1, self.r_max)])
        self.masks.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape

        if self.masks is None or self.masks.shape[1:] != (H, W):
            self._build_masks(H, W, pred.device)

        # Compute FFT and shift zero-frequency to center
        # Use torch.fft.rfft2 if inputs are purely real and symmetric optimizations are preferred,
        # but fft2 matches your NumPy prototype.
        f_pred = torch.fft.fftshift(torch.fft.fft2(pred), dim=(-2, -1))
        f_gt = torch.fft.fftshift(torch.fft.fft2(target), dim=(-2, -1))

        # f_pred shape: (B, C, H, W), masks shape: (r_max-1, H, W)
        # Broadcasting masks to match batch and channel dims
        masks = self.masks.unsqueeze(1).unsqueeze(1)  # (r_max-1, 1, 1, H, W)

        # Vectorized ring extraction
        # Element-wise multiply spectra with masks, sum over spatial dims
        # num: Re( sum( P * conj(G) ) )
        cross_power = f_pred.unsqueeze(0) * torch.conj(f_gt.unsqueeze(0))
        cross_power_masked = cross_power * masks
        num = torch.real(cross_power_masked.sum(dim=(-2, -1)))  # (r_max-1, B, C)

        # den: sqrt( sum(|P|^2) * sum(|G|^2) )
        p_mag_sq = (torch.abs(f_pred.unsqueeze(0)) ** 2) * masks
        g_mag_sq = (torch.abs(f_gt.unsqueeze(0)) ** 2) * masks
        den = torch.sqrt(p_mag_sq.sum(dim=(-2, -1)) * g_mag_sq.sum(dim=(-2, -1)))

        # Compute FRC per ring
        frc_r = num / (den + 1e-8)
        frc_r = torch.clamp(frc_r, 0.0, 1.0)

        # Mean across rings (dim=0), then batch and channels
        aufrc = frc_r.mean()
        return 1.0 - aufrc


class ComprehensiveEMLoss(nn.Module):
    """
    Hybrid Objective Function:
    Loss = λ_1(L1) + λ_2(SSIM) + λ_3(ESPI) + λ_4(AUFRC)
    """

    def __init__(
        self,
        w_l1: float = 0.5,
        w_ssim: float = 0.3,
        w_espi: float = 0.1,
        w_aufrc: float = 0.1,
        channels: int = 1,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_espi = w_espi
        self.w_aufrc = w_aufrc

        self.l1_loss = nn.L1Loss(reduction="mean")
        self.ssim_loss = SSIMLoss(channels=channels)
        self.espi_loss = ESPILoss(channels=channels)
        self.aufrc_loss = AUFRCLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.w_l1 > 0:
            loss += self.w_l1 * self.l1_loss(pred, target)
        if self.w_ssim > 0:
            loss += self.w_ssim * self.ssim_loss(pred, target)
        if self.w_espi > 0:
            loss += self.w_espi * self.espi_loss(pred, target)
        if self.w_aufrc > 0:
            loss += self.w_aufrc * self.aufrc_loss(pred, target)
        return loss
