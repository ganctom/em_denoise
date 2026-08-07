# src/em_denoise_refactor/models/cosdd/cosdd_model.py

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lvae import LadderVAE
from .pixelcnn import PixelCNN
from .s_decoder import SDecoder
from .unet import UNet


class COSDDModel(nn.Module):
    """
    Unified PyTorch wrapper for COSDD (Correlated and Signal-Dependent Denoising).

    Integrates Ladder VAE (LVAE), Autoregressive PixelCNN Noise Decoder,
    Signal Decoder (SDecoder), and optional Direct UNet Denoiser.
    """

    def __init__(
        self,
        in_channels: int = 1,
        crop_size: Tuple[int, int] = (128, 128),
        s_code_channels: int = 64,
        number_layers: int = 14,
        number_components: int = 3,
        scale_initialisation: bool = True,
        noise_direction: str = "x",
        use_direct_denoiser: bool = False,
        direct_denoiser_loss: str = "MSE",
        dimensions: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.crop_size = crop_size
        self.s_code_channels = s_code_channels
        self.number_layers = number_layers
        self.number_components = number_components

        z_dims = [s_code_channels // 2] * number_layers

        def find_num_halves(s: int) -> int:
            return math.floor(math.log2(s)) - 1

        def find_difference(nh: int) -> int:
            return max(number_layers - nh, 0)

        num_halves = [find_num_halves(s) for s in crop_size]
        difference = [find_difference(nh) for nh in num_halves]
        n_dim = len(crop_size)

        downsampling = [[1 for _ in range(n_dim)] for _ in range(number_layers)]
        for i in range(n_dim):
            j = 0
            while difference[i] > 0:
                for k in range(number_layers // 2):
                    downsampling[j + k * 2][i] = 0
                    difference[i] -= 1
                    if difference[i] == 0:
                        break
                j += 1

        self.lvae = LadderVAE(
            colour_channels=in_channels,
            img_size=crop_size,
            s_code_channels=s_code_channels,
            n_filters=s_code_channels,
            scale_initialisation=scale_initialisation,
            z_dims=z_dims,
            downsampling=downsampling,
            dimensions=dimensions,
        )

        self.ar_decoder = PixelCNN(
            colour_channels=in_channels,
            s_code_channels=s_code_channels,
            kernel_size=5,
            noise_direction=noise_direction,
            n_filters=64,
            n_layers=4,
            n_components=number_components,
            dimensions=dimensions,
        )

        self.s_decoder = SDecoder(
            colour_channels=in_channels,
            s_code_channels=s_code_channels,
            n_filters=s_code_channels,
            dimensions=dimensions,
        )

        if use_direct_denoiser:
            self.direct_denoiser = UNet(
                colour_channels=in_channels,
                n_filters=s_code_channels,
                n_layers=number_layers,
                downsampling=downsampling,
                loss_fn=direct_denoiser_loss,
                dimensions=dimensions,
            )
        else:
            self.direct_denoiser = None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Forward pass computing denoised output and VAE ELBO + SDecoder losses.
        """
        vae_out = self.lvae(x)
        s_code = vae_out["s_code"]

        x_params = self.ar_decoder(x, s_code)
        s_hat = self.s_decoder(s_code)

        kl_div = self.lvae.kl_divergence(vae_out["q_list"], vae_out["p_list"])
        kl_div = kl_div / x.numel()
        nll = -self.ar_decoder.loglikelihood(x, x_params).mean()
        elbo = kl_div + nll

        sd_loss = F.mse_loss(s_hat, x)
        total_loss = elbo + sd_loss

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_elbo": float(elbo.item()),
            "loss_kl": float(kl_div.item()),
            "loss_nll": float(nll.item()),
            "loss_s_decoder": float(sd_loss.item()),
        }

        return s_hat, (total_loss, metrics)

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """Runs fast inference returning predicted clean signal s_hat."""
        with torch.no_grad():
            vae_out = self.lvae(x)
            s_code = vae_out["s_code"]
            s_hat = self.s_decoder(s_code)
            return s_hat
