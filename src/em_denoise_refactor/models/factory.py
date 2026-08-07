# src/em_denoise_refactor/models/factory.py

import torch.nn as nn
from em_denoise_refactor.models.connectomics import ConnectomicsDenoisingNet
from em_denoise_refactor.models.residual_unet import (
    ResidualUNet,
    ModernUNet,
    UNetConfig,
)


def build_model(
    model_type: str = "connectomics",
    in_channels: int = 1,
    out_channels: int = 1,
    base_dim: int = 32,
    **kwargs,
) -> nn.Module:
    """
    Factory function to instantiate neural network architectures.

    Supported model_types:
      - "connectomics", "nafnet": NAFBlock + LaplacianEdgeAttention global residual network
      - "residual_unet", "res_unet": Deep Pre-activation Residual UNet with GroupNorm & GELU
      - "modern_unet": Modern ResConvBlock UNet with bilinear upsampling
    """
    m_type = model_type.lower().strip()

    if m_type in ("connectomics", "nafnet"):
        return ConnectomicsDenoisingNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=base_dim,
        )
    elif m_type in ("residual_unet", "res_unet", "residual"):
        config = UNetConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_dim,
            channel_multipliers=(1, 2, 4, 8),
            num_res_blocks=2,
            global_residual=True,
        )
        return ResidualUNet(config=config)
    elif m_type in ("modern_unet", "modern"):
        return ModernUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_dim=base_dim,
        )
    elif m_type in ("cosdd", "cosdd_lvae"):
        from em_denoise_refactor.models.cosdd import COSDDModel

        return COSDDModel(
            in_channels=in_channels,
            crop_size=kwargs.get("crop_size", (128, 128)),
            s_code_channels=kwargs.get("s_code_channels", 64),
            number_layers=kwargs.get("number_layers", 14),
            number_components=kwargs.get("number_components", 3),
            scale_initialisation=kwargs.get("scale_initialisation", True),
            noise_direction=kwargs.get("noise_direction", "x"),
            use_direct_denoiser=kwargs.get("use_direct_denoiser", False),
        )
    else:
        valid_models = ["connectomics", "residual_unet", "modern_unet", "cosdd"]
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: {valid_models}"
        )
