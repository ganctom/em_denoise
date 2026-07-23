from pydantic import BaseModel, Field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConfig(BaseModel):
    in_channels: int = Field(
        1, description="Number of input channels (1 for grayscale EM)"
    )
    out_channels: int = Field(1, description="Number of output channels")
    base_channels: int = Field(64, description="Channels in the first encoder block")
    channel_multipliers: Tuple[int, ...] = Field(
        (1, 2, 4, 8), description="Multiplier per depth level"
    )
    num_res_blocks: int = Field(
        2, description="Number of residual blocks per resolution level"
    )
    use_attention: bool = Field(
        False, description="Enable spatial attention at the bottleneck"
    )
    dropout_rate: float = Field(0.1, ge=0.0, le=1.0)


class ResBlock(nn.Module):
    """
    Pre-activation Residual Block using GroupNorm and GELU.
    Follows modern conventions (e.g., ConvNeXt, Vision Transformers) favoring
    smoother activations and batch-independent normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 1x1 conv projection if channel dimensions change across the residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    """Strided convolution for downsampling instead of MaxPool to retain spatial phase."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Bilinear interpolation followed by convolution to prevent checkerboard artifacts."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class ResidualUNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config

        # Initial projection
        self.init_conv = nn.Conv2d(
            config.in_channels, config.base_channels, kernel_size=3, padding=1
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # --- Encoder ---
        in_ch = config.base_channels
        for mult in config.channel_multipliers:
            out_ch = config.base_channels * mult
            blocks = nn.Sequential(
                *[
                    ResBlock(in_ch if i == 0 else out_ch, out_ch, config.dropout_rate)
                    for i in range(config.num_res_blocks)
                ]
            )
            self.down_blocks.append(blocks)
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            ResBlock(in_ch, in_ch, config.dropout_rate),
            ResBlock(in_ch, in_ch, config.dropout_rate),
        )

        # --- Decoder ---
        reversed_mults = list(reversed(config.channel_multipliers))
        for i, mult in enumerate(reversed_mults):
            out_ch = config.base_channels * mult
            # Input to up block is current channels + skip connection channels
            skip_ch = config.base_channels * reversed_mults[i]

            self.up_blocks.append(Upsample(in_ch, out_ch))

            blocks = nn.Sequential(
                *[
                    ResBlock(
                        out_ch + skip_ch if j == 0 else out_ch,
                        out_ch,
                        config.dropout_rate,
                    )
                    for j in range(config.num_res_blocks)
                ]
            )
            self.up_blocks.append(blocks)
            in_ch = out_ch

        # Final projection to output space
        self.final_norm = nn.GroupNorm(8, in_ch)
        self.final_act = nn.GELU()
        self.final_conv = nn.Conv2d(
            in_ch, config.out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)

        skips = []
        # Encoder pass
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)  # Downsample

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass
        for i in range(0, len(self.up_blocks), 2):
            up_layer = self.up_blocks[i]
            res_blocks = self.up_blocks[i + 1]

            x = up_layer(x)
            skip = skips.pop()

            # Handle potential padding differences
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear")

            x = torch.cat([x, skip], dim=1)
            x = res_blocks(x)

        x = self.final_conv(self.final_act(self.final_norm(x)))
        return x
