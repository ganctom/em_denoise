# src/em_denoise_refactor/models/residual_unet.py

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    dropout_rate: float = 0.0
    global_residual: bool = True
    norm_transitions: bool = True


def _validate_residual_channels(in_channels: int, out_channels: int) -> None:
    """Validate that global residual skip is well-defined for (in_channels, out_channels)."""
    if in_channels == out_channels:
        return
    if out_channels == 1 and in_channels > 0 and in_channels % 2 == 1:
        return
    raise ValueError(
        f"Global residual skip requires in_channels == out_channels or "
        f"(out_channels == 1 and odd in_channels for 2.5D center-slice skip), "
        f"got in_channels={in_channels}, out_channels={out_channels}."
    )


def _extract_residual_skip(x: torch.Tensor, out_channels: int) -> torch.Tensor:
    """Return full input for 2D (C_in == C_out) or center slice for 2.5D (odd C_in -> 1)."""
    c_in = x.shape[1]
    if c_in == out_channels:
        return x
    center = c_in // 2
    return x[:, center : center + 1]


def _group_count(channels: int) -> int:
    """Return a valid GroupNorm group count dividing `channels` (preferring 8)."""
    for g in (8, 4, 2):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """
    Pre-activation Residual Block using GroupNorm and GELU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_transitions: bool = True,
    ):
        super().__init__()
        num_groups = _group_count(in_channels)
        out_groups = _group_count(out_channels)

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )

        self.norm2 = nn.GroupNorm(num_groups=out_groups, num_channels=out_channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_channels != out_channels:
            if norm_transitions:
                # Normalize channel-changing shortcuts so skip variance stays O(1) across stages.
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=out_groups, num_channels=out_channels),
                )
            else:
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
    """Strided convolution with optional GroupNorm to keep multi-scale skip variance O(1)."""

    def __init__(self, channels: int, norm_transitions: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=not norm_transitions,
        )
        self.norm: nn.Module = (
            nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels)
            if norm_transitions
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class Upsample(nn.Module):
    """Bilinear interpolation followed by convolution and optional GroupNorm to keep skip variance O(1)."""

    def __init__(
        self, in_channels: int, out_channels: int, norm_transitions: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not norm_transitions,
        )
        self.norm: nn.Module = (
            nn.GroupNorm(
                num_groups=_group_count(out_channels), num_channels=out_channels
            )
            if norm_transitions
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.norm(self.conv(x))


class ResidualUNet(nn.Module):
    """
    Multi-scale Pre-activation Residual UNet with configurable depth,
    GroupNorm, GELU, and global residual connection.
    """

    def __init__(self, config: UNetConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = UNetConfig(**kwargs)
        self.config = config
        if config.global_residual:
            _validate_residual_channels(config.in_channels, config.out_channels)

        self.init_conv = nn.Conv2d(
            config.in_channels, config.base_channels, kernel_size=3, padding=1
        )

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # --- Encoder ---
        in_ch = config.base_channels
        for mult in config.channel_multipliers:
            out_ch = config.base_channels * mult
            blocks = nn.Sequential(
                *[
                    ResBlock(
                        in_ch if i == 0 else out_ch,
                        out_ch,
                        config.dropout_rate,
                        norm_transitions=config.norm_transitions,
                    )
                    for i in range(config.num_res_blocks)
                ]
            )
            self.down_blocks.append(blocks)
            self.down_samples.append(
                Downsample(out_ch, norm_transitions=config.norm_transitions)
            )
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            ResBlock(
                in_ch,
                in_ch,
                config.dropout_rate,
                norm_transitions=config.norm_transitions,
            ),
            ResBlock(
                in_ch,
                in_ch,
                config.dropout_rate,
                norm_transitions=config.norm_transitions,
            ),
        )

        # --- Decoder ---
        reversed_mults = list(reversed(config.channel_multipliers))
        for i, mult in enumerate(reversed_mults):
            out_ch = config.base_channels * mult
            skip_ch = config.base_channels * reversed_mults[i]

            self.up_blocks.append(
                Upsample(in_ch, out_ch, norm_transitions=config.norm_transitions)
            )

            blocks = nn.Sequential(
                *[
                    ResBlock(
                        out_ch + skip_ch if j == 0 else out_ch,
                        out_ch,
                        config.dropout_rate,
                        norm_transitions=config.norm_transitions,
                    )
                    for j in range(config.num_res_blocks)
                ]
            )
            self.up_blocks.append(blocks)
            in_ch = out_ch

        final_groups = _group_count(in_ch)
        self.final_norm = nn.GroupNorm(final_groups, in_ch)
        self.final_act = nn.GELU()
        self.final_conv = nn.Conv2d(
            in_ch, config.out_channels, kernel_size=3, padding=1
        )
        # Start global residual at identity (pred == input_x) to avoid step-0 overshoot.
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.init_conv(x)

        skips = []
        # Encoder pass with learned strided downsampling
        for block, downsample in zip(self.down_blocks, self.down_samples):
            x = block(x)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass
        for i in range(0, len(self.up_blocks), 2):
            up_layer = self.up_blocks[i]
            res_blocks = self.up_blocks[i + 1]

            x = up_layer(x)
            skip = skips.pop()

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear")

            x = torch.cat([x, skip], dim=1)
            x = res_blocks(x)

        out = self.final_conv(self.final_act(self.final_norm(x)))
        if self.config.global_residual:
            out = out + _extract_residual_skip(input_x, self.config.out_channels)
        return out


class ResConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        num_groups = 8 if out_c % 8 == 0 else min(out_c, 4)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_c)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_c)
        self.shortcut = (
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
            if in_c != out_c
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + res)


class ModernUNet(nn.Module):
    """
    Modern UNet architecture using ResConvBlocks, GroupNorm, and GELU activations.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_dim: int = 32):
        super().__init__()
        _validate_residual_channels(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enc1 = ResConvBlock(in_channels, base_dim)
        self.enc2 = ResConvBlock(base_dim, base_dim * 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ResConvBlock(base_dim * 2, base_dim * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv2 = nn.Conv2d(base_dim * 4, base_dim * 2, kernel_size=1)
        self.dec2 = ResConvBlock(base_dim * 4, base_dim * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv1 = nn.Conv2d(base_dim * 2, base_dim, kernel_size=1)
        self.dec1 = ResConvBlock(base_dim * 2, base_dim)

        self.final = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        b = self.bottleneck(self.pool(x2))

        u2 = self.up_conv2(self.up2(b))
        dec2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = self.up_conv1(self.up1(dec2))
        dec1 = self.dec1(torch.cat([u1, x1], dim=1))

        return self.final(dec1) + _extract_residual_skip(x, self.out_channels)
