import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Core Primitives (NAFBlock)
# -------------------------------------------------------------------------


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels)

        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pw_expand = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.sg = SimpleGate()
        self.pw_reduce = nn.Conv2d(channels, channels, kernel_size=1)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, kernel_size=1)
        )

        # Initialize beta to 1.0 so residual signal passes immediately
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.permute(0, 2, 3, 1)
        out = self.layer_norm(out)
        out = out.permute(0, 3, 1, 2)

        out = self.dw_conv(out)
        out = self.pw_expand(out)
        out = self.sg(out)
        out = out * self.sca(out)
        out = self.pw_reduce(out)

        return x + out * self.beta


# -------------------------------------------------------------------------
# High-Frequency Attention
# -------------------------------------------------------------------------


class LaplacianEdgeAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        laplacian_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("laplacian", laplacian_kernel)
        self.attention_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        channels = feature_map.size(1)
        laplacian_weight = self.laplacian.repeat(channels, 1, 1, 1)

        edges = F.conv2d(feature_map, laplacian_weight, padding=1, groups=channels)
        concat = torch.cat([feature_map, edges], dim=1)
        mask = self.sigmoid(self.attention_conv(concat))

        # Direct multiplicative residual gating
        return feature_map * (1.0 + mask)


# -------------------------------------------------------------------------
# Macro Architecture
# -------------------------------------------------------------------------


class ConnectomicsDenoisingNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_dim: int = 32):
        super().__init__()

        self.intro = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # Encoder
        self.enc1 = NAFBlock(base_dim)
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=2, stride=2)

        self.enc2 = NAFBlock(base_dim * 2)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = NAFBlock(base_dim * 4)
        self.edge_attention = LaplacianEdgeAttention(base_dim * 4)

        # Decoder
        self.up1 = nn.ConvTranspose2d(
            base_dim * 4, base_dim * 2, kernel_size=2, stride=2
        )
        self.dec1 = NAFBlock(base_dim * 2)

        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.dec2 = NAFBlock(base_dim)

        self.outro = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.intro(x)

        e1 = self.enc1(out)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        bot = self.bottleneck(d2)
        bot = self.edge_attention(bot)

        u1 = self.up1(bot) + e2
        dec1 = self.dec1(u1)

        u2 = self.up2(dec1) + e1
        dec2 = self.dec2(u2)

        # Global residual reconstruction
        return self.outro(dec2) + x
