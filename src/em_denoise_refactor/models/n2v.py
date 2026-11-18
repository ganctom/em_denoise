# src/em_denoise_refactor/models/n2v.py
"""
Noise2Void 2 (N2V2) UNet architecture for self-supervised EM denoising.

Key N2V2 improvements over the original N2V baseline:
  1. Deterministic 3x3 median masking  (in data/n2v_dataset.py)
  2. Anti-aliased MaxBlurPool downsampling  (MaxBlurPool2d below)
  3. Severable top encoder-to-decoder skip connection  (blind_spot_strict flag)

Architecture reuses NAFBlock and LaplacianEdgeAttention from connectomics.py to
avoid duplicating primitives and to keep the backbone competitive.

References:
  - Krull et al. 2019  "Noise2Void — Learning Denoising from Single Noisy Images"
  - Höck et al. 2022   "N2V2 — Fixing Noise2Void Checkerboard Artifacts"
  - Broaddus et al.    "Removing Structured Noise with Self-Supervised Blind-Spot Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from em_denoise_refactor.models.connectomics import NAFBlock, LaplacianEdgeAttention


# ---------------------------------------------------------------------------
# MaxBlurPool2d — anti-aliased downsampling (N2V2 requirement)
# ---------------------------------------------------------------------------


class MaxBlurPool2d(nn.Module):
    """
    Anti-aliased 2×2 max-pooling followed by a depthwise 3×3 binomial blur.

    Replaces MaxPool2d to eliminate shift-variance artifacts that produce
    checkerboard patterns in N2V reconstructions (Höck et al. 2022, Sec. 3.2).

    The blur kernel is a fixed, non-trainable 3×3 binomial approximation:
        [1 2 1]
        [2 4 2] / 16
        [1 2 1]
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fixed 3x3 binomial blur kernel — registered as buffer (not a parameter)
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = torch.outer(kernel_1d, kernel_1d) / 16.0
        # Shape: (channels, 1, 3, 3) for depthwise convolution
        blur_kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer("blur_kernel", blur_kernel)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x)
        # Depthwise convolution: each channel blurred independently
        return F.conv2d(pooled, self.blur_kernel, padding=1, groups=self.channels)


# ---------------------------------------------------------------------------
# N2VUNet — the full backbone
# ---------------------------------------------------------------------------


class N2VUNet(nn.Module):
    """
    NAFBlock-based UNet satisfying N2V2 architectural constraints.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (1 for grayscale EM).
    out_channels : int
        Number of output channels (1 for single-channel denoising).
    base_dim : int
        Base channel width. Encoder stages use [base_dim, base_dim*2, base_dim*4].
    blind_spot_strict : bool
        If True (default, N2V2-correct), severs the top encoder-to-decoder skip
        connection so that high-frequency noise cannot leak from enc2 directly to
        the first decoder stage.  Set False to re-enable it for ablation — useful
        when fine sub-pixel features (e.g., synaptic densities) benefit from it.

    Notes
    -----
    The *bottom* skip connection (enc1 → dec2) is always preserved regardless of
    blind_spot_strict.  This asymmetric design retains coarse spatial context
    while preventing the most direct noise leakage path at the top.

    A global residual (output = network(x) + x) is applied identically to
    ConnectomicsDenoisingNet.  This is compatible with N2V because the residual
    is with respect to the masked input (the blind-spot substituted value at
    masked positions), not the original clean pixel, so the loss gradient still
    flows only through the reconstructed masked positions.
    """

    # Class-level flag so DenoisingTrainer can detect N2V mode without isinstance()
    is_n2v: bool = True

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_dim: int = 32,
        blind_spot_strict: bool = True,
    ) -> None:
        super().__init__()
        self.blind_spot_strict = blind_spot_strict

        # --- Stem ---
        self.intro = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # --- Encoder ---
        self.enc1 = NAFBlock(base_dim)
        self.down1 = MaxBlurPool2d(base_dim)  # anti-aliased ↓2

        self.enc2 = NAFBlock(base_dim * 2)
        self.down2 = MaxBlurPool2d(base_dim * 2)  # anti-aliased ↓2

        # Strided conv to expand channels when moving enc1→enc2
        self.channel_expand = nn.Conv2d(base_dim, base_dim * 2, kernel_size=1)

        # --- Bottleneck ---
        self.bottleneck = NAFBlock(base_dim * 4)
        self.edge_attention = LaplacianEdgeAttention(base_dim * 4)

        # Strided conv to expand channels before bottleneck
        self.channel_expand2 = nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=1)

        # --- Decoder ---
        # dec1: after up1, optionally merges enc2 skip
        self.up1 = nn.ConvTranspose2d(
            base_dim * 4, base_dim * 2, kernel_size=2, stride=2
        )
        # When blind_spot_strict=False, concatenate enc2 skip → need extra conv to restore channels
        self.dec1_skip_merge = nn.Conv2d(base_dim * 4, base_dim * 2, kernel_size=1)
        self.dec1 = NAFBlock(base_dim * 2)

        # dec2: after up2, always merges enc1 skip
        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.dec2_skip_merge = nn.Conv2d(base_dim * 2, base_dim, kernel_size=1)
        self.dec2 = NAFBlock(base_dim)

        # --- Output ---
        self.outro = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Stem ---
        stem = self.intro(x)

        # --- Encoder ---
        e1 = self.enc1(stem)  # [B, D, H, W]
        d1 = self.down1(e1)  # [B, D, H/2, W/2]
        d1 = self.channel_expand(d1)  # [B, 2D, H/2, W/2]

        e2 = self.enc2(d1)  # [B, 2D, H/2, W/2]
        d2 = self.down2(e2)  # [B, 2D, H/4, W/4]
        d2 = self.channel_expand2(d2)  # [B, 4D, H/4, W/4]

        # --- Bottleneck ---
        bot = self.bottleneck(d2)
        bot = self.edge_attention(bot)

        # --- Decoder stage 1 ---
        u1 = self.up1(bot)  # [B, 2D, H/2, W/2]
        if self.blind_spot_strict:
            # Top skip SEVERED: prevents direct noise leakage from enc2 → dec1
            dec1_in = u1
        else:
            # Top skip ENABLED: re-adds enc2 high-freq features (ablation mode)
            cat1 = torch.cat([u1, e2], dim=1)  # [B, 4D, H/2, W/2]
            dec1_in = self.dec1_skip_merge(cat1)  # [B, 2D, H/2, W/2]
        dec1 = self.dec1(dec1_in)

        # --- Decoder stage 2 (bottom skip always kept) ---
        u2 = self.up2(dec1)  # [B, D, H, W]
        cat2 = torch.cat([u2, e1], dim=1)  # [B, 2D, H, W]
        dec2_in = self.dec2_skip_merge(cat2)  # [B, D, H, W]
        dec2 = self.dec2(dec2_in)

        # --- Global residual (compatible with N2V: residual w.r.t. masked input) ---
        return self.outro(dec2) + x
