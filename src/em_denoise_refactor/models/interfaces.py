# src/em_denoise_refactor/models/interfaces.py

import torch
import torch.nn as nn
from typing import Protocol, runtime_checkable, Tuple


@runtime_checkable
class HighFreqAttentionBlock(Protocol):
    """
    Contract for attention mechanisms specifically tuned to preserve high-frequency
    connectomics features (axon cross-sections, lipid bi-layers, synaptic vesicles).
    """

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class FeatureExtractionEncoder(Protocol):
    """Contract for the downsampling path of the topology."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]: ...


@runtime_checkable
class FeatureReconstructionDecoder(Protocol):
    """Contract for the upsampling path, utilizing skip connections."""

    def forward(
        self, x: torch.Tensor, skip_connections: list[torch.Tensor]
    ) -> torch.Tensor: ...


class ConnectomicsDenoisingNet(nn.Module):
    """
    Main Neural Network Topology.
    Dependencies (Encoder, Decoder, Attention) are injected to allow easy A/B testing
    of architectures (e.g., SwinIR variants, NAFNet, U-Net) during grid search.
    """

    def __init__(
        self,
        encoder: FeatureExtractionEncoder,
        decoder: FeatureReconstructionDecoder,
        bottleneck_attention: HighFreqAttentionBlock,
    ) -> None: ...

    def forward(self, noisy_input: torch.Tensor) -> torch.Tensor: ...
