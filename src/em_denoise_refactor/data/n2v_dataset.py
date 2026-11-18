# src/em_denoise_refactor/data/n2v_dataset.py
"""
N2V2 / StructN2V blind-spot dataset wrappers for self-supervised EM denoising.

Design principles
-----------------
1. Thin wrapper pattern: Accepts any existing Dataset that yields (noisy, clean, fname)
   triples and applies masking in __getitem__.  Zero changes required to the underlying
   LMDBPatchDataset or PatchFolderDataset.

2. For N2V the LMDB 'clean' patches ARE the raw noisy images — the naming reflects the
   supervised pipeline convention.  In N2V mode the caller should pass a dataset
   initialized WITHOUT a noise_injector (so noisy == clean == raw EM patch).

3. Three masking strategies:
   - "random"     N2V2 — random point mask, 3x3 median substitution
   - "horizontal" StructN2V — full horizontal stripe mask, median substitution along rows
   - "vertical"   StructN2V — full vertical stripe mask, median substitution along cols

4. Returns (masked_input, original_noisy, mask, fname) where:
   - masked_input  : the patch with selected pixels replaced by median neighbours
   - original_noisy: the unmasked raw patch (used as the regression target at mask sites)
   - mask          : boolean tensor marking which pixels were replaced (True = masked)
   - fname         : passthrough filename string for logging / evaluation

References
----------
- Krull et al. 2019  "Noise2Void — Learning Denoising from Single Noisy Images"
- Höck et al. 2022   "N2V2 — Fixing Noise2Void Checkerboard Artifacts"
- Broaddus et al.    "Removing Structured Noise with Self-Supervised Blind-Spot Networks"
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple


class N2VDatasetWrapper(Dataset):
    """
    Wraps an existing (noisy, clean, fname) Dataset with blind-spot masking.

    In N2V mode:
      - `noisy` from the inner dataset should equal `clean` (no noise injection).
      - Masking is applied to the noisy tensor, producing (masked_input, original_noisy, mask).

    Parameters
    ----------
    inner_dataset : Dataset
        Any Dataset whose __getitem__ returns (noisy_tensor, clean_tensor, fname_str).
        For N2V, pass a dataset WITHOUT a noise_injector so that noisy == clean == raw EM.
    mask_ratio : float
        Fraction of pixels selected for masking.  Default 0.015 (N2V standard).
    mask_mode : str
        Masking strategy: "random" (N2V2), "horizontal" (StructN2V), "vertical" (StructN2V).
    """

    def __init__(
        self,
        inner_dataset: Dataset,
        mask_ratio: float = 0.015,
        mask_mode: str = "random",
    ) -> None:
        self.inner = inner_dataset
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode.lower()

        if self.mask_mode not in ("random", "horizontal", "vertical"):
            raise ValueError(
                f"mask_mode must be 'random', 'horizontal', or 'vertical', "
                f"got '{mask_mode}'"
            )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Returns
        -------
        masked_input : Tensor [C, H, W]
            The patch with selected pixels replaced by their 3x3 median neighbours.
        original_noisy : Tensor [C, H, W]
            The raw unmasked patch — the regression target at masked positions.
        mask : Tensor [C, H, W] bool
            True at positions that were replaced (loss is computed only here).
        fname : str
            Filename string from the inner dataset.
        """
        # Inner dataset returns (noisy, clean, fname).
        # In N2V mode the caller passes a dataset with no noise_injector, so
        # noisy == clean == raw EM patch.  We use the 'noisy' slot as the raw input.
        raw_noisy, _clean, fname = self.inner[idx]  # [C, H, W]

        if self.mask_mode == "random":
            masked_input, mask = _apply_n2v2_median_mask(raw_noisy, self.mask_ratio)
        elif self.mask_mode == "horizontal":
            masked_input, mask = _apply_struct_mask(raw_noisy, self.mask_ratio, axis=0)
        else:  # "vertical"
            masked_input, mask = _apply_struct_mask(raw_noisy, self.mask_ratio, axis=1)

        return masked_input, raw_noisy, mask, fname


# ---------------------------------------------------------------------------
# Masking strategies
# ---------------------------------------------------------------------------


def _apply_n2v2_median_mask(
    tensor: torch.Tensor,
    mask_ratio: float = 0.015,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    N2V2 deterministic masking: randomly selects ~mask_ratio of pixels and replaces
    each with the median of its 3x3 Moore neighbourhood (excluding the centre pixel).

    This is the critical N2V2 improvement over the original N2V:
    - Original N2V: replaces with a uniformly-sampled random *single* neighbour.
      → Introduces a data-dependent bias into the substitute value that the network
        can learn to partially exploit (leakage), producing checkerboard artifacts.
    - N2V2: replaces with the *median* of all 8 neighbours.
      → The median is a strictly local, content-adaptive estimate that cannot be
        predicted from the centre pixel, eliminating the checkerboard bias.

    Parameters
    ----------
    tensor : Tensor [C, H, W]
    mask_ratio : float
        Fraction of pixels to mask.

    Returns
    -------
    masked : Tensor [C, H, W]
    mask   : Tensor [C, H, W] bool
    """
    C, H, W = tensor.shape

    # --- Compute 8-neighbour median via unfold ---
    # Pad tensor to handle border pixels
    # Shape after unsqueeze: [1, C, H, W]
    t = tensor.unsqueeze(0)
    # Pad with reflect to avoid border bias
    t_padded = F.pad(t, (1, 1, 1, 1), mode="reflect")  # [1, C, H+2, W+2]

    # Unfold the 3x3 neighbourhood: [1, C, H, W, 9]
    unfolded = t_padded.unfold(2, 3, 1).unfold(3, 3, 1)
    # Reshape to [1, C, H, W, 9]
    unfolded = unfolded.contiguous().view(1, C, H, W, 9)
    # Remove centre pixel (index 4 in row-major 3x3)
    neighbours = torch.cat(
        [unfolded[..., :4], unfolded[..., 5:]], dim=-1
    )  # [1,C,H,W,8]
    # Median over the 8 neighbours
    median_neighbours = neighbours.median(dim=-1).values.squeeze(0)  # [C, H, W]

    # --- Generate binary mask ---
    mask = torch.rand(C, H, W, device=tensor.device) < mask_ratio

    # --- Replace masked pixels with local median ---
    masked = tensor.clone()
    masked[mask] = median_neighbours[mask]

    return masked, mask


def _apply_struct_mask(
    tensor: torch.Tensor,
    mask_ratio: float = 0.015,
    axis: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    StructN2V masking: selects entire rows (axis=0 → horizontal) or entire columns
    (axis=1 → vertical) at random and replaces them with the median of their
    immediate perpendicular neighbours.

    Designed to suppress structured (correlated) noise patterns such as the
    horizontal raster charging streaks common in SBF-SEM acquisitions.

    Parameters
    ----------
    tensor : Tensor [C, H, W]
    mask_ratio : float
        Fraction of rows (or columns) to mask.
    axis : int
        0 → mask rows (horizontal stripes, targets horizontal raster noise).
        1 → mask columns (vertical stripes).

    Returns
    -------
    masked : Tensor [C, H, W]
    mask   : Tensor [C, H, W] bool
    """
    C, H, W = tensor.shape
    masked = tensor.clone()
    mask = torch.zeros(C, H, W, dtype=torch.bool, device=tensor.device)

    if axis == 0:
        # Horizontal: select rows to mask
        n_mask = max(1, int(H * mask_ratio))
        row_indices = torch.randperm(H, device=tensor.device)[:n_mask]

        for r in row_indices.tolist():
            # Replace row with average of immediate neighbours (row above + row below)
            r_prev = max(r - 1, 0)
            r_next = min(r + 1, H - 1)
            masked[:, r, :] = (tensor[:, r_prev, :] + tensor[:, r_next, :]) * 0.5
            mask[:, r, :] = True
    else:
        # Vertical: select columns to mask
        n_mask = max(1, int(W * mask_ratio))
        col_indices = torch.randperm(W, device=tensor.device)[:n_mask]

        for c in col_indices.tolist():
            c_prev = max(c - 1, 0)
            c_next = min(c + 1, W - 1)
            masked[:, :, c] = (tensor[:, :, c_prev] + tensor[:, :, c_next]) * 0.5
            mask[:, :, c] = True

    return masked, mask
