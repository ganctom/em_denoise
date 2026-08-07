# src/data/dataset.py

import warnings
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, List

from em_denoise_refactor.data.interfaces import LocalPatchStore, NoiseSynthesizer


class LMDBPatchStore:
    """
    Zero-copy local LMDB store optimized for M4 Pro unified memory bandwidth.
    Satisfies the LocalPatchStore protocol via structural typing.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        map_size: int = 1_099_511_627_776,
    ) -> None:
        self.map_size = map_size
        self.path: Optional[str] = None
        self.env: Optional[lmdb.Environment] = None
        self.txn: Optional[lmdb.Transaction] = None
        self._length: int = -1

        if db_path is not None:
            self.open(db_path)

    def open(self, path: Path) -> None:
        """Opens the LMDB store in read-only mode for high-speed M4 Pro mmap access."""
        # Ensure any previously open handle is closed before opening a new environment
        self.close()

        self.path = str(path)
        self.env = lmdb.open(
            self.path,
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=128,
            map_size=self.map_size,
        )
        self.txn = self.env.begin(write=False, buffers=True)

    def __len__(self) -> int:
        if self.env is None:
            raise RuntimeError("Store must be opened before accessing length.")

        if self._length == -1:
            # Query active transaction/environment directly without opening a secondary handle
            if self.txn is not None:
                len_bytes = self.txn.get(b"len")
                if len_bytes is not None:
                    self._length = int(len_bytes.decode("utf-8"))
                else:
                    self._length = self.env.stat()["entries"]
            else:
                self._length = self.env.stat()["entries"]

        return self._length

    def get_patch(self, index: int) -> torch.Tensor:
        if self.txn is None:
            raise RuntimeError("Transaction is null. Ensure open() was called.")

        key = f"{index:08d}".encode("ascii")
        buf = self.txn.get(key)
        if buf is None:
            raise IndexError(f"Patch index {index} out of bounds or missing.")

        # Zero-copy uint8 view directly from memory-mapped buffer
        arr = np.frombuffer(buf, dtype=np.uint8)
        side_len = int(np.sqrt(len(arr)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return torch.from_numpy(arr).view(1, side_len, side_len)

    def close(self) -> None:
        """Safely aborts active transactions and closes the LMDB environment handle."""
        if self.txn is not None:
            try:
                self.txn.abort()
            except Exception:
                pass
            self.txn = None

        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self) -> None:
        self.close()


class SBFSEMNoiseSynthesizer(NoiseSynthesizer):
    """
    Vectorized procedural degradation generator simulating SBF-SEM acquisition physics.

    Noise model ported from add_em_noise (em_denoise), adapted for online PyTorch
    synthesis. All Poisson and readout noise operations are performed in uint8-equivalent
    scale [0, 255] for physically correct shot-noise statistics, then normalized back
    to [0, 1] for network consumption.

    Parameters
    ----------
    photon_scale : float
        Simulated photon count per unit pixel intensity.  Lower values produce
        stronger shot noise.  Equivalent to add_em_noise's ``photon_scale``.
        Typical range: 0.02 – 0.10.
    read_std : float
        Additive Gaussian read-out noise standard deviation in uint8 [0, 255] space.
        Equivalent to add_em_noise's ``read_std``.
    gain_variance : float
        Standard deviation of the per-pixel detector gain map (mean = 1.0).
        Models fixed-pattern sensitivity variation across the detector.
    streak_probability : float
        Per-image probability of injecting horizontal raster charging streaks.
    """

    def __init__(
        self,
        photon_scale: Union[float, Tuple[float, float], List[float]] = 0.04,
        read_std: Union[float, Tuple[float, float], List[float]] = 8.0,
        gain_variance: Union[float, Tuple[float, float], List[float]] = 0.02,
        streak_probability: Union[float, Tuple[float, float], List[float]] = 0.0,
        inverted_bse: bool = True,
    ) -> None:
        self.photon_scale = photon_scale
        self.read_std = read_std
        self.gain_variance = gain_variance
        self.streak_p = streak_probability
        self.inverted_bse = inverted_bse

    def _sample_param(
        self,
        param: Union[float, Tuple[float, float], List[float]],
        clean_tensor: Tensor,
    ) -> Union[float, Tensor]:
        if isinstance(param, (list, tuple)):
            # Support batched inputs [B, C, H, W] or unbatched [C, H, W]
            shape = (
                (clean_tensor.shape[0], 1, 1, 1)
                if clean_tensor.ndim == 4
                else (1, 1, 1)
            )
            return torch.empty(shape, device=clean_tensor.device).uniform_(
                param[0], param[1]
            )
        return float(param)

    def _sample_prob(
        self, param: Union[float, Tuple[float, float], List[float]]
    ) -> float:
        if isinstance(param, (list, tuple)):
            return torch.empty(1).uniform_(param[0], param[1]).item()
        return float(param)

    def inject(self, clean_tensor: Tensor) -> Tensor:
        # Lift to uint8-equivalent scale for correct Poisson statistics.
        img_255 = clean_tensor * 255.0

        # Un-invert to raw BSE electron signal space if inverted_bse is True (dark membranes -> high electron count)
        if self.inverted_bse:
            raw_bse = 255.0 - img_255
        else:
            raw_bse = img_255

        # Sample randomized parameters for this patch/batch
        photon_scale = self._sample_param(self.photon_scale, clean_tensor)
        read_std = self._sample_param(self.read_std, clean_tensor)
        gain_variance = self._sample_param(self.gain_variance, clean_tensor)
        streak_p = self._sample_prob(self.streak_p)

        # 1. Shot noise (Poisson process in raw electron space)
        photons = torch.clamp(raw_bse * photon_scale, min=1e-5)
        noisy_raw = torch.poisson(photons) / photon_scale

        # 2. Per-pixel detector gain variation (fixed-pattern noise)
        if isinstance(gain_variance, Tensor) or gain_variance > 0:
            # We can't just check > 0 easily if it's a tensor, so we broadcast
            if isinstance(gain_variance, float) and gain_variance <= 0:
                pass
            else:
                # torch.normal doesn't broadcast well with tensor std, so we do it manually:
                gain = 1.0 + torch.randn_like(noisy_raw) * gain_variance
                noisy_raw = noisy_raw * gain

        # 3. Additive Gaussian read-out noise (in uint8 scale)
        noisy_raw = noisy_raw + torch.randn_like(noisy_raw) * read_std

        # 4. Vectorized raster streak anomalies (only if streak_p > 0)
        if streak_p > 0 and torch.rand(1).item() < streak_p:
            h, w = clean_tensor.shape[-2:]
            num_streaks = int(torch.randint(1, 4, (1,)).item())
            row_indices = torch.randint(0, h, (num_streaks,), device=noisy_raw.device)
            intensities = torch.randn(num_streaks, 1, device=noisy_raw.device) * (
                0.3 * 255.0
            )
            noisy_raw[..., row_indices, :] += intensities.expand(-1, w)

        # Re-invert back to display space if inverted_bse is True
        if self.inverted_bse:
            noisy = 255.0 - noisy_raw
        else:
            noisy = noisy_raw

        # Normalize back to [0, 1] and clamp to valid range
        return (noisy / 255.0).clamp_(0.0, 1.0)

    def __call__(self, clean_tensor: Tensor) -> Tensor:
        """Forwarding interface to satisfy Callable semantics."""
        return self.inject(clean_tensor)


class EmpiricalNoiseSynthesizer(NoiseSynthesizer):
    """
    Synthesizes noise by sampling from a calibrated 2D empirical histogram
    P(noisy | clean) using inverse transform sampling.
    """

    def __init__(
        self, model_path: Union[str, Path], noise_amplification: float = 1.0
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Empirical noise model not found at {model_path}")

        # Load P(x | s) shape (256, 256)
        p_x_given_s = np.load(model_path)
        # Compute CDF for inverse transform sampling
        cdf = np.cumsum(p_x_given_s, axis=1)
        # Ensure it ends exactly at 1.0 to avoid sampling out of bounds
        cdf[:, -1] = 1.0

        self.cdf = torch.tensor(cdf, dtype=torch.float32)
        self.noise_amplification = noise_amplification
        self.device = None

    def inject(self, clean_tensor: Tensor) -> Tensor:
        if self.device is None or self.device != clean_tensor.device:
            self.cdf = self.cdf.to(clean_tensor.device)
            self.device = clean_tensor.device

        # Map clean to [0, 255] index
        clean_idx = torch.clamp(torch.round(clean_tensor * 255), 0, 255).long()

        # Fetch the 256-bin CDF for each pixel based on its clean intensity
        # shape: [..., H, W, 256]
        pixel_cdfs = self.cdf[clean_idx]

        # Sample uniform random values [0, 1)
        u = torch.rand_like(clean_tensor).unsqueeze(-1)

        # Find the index where u falls in the CDF (empirical noisy intensity in [0, 255])
        noisy_idx = torch.searchsorted(pixel_cdfs, u).squeeze(-1).float()

        # Amplify noise deviation: push noisy further from clean by factor alpha.
        # alpha=1.0: exact empirical noise; alpha=2.0: doubled deviation; alpha=0.5: halved.
        if self.noise_amplification != 1.0:
            clean_float = clean_idx.float()
            deviation = noisy_idx - clean_float
            noisy_idx = torch.clamp(
                clean_float + self.noise_amplification * deviation, 0.0, 255.0
            )

        # Convert back to [0, 1] float
        noisy = noisy_idx / 255.0
        return noisy

    def __call__(self, clean_tensor: Tensor) -> Tensor:
        """Forwarding interface to satisfy Callable semantics."""
        return self.inject(clean_tensor)


class SBFSEMDataset(Dataset):
    """
    PyTorch Dataset resolving dependencies via Protocol injection or path initialization,
    supporting zero-copy RAM caching and D4 dihedral data augmentation.
    """

    def __init__(
        self,
        store: Optional[LocalPatchStore] = None,
        noise_injector: Optional[NoiseSynthesizer] = None,
        store_path: Optional[Union[str, Path]] = None,
        cache_in_ram: bool = False,
        augment: bool = False,
    ) -> None:
        if store is None:
            if store_path is None:
                raise ValueError(
                    "Must provide either an instantiated 'store' or a 'store_path'."
                )

            # Auto-instantiate concrete store from path
            db_store = LMDBPatchStore()
            db_store.open(Path(store_path))
            self.store: LocalPatchStore = db_store
        else:
            self.store = store

        self.noise_injector: Optional[NoiseSynthesizer] = noise_injector
        self.cache_in_ram = cache_in_ram
        self.augment = augment

        self.ram_cache: Dict[int, torch.Tensor] = {}
        if self.cache_in_ram:
            print(f"Pre-caching {len(self.store)} LMDB patches into RAM...")
            for idx in range(len(self.store)):
                self.ram_cache[idx] = self.store.get_patch(idx)

    def __len__(self) -> int:
        return len(self.store)

    def _apply_augmentation(
        self, noisy: torch.Tensor, clean: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # D4 Dihedral Symmetry: Random flips + 90-degree rotations
        if torch.rand(1).item() > 0.5:
            noisy = torch.flip(noisy, dims=[-1])
            clean = torch.flip(clean, dims=[-1])
        if torch.rand(1).item() > 0.5:
            noisy = torch.flip(noisy, dims=[-2])
            clean = torch.flip(clean, dims=[-2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            noisy = torch.rot90(noisy, k, dims=[-2, -1])
            clean = torch.rot90(clean, k, dims=[-2, -1])
        return noisy, clean

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_in_ram and idx in self.ram_cache:
            target = self.ram_cache[idx].clone()
        else:
            target = self.store.get_patch(idx)

        if self.noise_injector is not None:
            noisy = self.noise_injector.inject(target)
        else:
            noisy = target.clone()

        if self.augment:
            noisy, target = self._apply_augmentation(noisy, target)

        return noisy, target
