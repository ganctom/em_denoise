# src/data/dataset.py

import warnings
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from pathlib import Path
from typing import Tuple, Optional, Union, Dict

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
        photon_scale: float = 0.04,
        read_std: float = 8.0,
        gain_variance: float = 0.02,
        streak_probability: float = 0.15,
    ) -> None:
        self.photon_scale = photon_scale
        self.read_std = read_std
        self.gain_variance = gain_variance
        self.streak_p = streak_probability

    def inject(self, clean_tensor: Tensor) -> Tensor:
        # Lift to uint8-equivalent scale for correct Poisson statistics.
        # In 8-bit EM images, pixel intensity directly encodes photon counts
        # (after gain and ADC), so working in [0, 255] gives σ_shot ∝ √photons.
        img_255 = clean_tensor * 255.0

        # 1. Shot noise (Poisson process in photon space)
        # λ = pixel_value * photon_scale  →  lower photon_scale = higher relative noise
        photons = torch.clamp(img_255 * self.photon_scale, min=1e-5)
        noisy = torch.poisson(photons) / self.photon_scale

        # 2. Per-pixel detector gain variation (fixed-pattern noise)
        # Models pixel-to-pixel sensitivity variation of the BSE detector.
        if self.gain_variance > 0:
            gain = torch.normal(
                mean=torch.ones_like(noisy),
                std=torch.full_like(noisy, self.gain_variance),
            )
            noisy = noisy * gain

        # 3. Additive Gaussian read-out noise (in uint8 scale)
        noisy = noisy + torch.randn_like(noisy) * self.read_std

        # 4. Vectorized raster streak anomalies (beam charging / curtaining)
        if torch.rand(1).item() < self.streak_p:
            h = clean_tensor.shape[-2]
            w = clean_tensor.shape[-1]
            num_streaks = int(torch.randint(1, 4, (1,)).item())

            row_indices = torch.randint(0, h, (num_streaks,), device=noisy.device)
            # Streak intensity in uint8 scale: 0.3 * 255 ≈ 76 → same perceptual magnitude
            # as the original 0.3 std in [0, 1] space.
            intensities = torch.randn(num_streaks, 1, device=noisy.device) * (
                0.3 * 255.0
            )
            noisy[..., row_indices, :] += intensities.expand(-1, w)

        # Normalize back to [0, 1] and clamp to valid range
        return (noisy / 255.0).clamp_(0.0, 1.0)

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
