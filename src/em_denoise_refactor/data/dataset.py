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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return torch.from_numpy(arr).view(1, 512, 512)

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
    Strictly implements the NoiseSynthesizer protocol.
    """

    def __init__(
        self,
        poisson_alpha: float = 20.0,
        gaussian_sigma: float = 0.05,
        streak_probability: float = 0.15,
    ) -> None:
        self.alpha = poisson_alpha
        self.sigma = gaussian_sigma
        self.streak_p = streak_probability

    def inject(self, clean_tensor: Tensor) -> Tensor:
        # 1. Shot noise (Poisson process)
        # Allocate once out-of-place to maintain immutability of the input tensor
        noisy = torch.poisson(clean_tensor * self.alpha)

        # In-place division to avoid allocating a new tensor
        noisy.div_(self.alpha)

        # 2. Gaussian sensor noise
        # add_ computes the scaled random tensor and accumulates in-place
        noisy.add_(torch.randn_like(noisy), alpha=self.sigma)

        # 3. Vectorized raster anomalies (charging/streaks)
        if torch.rand(1).item() < self.streak_p:
            h = clean_tensor.shape[-2]
            w = clean_tensor.shape[-1]
            num_streaks = torch.randint(1, 4, (1,)).item()

            # Vectorized index and intensity generation on the target device
            row_indices = torch.randint(0, h, (num_streaks,), device=noisy.device)
            intensities = torch.randn(num_streaks, 1, device=noisy.device) * 0.3

            # Broadcast intensities across the width dimension and add in-place
            # Note: rare index collisions will apply only the last computed intensity
            noisy[..., row_indices, :] += intensities.expand(-1, w)

        # In-place clamp
        return noisy.clamp_(0.0, 1.0)

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
