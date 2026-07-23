import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from tqdm import tqdm


class SupervisedEMDataset(Dataset):
    """
    PyTorch Dataset interfacing with paired 'clean' and 'noisy' 8-bit TIFF files.
    Includes in-RAM pre-loading to eliminate I/O bottlenecks during training iterations.
    """

    def __init__(
        self, data_dir: Path, augment: bool = False, cache_in_ram: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.clean_dir = self.data_dir / "clean"
        self.noisy_dir = self.data_dir / "noisy"
        self.augment = augment
        self.cache_in_ram = cache_in_ram

        if not self.clean_dir.exists() or not self.noisy_dir.exists():
            raise FileNotFoundError(
                f"Invalid directory layout under {self.data_dir}. Expected 'clean' and 'noisy' subdirectories."
            )

        self.filenames = sorted([p.name for p in self.clean_dir.glob("*.tif")])
        if not self.filenames:
            raise FileNotFoundError(f"No TIFF files located in {self.clean_dir}")

        noisy_set = set(p.name for p in self.noisy_dir.glob("*.tif"))
        for fname in self.filenames:
            if fname not in noisy_set:
                raise ValueError(f"Missing corresponding noisy pair for file: {fname}")

        # In-RAM storage buffers
        self.cached_clean: list[np.ndarray] = []
        self.cached_noisy: list[np.ndarray] = []

        if self.cache_in_ram:
            print(
                f"[*] Pre-loading {len(self.filenames)} image pairs into RAM ({self.data_dir.name} split)..."
            )
            for fname in tqdm(
                self.filenames, desc=f"Caching {self.data_dir.name}", leave=False
            ):
                clean_img = tifffile.imread(str(self.clean_dir / fname))
                noisy_img = tifffile.imread(str(self.noisy_dir / fname))
                self.cached_clean.append(clean_img)
                self.cached_noisy.append(noisy_img)

    def __len__(self) -> int:
        return len(self.filenames)

    def _apply_spatial_augmentations(
        self, noisy: torch.Tensor, clean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            noisy = torch.flip(noisy, dims=[2])
            clean = torch.flip(clean, dims=[2])

        if random.random() > 0.5:
            noisy = torch.flip(noisy, dims=[1])
            clean = torch.flip(clean, dims=[1])

        k = random.randint(0, 3)
        if k > 0:
            noisy = torch.rot90(noisy, k, dims=[1, 2])
            clean = torch.rot90(clean, k, dims=[1, 2])

        return noisy, clean

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_in_ram:
            clean_np = self.cached_clean[idx]
            noisy_np = self.cached_noisy[idx]
        else:
            fname = self.filenames[idx]
            clean_np = tifffile.imread(str(self.clean_dir / fname))
            noisy_np = tifffile.imread(str(self.noisy_dir / fname))

        max_val = (
            float(np.iinfo(clean_np.dtype).max)
            if np.issubdtype(clean_np.dtype, np.integer)
            else 255.0
        )

        # Cast to float32 tensor and normalize [0.0, 1.0]
        clean_tensor = (
            torch.from_numpy(clean_np.astype("float32")).unsqueeze(0) / max_val
        )
        noisy_tensor = (
            torch.from_numpy(noisy_np.astype("float32")).unsqueeze(0) / max_val
        )

        if self.augment:
            noisy_tensor, clean_tensor = self._apply_spatial_augmentations(
                noisy_tensor, clean_tensor
            )

        return noisy_tensor, clean_tensor
