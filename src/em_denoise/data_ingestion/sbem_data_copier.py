from pathlib import Path
import random
import numpy as np
import tifffile
from tqdm import tqdm
import itertools
from typing import Set, Optional, Dict


def load_tile_ids(txt_path: str | Path) -> Set[int]:
    """Load tile IDs from all_tile_ids.txt (one integer per line)."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Tile ID file not found: {txt_path}")

    tile_ids = set()
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    tile_ids.add(int(line))
                except ValueError:
                    print(f"Warning: Skipping invalid line: {line}")
    return tile_ids


def is_informative_patch(
    patch: np.ndarray,
    min_std: float = 5.0,
    max_mean_ratio: float = 0.95,
    dark_pixel_thresh: float = 15.0,
    max_dark_pixel_ratio: float = 0.05,
) -> bool:
    """
    Evaluates whether an EM patch contains valid structural information while filtering
    out resin, empty background, and silver precipitate contamination.

    Parameters
    ----------
    patch : np.ndarray
        Extracted image patch.
    min_std : float
        Minimum pixel standard deviation threshold to reject flat/empty resin.
    max_mean_ratio : float
        Maximum allowed mean intensity relative to dynamic range maximum.
    dark_pixel_thresh : float
        Absolute pixel intensity boundary below which a pixel is classified as 'dark/silver'.
        For 8-bit images, values < 10-20 typically represent silver precipitates.
    max_dark_pixel_ratio : float
        Maximum allowed fraction (0.0 to 1.0) of dark pixels in a patch. If exceeded,
        the patch is classified as contaminated resin and rejected.
    """
    dtype_info = (
        np.iinfo(patch.dtype) if np.issubdtype(patch.dtype, np.integer) else None
    )
    max_val = dtype_info.max if dtype_info else 1.0

    # 1. Reject saturated / bright empty resin background
    if np.mean(patch) > (max_val * max_mean_ratio):
        return False

    # 2. Reject featureless patches (low variance)
    if np.std(patch) < min_std:
        return False

    # 3. Reject resin patches contaminated with dark silver particles
    # Calculate fraction of pixels falling below dark threshold
    dark_pixel_count = np.count_nonzero(patch < dark_pixel_thresh)
    dark_pixel_ratio = dark_pixel_count / patch.size

    if dark_pixel_ratio > max_dark_pixel_ratio:
        return False

    return True


def extract_and_save_sbem_patches_split(
    root: str | Path,
    dest_root: str | Path,
    tile_ids_file: str | Path = "all_tile_ids.txt",
    total_images: int = 1000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    patches_per_image: int = 10,
    patch_size: int = 256,
    section_range: range = range(0, 21000),
    random_seed: Optional[int] = 42,
    max_attempts: int = 10_000_000,
    **filter_kwargs,
) -> Dict[str, int]:
    """
    Extract patches with proper train/val/test split at IMAGE level.
    """
    root = Path(root)
    dest_root = Path(dest_root)

    # Create three folders
    for split in ["train", "val", "test"]:
        (dest_root / split).mkdir(parents=True, exist_ok=True)

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    tile_ids = load_tile_ids(tile_ids_file)
    tile_list = sorted(tile_ids)
    section_list = list(section_range)

    # === Step 1: Generate all possible image pairs and shuffle ===
    all_possible_images = list(itertools.product(tile_list, section_list))
    random.shuffle(all_possible_images)

    # === Step 2: Assign full images to splits ===
    n_train = int(total_images * train_ratio)
    n_val = int(total_images * val_ratio)
    n_test = total_images - n_train - n_val

    train_images = all_possible_images[:n_train]
    val_images = all_possible_images[n_train : n_train + n_val]
    test_images = all_possible_images[n_train + n_val : n_train + n_val + n_test]

    print("Split summary:")
    print(f"  Train images : {len(train_images)}")
    print(f"  Val images   : {len(val_images)}")
    print(f"  Test images  : {len(test_images)}")

    # === Step 3: Process each split separately ===
    splits = {"train": train_images, "val": val_images, "test": test_images}

    total_saved = {}

    for split_name, image_list in splits.items():
        print(f"\n=== Extracting {split_name} patches ===")
        saved_count = 0
        rejected = 0
        pbar = tqdm(total=len(image_list), desc=f"{split_name}", unit="img")

        for tile_id, section in image_list:
            tile_str = f"{tile_id:04d}"
            slice_str = f"{section:05d}"
            folder_name = f"t{tile_str}"
            filename = f"20260201_RoLi_F1_g0000_t{tile_str}_s{slice_str}.tif"

            src_path = root / folder_name / filename
            if not src_path.exists():
                continue

            try:
                img = tifffile.imread(str(src_path))
                h, w = img.shape[-2:]

                if h < patch_size or w < patch_size:
                    continue

                y_bins = h // patch_size
                x_bins = w // patch_size
                possible_coords = list(itertools.product(range(y_bins), range(x_bins)))
                random.shuffle(possible_coords)

                saved_from_this = 0
                for by, bx in possible_coords:
                    if saved_from_this >= patches_per_image:
                        break

                    y = by * patch_size
                    x = bx * patch_size
                    patch = img[y : y + patch_size, x : x + patch_size]

                    if not is_informative_patch(patch, **filter_kwargs):
                        rejected += 1
                        continue

                    patch_filename = (
                        f"t{tile_str}_s{slice_str}_p{saved_from_this:02d}.tif"
                    )
                    dest_path = dest_root / split_name / patch_filename

                    tifffile.imwrite(str(dest_path), patch)
                    saved_count += 1
                    saved_from_this += 1

                if saved_from_this > 0:
                    pbar.update(1)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        pbar.close()
        total_saved[split_name] = saved_count
        print(f"  Saved {saved_count:,} patches | Rejected {rejected:,}")

    return total_saved


# ========================== Usage ==========================
if __name__ == "__main__":
    sbem_dir = "/Volumes/tachyon/groups/scratch/gmicro_sem/gfriedri/_EM_acquisitions/20260201_RoLi_F1/tiles/g0000"
    root_dir = "/Users/ganctoma/SW/sort/N2V/run_22/supervised_sandbox"
    raw_dir = "raw_patches"
    fn_tile_ids = "all_tile_ids.txt"
    section_start = 2000
    section_end = 9000
    nr_of_patches = 80

    sbem_dir, root_dir = map(Path, (sbem_dir, root_dir))

    extract_and_save_sbem_patches_split(
        root=sbem_dir,
        dest_root=root_dir,
        tile_ids_file="/Users/ganctoma/SW/sort/N2V/all_tile_ids.txt",
        total_images=1000,  # total full images to sample
        train_ratio=0.8,
        val_ratio=0.1,
        patches_per_image=12,  # you can use more for train
        patch_size=256,
        section_range=range(2000, 9000),
        random_seed=43,
        min_std=20.0,
        max_mean_ratio=0.55,
        dark_pixel_thresh=15.0,
        max_dark_pixel_ratio=0.05,
    )
