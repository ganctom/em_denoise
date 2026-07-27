import itertools
from pathlib import Path
import random
import shutil
from typing import Dict, Optional, Set

import numpy as np
import tifffile
from tqdm import tqdm


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
    """Evaluates whether an EM patch contains valid structural information while

    filtering out resin, empty background, and silver precipitate
    contamination.
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
    n_test_raw: Optional[int] = None,
    **filter_kwargs,
) -> Dict[str, int]:
    """Extract patches with proper train/val/test split at IMAGE level.

    Creates two distinct test directories: - test_patches: patch-level crop dataset
    (like train/val) - test_raw: full-scale raw .tif image targets
    """
    root = Path(root)
    dest_root = Path(dest_root)

    # Directories structure setup
    split_dirs = {
        "train": dest_root / "train",
        "val": dest_root / "val",
        "test_patches": dest_root / "test_patches",
        "test_raw": dest_root / "test_raw",
    }
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

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
    print(f"  Train images      : {len(train_images)}")
    print(f"  Val images        : {len(val_images)}")
    print(f"  Test images total : {len(test_images)}")

    splits = {"train": train_images, "val": val_images, "test": test_images}
    total_saved = {}

    # Set threshold for how many full-scale test files to dump to test_raw
    max_raw_test_count = n_test_raw if n_test_raw is not None else len(test_images)

    for split_name, image_list in splits.items():
        print(f"\n=== Extracting {split_name} data ===")
        saved_count = 0
        raw_saved_count = 0
        rejected = 0
        pbar = tqdm(total=len(image_list), desc=f"{split_name}", unit="img")

        # Map current split target directory for standard patches
        patch_dest_dir = (
            split_dirs["test_patches"]
            if split_name == "test"
            else split_dirs[split_name]
        )

        for tile_id, section in image_list:
            tile_str = f"{tile_id:04d}"
            slice_str = f"{section:05d}"
            folder_name = f"t{tile_str}"
            filename = f"20260201_RoLi_F1_g0000_t{tile_str}_s{slice_str}.tif"

            src_path = root / folder_name / filename
            if not src_path.exists():
                continue

            # Process full-scale image output for test dataset split
            if split_name == "test" and raw_saved_count < max_raw_test_count:
                dest_raw_path = split_dirs["test_raw"] / filename
                shutil.copy2(src_path, dest_raw_path)
                raw_saved_count += 1

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
                    dest_path = patch_dest_dir / patch_filename

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
        if split_name == "test":
            print(f"  Saved {raw_saved_count:,} full-scale .tif images to test_raw")

    return total_saved


# ========================== Usage ==========================
if __name__ == "__main__":
    sbem_dir = "/Volumes/tachyon/groups/scratch/gmicro_sem/gfriedri/_EM_acquisitions/20260201_RoLi_F1/tiles/g0000"
    root_dir = "/Users/ganctoma/SW/sort/N2V/run_25/"
    raw_dir = "raw_patches"
    sandbox_dir = "supervised_sandbox"
    fn_tile_ids = "all_tile_ids.txt"

    sbem_dir, root_dir = map(Path, (sbem_dir, root_dir))

    extract_and_save_sbem_patches_split(
        root=sbem_dir,
        dest_root=root_dir / raw_dir,
        tile_ids_file="/Users/ganctoma/SW/sort/N2V/all_tile_ids.txt",
        total_images=800,  # total full images to sample
        train_ratio=0.8,
        val_ratio=0.1,
        patches_per_image=12,
        patch_size=128,
        section_range=range(2000, 9000),
        random_seed=43,
        n_test_raw=20,  # Define explicit number of raw full-scale .tif files for test_raw
        min_std=20.0,
        max_mean_ratio=0.55,
        dark_pixel_thresh=15.0,
        max_dark_pixel_ratio=0.05,
    )

    from create_nn_datasets import prepare_supervised_from_patches

    prepare_supervised_from_patches(
        patches_root=root_dir / raw_dir,
        output_base_dir=root_dir / sandbox_dir,
        photon_scale=0.04,
        read_std=8.0,
        gain_variance=0.02,
        random_seed=42,
    )
