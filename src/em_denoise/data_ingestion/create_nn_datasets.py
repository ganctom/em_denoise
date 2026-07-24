import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import random


# Keep your noise function as-is (it's good)
def add_em_noise(
    clean_patch: np.ndarray,
    photon_scale: float = 0.05,
    read_std: float = 8.0,
    gain_variance: float = 0.02,
) -> np.ndarray:
    """Simulates realistic EM Poisson-Gaussian + gain noise."""
    img_float = clean_patch.astype(np.float32)

    photons = np.maximum(img_float * photon_scale, 1e-5)
    noisy_photons = np.random.poisson(photons).astype(np.float32)
    noisy_signal = noisy_photons / photon_scale

    if gain_variance > 0:
        gain_factor = np.random.normal(1.0, gain_variance, size=img_float.shape).astype(
            np.float32
        )
        noisy_signal *= gain_factor

    read_noise = np.random.normal(0.0, read_std, size=img_float.shape).astype(
        np.float32
    )
    noisy = noisy_signal + read_noise

    return np.clip(noisy, 0, 255).astype(clean_patch.dtype)


def prepare_supervised_from_patches(
    patches_root: Path,
    output_base_dir: Path,
    photon_scale: float = 0.05,
    read_std: float = 8.0,
    gain_variance: float = 0.02,
    random_seed: int = 42,
):
    """
    Takes existing train/val/test patch folders and creates clean/noisy pairs
    inside each split.
    """
    patches_root = Path(patches_root)
    output_base_dir = Path(output_base_dir)

    random.seed(random_seed)
    np.random.seed(random_seed)  # for reproducible noise

    splits = ["train", "val", "test"]

    for split_name in splits:
        clean_source_dir = patches_root / split_name

        if not clean_source_dir.exists():
            print(f"Warning: {clean_source_dir} does not exist, skipping...")
            continue

        # Output directories
        split_clean_dir = output_base_dir / split_name / "clean"
        split_noisy_dir = output_base_dir / split_name / "noisy"

        split_clean_dir.mkdir(parents=True, exist_ok=True)
        split_noisy_dir.mkdir(parents=True, exist_ok=True)

        patch_files = sorted(list(clean_source_dir.glob("*.tif")))
        print(
            f"[{split_name.upper()}] Found {len(patch_files)} patches → creating clean/noisy pairs"
        )

        for fp in tqdm(patch_files, desc=f"Processing {split_name}"):
            try:
                clean_img = tifffile.imread(str(fp))

                # Add noise
                noisy_img = add_em_noise(
                    clean_img,
                    photon_scale=photon_scale,
                    read_std=read_std,
                    gain_variance=gain_variance,
                )

                # Save both
                tifffile.imwrite(str(split_clean_dir / fp.name), clean_img)
                tifffile.imwrite(str(split_noisy_dir / fp.name), noisy_img)

            except Exception as e:
                print(f"Error processing {fp.name}: {e}")

    print("\n✓ Done! Supervised dataset ready at:", output_base_dir)


# ========================== Usage ==========================
if __name__ == "__main__":
    PATCHES_ROOT = Path(
        "/Users/ganctoma/SW/sort/N2V/run_22/patches"
    )  # your new split folders
    SANDBOX_ROOT = Path("/Users/ganctoma/SW/sort/N2V/run_22/supervised_sandbox")

    prepare_supervised_from_patches(
        patches_root=PATCHES_ROOT,
        output_base_dir=SANDBOX_ROOT,
        photon_scale=0.04,  # adjust for stronger/weaker noise
        read_std=8.0,
        gain_variance=0.02,
        random_seed=42,
    )
