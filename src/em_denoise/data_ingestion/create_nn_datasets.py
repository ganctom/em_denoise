# import numpy as np
# import tifffile
# from pathlib import Path
# from tqdm import tqdm
# import random
#
# def add_em_noise(
#         clean_patch: np.ndarray,
#         photon_scale: float = 0.05,
#         read_std: float = 8.0,
#         gain_variance: float = 0.02
# ) -> np.ndarray:
#     """
#     Simulates aggressive, realistic EM Poisson-Gaussian + Multiplicative Gain noise.
#
#     Parameters
#     ----------
#     clean_patch : np.ndarray
#         Input array (uint8).
#     photon_scale : float
#         Scaling factor converting 8-bit intensity to photoelectron count.
#         Lower values (e.g., 0.01 - 0.05) simulate lower electron dose / high beam speed,
#         resulting in severe shot noise.
#     read_std : float
#         Standard deviation of Gaussian read noise (detector readout/amplifier floor).
#     gain_variance : float
#         Multiplicative gain variation (scintillator/photomultiplier gain fluctuations).
#     """
#     img_float = clean_patch.astype(np.float32)
#
#     # 1. Map intensity to physical electron count baseline
#     # Lower photon count -> higher relative variance
#     photons = np.maximum(img_float * photon_scale, 1e-5)
#
#     # 2. Apply primary Poisson shot noise (beam electron arrival)
#     noisy_photons = np.random.poisson(photons).astype(np.float32)
#
#     # 3. Convert back to original intensity domain
#     noisy_signal = noisy_photons / photon_scale
#
#     # 4. Multiplicative gain noise (scintillator/PMT conversion non-uniformity)
#     if gain_variance > 0:
#         gain_factor = np.random.normal(1.0, gain_variance, size=img_float.shape).astype(np.float32)
#         noisy_signal *= gain_factor
#
#     # 5. Additive Gaussian read noise (amplifier electronics)
#     read_noise = np.random.normal(0.0, read_std, size=img_float.shape).astype(np.float32)
#
#     # 6. Recombine and clamp to 8-bit dynamic range
#     noisy = noisy_signal + read_noise
#     return np.clip(noisy, 0, 255).astype(clean_patch.dtype)
#
# def prepare_supervised_sandbox(
#         source_clean_dir: Path,
#         output_base_dir: Path,
#         train_ratio: float = 0.8,
#         val_ratio: float = 0.1,
#         random_seed: int = 42
# ):
#     source_clean_dir = Path(source_clean_dir)
#     output_base_dir = Path(output_base_dir)
#
#     all_files = sorted(list(source_clean_dir.glob("*.tif")))
#     if not all_files:
#         raise FileNotFoundError(f"No TIFF files found in {source_clean_dir}")
#
#     # Shuffle deterministically before splitting
#     random.seed(random_seed)
#     random.shuffle(all_files)
#
#     n_total = len(all_files)
#     n_train = int(n_total * train_ratio)
#     n_val = int(n_total * val_ratio)
#
#     splits = {
#         "train": all_files[:n_train],
#         "val": all_files[n_train:n_train + n_val],
#         "test": all_files[n_train + n_val:]
#     }
#
#     print(f"[*] Total files: {n_total}")
#     print(
#         f"[*] Split distribution -> Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
#
#     for split_name, file_list in splits.items():
#         split_clean_dir = output_base_dir / split_name / "clean"
#         split_noisy_dir = output_base_dir / split_name / "noisy"
#
#         split_clean_dir.mkdir(parents=True, exist_ok=True)
#         split_noisy_dir.mkdir(parents=True, exist_ok=True)
#
#         for fp in tqdm(file_list, desc=f"Processing {split_name} set"):
#             clean_img = tifffile.imread(str(fp))
#             noisy_img = add_em_noise(clean_img)
#
#             # Save paired files
#             tifffile.imwrite(str(split_clean_dir / fp.name), clean_img)
#             tifffile.imwrite(str(split_noisy_dir / fp.name), noisy_img)
#
#
# if __name__ == "__main__":
#     # Point to your raw 2000 high-quality patches
#     RAW_PATCHES = Path("/Users/ganctoma/SW/sort/N2V/run_19/raw_patches")
#     SANDBOX_ROOT = Path("/Users/ganctoma/SW/sort/N2V/run_19/supervised_sandbox")
#
#     prepare_supervised_sandbox(
#         source_clean_dir=RAW_PATCHES,
#         output_base_dir=SANDBOX_ROOT,
#         train_ratio=0.8,
#         val_ratio=0.1
#     )


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
    PATCHES_ROOT = Path("/Users/ganctoma/SW/sort/N2V/run_20/")  # your new split folders
    SANDBOX_ROOT = Path("/Users/ganctoma/SW/sort/N2V/run_20/supervised_sandbox")

    prepare_supervised_from_patches(
        patches_root=PATCHES_ROOT,
        output_base_dir=SANDBOX_ROOT,
        photon_scale=0.04,  # adjust for stronger/weaker noise
        read_std=8.0,
        gain_variance=0.02,
        random_seed=42,
    )
