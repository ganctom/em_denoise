import gc
import time
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.em_denoise_refactor.etl.lmdb_store import LMDBStoreWriter
from src.em_denoise_refactor.data.dataset import (
    LMDBPatchStore,
    SBFSEMDataset,
    SBFSEMNoiseSynthesizer,
)


def validate_lmdb_integrity(db_path: Path):
    print("--- 1. Testing LMDB Write/Read Integrity ---")
    np.random.seed(42)
    patch_size = (1, 512, 512)
    num_patches = 100

    # 1. Initialize Writer via context manager
    with LMDBStoreWriter(map_size=1024 * 1024 * 50) as writer:
        writer.initialize(db_path)
        ground_truth = {}
        for i in range(num_patches):
            patch_id_str = f"{i:08d}"
            patch_data = np.random.randint(0, 255, size=patch_size, dtype=np.uint8)
            ground_truth[i] = patch_data
            writer.write_patch(patch_id_str, patch_data)
        writer.commit_transaction()

    print(f"Successfully wrote {num_patches} patches to {db_path}")

    # 2. Initialize Reader using context manager
    with LMDBPatchStore() as reader:
        reader.open(db_path)
        for patch_index, original_data in ground_truth.items():
            read_tensor = reader.get_patch(patch_index)
            read_data_np = read_tensor.numpy()

            np.testing.assert_array_equal(
                original_data,
                read_data_np,
                err_msg=f"Data corruption detected on patch index {patch_index}",
            )

    print("SUCCESS: Read data matches ground truth perfectly.\n")


def validate_throughput_benchmark(db_path: Path):
    print("--- 2. Executing DataLoader Throughput Benchmark ---")
    gc.collect()

    with LMDBPatchStore() as store:
        store.open(db_path)

        # Benchmark RAW zero-copy I/O throughput without CPU noise synthesis bottleneck
        dataset = SBFSEMDataset(store=store, noise_injector=None)

        # Enable multiprocessing workers if needed, or use optimized batch reads
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            num_workers=0,  # Zero-copy mmap works fastest in-process
            shuffle=False,
            pin_memory=False,
        )

        target_iterations = 10000
        patches_processed = 0

        start_time = time.perf_counter()

        while patches_processed < target_iterations:
            for batch in dataloader:
                current_batch_size = (
                    batch.size(0) if isinstance(batch, torch.Tensor) else len(batch[0])
                )
                patches_processed += current_batch_size

                if patches_processed >= target_iterations:
                    break

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        throughput = patches_processed / elapsed_time

        print(f"Processed {patches_processed} patches in {elapsed_time:.2f} seconds.")
        print(f"Throughput: {throughput:.2f} patches/second.")

        if throughput > 10000:
            print("SUCCESS: Target throughput (>10,000 patches/sec) achieved.\n")
        else:
            print(
                "WARNING: Throughput fell below 10,000 patches/sec limit. Check memory mapping overhead.\n"
            )


def validate_noise_visualization():
    print("--- 3. Visualizing Procedural Noise Injection ---")

    clean_patch = torch.ones(1, 1, 512, 512) * 0.5
    clean_patch[:, :, 250:260, :] = 0.1

    synthesizer = SBFSEMNoiseSynthesizer()
    noisy_patch = synthesizer(clean_patch)

    clean_np = clean_patch.squeeze().numpy()
    noisy_np = noisy_patch.squeeze().numpy()

    plt.figure(figsize=(12.0, 6.0))

    plt.subplot(1, 2, 1)
    plt.title("Pristine Ground Truth")
    plt.imshow(clean_np, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Degraded Input (Synthetic Noise)")
    plt.imshow(noisy_np, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    plt.tight_layout()
    print("Displaying matplotlib window. Close the window to complete the test.")
    plt.show()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_store_dir:
        db_path = Path(temp_store_dir) / "patch_store"

        try:
            validate_lmdb_integrity(db_path)
            validate_throughput_benchmark(db_path)
            validate_noise_visualization()

            print("Validation Step 1 Completed Successfully.")
        except AssertionError as e:
            print(f"\nINTEGRITY FAILURE: {e}")
        except Exception as e:
            print(f"\nRUNTIME ERROR: {e}")
