# validate_step5.py
import time
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader

# ==========================================
# IMPORT YOUR CONCRETE IMPLEMENTATIONS HERE
# ==========================================
# Data & Storage
from em_denoise_refactor.etl.lmdb_store import LMDBStoreWriter
from em_denoise_refactor.data.dataset import LMDBPatchStore, SBFSEMDataset
from em_denoise_refactor.data.dataset import SBFSEMNoiseSynthesizer

# ETL Phase 1
from em_denoise_refactor.etl.lmdb_store import SMBStorageClient, ExtractionConfig
from em_denoise_refactor.etl.lmdb_store import VarianceAwareROISampler
from em_denoise_refactor.etl.lmdb_store import SMBHighFreqPatchExtractor


def run_phase1_remote_extraction(db_path: Path, extr_cfg: ExtractionConfig):
    """
    Connects to the remote server, parses metadata, and stages a mini-extraction to LMDB.
    """
    print("\n--- 1. Testing Remote Metadata Fetch ---")

    # Initialize the concrete remote client (e.g., S3 / HTTP)
    client = SMBStorageClient(extr_cfg)

    # Ensure it successfully connects to the server and parses the DatasetMetadata [cite: 107]
    metadata = client.fetch_metadata(force_rescan=True)
    print(f"SUCCESS: Metadata fetched. Total sections: {metadata.total_sections}")

    print("\n--- 2. Executing Mini-Extraction (5 ROIs) ---")
    sampler = VarianceAwareROISampler()

    # Pull a tiny subset (e.g., 5 ROIs) [cite: 108]
    rois = sampler.generate_rois(metadata, num_regions=5)
    extractor = SMBHighFreqPatchExtractor()

    # Initialize the LMDB writer (using the fixes from Step 1)
    writer = LMDBStoreWriter(map_size=1024 * 1024 * 500)  # 500MB ceiling for dry run
    writer.initialize(db_path)

    extracted_count = 0

    # Stream raw tiles and crop sub-volumes directly into the BinaryStoreWriter [cite: 314]
    for roi in rois:
        patch_generator: Iterator[np.ndarray] = extractor.extract_from_roi(
            client, roi, patch_size=(512, 512)
        )
        for patch_data in patch_generator:
            patch_id = f"{extracted_count:08d}"
            writer.write_patch(patch_id, patch_data)
            extracted_count += 1

    # Ensure the BinaryStoreWriter commits the transaction without throwing connection timeouts [cite: 109]
    writer.commit_transaction()
    writer.close()

    print(f"SUCCESS: Extracted {extracted_count} patches and staged to local SSD.")
    return extracted_count


def create_local_synthetic_dataset(
    sbem_root: Path, sections: int = 3, tiles_per_section: int = 2
) -> None:
    """Create a tiny SMB-style local tile hierarchy with synthetic TIFFs for offline validation."""
    tiles_root = sbem_root / "tiles" / "g0000"
    tiles_root.mkdir(parents=True, exist_ok=True)

    for section_idx in range(1, sections + 1):
        for tile_idx in range(1, tiles_per_section + 1):
            tile_dir = tiles_root / f"t{tile_idx:04d}"
            tile_dir.mkdir(parents=True, exist_ok=True)

            image = np.linspace(0, 255, 1024 * 1024, dtype=np.uint8).reshape(1024, 1024)
            image = (
                (image.astype(np.int16) + section_idx * 20 + tile_idx * 5) % 256
            ).astype(np.uint8)

            filename = f"{sbem_root.name}_g0000_t{tile_idx:04d}_s{section_idx:05d}.tif"
            tifffile.imwrite(str(tile_dir / filename), image)


def run_phase2_offline_training(db_path: Path):
    """
    Executes a dummy training loop reading strictly from the LMDB store on the MPS backend.
    """
    print("\n--- 3. Testing Offline Training Execution (MPS) ---")

    # Open the local patch store strictly in read-only zero-copy mode
    store = LMDBPatchStore()
    store.open(db_path)

    noise_injector = SBFSEMNoiseSynthesizer()
    dataset = SBFSEMDataset(store=store, noise_injector=noise_injector)

    # Ensure num_workers=0 for this test to easily trace memory mapping
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = time.time()
    for batch_idx, (noisy_inputs, clean_targets) in enumerate(dataloader):
        # Cast from ByteTensor to FloatTensor and normalize to [0.0, 1.0]
        noisy_inputs = (
            noisy_inputs.to(device, dtype=torch.float32, non_blocking=True) / 255.0
        )
        clean_targets = (
            clean_targets.to(device, dtype=torch.float32, non_blocking=True) / 255.0
        )

        # Utilize clean_targets in a mock loss calculation
        loss_placeholder = torch.mean((noisy_inputs - clean_targets) ** 2)

        print(
            f"Batch {batch_idx:02d} loaded to device | "
            f"Input Shape: {noisy_inputs.shape} | "
            f"Target Shape: {clean_targets.shape} | "
            f"Dummy Loss: {loss_placeholder.item():.4f} | "
            f"Device: {noisy_inputs.device}"
        )

        if batch_idx >= 4:  # Stop after 5 batches for the dry run
            break

    store.close()
    elapsed = time.time() - start_time
    print(f"SUCCESS: Offline execution completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    # Use a temporary directory to keep the test environment clean
    with tempfile.TemporaryDirectory() as temp_root:
        temp_root_path = Path(temp_root)
        local_sbem_root = temp_root_path / "local_sbem"
        create_local_synthetic_dataset(local_sbem_root)

        extr_cfg = ExtractionConfig(sbem_root=local_sbem_root, grid_num=0)

        lmdb_path = temp_root_path / "patch_store.lmdb"

        print("==================================================")
        print(" STARTING PIPELINE VALIDATION STEP 5")
        print("==================================================")
        patches_staged = run_phase1_remote_extraction(lmdb_path, extr_cfg)

        if patches_staged > 0:
            # Step 5B: Test offline execution
            run_phase2_offline_training(lmdb_path)
        else:
            print("FAILED: No patches were extracted. Skipping offline test.")
