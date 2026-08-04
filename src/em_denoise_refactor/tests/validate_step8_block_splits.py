# src/em_denoise_refactor/tests/validate_step8_block_splits.py

from pathlib import Path
import tempfile
from em_denoise_refactor.etl.lmdb_store import SQLiteMetadataStore


def test_volumetric_block_splits_with_guard_bands():
    """
    Validates:
    1. Volumetric block partitioning across sections.
    2. Guard band buffer zones placed between train, val, and test blocks.
    3. Zero section overlap or leakage between train and val/test splits.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_block_splits.db"
        store = SQLiteMetadataStore(db_path)

        # Populate mock tiles for 200 contiguous sections (sections 1000 to 1199)
        inserts = []
        for sec in range(1000, 1200):
            fn = f"tiles/g0000/t0000/exp_g0000_t0000_s{sec:05d}.tif"
            inserts.append((fn, fn, sec, 0))

        with store.conn:
            store.conn.executemany(
                "INSERT INTO tiles (filename, filepath, section_id, tile_id) VALUES (?, ?, ?, ?)",
                inserts,
            )

        print(
            "1. Testing Volumetric Block Splitting with Guard Bands (block_size=100, guard=5)..."
        )
        store.assign_splits(
            strategy="block",
            block_size=100,
            train_ratio=0.8,
            val_ratio=0.1,
            guard_band_size=5,
        )

        cursor = store.conn.execute(
            "SELECT section_id, split FROM tiles GROUP BY section_id ORDER BY section_id"
        )
        sec_splits = dict(cursor.fetchall())

        train_secs = [s for s, split in sec_splits.items() if split == "train"]
        val_secs = [s for s, split in sec_splits.items() if split == "val"]
        test_secs = [s for s, split in sec_splits.items() if split == "test"]
        guard_secs = [s for s, split in sec_splits.items() if split == "guard_band"]

        print(
            f"   Splits summary: {len(train_secs)} train, {len(val_secs)} val, {len(test_secs)} test, {len(guard_secs)} guard_band sections."
        )

        # Verify 3D Guard Band distance between splits
        for train_s in train_secs:
            for val_s in val_secs:
                dist = abs(train_s - val_s)
                assert dist >= 5, (
                    f"3D Spatial Leakage detected! Train sec {train_s} and Val sec {val_s} are only {dist} sections apart (expected >= 5)"
                )

            for test_s in test_secs:
                dist = abs(train_s - test_s)
                assert dist >= 5, (
                    f"3D Spatial Leakage detected! Train sec {train_s} and Test sec {test_s} are only {dist} sections apart (expected >= 5)"
                )

        print(
            "✓ Zero 3D feature leakage verified! All train sections are >= 5 sections away from val/test."
        )

        # Verify metadata retrieval filters out guard_band sections
        train_meta = store.get_dataset_metadata(split="train")
        assert set(train_meta.tiles_per_section.keys()) == set(train_secs), (
            "Metadata for 'train' split contains non-train sections!"
        )

        print("✓ Metadata filtering verified!")


if __name__ == "__main__":
    test_volumetric_block_splits_with_guard_bands()
    print("\nAll volumetric block split validation checks passed!")
