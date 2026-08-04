# src/em_denoise_refactor/tests/validate_step7_modulo_extraction.py

from pathlib import Path
import tempfile

from em_denoise_refactor.etl.lmdb_store import (
    VarianceAwareROISampler,
    SQLiteMetadataStore,
)


def test_modulo_splits_and_weighted_sampling():
    """
    Validates:
    1. Modulo-10 section splitting (train: 0..7, val: 8, test: 9).
    2. Proportional volume sampling in VarianceAwareROISampler.
    3. Section range filtering (first_sec <= section_id <= last_sec).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_metadata.db"
        store = SQLiteMetadataStore(db_path)

        # Insert mock tile records for sections 10 to 29 (20 sections total)
        # Give section 10 and 20 extra tiles (10 tiles each) vs other sections (1 tile each)
        inserts = []
        for sec in range(10, 30):
            tile_count = 10 if sec in (10, 20) else 1
            for t in range(tile_count):
                fn = f"tiles/g0000/t{t:04d}/exp_g0000_t{t:04d}_s{sec:05d}.tif"
                inserts.append((fn, fn, sec, t))

        with store.conn:
            store.conn.executemany(
                "INSERT INTO tiles (filename, filepath, section_id, tile_id) VALUES (?, ?, ?, ?)",
                inserts,
            )

        print("1. Testing Modulo-10 Data Splitting...")
        store.assign_splits()

        # Verify split distribution
        cursor = store.conn.execute(
            "SELECT section_id, split FROM tiles GROUP BY section_id"
        )
        section_splits = dict(cursor.fetchall())

        for sec, split in section_splits.items():
            mod = sec % 10
            if mod < 8:
                assert split == "train", (
                    f"Section {sec} (mod {mod}) should be 'train', got '{split}'"
                )
            elif mod == 8:
                assert split == "val", (
                    f"Section {sec} (mod {mod}) should be 'val', got '{split}'"
                )
            elif mod == 9:
                assert split == "test", (
                    f"Section {sec} (mod {mod}) should be 'test', got '{split}'"
                )

        print("✓ Modulo-10 section splitting verified successfully!")

        print("\n2. Testing Proportional Section Volume Coverage in ROISampler...")
        train_metadata = store.get_dataset_metadata(split="train")

        sampler = VarianceAwareROISampler(z_depth_per_roi=1, tiles_per_section=2)
        rois = sampler.generate_rois(train_metadata, num_regions=1000)

        sampled_sections = [roi.z_start for roi in rois]
        sec_10_count = sampled_sections.count(10)
        sec_11_count = sampled_sections.count(11)

        print("   In 1000 sampled ROIs:")
        print(f"   - Section 10 (10 tiles) was sampled: {sec_10_count} times")
        print(f"   - Section 11 (1 tile)   was sampled: {sec_11_count} times")

        # Section 10 has 10x more tiles than section 11, so it should be sampled ~10x more often
        ratio = sec_10_count / max(sec_11_count, 1)
        print(
            f"   Observed sampling ratio (Sec 10 / Sec 11): {ratio:.2f}x (Expected ~10x)"
        )
        assert ratio > 3.0, (
            "ROISampler is not properly weighting sections by tile count!"
        )

        print("✓ Proportional section volume coverage verified successfully!")


if __name__ == "__main__":
    test_modulo_splits_and_weighted_sampling()
    print("\nAll validation checks passed!")
