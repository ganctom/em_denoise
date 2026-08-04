from pathlib import Path
import tempfile

from em_denoise_refactor.etl.lmdb_store import SQLiteMetadataStore


def test_sqlite_metadata_indexing():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_path = temp_path / "test.db"

        # Create mock metadata logs
        meta_logs_dir = temp_path / "meta" / "logs"
        meta_logs_dir.mkdir(parents=True)

        log_file1 = meta_logs_dir / "metadata_1.txt"
        with open(log_file1, "w") as f:
            f.write("SESSION: {'timestamp': 1769784184}\n")
            f.write(
                "TILE: {'tileid': '0000.0411.00146', 'filename': 'tiles/g0000/t0411/20260201_s00146.tif'}\n"
            )
            f.write(
                "TILE: {'tileid': '0000.0412.00146', 'filename': 'tiles/g0000/t0412/20260201_s00146.tif'}\n"
            )

        log_file2 = meta_logs_dir / "metadata_2.txt"
        with open(log_file2, "w") as f:
            # Different section
            f.write(
                "TILE: {'tileid': '0000.0411.00156', 'filename': 'tiles/g0000/t0411/20260201_s00156.tif'}\n"
            )

        store = SQLiteMetadataStore(db_path)

        # 1. Test update_db
        store.update_db(meta_logs_dir, first_sec=100, last_sec=1200)

        cursor = store.conn.execute("SELECT COUNT(*) FROM tiles")
        assert cursor.fetchone()[0] == 3, "Expected 3 tiles inserted"

        # Test exact lookups
        filepath = store.get_filepath(146, 411)
        assert filepath == "tiles/g0000/t0411/20260201_s00146.tif"

        # 2. Test splits
        store.assign_splits(val_stride=10)

        cursor = store.conn.execute(
            "SELECT section_id, split FROM tiles ORDER BY section_id"
        )
        rows = cursor.fetchall()

        # First section is 146 (index 0) => val
        assert rows[0][1] == "val"
        assert rows[1][1] == "val"  # Also 146

        # Second section is 156 (index 1). Since val_stride=10, 1 % 10 != 0, so 'train'
        assert rows[2][1] == "train"

        # 3. Test DatasetMetadata retrieval
        metadata = store.get_dataset_metadata()
        assert metadata.total_sections == 156
        assert len(metadata.tiles_per_section[146]) == 2
        assert len(metadata.tiles_per_section[156]) == 1


if __name__ == "__main__":
    test_sqlite_metadata_indexing()
    print("All tests passed!")
