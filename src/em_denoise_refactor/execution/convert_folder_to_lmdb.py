import argparse
from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm
from em_denoise_refactor.etl.lmdb_store import LMDBStoreWriter


def convert_tiff_folder_to_lmdb(
    input_dir: Path | str, output_lmdb_path: Path | str
) -> None:
    input_dir = Path(input_dir)
    output_lmdb_path = Path(output_lmdb_path)

    # Normalize root_dir (handles /train or /train/clean input paths gracefully)
    root_dir = (
        input_dir.parent
        if input_dir.name in ["train", "val", "test", "clean"]
        else input_dir
    )
    if root_dir.name in ["clean", "noisy"]:
        root_dir = root_dir.parent.parent

    print(
        f"Converting TIFF patch dataset from {root_dir} to LMDB at {output_lmdb_path}..."
    )

    writer = LMDBStoreWriter(map_size=100 * 1024 * 1024 * 1024)
    writer.initialize(output_lmdb_path)

    splits = ["train", "val", "test"]
    total_written = 0

    for split in splits:
        clean_dir = root_dir / split / "clean"
        if not clean_dir.exists():
            if split == "train" and (root_dir / "clean").exists():
                clean_dir = root_dir / "clean"
            else:
                continue

        tiffs = sorted(list(clean_dir.glob("*.tif")) + list(clean_dir.glob("*.tiff")))
        if not tiffs:
            continue

        for p in tqdm(tiffs, desc=f"Converting {split}"):
            arr = tifffile.imread(str(p))
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)

            patch_key = f"{split}_{p.stem}"
            writer.write_patch(patch_key, arr)
            total_written += 1

    writer.commit_transaction()
    writer.close()
    print(f"✓ Successfully wrote {total_written} patches to LMDB at {output_lmdb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TIFF patch directory structure to zero-copy LMDB store."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing train/val/test clean TIFF patches.",
    )
    parser.add_argument(
        "-o",
        "--output-lmdb",
        type=str,
        required=True,
        help="Path to output patch_store.lmdb file/directory.",
    )
    args = parser.parse_args()

    convert_tiff_folder_to_lmdb(args.input_dir, args.output_lmdb)
