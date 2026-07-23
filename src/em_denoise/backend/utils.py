#!/usr/bin/env python3
from pathlib import Path


def get_relative_files(directory: Path) -> set[Path]:
    """
    Traverses the directory structure recursively and returns a set
    of relative file paths to normalize comparison against the base path.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Target is not a resolvable directory: {directory}")

    return {
        path.relative_to(directory) for path in directory.rglob("*") if path.is_file()
    }


def compute_intersection(dir_a: str | Path, dir_b: str | Path) -> set[Path]:
    path_a = Path(dir_a).resolve()
    path_b = Path(dir_b).resolve()

    files_a = get_relative_files(path_a)
    files_b = get_relative_files(path_b)

    return files_a & files_b
