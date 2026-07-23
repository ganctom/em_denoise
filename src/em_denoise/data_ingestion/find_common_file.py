import sys
from pathlib import Path

from em_denoise.backend.utils import compute_intersection


def main() -> None:
    # Hardcoded target directories
    dir1 = Path(
        "/Users/ganctoma/SW/sort/N2V/run_19/experiments/run_00_epochs_10/predictions"
    )
    dir2 = Path("/Users/ganctoma/SW/sort/N2V/run_20/experiments/run_0/predictions")

    try:
        common_files = compute_intersection(dir1, dir2)

        if not common_files:
            print(
                "Intersection yields an empty set. No common files detected.",
                file=sys.stdout,
            )
            sys.exit(0)

        for file_path in sorted(common_files):
            print(file_path)

    except PermissionError as e:
        print(f"I/O Permission Error: {e}", file=sys.stderr)
        sys.exit(
            126
        )  # Standard bash code for command invoked cannot execute / permission denied
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
