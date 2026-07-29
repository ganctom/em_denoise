import json
from pathlib import Path
from typing import NamedTuple
from concurrent.futures import ProcessPoolExecutor
from pydantic import BaseModel, Field
from tqdm import tqdm

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False


class Config(BaseModel):
    dir_sections: Path
    dir_out: Path
    fn_tiles_per_section: str = "tiles_per_section.csv"
    start_sec_num: int
    end_sec_num: int
    grid_num: int
    max_workers: int = Field(default=8, description="Process pool parallelism depth")

    def get_fp_out(self) -> Path:
        """Return full Path to the output file."""
        return self.dir_out / self.fn_tiles_per_section


class SectionResult(NamedTuple):
    sec_num: int
    count: int
    missing: bool


def parse_sec_num_from_sec_dir(input_dir: Path) -> int:
    sec_id = input_dir.name.split("_")[0]
    return int(sec_id[1:])


def count_non_negative_tiles_fast(data: list) -> int:
    """Flatten nested JSON iterable structure and compute non-negative unique tile count."""
    unique_ids = set()

    def _extract(elem):
        if isinstance(elem, list):
            for sub in elem:
                _extract(sub)
        elif elem != -1:
            unique_ids.add(elem)

    _extract(data)
    return len(unique_ids)


def process_single_section(args: tuple[Path, int, int]) -> SectionResult:
    """Worker function for parallel section processing."""
    dir_sections, sec_num, grid_num = args
    json_file = dir_sections / f"s{sec_num}_g{grid_num}" / "tile_id_map.json"

    if not json_file.is_file():
        return SectionResult(sec_num=sec_num, count=0, missing=True)

    try:
        with open(json_file, "rb") as f:
            data = orjson.loads(f.read()) if HAS_ORJSON else json.load(f)
        count = count_non_negative_tiles_fast(data)
        return SectionResult(sec_num=sec_num, count=count, missing=False)
    except Exception:
        return SectionResult(sec_num=sec_num, count=0, missing=False)


def process_sections(config: Config) -> tuple[dict[int, int], list[int]]:
    """Process sections concurrently across CPU cores."""
    if not config.dir_sections.exists():
        raise FileNotFoundError(f"Sections directory not found: {config.dir_sections}")

    tasks = [
        (config.dir_sections, sec_num, config.grid_num)
        for sec_num in range(config.start_sec_num, config.end_sec_num + 1)
    ]

    result: dict[int, int] = {}
    missing_tile_id_maps: list[int] = []

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        for res in tqdm(
            executor.map(process_single_section, tasks, chunksize=128),
            total=len(tasks),
            desc="Processing sections",
            unit="section",
            ncols=100,
            smoothing=0.1,
        ):
            if res.missing:
                missing_tile_id_maps.append(res.sec_num)
            else:
                result[res.sec_num] = res.count

    return result, missing_tile_id_maps


def store_result(config: Config, data: dict[int, int]) -> None:
    """
    Store section -> tile count in comma-separated format.
    Includes a '#' commented header for metadata clarity while remaining 100%
    compatible with downstream line.split(',') deserialization.
    """
    config.dir_out.mkdir(parents=True, exist_ok=True)
    fp = config.get_fp_out()

    lines = ["#section_num,tile_count\n"]
    lines.extend(f"{sec_num},{count}\n" for sec_num, count in data.items())
    fp.write_text("".join(lines), encoding="utf-8")


def load_result(config: Config) -> dict[int, int]:
    """Load comma-separated section tile database back into a dictionary."""
    fp = config.get_fp_out()
    if not fp.is_file():
        return {}

    result = {}
    try:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    try:
                        result[int(parts[0].strip())] = int(parts[1].strip())
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warning: Failed to load result file {fp}: {e}")
        return {}

    return result


def main(config: Config) -> None:
    print(f"Processing sections from: {config.dir_sections}")
    print(f"CSV output target: {config.get_fp_out()}")

    result, missing = process_sections(config)
    store_result(config, result)

    print(f"Done! Successfully processed {len(result)} sections.")
    if missing:
        print(f"Missing tile_id_map.json files count: {len(missing)}")


if __name__ == "__main__":
    config = Config(
        dir_sections=Path(
            "/Volumes/tachyon/groups/scratch/gmicro_sem/gfriedri/_processing/SOFIMA/nextflow/ganctoma/gfriedri-em-alignment-flows/runs/roli-f1/all-01240_22814/sections"
        ),
        dir_out=Path("/Users/ganctoma/SW/sort/N2V/tst/"),
        fn_tiles_per_section="section_tile_counts.csv",
        start_sec_num=1250,
        end_sec_num=1260,
        grid_num=0,
        max_workers=16,
    )

    main(config)
