from pathlib import Path
from em_denoise.data_ingestion.tiles_per_section import (
    Config,
    process_sections,
    store_result,
    parse_sec_num_from_sec_dir,
    load_result,
)


def test_config_creation():
    config = Config(dir_sections="/path/to/sections", dir_out="/path/to/output")
    assert config.dir_sections == "/path/to/sections"
    assert config.dir_out == "/path/to/output"
    assert config.fn_tiles_per_section == "tiles_per_section.txt"


def test_get_fp_out():
    config = Config(dir_sections="/tmp/sections", dir_out="/tmp/output")
    fp = config.get_fp_out()
    expected = str(Path("/tmp/output") / "tiles_per_section.txt")
    assert fp == expected


def test_store_result_creates_txt_file_with_correct_format(tmp_path):
    config = Config(
        dir_sections=str(tmp_path / "sections"), dir_out=str(tmp_path / "output")
    )

    test_data = {"001": 45, "002": 0, "010": 128, "042": 67}

    store_result(config, test_data)

    output_file = Path(config.get_fp_out())

    # File exists and has correct name
    assert output_file.exists()
    assert output_file.name == "tiles_per_section.txt"

    # Read and check content
    content = output_file.read_text().strip()
    lines = content.splitlines()

    assert len(lines) == 4
    assert lines[0].startswith("001")
    assert "45" in lines[0]
    assert "\t" in lines[0] or " " in lines[0]  # tab or space separated

    # Check all sections are present and sorted
    sections = [line.split()[0] for line in lines]
    assert sections == ["001", "002", "010", "042"]


def test_store_result_uses_tab_separation(tmp_path):
    config = Config(dir_sections="/in", dir_out=str(tmp_path))
    data = {"005": 23, "012": 87}

    store_result(config, data)

    content = Path(config.get_fp_out()).read_text()
    assert "005\t23" in content
    assert "012\t87" in content


def test_parse_sec_num_from_sec_dir():
    sec_dirs = [
        "/Users/ganctoma/SW/sort/N2V/tst/sections/s20001_g0",
        "/Users/ganctoma/SW/sort/N2V/tst/sections/s1250_g0",
    ]

    sec_num = parse_sec_num_from_sec_dir(sec_dirs[0])
    assert isinstance(sec_num, int)
    assert sec_num == 20001

    sec_num = parse_sec_num_from_sec_dir(sec_dirs[1])
    assert isinstance(sec_num, int)
    assert sec_num == 1250


def test_load_result_returns_empty_on_missing_file(tmp_path):
    config = Config(dir_sections=str(tmp_path), dir_out=str(tmp_path / "nonexistent"))

    result = load_result(config)
    assert result == {}


def test_load_result_handles_empty_file(tmp_path):
    config = Config(dir_sections="/in", dir_out=str(tmp_path))

    # Create empty file
    Path(config.get_fp_out()).touch()

    result = load_result(config)
    assert result == {}


def test_process_sections():
    sections_dir = "/Users/ganctoma/SW/sort/N2V/tst/sections"

    config = Config(
        dir_sections=str(sections_dir), dir_out=str(Path(sections_dir).root)
    )

    result = process_sections(config)

    assert result == {
        1250: 18,
        1251: 18,
        20001: 55,
        20002: 55,
        20003: 55,
        20004: 55,
    }


def test_load_result():
    sections_dir = "/Users/ganctoma/SW/sort/N2V/tst/sections/"

    config = Config(dir_sections=sections_dir, dir_out=sections_dir)

    filepath = config.get_fp_out()
    assert filepath == "/Users/ganctoma/SW/sort/N2V/tst/sections/tiles_per_section.txt"

    result = load_result(str(config.get_fp_out()))
    assert result == {
        1250: 18,
        1251: 18,
        20001: 55,
        20002: 55,
        20003: 55,
        20004: 55,
    }
