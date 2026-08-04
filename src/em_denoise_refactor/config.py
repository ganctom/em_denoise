# src/em_denoise_refactor/config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
import yaml


@dataclass
class ExperimentConfig:
    name: str = "N2V_run_32"
    output_dir: Path = field(
        default_factory=lambda: Path("/Users/ganctoma/SW/sort/N2V/run_32")
    )

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class DataConfig:
    sbem_root: Path = field(
        default_factory=lambda: Path(
            "/Volumes/tachyon/groups/scratch/gmicro_sem/gfriedri/_EM_acquisitions/20260201_RoLi_F1"
        )
    )
    grid_num: int = 0
    first_sec: int = 1250
    last_sec: int = 3000

    def __post_init__(self):
        self.sbem_root = Path(self.sbem_root)


@dataclass
class SplittingConfig:
    strategy: str = "block"
    block_size: int = 100
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    guard_band_size: int = 5


@dataclass
class ExtractionSettingsConfig:
    patch_size: int = 512
    patches_per_tile: int = 4
    tiles_per_section: int = 4
    z_depth_per_roi: int = 1
    target_train_patches: int = 100
    target_val_patches: int = 10
    target_test_patches: int = 10


@dataclass
class NoiseConfig:
    poisson_alpha: float = 20.0
    gaussian_sigma: float = 0.05
    streak_probability: float = 0.15


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.0001
    device: str = "mps"


@dataclass
class PipelineConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    extraction: ExtractionSettingsConfig = field(
        default_factory=ExtractionSettingsConfig
    )
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            experiment=ExperimentConfig(**d.get("experiment", {})),
            data=DataConfig(**d.get("data", {})),
            splitting=SplittingConfig(**d.get("splitting", {})),
            extraction=ExtractionSettingsConfig(**d.get("extraction", {})),
            noise=NoiseConfig(**d.get("noise", {})),
            training=TrainingConfig(**d.get("training", {})),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str = "config.yaml") -> "PipelineConfig":
        path = Path(yaml_path)
        if not path.exists():
            candidates = [
                Path("src/em_denoise_refactor/config.yaml"),
                Path("config.yaml"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"Configuration file not found at {path.resolve()}"
                )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data or {})
