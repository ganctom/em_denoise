# src/em_denoise_refactor/config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
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
    synthesizer_type: str = "parametric"  # "parametric" or "empirical"
    empirical_model_path: str = "noise_model.npy"
    calibrate_on_the_fly: bool = False
    filter_type: str = "gaussian"  # "gaussian" or "median" for pseudo-clean estimation
    gaussian_sigma: float = 0.7  # Sigma for gaussian filter
    photon_scale: float | list[float] = field(default_factory=lambda: [0.02, 0.08])
    read_std: float | list[float] = field(default_factory=lambda: [5.0, 15.0])
    gain_variance: float | list[float] = field(default_factory=lambda: [0.01, 0.05])
    streak_probability: float | list[float] = field(
        default_factory=lambda: [0.05, 0.20]
    )


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.0001
    cache_in_ram: bool = False
    device: str = "mps"


@dataclass
class InferenceConfig:
    custom_input_dir: Optional[Path] = None

    def __post_init__(self):
        if self.custom_input_dir is not None and self.custom_input_dir != "":
            self.custom_input_dir = Path(self.custom_input_dir)
        else:
            self.custom_input_dir = None


@dataclass
class LossConfig:
    mse_weight: float = 1.0
    edge_weight: float = 0.2
    ssim_weight: float = 0.1


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
    loss: LossConfig = field(default_factory=LossConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            experiment=ExperimentConfig(**d.get("experiment", {})),
            data=DataConfig(**d.get("data", {})),
            splitting=SplittingConfig(**d.get("splitting", {})),
            extraction=ExtractionSettingsConfig(**d.get("extraction", {})),
            noise=NoiseConfig(**d.get("noise", {})),
            training=TrainingConfig(**d.get("training", {})),
            loss=LossConfig(**d.get("loss", {})),
            inference=InferenceConfig(**d.get("inference", {})),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str = "config.yaml") -> "PipelineConfig":
        path_str = str(yaml_path).lstrip("=")
        path = Path(path_str)
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

        resolved_path = path.resolve()
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = cls.from_dict(data or {})
        config._loaded_from = resolved_path
        return config
