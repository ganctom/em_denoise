from pathlib import Path
from typing import Tuple
import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    sandbox_root: Path
    experiment_dir: Path


class DatasetConfig(BaseModel):
    batch_size: int = Field(16, gt=0)
    num_workers_train: int = Field(4, ge=0)
    num_workers_val: int = Field(2, ge=0)


class UNetConfigSchema(BaseModel):
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    dropout_rate: float = Field(0.05, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    epochs: int = Field(50, gt=0)
    lr: float = Field(1e-3, gt=0.0)
    eta_min: float = Field(1e-6, ge=0.0)
    weight_decay: float = Field(
        default=1e-4, description="L2 penalty decoupled via AdamW."
    )
    warmup_epochs: int = Field(
        default=5, description="Number of epochs for linear LR warmup."
    )
    min_lr: float = Field(
        default=1e-6, description="Minimum learning rate for CosineAnnealingLR."
    )
    max_grad_norm: float = Field(
        default=1.0, description="Threshold for gradient clipping."
    )


class EMLossConfig(BaseModel):
    l1: float = 0.5
    ssim: float = 0.3
    espi: float = 0.1
    aufrc: float = 0.1


class PipelineConfig(BaseModel):
    paths: PathsConfig
    checkpoints: dict[str, Path] = Field(default_factory=dict)
    dataset: DatasetConfig
    unet: UNetConfigSchema
    training: TrainingConfig
    em_loss: EMLossConfig

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "PipelineConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file missing: {yaml_path}")
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
