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
    calibration_samples: int = 100
    calibration_workers: int = 8
    filter_type: str = "gaussian"  # "gaussian" or "median" for pseudo-clean estimation
    gaussian_sigma: float = 0.7  # Sigma for gaussian filter
    noise_amplification: float = (
        1.0  # Scale empirical noise deviation: >1.0 stronger, <1.0 weaker
    )
    photon_scale: float | list[float] = field(default_factory=lambda: [0.02, 0.08])
    read_std: float | list[float] = field(default_factory=lambda: [5.0, 15.0])
    gain_variance: float | list[float] = field(default_factory=lambda: [0.01, 0.05])
    streak_probability: float | list[float] = 0.0
    inverted_bse: bool = True  # True if membranes are dark (inverted BSE mode)


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.0001
    lr_scheduler: str = "none"  # "none" or "cosine"
    lr_min: float = 1e-6
    warmup_epochs: int = 0
    seed: Optional[int] = 42
    cache_in_ram: bool = False
    device: str = "mps"
    num_workers: int = 0
    save_all_epochs: bool = True
    compile_model: bool = False


@dataclass
class InferenceConfig:
    custom_input_dir: Optional[Path] = None
    preview_image: Optional[Path] = None
    preview_crop_size: int = 512
    preview_crop_y: Optional[int] = None
    preview_crop_x: Optional[int] = None

    def __post_init__(self):
        if self.custom_input_dir is not None and self.custom_input_dir != "":
            self.custom_input_dir = Path(self.custom_input_dir)
        else:
            self.custom_input_dir = None
        if self.preview_image is not None and self.preview_image != "":
            self.preview_image = Path(self.preview_image)
        else:
            self.preview_image = None


@dataclass
class LossConfig:
    loss_type: str = "composite"  # "composite" (legacy), "enhanced", or "dynamic"
    # Legacy composite weights (used when loss_type="composite")
    mse_weight: float = 1.0
    edge_weight: float = 0.2
    ssim_weight: float = 0.1
    # Enhanced composite weights (used when loss_type="enhanced")
    charbonnier_weight: float = 0.3
    laplacian_weight: float = 0.15
    ffl_weight: float = 0.15
    espi_weight: float = 0.1


@dataclass
class COSDDConfig:
    s_code_channels: int = 64
    number_layers: int = 14
    number_components: int = 3
    scale_initialisation: bool = True
    noise_direction: str = "x"
    use_direct_denoiser: bool = False


@dataclass
class N2VConfig:
    """
    Configuration for the N2V2 / StructN2V self-supervised blind-spot training regime.

    Attributes
    ----------
    blind_spot_strict : bool
        If True (default), the top encoder-to-decoder skip connection in N2VUNet is
        severed, preserving the blind-spot invariant as required by N2V2 theory.
        Set False to re-enable it for ablation on sub-pixel EM features.
    mask_ratio : float
        Fraction of pixels replaced per patch per forward pass. N2V default is 0.015
        (~1 in 66 pixels). Higher values increase the self-supervised signal but reduce
        spatial resolution of the learned representation.
    mask_mode : str
        Masking strategy:
          - "random"     : N2V2 mode. Random point mask with 3x3 median substitution.
          - "horizontal" : StructN2V mode. Horizontal stripe mask targeting raster
                           charging artifacts (SBF-SEM horizontal streak noise).
          - "vertical"   : StructN2V mode. Vertical stripe mask.
    """

    blind_spot_strict: bool = True
    mask_ratio: float = 0.015
    mask_mode: str = "random"  # "random", "horizontal", "vertical"


@dataclass
class N2NConfig:
    """
    Configuration for Noise2Noise (N2N) training on sbem-n2n-overlaps shards
    or duplicated SBEMimage tiles.
    """

    enabled: bool = False
    source: str = "overlaps"  # "overlaps" or "duplicates"
    overlaps_dir: str = ""
    duplicates_dir: str = ""
    pair_mode: str = "all"  # "all" or "neighbours" (duplicates mode)
    split_unit: str = "group"  # "group", "image", or "patch" (duplicates mode)
    split_seed: int = 42
    max_cache_tiles: int = (
        32  # Bounded LRU tile cache per worker when cache_in_ram=False
    )
    pair_direction: str = (
        "random_swap"  # "random_swap", "first_to_second", "second_to_first"
    )
    orientation: str = "all"  # "all", "horizontal", "vertical"
    max_train_pairs: Optional[int] = 15000
    max_val_pairs: Optional[int] = 1500
    max_test_pairs: Optional[int] = 500
    seed: int = 42
    augment_rot90: bool = False
    align_patches: bool = False
    align_subpixel: bool = False
    align_max_shift_px: int = 16
    align_blur_sigma: float = 1.0
    patch_sampling: str = (
        "random"  # "random" (1 crop/stripe/epoch) or "tile" (full stripe tiling)
    )
    tile_stride: Optional[int] = None  # Defaults to extraction.patch_size when null
    trim_enabled: bool = False
    v_trim_top_px: int = 50
    v_trim_left_px: int = 500
    v_trim_bottom_px: int = 0
    v_trim_right_px: int = 0
    h_trim_left_px: int = 32
    h_trim_top_px: int = 0
    h_trim_bottom_px: int = 0
    h_trim_right_px: int = 0
    quality_filter_enabled: bool = False
    quality_metric: str = "ncc"  # "ncc" (smoothed Pearson correlation) or "ssim"
    min_pair_quality: float = 0.60
    quality_blur_sigma: float = 1.5
    min_patch_std: float = (
        0.0  # Optional min smoothed std (0-255 space) to drop flat/empty patches
    )
    streaming: bool = False
    shuffle_buffer_size: int = 2048


@dataclass
class ModelConfig:
    model_type: str = "connectomics"  # Options: "connectomics", "residual_unet", "modern_unet", "cosdd", "n2v"
    base_dim: int = 32


@dataclass
class PipelineConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    extraction: ExtractionSettingsConfig = field(
        default_factory=ExtractionSettingsConfig
    )
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cosdd: COSDDConfig = field(default_factory=COSDDConfig)
    n2v: N2VConfig = field(default_factory=N2VConfig)
    n2n: N2NConfig = field(default_factory=N2NConfig)
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
            model=ModelConfig(**d.get("model", {})),
            cosdd=COSDDConfig(**d.get("cosdd", {})),
            n2v=N2VConfig(**d.get("n2v", {})),
            n2n=N2NConfig(**d.get("n2n", {})),
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
