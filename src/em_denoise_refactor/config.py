# src/em_denoise_refactor/config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Tuple
import yaml


def normalize_hw_pair(
    value: int | Sequence[int], name: str = "patch_size"
) -> Tuple[int, int]:
    """
    Normalize a scalar int or 2-element (H, W) sequence into a positive (height, width) tuple.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive int or 2-element (H, W) sequence, got bool"
        )
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
        return int(value), int(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        h, w = value
        if (
            isinstance(h, bool)
            or isinstance(w, bool)
            or not isinstance(h, int)
            or not isinstance(w, int)
        ):
            raise ValueError(f"{name} elements must be positive ints, got {value!r}")
        if h <= 0 or w <= 0:
            raise ValueError(f"{name} dimensions must be > 0, got {(h, w)}")
        return int(h), int(w)
    raise ValueError(
        f"{name} must be a positive int or 2-element (H, W) sequence, got {value!r}"
    )


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
    origin: Optional[int] = (
        None  # Absolute block-grid anchor (e.g. 0); required when z_context_radius > 0
    )


@dataclass
class ExtractionSettingsConfig:
    patch_size: int | list[int] = 512
    patches_per_tile: int = 4
    tiles_per_section: int = 4
    z_depth_per_roi: int = 1
    target_train_patches: int = 100
    target_val_patches: int = 10
    target_test_patches: int = 10

    def __post_init__(self):
        ph, pw = normalize_hw_pair(self.patch_size, "extraction.patch_size")
        self.patch_size = ph if ph == pw else [ph, pw]

    @property
    def patch_hw(self) -> Tuple[int, int]:
        return normalize_hw_pair(self.patch_size, "extraction.patch_size")


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
    samples_per_epoch: Optional[int] = None


@dataclass
class InferenceConfig:
    custom_input_dir: Optional[Path] = None
    preview_image: Optional[Path] = None
    preview_crop_size: int = 512
    preview_crop_y: Optional[int] = None
    preview_crop_x: Optional[int] = None
    tiled: bool = False
    tile_size: Optional[int | Tuple[int, int]] = None
    z_register: bool = True
    z_register_max_shift_px: int = 64
    z_register_block_size: int = 1024

    def __post_init__(self):
        if self.custom_input_dir is not None and self.custom_input_dir != "":
            self.custom_input_dir = Path(self.custom_input_dir)
        else:
            self.custom_input_dir = None
        if self.preview_image is not None and self.preview_image != "":
            self.preview_image = Path(self.preview_image)
        else:
            self.preview_image = None
        self.tiled = bool(self.tiled)
        if self.tile_size is not None:
            th, tw = normalize_hw_pair(self.tile_size, "inference.tile_size")
            self.tile_size = th if th == tw else (th, tw)


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
    tile_stride: Optional[int | list[int]] = (
        None  # Defaults to extraction.patch_size when null
    )
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
    crop_cache_dir: Optional[str] = (
        None  # Optional persistent .npy mmap crop cache dir for tile mode
    )
    max_tiles_per_stripe: Optional[int] = (
        None  # Max kept tiles per stripe in streaming tile mode (null = auto)
    )
    prep_workers: int = 1  # Threads for alignment & QA during tile materialization
    streaming: bool = False
    shuffle_buffer_size: int = 2048
    # True 2.5D Noise2Noise configuration (0 = standard 2D N2N, 1 = 3-slice, 2 = 5-slice)
    z_context_radius: int = 0
    z_context_view: str = "full"  # "full" | "center_only" (same-patch 2D control) | "center_filled" (ablation)
    z_eval_radius: Optional[int] = (
        None  # Optional inner-channel slice when evaluating R=1 on a frozen R=2 cache
    )
    z_neighbor_dropout_prob: float = (
        0.15  # Per-neighbor center-fill dropout probability during training
    )
    z_flip_augment: bool = (
        False  # Default False (SBEM subsurface BSE asymmetry); opt-in A/B
    )
    # Locked Phase 0 registration thresholds
    z_align_prior_search_px: int = (
        32  # Prior-centered search half-width around (mov.fy0-ref.fy0, mov.fx0-ref.fx0)
    )
    z_align_min_dog_ncc: float = 0.15  # Fine-stage 1x DoG-NCC threshold
    z_align_min_psr: float = (
        7.5  # Coarse-stage fixed-window Peak-to-Sidelobe Ratio threshold
    )
    z_align_high_ncc_override: float = 0.45  # Wide-lobe soma high-NCC override
    z_align_high_ncc_min_psr: float = 6.0  # Minimum coarse PSR for high-NCC override
    z_align_missed_cut_max_ncc: float = (
        0.85  # Section-median hop DoG-NCC upper bound (missed-cut rejection)
    )
    z_align_closure_tol_px: int = 2  # Cycle closure max error (px)
    z_align_tile_consistency_tol_px: int = (
        2  # Per-first_exposed_tile_id _v/_h shift consistency tolerance (px)
    )
    z_align_consensus_tol_px: float = (
        12.0  # Outer section-median physical shift sanity tolerance (<=0 disables)
    )
    z_align_local_refine_px: int = (
        0  # Per-patch local integer Z-refine (0=stripe-rigid default; benchmark 0 vs 4)
    )

    def __post_init__(self):
        if self.tile_stride is not None:
            sh, sw = normalize_hw_pair(self.tile_stride, "n2n.tile_stride")
            self.tile_stride = sh if sh == sw else [sh, sw]
        if isinstance(self.z_context_radius, bool) or self.z_context_radius not in (
            0,
            1,
            2,
        ):
            raise ValueError(
                f"n2n.z_context_radius must be 0, 1, or 2, got {self.z_context_radius!r}"
            )
        self.z_context_view = str(self.z_context_view).lower().strip()
        if self.z_context_view not in ("full", "center_only", "center_filled"):
            raise ValueError(
                f"n2n.z_context_view must be 'full', 'center_only', or 'center_filled', "
                f"got {self.z_context_view!r}"
            )
        if not (0.0 <= float(self.z_neighbor_dropout_prob) < 1.0):
            raise ValueError(
                f"n2n.z_neighbor_dropout_prob must be in [0.0, 1.0), got {self.z_neighbor_dropout_prob}"
            )
        if self.z_align_prior_search_px <= 0:
            raise ValueError(
                f"n2n.z_align_prior_search_px must be > 0, got {self.z_align_prior_search_px}"
            )
        if self.z_align_local_refine_px < 0:
            raise ValueError(
                f"n2n.z_align_local_refine_px must be >= 0, got {self.z_align_local_refine_px}"
            )
        if self.z_eval_radius is not None:
            if isinstance(self.z_eval_radius, bool) or not (
                0 <= int(self.z_eval_radius) <= self.z_context_radius
            ):
                raise ValueError(
                    f"n2n.z_eval_radius must be in [0, z_context_radius={self.z_context_radius}], "
                    f"got {self.z_eval_radius!r}"
                )


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

    def validate(self) -> None:
        """Validate cross-section configuration invariants (especially for 2.5D N2N)."""
        import logging

        r = int(self.n2n.z_context_radius)
        if r == 0:
            if self.n2n.z_context_view != "full":
                raise ValueError(
                    f"n2n.z_context_view='{self.n2n.z_context_view}' requires n2n.z_context_radius > 0."
                )
            return

        if not self.n2n.enabled:
            raise ValueError(f"n2n.z_context_radius={r} requires n2n.enabled=True.")
        source = str(self.n2n.source).lower().strip()
        if source != "overlaps":
            raise ValueError(
                f"n2n.z_context_radius={r} is only supported with n2n.source='overlaps', got '{source}'."
            )
        pair_dir = str(self.n2n.pair_direction).lower().strip()
        if pair_dir != "first_to_second":
            raise ValueError(
                f"n2n.z_context_radius={r} requires n2n.pair_direction='first_to_second' so all input "
                f"channels are virgin single-exposure tissue, got '{pair_dir}'."
            )
        if self.n2n.streaming:
            raise ValueError(
                f"n2n.z_context_radius={r} is incompatible with n2n.streaming=True; "
                "use patch_sampling='tile' with crop_cache_dir or cache_in_ram=True."
            )
        if str(self.n2n.patch_sampling).lower().strip() != "tile":
            raise ValueError(
                f"n2n.z_context_radius={r} requires n2n.patch_sampling='tile', got '{self.n2n.patch_sampling}'."
            )
        m_type = str(self.model.model_type).lower().strip()
        if m_type in ("n2v", "noise2void", "n2v2", "cosdd", "cosdd_lvae"):
            raise ValueError(
                f"n2n.z_context_radius={r} is incompatible with model_type='{self.model.model_type}'. "
                "Use 'residual_unet', 'connectomics', or 'modern_unet'."
            )
        if str(self.splitting.strategy).lower().strip() != "block":
            raise ValueError(
                f"n2n.z_context_radius={r} requires splitting.strategy='block' to prevent Z-leakage, "
                f"got '{self.splitting.strategy}'."
            )
        if int(self.splitting.guard_band_size) < r:
            raise ValueError(
                f"n2n.z_context_radius={r} requires splitting.guard_band_size >= {r} to guarantee "
                f"disjoint Z-support sets across splits, got {self.splitting.guard_band_size}."
            )
        if self.splitting.origin is None:
            raise ValueError(
                f"n2n.z_context_radius={r} requires an explicit absolute splitting.origin (e.g. 0) "
                "so block boundaries cannot shift across filtered queries or superset cache slices."
            )
        if self.n2n.trim_enabled and int(self.n2n.v_trim_top_px) > 35:
            logging.getLogger("train_and_infer").warning(
                f"n2n.v_trim_top_px={self.n2n.v_trim_top_px} is tight for 2.5D N2N on ~180px vertical "
                "stripes under inter-section shift; v_trim_top_px <= 25 is recommended."
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        cfg = cls(
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
        cfg.validate()
        return cfg

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
