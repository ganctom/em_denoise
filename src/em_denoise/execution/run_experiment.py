import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from em_denoise.backend.dataset import SupervisedEMDataset
from em_denoise.backend.architecture import ResidualUNet, UNetConfig
from em_denoise.backend.trainer import SupervisedTrainer
from em_denoise.backend.config_schema import PipelineConfig
from em_denoise.backend.loss import CombinedL1SSIMLoss


def main(config_path: Path):
    cfg = PipelineConfig.from_yaml(config_path)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    train_dataset = SupervisedEMDataset(
        data_dir=cfg.paths.sandbox_root / "train", augment=True
    )
    val_dataset = SupervisedEMDataset(
        data_dir=cfg.paths.sandbox_root / "val", augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers_train,
        pin_memory=(device.type != "cpu"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers_val,
    )

    unet_kwargs = cfg.unet.model_dump()
    unet_config = UNetConfig(**unet_kwargs)
    model = ResidualUNet(unet_config).to(device)

    criterion = CombinedL1SSIMLoss(
        alpha=0.84,
        channels=unet_config.in_channels,
        data_range=1.0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    warmup_epochs = getattr(cfg.training, "warmup_epochs", 5)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

    cosine_epochs = cfg.training.epochs - warmup_epochs
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, cosine_epochs),
        eta_min=cfg.training.eta_min,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    cfg.paths.experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.paths.experiment_dir / "resolved_config.yaml", "w") as f:
        f.write(config_path.read_text())

    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        output_dir=cfg.paths.experiment_dir,
        predict_residual=True,
    )

    trainer.fit(num_epochs=cfg.training.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Supervised Denoising Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    main(args.config)
