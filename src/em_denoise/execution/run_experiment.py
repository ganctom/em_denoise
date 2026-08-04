import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from em_denoise.backend.dataset import SupervisedEMDataset
from em_denoise.backend.architecture import ResidualUNet, UNetConfig
from em_denoise.backend.trainer import SupervisedTrainer
from em_denoise.backend.config_schema import PipelineConfig
from em_denoise.backend.loss import DynamicEMLoss
from em_denoise.execution import run_inference


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

    # # Initialize the new multi-objective criterion
    # criterion = ComprehensiveEMLoss(
    #     w_l1=cfg.em_loss.l1,
    #     w_ssim=cfg.em_loss.ssim,
    #     w_espi=cfg.em_loss.espi,
    #     w_aufrc=cfg.em_loss.aufrc,
    #     channels=unet_config.in_channels,
    # ).to(device)
    #
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=cfg.training.lr,
    #     weight_decay=cfg.training.weight_decay,
    # )

    criterion = DynamicEMLoss(channels=unet_config.in_channels)

    # Critical: chain the parameters of the model and the loss module, otherwise the
    # learnable log-variances of DynamicEMLoss never receive updates.
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    warmup_epochs = cfg.training.warmup_epochs
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
        max_grad_norm=cfg.training.max_grad_norm,
    )

    trainer.fit(num_epochs=cfg.training.epochs)

    # INFERENCE on the untouched full-size tiles copied out by the ingestion pipeline.
    test_dir = Path(cfg.paths.sandbox_root).parent / "raw_patches/test_raw"
    checkpoint_path = Path(cfg.paths.experiment_dir) / "checkpoints/best_model.pt"
    if test_dir.exists():
        run_inference.run_from_config(
            config_path=config_path,
            test_dir=test_dir,
            ckpt_path=checkpoint_path,
        )
    else:
        print(f"[!] Skipping post-training inference, no raw test tiles at {test_dir}")


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
