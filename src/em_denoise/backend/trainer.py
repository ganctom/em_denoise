from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from em_denoise.backend.metrics import compute_psnr, compute_ssim
from em_denoise.backend.loss import ESPILoss, AUFRCLoss


class SupervisedTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        criterion: nn.Module,
        device: torch.device,
        output_dir: Path,
        predict_residual: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion.to(device)
        self.device = device
        self.predict_residual = predict_residual
        self.max_grad_norm = max_grad_norm

        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.best_val_score = -float("inf")

        # Instantiate metric evaluators for validation
        self.espi_eval = ESPILoss(channels=1).to(device)
        self.aufrc_eval = AUFRCLoss().to(device)

    def _get_denoised_estimate(
        self, noisy: torch.Tensor, model_output: torch.Tensor
    ) -> torch.Tensor:
        if self.predict_residual:
            return noisy - model_output
        return model_output

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        running_metrics = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [Train]")
        for noisy, clean in pbar:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            batch_size = noisy.size(0)

            self.optimizer.zero_grad()
            model_output = self.model(noisy)
            denoised = self._get_denoised_estimate(noisy, model_output)

            # Unpack dynamic loss and component metrics
            loss, batch_metrics = self.criterion(denoised, clean)

            loss.backward()
            # Clip model and criterion parameters together: DynamicEMLoss carries
            # trainable log-variances that are part of the same optimizer.
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.criterion.parameters()),
                max_norm=self.max_grad_norm,
            )
            self.optimizer.step()

            running_loss += loss.item() * batch_size

            # Accumulate dynamic weights and raw loss components for TensorBoard
            for k, v in batch_metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v * batch_size

            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        N = len(self.train_loader.dataset)
        epoch_loss = running_loss / N
        self.writer.add_scalar("Loss/Train", epoch_loss, epoch)

        for k, v in running_metrics.items():
            self.writer.add_scalar(f"DynamicLoss/{k}", v / N, epoch)

        return epoch_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, dict]:
        self.model.eval()
        running_loss = 0.0

        total_psnr = 0.0
        total_ssim = 0.0
        total_espi = 0.0
        total_aufrc = 0.0

        for noisy, clean in self.val_loader:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            batch_size = noisy.size(0)

            model_output = self.model(noisy)
            denoised = self._get_denoised_estimate(noisy, model_output)

            # Discard metrics dict during validation loop
            loss, _ = self.criterion(denoised, clean)
            running_loss += loss.item() * batch_size

            denoised_clamped = torch.clamp(denoised, 0.0, 1.0)

            total_psnr += compute_psnr(denoised_clamped, clean).item() * batch_size
            total_ssim += compute_ssim(denoised_clamped, clean).item() * batch_size

            # Subtracted from 1.0 to convert loss-form back to standard metric [0, 1]
            total_espi += (
                1.0 - self.espi_eval(denoised_clamped, clean).item()
            ) * batch_size
            total_aufrc += (
                1.0 - self.aufrc_eval(denoised_clamped, clean).item()
            ) * batch_size

        N = len(self.val_loader.dataset)
        val_loss = running_loss / N

        metrics = {
            "psnr": total_psnr / N,
            "ssim": total_ssim / N,
            "espi": total_espi / N,
            "aufrc": total_aufrc / N,
        }

        self.writer.add_scalar("Loss/Validation", val_loss, epoch)
        self.writer.add_scalar("Metrics/PSNR", metrics["psnr"], epoch)
        self.writer.add_scalar("Metrics/SSIM", metrics["ssim"], epoch)
        self.writer.add_scalar("Metrics/ESPI", metrics["espi"], epoch)
        self.writer.add_scalar("Metrics/AUFRC", metrics["aufrc"], epoch)

        if self.scheduler is not None:
            self.writer.add_scalar(
                "Params/LearningRate", self.optimizer.param_groups[0]["lr"], epoch
            )

        return val_loss, metrics

    def _compute_compound_score(self, metrics: dict) -> float:
        """
        Synthesizes a compound score for checkpoint acceptance.
        Uses static heuristic weights to evaluate structural preservation quality,
        decoupled from the dynamic loss log-variances.
        """
        psnr_norm = min(metrics["psnr"] / 40.0, 1.0)

        # Fixed validation metric priorities
        w_psnr = 0.2
        w_ssim = 0.3
        w_espi = 0.25
        w_aufrc = 0.25

        score = float(
            w_psnr * psnr_norm
            + w_ssim * metrics["ssim"]
            + w_espi * metrics["espi"]
            + w_aufrc * metrics["aufrc"]
        )
        return score

    def fit(self, num_epochs: int):
        print(
            f"[*] Starting training loop | Device: {self.device} | Residual Mode: {self.predict_residual}"
        )

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            compound_score = self._compute_compound_score(metrics)

            print(
                f"Epoch {epoch:03d}/{num_epochs:03d} | "
                f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f} | "
                f"ESPI: {metrics['espi']:.4f} | AUFRC: {metrics['aufrc']:.4f} | "
                f"Score: {compound_score:.4f}"
            )

            if compound_score > self.best_val_score:
                self.best_val_score = compound_score
                ckpt_path = self.checkpoint_dir / "best_model.pt"

                # Bundle criterion state_dict to save learned variances
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "criterion_state_dict": self.criterion.state_dict(),
                        "metrics": metrics,
                        "compound_score": compound_score,
                        "predict_residual": self.predict_residual,
                    },
                    ckpt_path,
                )
                print(
                    f"  [+] Saved new best checkpoint -> {ckpt_path.name} (Score: {compound_score:.4f})"
                )

        self.writer.close()
        print("[*] Training pipeline finished successfully.")
