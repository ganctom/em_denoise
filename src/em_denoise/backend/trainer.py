from pathlib import Path
from typing import Tuple

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
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        criterion: nn.Module,
        device: torch.device,
        output_dir: Path,
        predict_residual: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.predict_residual = predict_residual

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

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [Train]")
        for noisy, clean in pbar:
            noisy, clean = noisy.to(self.device), clean.to(self.device)

            self.optimizer.zero_grad()
            model_output = self.model(noisy)

            denoised = self._get_denoised_estimate(noisy, model_output)
            loss = self.criterion(denoised, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * noisy.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.writer.add_scalar("Loss/Train", epoch_loss, epoch)
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
            model_output = self.model(noisy)

            denoised = self._get_denoised_estimate(noisy, model_output)
            loss = self.criterion(denoised, clean)
            running_loss += loss.item() * noisy.size(0)

            denoised_clamped = torch.clamp(denoised, 0.0, 1.0)
            batch_size = noisy.size(0)

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
        self.writer.add_scalar(
            "Params/LearningRate", self.optimizer.param_groups[0]["lr"], epoch
        )

        return val_loss, metrics

    def _compute_compound_score(self, metrics: dict) -> float:
        """
        Synthesizes a compound score for checkpoint acceptance.
        Assumes PSNR practical max is ~40.0 dB for normalization mapping.
        Weights can be adapted based on artifact sensitivity parameters.
        """
        psnr_norm = min(metrics["psnr"] / 40.0, 1.0)

        fct_psnr = 0.5
        fct_ssim = 0.30
        fct_espi = 0.1
        fct_aufrc = 0.1

        # fct_psnr = 0.40
        # fct_ssim = 0.50
        # fct_espi = 0.05
        # fct_aufrc = 0.05

        score = float(
            fct_psnr * psnr_norm
            + fct_ssim * metrics["ssim"]
            + fct_espi * metrics["espi"]
            + fct_aufrc * metrics["aufrc"]
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
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
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
