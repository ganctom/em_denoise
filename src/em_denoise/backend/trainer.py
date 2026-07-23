import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

from metrics import compute_psnr, compute_ssim


class SupervisedTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.best_val_psnr = -float("inf")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [Train]")
        for noisy, clean in pbar:
            noisy, clean = noisy.to(self.device), clean.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(noisy)
            loss = self.criterion(output, clean)

            loss.backward()
            # Gradient clipping prevents exploding gradients during deep U-Net backprop
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * noisy.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.writer.add_scalar("Loss/Train", epoch_loss, epoch)
        return epoch_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple[float, float, float]:
        self.model.eval()
        running_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0

        for noisy, clean in self.val_loader:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            output = self.model(noisy)

            loss = self.criterion(output, clean)
            running_loss += loss.item() * noisy.size(0)

            # Clamp output to valid dynamic range before calculating evaluation metrics
            output_clamped = torch.clamp(output, 0.0, 1.0)
            total_psnr += compute_psnr(output_clamped, clean) * noisy.size(0)
            total_ssim += compute_ssim(output_clamped, clean) * noisy.size(0)

        val_loss = running_loss / len(self.val_loader.dataset)
        val_psnr = total_psnr / len(self.val_loader.dataset)
        val_ssim = total_ssim / len(self.val_loader.dataset)

        self.writer.add_scalar("Loss/Validation", val_loss, epoch)
        self.writer.add_scalar("Metrics/PSNR", val_psnr, epoch)
        self.writer.add_scalar("Metrics/SSIM", val_ssim, epoch)
        self.writer.add_scalar(
            "Params/LearningRate", self.optimizer.param_groups[0]["lr"], epoch
        )

        return val_loss, val_psnr, val_ssim

    def fit(self, num_epochs: int):
        print(
            f"[*] Starting training loop for {num_epochs} epochs on device: {self.device}"
        )

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_psnr, val_ssim = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {epoch:03d}/{num_epochs:03d} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Val PSNR: {val_psnr:.2f} dB | "
                f"Val SSIM: {val_ssim:.4f}"
            )

            # Checkpoint management: Save state on validation PSNR improvements
            if val_psnr > self.best_val_psnr:
                self.best_val_psnr = val_psnr
                ckpt_path = self.checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                    },
                    ckpt_path,
                )
                print(
                    f"  [+] Saved new best checkpoint -> {ckpt_path.name} (PSNR: {val_psnr:.2f} dB)"
                )

        self.writer.close()
        print("[*] Training pipeline finished successfully.")
