import torch
import torch.nn as nn
import torch.optim as optim
import logging

from src.em_denoise_refactor.models.connectomics import ConnectomicsDenoisingNet
from src.em_denoise_refactor.execution.mps_manager import AppleSiliconManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def validate_network_topology():
    manager = AppleSiliconManager()
    device = manager.get_device()

    logging.info(f"Targeting device: {device}")

    # Instantiate architecture mapping to the defined topology protocol
    model = ConnectomicsDenoisingNet(in_channels=1, out_channels=1).to(device)

    # -------------------------------------------------------------------------
    # 1. Tensor Flow-Through Validation
    # -------------------------------------------------------------------------
    B, C, H, W = 4, 1, 512, 512
    logging.info(f"Executing Tensor Flow-Through with shape ({B}, {C}, {H}, {W})")

    x = torch.randn(B, C, H, W, device=device)

    model.eval()
    with torch.no_grad():
        y = model(x)

    if x.shape != y.shape:
        logging.error(
            f"Dimensionality invariant violated: Input {x.shape} -> Output {y.shape}"
        )
        raise RuntimeError("Tensor flow-through failed.")

    logging.info("Tensor Flow-Through: PASSED. Dimensionality invariant maintained.")

    # -------------------------------------------------------------------------
    # 2. Single-Batch Overfit Validation
    # -------------------------------------------------------------------------
    logging.info("Executing Single-Batch Overfit (Gradient Propagation Sanity Check)")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    # Note: Replace nn.MSELoss with your specific composite structural loss if implemented
    criterion = nn.MSELoss()

    # Generate fixed deterministic batch simulating SBF-SEM targets and noise
    clean_target = torch.randn(B, C, H, W, device=device)
    noisy_input = clean_target + 0.5 * torch.randn(B, C, H, W, device=device)

    iterations = 100
    initial_loss = None

    for i in range(iterations):
        optimizer.zero_grad(set_to_none=True)

        predictions = model(noisy_input)
        loss = criterion(predictions, clean_target)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        if i == 0:
            initial_loss = current_loss
            logging.info(f"Iteration {i:03d} | Initial Loss: {initial_loss:.6f}")
        elif (i + 1) % 20 == 0 or i == iterations - 1:
            logging.info(f"Iteration {i:03d} | Current Loss: {current_loss:.6f}")

        manager.optimize_memory()

    convergence_ratio = current_loss / (initial_loss + 1e-8)

    # Hard threshold for gradient validation
    if convergence_ratio > 0.05:
        logging.error(
            f"Single-Batch Overfit: FAILED. Loss failed to converge (Ratio: {convergence_ratio:.4f})"
        )
        raise RuntimeError(
            "Gradient flow obstructed. Check HighFreqAttentionBlock implementation."
        )
    else:
        logging.info(
            f"Single-Batch Overfit: PASSED. Convergence ratio: {convergence_ratio:.4f}"
        )


if __name__ == "__main__":
    validate_network_topology()
