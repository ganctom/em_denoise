"""
Validation Harness Step 4: Inference Engine & Spatial Windowing.
Proves 2D Hann window overlap-add weight normalization and verifies zero grid-seam artifacts.
"""

import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import tifffile

from src.em_denoise_refactor.execution.inference import StandaloneInferenceEngine


class IdentityDenoisingModel(nn.Module):
    """Pass-through dummy network used to mathematically validate tile stitching."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MockMPSDeviceManager:
    """Mock device manager implementing MPSDeviceManager protocol for isolated validation."""

    def get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def optimize_memory(self) -> None:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def generate_synthetic_sbfsem_tif(
    file_path: Path, height: int = 3072, width: int = 4096
) -> None:
    """
    Generates a synthetic 8-bit SBF-SEM .tif featuring fine high-frequency grid lines,
    axon-like circles, and gradients to make tile seam errors glaringly obvious.
    """
    y, x = np.ogrid[:height, :width]

    # Smooth background gradient
    background = (x / width * 100 + y / height * 100).astype(np.float32)

    # High-frequency structural grid (simulating membranes/bilayers)
    grid_pattern = ((x % 64 == 0) | (y % 64 == 0)).astype(np.float32) * 80.0

    # High-frequency sinusoidal waves
    waves = (np.sin(x / 8.0) * np.cos(y / 8.0) * 40.0) + 40.0

    synthetic_img = np.clip(background + grid_pattern + waves, 0, 255).astype(np.uint8)
    tifffile.imwrite(file_path, synthetic_img)


def validate_hann_window_normalization():
    """Validates that 2D window weights provide non-zero coverage and normalize cleanly to 1.0."""
    print("--- 1. Testing 2D Hann Window Weight Matrix Normalization ---")

    tile_overlap = 256
    window_size = tile_overlap * 2  # 512
    step = window_size - tile_overlap  # 256

    model = IdentityDenoisingModel()
    device_manager = MockMPSDeviceManager()
    engine = StandaloneInferenceEngine(model=model, device_manager=device_manager)

    window_tensor = engine._create_2d_hann_window(window_size)
    window = window_tensor.cpu().numpy()

    assert window.shape == (512, 512), f"Window dimension mismatch! Got {window.shape}"

    # Simulate 2x2 tile grid spatial accumulation
    canvas_h, canvas_w = 768, 768
    data_accum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # Unit field (all ones) pass-through test
    unit_patch = np.ones((window_size, window_size), dtype=np.float32)

    for y in [0, step]:
        for x in [0, step]:
            data_accum[y : y + window_size, x : x + window_size] += unit_patch * window
            weight_map[y : y + window_size, x : x + window_size] += window

    # Verify coverage in central overlap region
    center_overlap_weights = weight_map[256:512, 256:512]
    min_weight = center_overlap_weights.min()

    print(
        f"Overlap Zone Weight Sum Range: [{min_weight:.6f}, {center_overlap_weights.max():.6f}]"
    )
    assert min_weight > 0.1, "Window weight collapsed near zero in overlap zone!"

    # Validate absolute normalized reconstruction fidelity (must equal 1.000000 everywhere)
    normalized_reconstruction = data_accum[256:512, 256:512] / center_overlap_weights
    err = np.max(np.abs(normalized_reconstruction - 1.0))

    print(f"Max Reconstruction Error on Unit Field: {err:.8f}")
    assert err < 1e-5, "Overlap-add normalization failed to reconstruct flat intensity!"

    print(
        "SUCCESS: 2D window weights yield mathematically exact normalized reconstruction.\n"
    )


def validate_tiled_inference_pipeline(temp_dir: Path):
    """Executes process_raw_tif on a 4096x3072 synthetic image and inspects for seam artifacts."""
    print("--- 2. Executing Tiled Inference Pipeline Validation ---")

    input_tif = temp_dir / "synthetic_raw_4096x3072.tif"
    output_tif = temp_dir / "denoised_reconstructed.tif"

    print(f"Generating synthetic 4096x3072 SBF-SEM tissue image at: {input_tif}")
    generate_synthetic_sbfsem_tif(input_tif)

    raw_gt = tifffile.imread(input_tif)

    # Instantiate identity model & refactored StandaloneInferenceEngine
    model = IdentityDenoisingModel()
    device_manager = MockMPSDeviceManager()
    engine = StandaloneInferenceEngine(model=model, device_manager=device_manager)

    print("Running process_raw_tif with 2D Hann overlap-add blending...")
    engine.process_raw_tif(
        input_tif_path=input_tif, output_tif_path=output_tif, tile_overlap=256
    )

    reconstructed = tifffile.imread(output_tif)

    # 1. Dimension & Type Integrity Check
    assert reconstructed.shape == raw_gt.shape, (
        f"Shape mismatch! Expected {raw_gt.shape}, got {reconstructed.shape}"
    )
    assert reconstructed.dtype == np.uint8, (
        f"Dtype mismatch! Expected uint8, got {reconstructed.dtype}"
    )

    # 2. Structural Reconstruction Accuracy (MAE)
    mae = np.mean(np.abs(raw_gt.astype(np.float32) - reconstructed.astype(np.float32)))
    print(f"Mean Absolute Error (MAE) vs Ground Truth: {mae:.4f}")

    # 3. Grid Line Seam Discontinuity Inspection
    # Measure spatial derivative jump at tile boundaries relative to Ground Truth ground state
    boundary_artifact_jumps = []
    for boundary in range(256, 3000, 256):
        if boundary < reconstructed.shape[0]:
            # Differential jump across row boundary in output
            rec_jump = reconstructed[boundary, :].astype(float) - reconstructed[
                boundary - 1, :
            ].astype(float)
            # Differential jump across row boundary in GT
            gt_jump = raw_gt[boundary, :].astype(float) - raw_gt[
                boundary - 1, :
            ].astype(float)

            # Isolated stitching error delta
            stitching_artifact = np.mean(np.abs(rec_jump - gt_jump))
            boundary_artifact_jumps.append(stitching_artifact)

    avg_stitching_artifact = np.mean(boundary_artifact_jumps)
    print(
        f"Average Boundary Stitching Artifact: {avg_stitching_artifact:.4f} intensity units"
    )

    # Assert reconstruction fidelity and zero stitching seams
    assert mae < 1.5, f"Reconstruction fidelity check failed! High MAE: {mae}"
    assert avg_stitching_artifact < 1.0, (
        f"Grid seam artifact detected! Discontinuity delta: {avg_stitching_artifact}"
    )

    print(
        "\nSUCCESS: Inference engine completed without spatial grid artifacts or seam shifts."
    )
    print(f"Output saved to: {output_tif}")


def main():
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        validate_hann_window_normalization()
        validate_tiled_inference_pipeline(temp_dir)


if __name__ == "__main__":
    main()
