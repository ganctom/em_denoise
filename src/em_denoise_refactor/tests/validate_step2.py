import torch
import torch.nn as nn
from src.em_denoise_refactor.execution.mps_manager import AppleSiliconManager


def validate_mps_memory_harness():
    print("--- Validation Step 2: Stress-Test the MPS Memory Harness ---")

    # Instantiate your MPSDeviceManager
    manager = AppleSiliconManager()
    device = manager.get_device()
    print(f"Targeting device: {device}")

    # Validation parameters
    iterations = 200
    batch_size = 64
    channels = 1
    h, w = 512, 512

    print(f"Simulating Batch Size: {batch_size}, Spatial Dims: {h}x{w}")

    # Dummy convolutional layer to simulate graph generation
    dummy_net = nn.Conv2d(
        in_channels=channels, out_channels=32, kernel_size=3, padding=1
    ).to(device)

    memory_history = []

    try:
        # The Dummy Loop
        for i in range(iterations):
            # 1. Allocate massive tensor directly on MPS
            x = torch.randn(batch_size, channels, h, w, device=device)

            # 2. Forward pass to build a computational graph
            out = dummy_net(x)

            # 3. Simulate loss calculation and backward pass (optional but recommended for stress tests)
            loss = out.sum()
            loss.backward()

            # 4. Invoke the Purge
            manager.optimize_memory()

            # 5. Monitor the Allocator
            mem_stats = manager.log_unified_memory_usage()

            # Using current_allocated_mb as a proxy if driver_allocated_mb isn't directly exposed by PyTorch APIs
            allocated_mb = mem_stats.get(
                "driver_allocated_mb", torch.mps.current_allocated_memory() / (1024**2)
            )
            memory_history.append(allocated_mb)

            if i % 20 == 0 or i == iterations - 1:
                print(f"Iteration {i:03d} | Driver Allocated: {allocated_mb:.2f} MB")

        print("\n--- Benchmark Complete ---")

        # Diagnostics
        initial_spike = memory_history[5]  # Memory after first few iterations stabilize
        final_memory = memory_history[-1]
        drift = final_memory - initial_spike

        print(f"Post-Spike Memory: {initial_spike:.2f} MB")
        print(f"Final Memory:      {final_memory:.2f} MB")

        if (
            drift > 150.0
        ):  # Allowing minor fluctuation, but 150MB+ continuous drift is a leak
            print("\nDIAGNOSIS: LEAKING")
            print(
                "Tensors or computational graphs are being unintentionally stored in memory."
            )
        else:
            print("\nDIAGNOSIS: HEALTHY")
            print("Driver overhead stabilized. Ready for training.")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nDIAGNOSIS: CRASHING")
            print(
                "Batch size is too large for the 48GB ceiling, or empty_cache() is failing."
            )
        else:
            print(f"\nUNEXPECTED CRASH: {e}")


if __name__ == "__main__":
    # Ensure PyTorch is using the MPS backend
    if not torch.backends.mps.is_available():
        print("ERROR: MPS backend not found. Ensure you are running on Apple Silicon.")
    else:
        validate_mps_memory_harness()
