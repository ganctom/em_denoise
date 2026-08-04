import torch
import gc
from typing import Dict

from em_denoise_refactor.execution.interfaces import MPSDeviceManager


class AppleSiliconManager(MPSDeviceManager):
    """
    Concrete implementation of MPSDeviceManager for Apple M4 Pro.
    Manages unified memory allocation and mitigates PyTorch MPS caching leaks.
    """

    def __init__(self):
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS backend is not available on this machine. Ensure you are on Apple Silicon."
            )
        self.device = torch.device("mps")

    def get_device(self) -> torch.device:
        return self.device

    def optimize_memory(self) -> None:
        """
        Aggressively purges unreferenced memory to prevent OOM errors on unified memory.
        """
        # Force Python garbage collection first to release tensor references
        gc.collect()
        # Clear the MPS caching allocator
        torch.mps.empty_cache()

    def log_unified_memory_usage(self) -> Dict[str, float]:
        """
        Retrieves current memory metrics from the MPS driver.
        Returns values in Megabytes (MB).
        """
        # Allocated by PyTorch
        allocated = torch.mps.current_allocated_memory() / (1024**2)
        # Total allocated by the Metal driver (includes PyTorch + cache)
        driver_allocated = torch.mps.driver_allocated_memory() / (1024**2)

        return {
            "pytorch_allocated_mb": round(allocated, 2),
            "driver_allocated_mb": round(driver_allocated, 2),
            "driver_overhead_mb": round(driver_allocated - allocated, 2),
        }
