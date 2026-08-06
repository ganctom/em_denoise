import numpy as np
from scipy.stats import qmc


def generate_barycentric_grid(K: int):
    """Generates weight triplets summing to 1.0 with resolution 1/K."""
    grid = []
    for a in range(K + 1):
        for b in range(K + 1 - a):
            c = K - a - b
            grid.append((a / K, b / K, c / K))
    return grid


def generate_sobol_simplex(n_samples: int, seed: int | None = None) -> np.ndarray:
    """Generates n_samples points on the 2-simplex using a 2D Sobol sequence.

    Uses Kraemer (square-root) transformation to achieve uniform spatial coverage.

    Args:
        n_samples: Number of weight vectors to generate.
        seed: Random seed for scramble initialization.

    Returns:
        np.ndarray: Array of shape (n_samples, 3) where rows sum to 1.0.
    """
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    u = sampler.random(n_samples)

    # Kraemer mapping: [0, 1]^2 -> 2-Simplex
    sqrt_u0 = np.sqrt(u[:, 0])
    w_mse = 1.0 - sqrt_u0
    w_edge = sqrt_u0 * (1.0 - u[:, 1])
    w_ssim = sqrt_u0 * u[:, 1]

    return np.column_stack([w_mse, w_edge, w_ssim])


# Example Usage
if __name__ == "__main__":
    weights = generate_sobol_simplex(n_samples=8, seed=42)
    print("Sobol Simplex Samples (shape:", weights.shape, "):")
    print(weights)
    print("Row sums verification:", np.sum(weights, axis=1))
