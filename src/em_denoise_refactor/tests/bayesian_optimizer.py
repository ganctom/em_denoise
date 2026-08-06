import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from typing import cast


class SimplexBayesianOptimizer:
    """Bayesian Optimizer executing on a 2-simplex domain via ALR transformations."""

    def __init__(self, bounds_unbounded: float = 3.0):
        # Search range in ALR space: [-bounds_unbounded, bounds_unbounded]^2
        self.bounds = [(-bounds_unbounded, bounds_unbounded)] * 2
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), n_restarts_optimizer=5, normalize_y=True, alpha=1e-6
        )
        self.X_alr: list[np.ndarray] = []
        self.y_vals: list[float] = []

    @staticmethod
    def alr_fwd(w: np.ndarray) -> np.ndarray:
        """Forward ALR: Simplex S^3 -> R^2."""
        eps = 1e-12
        w_c = np.clip(w, eps, 1.0)
        return np.log(w_c[:2] / w_c[2])

    @staticmethod
    def alr_inv(y: np.ndarray) -> np.ndarray:
        """Inverse ALR (Softmax): R^2 -> Simplex S^3."""
        y_ext = np.append(y, 0.0)
        exp_y = np.exp(y_ext - np.max(y_ext))  # Shift for numerical stability
        return exp_y / np.sum(exp_y)

    def observe(self, w: np.ndarray, loss: float) -> None:
        """Records an evaluated simplex weight vector and corresponding validation loss."""
        self.X_alr.append(self.alr_fwd(w))
        self.y_vals.append(loss)

    def _expected_improvement(self, y_vec: np.ndarray, xi: float = 0.01) -> float:
        """Computes negative Expected Improvement for minimization."""
        X_arr = np.array(self.X_alr)
        y_arr = np.array(self.y_vals)

        self.gp.fit(X_arr, y_arr)
        pred = self.gp.predict(y_vec.reshape(1, -1), return_std=True)
        mu, sigma = cast(tuple[np.ndarray, np.ndarray], pred)

        best_y = np.min(y_arr)
        sigma = np.maximum(sigma, 1e-9)

        improvement = best_y - mu - xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return -float(ei[0])

    def suggest_next(self) -> np.ndarray:
        """Proposes the next optimal loss weights on the simplex."""
        # Initial cold start with random simplex points
        if len(self.X_alr) < 3:
            u = np.random.uniform(0, 1, size=2)
            sqrt_u0 = np.sqrt(u[0])
            return np.array([1.0 - sqrt_u0, sqrt_u0 * (1.0 - u[1]), sqrt_u0 * u[1]])

        best_val = float("inf")
        best_y_next: np.ndarray | None = None

        # Multi-start optimization over acquisition function in R^2
        for _ in range(10):
            y_init = np.random.uniform(self.bounds[0][0], self.bounds[0][1], size=2)
            res = minimize(
                self._expected_improvement,
                x0=y_init,
                bounds=self.bounds,
                method="L-BFGS-B",
            )
            if res.fun < best_val:
                best_val = res.fun
                best_y_next = res.x

        if best_y_next is None:
            best_y_next = np.zeros(2)

        return self.alr_inv(best_y_next)


# Example Training Loop Simulation
if __name__ == "__main__":
    # Simulated noisy objective function (e.g., neural network validation loss)
    def dummy_train_and_eval(w: np.ndarray) -> float:
        target_w = np.array([0.1, 0.3, 0.6])
        return float(np.sum((w - target_w) ** 2) + np.random.normal(0, 0.01))

    opt = SimplexBayesianOptimizer()

    for step in range(10):
        w_next = opt.suggest_next()
        loss = dummy_train_and_eval(w_next)
        opt.observe(w_next, loss)
        print(
            f"Iter {step + 1:02d} | Weights: {np.round(w_next, 3)} | Loss: {loss:.5f}"
        )
