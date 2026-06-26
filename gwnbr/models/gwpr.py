"""
gwnbr.models.gwpr
------------------
Geographically Weighted Poisson Regression (GWPR).

GWPR is a special case of GWNBR with alpha = 0 (no overdispersion).
Implemented here for direct comparison and as a baseline model.

Reference
---------
Nakaya, T., Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2005).
"Geographically Weighted Poisson Regression for Disease Association Mapping."
Statistics in Medicine, 24, 2695-2717.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.special import gammaln
from joblib import Parallel, delayed

from gwnbr.models.base import GWRBase, _nb_deviance
from gwnbr.kernels import compute_weights
from gwnbr.utils.distance import pairwise_distances
from gwnbr.utils.irls_solver import irls, _working_weights


def _fit_gwpr_tract(i, X, y, distances_i, offset,
                    bandwidth, kernel, beta_init,
                    max_irls, tol_irls):
    w_i = compute_weights(distances_i, bandwidth, kernel)
    beta_i, mu_i, cov_i, _ = irls(
        X, y, w_i, offset,
        alpha=0.0,      # Poisson: alpha = 0
        beta_init=beta_init,
        max_iter=max_irls, tol=tol_irls
    )
    A_i = _working_weights(mu_i, 0.0)
    W_combined = w_i * A_i
    XtWX = (X * W_combined[:, None]).T @ X
    if abs(np.linalg.det(XtWX)) < 1e-20:
        hat_row = np.zeros(len(y))
    else:
        hat_row = X[i] @ np.linalg.inv(XtWX) @ (X * W_combined[:, None]).T

    se_i = np.sqrt(np.maximum(np.diag(cov_i), 0.0))
    return {
        "i": i, "beta": beta_i, "se": se_i,
        "hat_row": hat_row, "y_hat_i": mu_i[i]
    }


class GWPR(GWRBase):
    """
    Geographically Weighted Poisson Regression (GWPR).

    Equivalent to GWNBR with alpha fixed at 0.
    Provided for comparison with GWNBR / GWNBRg.

    Parameters
    ----------
    Same as GWNBRg / GWNBR.
    """

    def __init__(self, coords, y, X, offset=None, variable_names=None):
        super().__init__(coords, y, X, offset, variable_names)

    def fit(self,
            bandwidth: float,
            kernel: str = "gaussian",
            n_jobs: int = -1,
            max_irls: int = 100,
            tol_irls: float = 1e-6,
            verbose: bool = True) -> "GWPR":
        """
        Fit the GWPR model.

        Parameters
        ----------
        bandwidth : float  Spatial bandwidth.
        kernel    : str    'gaussian', 'bisquare', or 'adaptive_nn'.
        n_jobs    : int    Parallel jobs. -1 = all CPUs.
        verbose   : bool

        Returns
        -------
        self
        """
        self.bandwidth = bandwidth
        self._kernel = kernel

        if verbose:
            print(f"[GWPR] Computing {self.n}×{self.n} distance matrix...")

        D = pairwise_distances(self.coords)

        # Simple Poisson init
        mu0 = np.maximum(self.y, 0.5)
        beta_init = np.zeros(self.p)
        beta_init[0] = np.log(np.mean(mu0))

        if verbose:
            print(f"[GWPR] Fitting {self.n} local Poisson regressions "
                  f"(bandwidth={bandwidth}, kernel='{kernel}') ...")

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_gwpr_tract)(
                i, self.X, self.y, D[i], self.offset,
                bandwidth, kernel, beta_init, max_irls, tol_irls
            )
            for i in range(self.n)
        )

        self.betas      = np.zeros((self.n, self.p))
        self.se_betas   = np.zeros((self.n, self.p))
        self.y_hat      = np.zeros(self.n)
        self.hat_matrix = np.zeros((self.n, self.n))

        for r in results:
            i = r["i"]
            self.betas[i]     = r["beta"]
            self.se_betas[i]  = r["se"]
            self.y_hat[i]     = r["y_hat_i"]
            self.hat_matrix[i] = r["hat_row"]

        self.alphas = np.zeros(self.n)
        self.t_stats  = self.betas / np.where(self.se_betas < 1e-20,
                                               1e-20, self.se_betas)
        self.p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(self.t_stats)))
        self.n_params = float(np.trace(self.hat_matrix))

        # Poisson deviance
        tt = np.where(self.y_hat > 0, self.y / self.y_hat, 1e-10)
        tt = np.where(tt == 0, 1e-10, tt)
        self.deviance = float(2.0 * np.sum(
            self.y * np.log(tt) - (self.y - self.y_hat)))

        # Poisson log-likelihood
        clgamma = gammaln(self.y + 1.0)
        self.log_likelihood = float(np.sum(
            -self.y_hat + self.y * np.log(np.maximum(self.y_hat, 1e-300))
            - clgamma))

        self._compute_diagnostics()
        self._fitted = True

        if verbose:
            print(f"[GWPR] Done.  AICc={self.AICc:.2f}  "
                  f"Pseudo-R2={self.pct_deviance:.4f}")

        return self
    
    def significant_betas(self, alpha_level: float = 0.05) -> np.ndarray:
        """
        Significance mask using standard p-value cutoff.
        No multiple-testing correction for GWPR (alpha not estimated).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.p_values < alpha_level
