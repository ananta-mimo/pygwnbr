"""
gwnbr.models.gwnbrg
--------------------
Geographically Weighted Negative Binomial Regression
with Global overdispersion (GWNBRg).

In GWNBRg, the overdispersion parameter alpha is estimated once
globally from a standard (non-spatial) Negative Binomial regression.
Local beta coefficients are then estimated via IRLS for each
focal location, using that fixed global alpha.

This is computationally lighter than full GWNBR and allows proper
AICc-based bandwidth selection (because k_2 = 1, not a surface).

Reference
---------
Silva & Rodrigues (2014). Statistics and Computing, 24, 769-783.
  Section: "GWNBRG MODEL"
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.special import gammaln
from joblib import Parallel, delayed

from gwnbr.models.base import GWRBase, _nb_log_likelihood, _nb_deviance
from gwnbr.kernels import compute_weights
from gwnbr.utils.distance import pairwise_distances
from gwnbr.utils.nr_solver import fit_alpha_nr
from gwnbr.utils.irls_solver import irls


# -----------------------------------------------------------------------
# Global NB fit (needed to initialise alpha for GWNBRg)
# -----------------------------------------------------------------------

def _fit_global_nb(X: np.ndarray,
                   y: np.ndarray,
                   offset: np.ndarray,
                   max_outer: int = 50,
                   max_irls: int = 100,
                   tol_outer: float = 1e-5,
                   tol_irls: float = 1e-6) -> tuple:
    """
    Fit a global (non-spatial) NB-2 regression using alternating
    NR (for alpha) and IRLS (for beta).

    Mirrors the global NB initialisation block in the SAS %gwnbr macro.

    Returns
    -------
    alpha_global : float   Global overdispersion.
    beta_global  : np.ndarray, shape (p,)
    mu_global    : np.ndarray, shape (n,)
    """
    n = len(y)
    ones = np.ones(n)

    # Initialise mu
    mu = np.full(n, max(np.mean(y), 0.5))
    alpha = 1.0   # start with alpha = 1

    for _outer in range(max_outer):
        alpha_old = alpha

        # --- NR step: update alpha (all weights = 1 for global) ---
        alpha_new, _, _ = fit_alpha_nr(
            y, mu, ones,
            theta_init=max(1.0 / alpha, 1e-10),
            max_iter=100, tol=1e-3
        )
        alpha = max(alpha_new, 1e-10)

        # --- IRLS step: update beta ---
        beta, mu, _, _ = irls(
            X, y, ones, offset, alpha,
            max_iter=max_irls, tol=tol_irls
        )

        if abs(alpha - alpha_old) < tol_outer:
            break

    return alpha, beta, mu


# -----------------------------------------------------------------------
# Single-tract fit helper (parallelisable)
# -----------------------------------------------------------------------

def _fit_single_tract(i: int,
                      X: np.ndarray,
                      y: np.ndarray,
                      distances_i: np.ndarray,
                      offset: np.ndarray,
                      alpha_global: float,
                      bandwidth: float,
                      kernel: str,
                      beta_init: np.ndarray,
                      max_irls: int,
                      tol_irls: float) -> dict:
    """
    Estimate local beta for focal tract i (GWNBRg: alpha is global).
    """
    w_i = compute_weights(distances_i, bandwidth, kernel)

    beta_i, mu_i, cov_i, converged = irls(
        X, y, w_i, offset,
        alpha=alpha_global,
        beta_init=beta_init,
        max_iter=max_irls,
        tol=tol_irls
    )

    # Hat-matrix row: x_i (X'W_s A X)^{-1} X' W_s A
    from gwnbr.utils.irls_solver import _working_weights
    A_i = _working_weights(mu_i, alpha_global)
    W_combined = w_i * A_i
    XtWX = (X * W_combined[:, None]).T @ X
    if abs(np.linalg.det(XtWX)) < 1e-20:
        hat_row = np.zeros(len(y))
    else:
        hat_row = X[i] @ np.linalg.inv(XtWX) @ (X * W_combined[:, None]).T

    se_i = np.sqrt(np.maximum(np.diag(cov_i), 0.0))
    y_hat_i = mu_i[i]

    return {
        "i": i,
        "beta": beta_i,
        "se": se_i,
        "hat_row": hat_row,
        "y_hat_i": y_hat_i,
        "converged": converged,
    }


# -----------------------------------------------------------------------
# GWNBRg class
# -----------------------------------------------------------------------

class GWNBRg(GWRBase):
    """
    Geographically Weighted Negative Binomial Regression
    with Global overdispersion (GWNBRg).

    Parameters
    ----------
    coords         : array-like, shape (n, 2)  [lon/x, lat/y].
    y              : array-like, shape (n,)    Count response variable.
    X              : array-like, shape (n, k)  Predictor matrix (no intercept).
    offset         : array-like, shape (n,) or None.
                     Log-scale offset (e.g. log(population)).
    variable_names : list of str, optional.    Names for predictors.

    Example
    -------
    >>> model = GWNBRg(coords=coords, y=crash_counts, X=X_vars,
    ...                offset=np.log(population),
    ...                variable_names=['income', 'unemployment'])
    >>> model.fit(bandwidth=50.0, kernel='gaussian')
    >>> print(model.summary())
    >>> df = model.to_dataframe()
    """

    def __init__(self, coords, y, X, offset=None, variable_names=None):
        super().__init__(coords, y, X, offset, variable_names)
        self.alpha_global = None   # set after fit()
        self._alpha_override = None   # used internally by StationarityTest

    def fit(self,
            bandwidth: float,
            kernel: str = "gaussian",
            n_jobs: int = -1,
            max_irls: int = 100,
            tol_irls: float = 1e-6,
            verbose: bool = True) -> "GWNBRg":
        """
        Fit the GWNBRg model.

        Parameters
        ----------
        bandwidth : float
            Spatial bandwidth (km for lat/lon, same units for projected;
            integer k for 'adaptive_nn' kernel).
        kernel    : str
            'gaussian' (fixed, default), 'bisquare' (adaptive),
            or 'adaptive_nn' (k-NN).
        n_jobs    : int
            Parallel jobs for tract loop. -1 = all CPUs.
        max_irls  : int    Max IRLS iterations per tract.
        tol_irls  : float  IRLS convergence tolerance.
        verbose   : bool   Print progress messages.

        Returns
        -------
        self  (for method chaining)
        """
        self.bandwidth = bandwidth
        self._kernel = kernel

        if verbose:
            print(f"[GWNBRg] Computing {self.n}×{self.n} distance matrix...")

        D = pairwise_distances(self.coords)    # (n, n)

        if self._alpha_override is not None:
            # Used by StationarityTest — skip global re-estimation so
            # permutation fits isolate spatial variation in beta only,
            # following the %estac SAS macro (Silva & Rodrigues, 2014).
            alpha_g = self._alpha_override
            # Need a beta_g starting point; quick global IRLS at fixed alpha
            from gwnbr.utils.irls_solver import irls
            ones = np.ones(self.n)
            beta_g, _, _, _ = irls(self.X, self.y, ones, self.offset,
                                    alpha=alpha_g, max_iter=max_irls,
                                    tol=tol_irls)
            if verbose:
                print(f"[GWNBRg] Using fixed alpha = {alpha_g:.6f} "
                      f"(stationarity test mode)")
        else:
            if verbose:
                print("[GWNBRg] Estimating global alpha via NB regression...")
            alpha_g, beta_g, _ = _fit_global_nb(self.X, self.y, self.offset)

        self.alpha_global = alpha_g

        if verbose:
            print(f"[GWNBRg] Global alpha = {alpha_g:.6f}  "
                  f"(phi = 1/alpha = {1/alpha_g:.4f})")
            print(f"[GWNBRg] Fitting local betas for {self.n} tracts "
                  f"(bandwidth={bandwidth}, kernel='{kernel}') ...")

        # --- Parallel tract loop ---
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_single_tract)(
                i, self.X, self.y, D[i], self.offset,
                alpha_g, bandwidth, kernel, beta_g,
                max_irls, tol_irls
            )
            for i in range(self.n)
        )

        # --- Collect results ---
        self.betas   = np.zeros((self.n, self.p))
        self.se_betas = np.zeros((self.n, self.p))
        self.y_hat   = np.zeros(self.n)
        self.hat_matrix = np.zeros((self.n, self.n))

        for r in results:
            i = r["i"]
            self.betas[i]    = r["beta"]
            self.se_betas[i] = r["se"]
            self.y_hat[i]    = r["y_hat_i"]
            self.hat_matrix[i] = r["hat_row"]

        # t-stats and p-values
        self.t_stats = self.betas / np.where(self.se_betas < 1e-20,
                                             1e-20, self.se_betas)
        self.p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(self.t_stats)))

        # Alpha (scalar for GWNBRg)
        self.alphas = np.full(self.n, alpha_g)

        # Effective parameters = trace(S) + 1  (the +1 is for global alpha)
        self.n_params = float(np.trace(self.hat_matrix)) + 1.0

        # Deviance and log-likelihood
        self.deviance = _nb_deviance(self.y, self.y_hat, alpha_g)
        self.log_likelihood = _nb_log_likelihood(self.y, self.y_hat,
                                                 np.full(self.n, alpha_g))

        self._compute_diagnostics()
        self._fitted = True

        if verbose:
            print(f"[GWNBRg] Done.  AICc={self.AICc:.2f}  "
                  f"Pseudo-R2={self.pct_deviance:.4f}")

        return self

    # ------------------------------------------------------------------
    # Multiple-testing corrected significance (Silva & Fotheringham 2015)
    # ------------------------------------------------------------------

    def significant_betas(self, alpha_level: float = 0.05) -> np.ndarray:
        """
        Return boolean mask (n, p) of locally significant coefficients
        using the multiple-testing correction from Silva & Fotheringham (2015):

            adjusted_alpha = alpha_level * (p / n_params)

        Parameters
        ----------
        alpha_level : float  Nominal significance level (default 0.05).

        Returns
        -------
        sig : np.ndarray, shape (n, p)  True where locally significant.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        adj_alpha = alpha_level * (self.p / self.n_params)
        return self.p_values < adj_alpha
