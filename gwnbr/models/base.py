"""
gwnbr.models.base
-----------------
Abstract base class for all GW regression models.
Stores results and provides shared summary/export methods.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.special import gammaln


class GWRBase(ABC):
    """
    Base class for Geographically Weighted regression models.

    Attributes set after fit()
    --------------------------
    betas        : np.ndarray, shape (n, p)   Local coefficients.
    se_betas     : np.ndarray, shape (n, p)   Standard errors.
    t_stats      : np.ndarray, shape (n, p)   t-statistics.
    p_values     : np.ndarray, shape (n, p)   Two-sided p-values.
    y_hat        : np.ndarray, shape (n,)     Fitted values.
    deviance     : float
    log_likelihood: float
    AIC          : float
    AICc         : float
    BIC          : float
    pct_deviance : float    Pseudo-R2 based on deviance.
    adj_pct_dev  : float    Adjusted pseudo-R2.
    bandwidth    : float    Fitted bandwidth.
    n_params     : float    Effective number of parameters (trace of hat).
    """

    def __init__(self,
                 coords: np.ndarray,
                 y: np.ndarray,
                 X: np.ndarray,
                 offset: np.ndarray = None,
                 variable_names: list = None):
        """
        Parameters
        ----------
        coords         : np.ndarray, shape (n, 2)  [lon/x, lat/y].
        y              : np.ndarray, shape (n,)    Count response.
        X              : np.ndarray, shape (n, p)  Covariates (NO intercept).
        offset         : np.ndarray, shape (n,)    Log-offset. None = zeros.
        variable_names : list of str               Names for X columns.
        """
        self.coords = np.asarray(coords, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n = len(y)

        # Prepend intercept column
        self.X = np.hstack([np.ones((self.n, 1)),
                            np.asarray(X, dtype=float)])
        self.p = self.X.shape[1]

        self.offset = (np.zeros(self.n) if offset is None
                       else np.asarray(offset, dtype=float))

        if variable_names is not None:
            self.var_names = ["Intercept"] + list(variable_names)
        else:
            self.var_names = [f"b{i}" for i in range(self.p)]

        # Results (filled by fit())
        self.betas      = None
        self.se_betas   = None
        self.t_stats    = None
        self.p_values   = None
        self.alphas     = None    # local or global alpha
        self.se_alphas  = None
        self.y_hat      = None
        self.hat_matrix = None   # (n, n) hat matrix S
        self.deviance   = None
        self.log_likelihood = None
        self.AIC        = None
        self.AICc       = None
        self.BIC        = None
        self.pct_deviance = None
        self.adj_pct_dev  = None
        self.n_params   = None
        self.bandwidth  = None
        self._fitted    = False

    @abstractmethod
    def fit(self, bandwidth: float, kernel: str = "gaussian",
            **kwargs) -> "GWRBase":
        """Fit the model. Must set all result attributes."""

    # ------------------------------------------------------------------
    # Shared diagnostics
    # ------------------------------------------------------------------

    def _compute_diagnostics(self):
        """
        Compute AIC, AICc, BIC, deviance-based pseudo-R2.
        Called at the end of each model's fit().
        """
        n = self.n
        k = self.n_params
        ll = self.log_likelihood

        self.AIC  = 2.0 * k - 2.0 * ll
        self.AICc = self.AIC + (2.0 * k * (k + 1)) / max(n - k - 1, 1e-6)
        self.BIC  = k * np.log(n) - 2.0 * ll

        # Pseudo-R2: deviance-based (Cameron & Windmeijer 1996)
        dev_null = self._null_deviance()
        self.pct_deviance = 1.0 - self.deviance / dev_null if dev_null != 0 else 0.0
        self.adj_pct_dev  = (1.0 - ((n - 1.0) / (n - k))
                             * (1.0 - self.pct_deviance))

    def _null_deviance(self) -> float:
        """Deviance of intercept-only NB model."""
        mu_null = np.full(self.n, np.mean(self.y))
        alpha_null = np.mean(self.alphas) if self.alphas is not None else 0.0
        return _nb_deviance(self.y, mu_null, alpha_null)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a text summary of model fit."""
        if not self._fitted:
            return "Model not yet fitted. Call fit() first."

        lines = [
            "=" * 62,
            "  Geographically Weighted Negative Binomial Regression",
            f"  Model type  : {self.__class__.__name__}",
            f"  n           : {self.n}",
            f"  Bandwidth   : {self.bandwidth:.4f}",
            f"  Eff. params : {self.n_params:.2f}",
            "-" * 62,
            "  Goodness of Fit",
            f"  Log-likelihood : {self.log_likelihood:.4f}",
            f"  Deviance       : {self.deviance:.4f}",
            f"  Pseudo-R2 (dev): {self.pct_deviance:.4f}",
            f"  Adj. R2  (dev) : {self.adj_pct_dev:.4f}",
            f"  AIC            : {self.AIC:.4f}",
            f"  AICc           : {self.AICc:.4f}",
            f"  BIC            : {self.BIC:.4f}",
            "-" * 62,
            "  Local Coefficient Summary",
            f"  {'Variable':<20} {'Mean':>8} {'Min':>8} "
            f"{'P25':>8} {'Median':>8} {'P75':>8} {'Max':>8}",
        ]
        for j, name in enumerate(self.var_names):
            b = self.betas[:, j]
            lines.append(
                f"  {name:<20} {np.mean(b):>8.4f} {np.min(b):>8.4f} "
                f"{np.percentile(b,25):>8.4f} {np.median(b):>8.4f} "
                f"{np.percentile(b,75):>8.4f} {np.max(b):>8.4f}"
            )
        lines.append("=" * 62)
        return "\n".join(lines)

    def coefficient_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame of local coefficient distribution statistics.

        Columns: Variable, Min, P25, Mean, Median, P75, Max, IQR, Std.

        Mirrors the quantile output table in the SAS %gwnbr macro
        (Silva & Rodrigues, 2014). Useful for comparing mean local
        coefficients against a global NB model (see GWNBRg.beta_global).

        Returns
        -------
        pd.DataFrame, shape (p, 9)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        rows = []
        for j, name in enumerate(self.var_names):
            b = self.betas[:, j]
            rows.append({
                "Variable": name,
                "Min":      round(float(np.min(b)),              4),
                "P25":      round(float(np.percentile(b, 25)),   4),
                "Mean":     round(float(np.mean(b)),             4),
                "Median":   round(float(np.median(b)),           4),
                "P75":      round(float(np.percentile(b, 75)),   4),
                "Max":      round(float(np.max(b)),              4),
                "IQR":      round(float(np.percentile(b, 75)
                                   - np.percentile(b, 25)),      4),
                "Std":      round(float(np.std(b)),              4),
            })
        return pd.DataFrame(rows)

    def local_r2(self) -> np.ndarray:
        """
        Local pseudo-R2 per tract.

        Uses the proportion of fitted value deviation from global mean,
        which is more numerically stable than single-observation deviance.

            r2_i = 1 - (y_i - yhat_i)^2 / (y_i - mean(y))^2

        This is analogous to local R2 in standard GWR (Fotheringham et al. 2002)
        and avoids numerical instability from single-point NB deviance.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        mu_null  = float(np.mean(self.y))
        residuals = self.y - self.y_hat
        null_dev  = self.y - mu_null

        # Avoid division by zero for tracts where y_i == mean(y)
        denom = null_dev ** 2
        denom = np.where(denom < 1e-10, np.nan, denom)

        r2_local = 1.0 - (residuals ** 2) / denom

        # Clip to [-1, 1] — extreme values indicate outlier tracts
        r2_local = np.clip(r2_local, -1.0, 1.0)

        # Replace nan (tracts at exact mean) with 0
        r2_local = np.where(np.isnan(r2_local), 0.0, r2_local)

        return r2_local

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export results as a tidy DataFrame.

        Columns: coords, y, y_hat, alpha, beta_*, se_*, t_*, p_*
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before to_dataframe().")

        df = pd.DataFrame({
            "x": self.coords[:, 0],
            "y_coord": self.coords[:, 1],
            "y_obs": self.y,
            "y_hat": self.y_hat,
        })

        if self.alphas is not None:
            if np.ndim(self.alphas) == 0 or len(np.atleast_1d(self.alphas)) == 1:
                df["alpha"] = float(np.atleast_1d(self.alphas)[0])
            else:
                df["alpha"] = self.alphas

        for j, name in enumerate(self.var_names):
            df[f"beta_{name}"]   = self.betas[:, j]
            df[f"se_{name}"]     = self.se_betas[:, j]
            df[f"t_{name}"]      = self.t_stats[:, j]
            df[f"p_{name}"]      = self.p_values[:, j]

        return df


# ------------------------------------------------------------------
# Module-level helpers used by subclasses
# ------------------------------------------------------------------

def _nb_log_likelihood(y: np.ndarray, mu: np.ndarray,
                       alpha: np.ndarray) -> float:
    """
    NB-2 log-likelihood (vectorised over alpha or scalar alpha).

    L = sum[ y*log(alpha*mu) - (y + 1/alpha)*log(1 + alpha*mu)
             + lgamma(y + 1/alpha) - lgamma(1/alpha) - lgamma(y+1) ]
    """
    alpha = np.atleast_1d(alpha)
    if len(alpha) == 1:
        alpha = np.full(len(y), float(alpha[0]))

    inv_a = 1.0 / np.where(alpha < 1e-10, 1e-10, alpha)
    term1 = y * np.log(np.where(alpha * mu < 1e-300, 1e-300, alpha * mu))
    term2 = (y + inv_a) * np.log(1.0 + alpha * mu)
    term3 = gammaln(y + inv_a)
    term4 = gammaln(inv_a)
    term5 = gammaln(y + 1.0)
    return float(np.sum(term1 - term2 + term3 - term4 - term5))


def _nb_deviance(y: np.ndarray, mu: np.ndarray,
                 alpha) -> float:
    """NB-2 deviance."""
    alpha = np.atleast_1d(alpha)
    if len(alpha) == 1:
        alpha = np.full(len(y), float(alpha[0]))

    tt = np.where(mu > 0, y / mu, 1e-10)
    tt = np.where(tt == 0, 1e-10, tt)

    inv_a = 1.0 / np.where(alpha < 1e-10, 1e-10, alpha)
    ratio1 = 1.0 + alpha * y
    ratio2 = 1.0 + alpha * mu
    ratio1 = np.where(ratio1 <= 0, 1e-10, ratio1)
    ratio2 = np.where(ratio2 <= 0, 1e-10, ratio2)
    return float(2.0 * np.sum(y * np.log(tt)
                              - (y + inv_a) * np.log(ratio1 / ratio2)))
