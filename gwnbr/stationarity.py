"""
gwnbr.stationarity
-------------------
Permutation-based stationarity test for local GWR coefficients.

Tests whether the spatial variation in each local coefficient is
statistically significant or merely noise, following the method
of Silva & Rodrigues (2014) and the %estac SAS macro.

Method
------
For each coefficient j, compute the observed variance statistic:

    Vk_j = (1/n) * sum_i (beta_ij - mean(beta_j))^2

Then permute y (shuffle observations across locations) R times,
refit the model each time, compute Vk_j for each permutation.

p-value = proportion of permuted Vk_j >= observed Vk_j

If p < alpha: coefficient j is spatially non-stationary.
              Its spatial variation is unlikely to be random noise.
If p >= alpha: coefficient j is stationary.
               Its spatial variation could be due to chance.

Reference
---------
Silva, A. R. and Rodrigues, T. C. V. (2014).
    Geographically Weighted Negative Binomial Regression —
    Incorporating Overdispersion. Statistics and Computing, 24, 769-783.

Silva, A. R. and Fotheringham, A. S. (2015).
    The Multiple Testing Issue in Geographically Weighted Regression.
    Geographical Analysis, 47(2), 118-136.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Type
from joblib import Parallel, delayed


def _compute_vk(betas: np.ndarray) -> np.ndarray:
    """
    Compute the variance statistic Vk for each coefficient.

    Vk_j = (1/n) * sum_i (beta_ij - mean(beta_j))^2

    This is the mean squared deviation of local betas from their
    global mean — a measure of how much the coefficient varies
    spatially. Larger Vk = more spatial variation.

    Parameters
    ----------
    betas : np.ndarray, shape (n, p)
        Local coefficient estimates.

    Returns
    -------
    vk : np.ndarray, shape (p,)
        Variance statistic per coefficient.
    """
    n = betas.shape[0]
    return np.sum((betas - betas.mean(axis=0)) ** 2, axis=0) / n


def _fit_permuted(
    model_class,
    y_perm: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    offset: np.ndarray,
    bandwidth: float,
    kernel: str,
    fit_kwargs: dict,
    fixed_alpha: float = None,
) -> np.ndarray:
    """
    Fit model on permuted y and return Vk statistics.

    Parameters
    ----------
    model_class : class   One of GWNBRg, GWNBR, GWPR.
    y_perm      : np.ndarray  Permuted response variable.
    X, coords, offset, bandwidth, kernel, fit_kwargs : as in main model.
    fixed_alpha : float or None
        If provided and model_class is GWNBRg, holds alpha fixed at
        this value instead of re-estimating it from permuted data.
        This isolates the test to spatial variation in beta only,
        following the %estac SAS macro (Silva & Rodrigues, 2014).
        Re-estimating alpha on permuted data adds unrelated noise
        from the permutation's altered dispersion structure, which
        biases the permutation distribution and can mask genuine
        spatial non-stationarity in beta.

    Returns
    -------
    vk_perm : np.ndarray, shape (p,)
    """
    model = model_class(coords, y_perm, X, offset=offset)
    if fixed_alpha is not None and hasattr(model, "_alpha_override"):
        model._alpha_override = fixed_alpha
    model.fit(bandwidth=bandwidth, kernel=kernel,
              verbose=False, **fit_kwargs)
    return _compute_vk(model.betas)


class StationarityTest:
    """
    Permutation-based stationarity test for GW regression coefficients.

    Translated from the %estac SAS macro of Silva & Rodrigues (2014).

    Parameters
    ----------
    model       : fitted GWNBRg, GWNBR, or GWPR object.
                  Must be fitted before passing.
    model_class : class   Same class as model (needed for refitting).
    n_permutations : int  Number of permutations (default 99).
                          99 gives p-value resolution of 0.01.
                          999 for publication-quality results.
    alpha_level : float   Significance level (default 0.05).
    n_jobs      : int     Parallel jobs for permutation fitting.
    random_seed : int     For reproducibility.
    verbose     : bool

    Example
    -------
    >>> from gwnbr.models import GWNBRg
    >>> from gwnbr.stationarity import StationarityTest
    >>>
    >>> model = GWNBRg(coords, y, X, offset=offset,
    ...                variable_names=["income", "unemployment"])
    >>> model.fit(bandwidth=215, kernel="adaptive_nn")
    >>>
    >>> test = StationarityTest(model, GWNBRg, n_permutations=99)
    >>> results = test.run()
    >>> print(results)
    """

    def __init__(self,
                 model,
                 model_class: Type,
                 n_permutations: int = 99,
                 alpha_level: float = 0.05,
                 n_jobs: int = -1,
                 random_seed: int = 42,
                 verbose: bool = True):

        if not model._fitted:
            raise RuntimeError("Model must be fitted before running "
                               "stationarity test.")

        self.model        = model
        self.model_class  = model_class
        self.n_perms      = n_permutations
        self.alpha_level  = alpha_level
        self.n_jobs       = n_jobs
        self.random_seed  = random_seed
        self.verbose      = verbose

        # Extract model internals needed for refitting
        self._coords    = model.coords
        self._y         = model.y
        self._X         = model.X[:, 1:]    # strip intercept (re-added in fit)
        self._offset    = model.offset
        self._bandwidth = model.bandwidth
        self._kernel    = model._kernel
        self._var_names = model.var_names
        self._n         = model.n
        self._p         = model.p

        # Determine fit kwargs based on model type
        self._fit_kwargs = {}
        self._fit_kwargs["n_jobs"] = 1   # inner parallelism off during perms

        # Capture alpha from the REAL fitted model so permutation fits
        # hold it fixed — isolates the test to spatial variation in
        # beta only, matching the %estac SAS macro behaviour.
        if hasattr(model, "alpha_global") and model.alpha_global is not None:
            self._fixed_alpha = model.alpha_global
        elif model.alphas is not None:
            self._fixed_alpha = float(np.mean(model.alphas))
        else:
            self._fixed_alpha = None

        # Results (set after run())
        self.vk_observed  = None
        self.vk_permuted  = None   # shape (n_perms, p)
        self.p_values     = None
        self.results_df   = None

    def run(self) -> pd.DataFrame:
        """
        Run the permutation stationarity test.

        Returns
        -------
        pd.DataFrame with columns:
            Variable, Vk_observed, p_value, Stationary, Interpretation
        """
        rng = np.random.default_rng(self.random_seed)

        # ── Observed Vk ──────────────────────────────────────────────
        self.vk_observed = _compute_vk(self.model.betas)

        if self.verbose:
            print(f"[StationarityTest] Running {self.n_perms} permutations...")
            print(f"  Model     : {self.model_class.__name__}")
            print(f"  Bandwidth : {self._bandwidth}")
            print(f"  Kernel    : {self._kernel}")
            print(f"  Variables : {self._var_names}")

        # ── Permutation distribution ──────────────────────────────────
        # Generate all permuted y arrays upfront for reproducibility
        permuted_ys = [
            rng.permutation(self._y)
            for _ in range(self.n_perms)
        ]

        vk_perms = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_permuted)(
                self.model_class,
                y_perm,
                self._X,
                self._coords,
                self._offset,
                self._bandwidth,
                self._kernel,
                self._fit_kwargs,
                self._fixed_alpha,
            )
            for i, y_perm in enumerate(permuted_ys)
        )

        self.vk_permuted = np.array(vk_perms)   # (n_perms, p)

        # ── P-values ──────────────────────────────────────────────────
        # p_j = proportion of permuted Vk_j >= observed Vk_j
        self.p_values = np.mean(
            self.vk_permuted >= self.vk_observed[None, :],
            axis=0
        )

        # ── Results table ─────────────────────────────────────────────
        rows = []
        for j, name in enumerate(self._var_names):
            stationary  = self.p_values[j] >= self.alpha_level
            interp = ("Stationary — spatial variation likely noise"
                      if stationary
                      else "Non-stationary — significant spatial variation")
            rows.append({
                "Variable":     name,
                "Vk_observed":  round(float(self.vk_observed[j]), 6),
                "p_value":      round(float(self.p_values[j]), 4),
                "Stationary":   stationary,
                "Interpretation": interp,
            })

        self.results_df = pd.DataFrame(rows)

        if self.verbose:
            print(f"\n[StationarityTest] Results (α={self.alpha_level}):")
            print(self.results_df.to_string(index=False))

        return self.results_df

    def summary(self) -> str:
        """Return a formatted summary string."""
        if self.results_df is None:
            return "Test not yet run. Call run() first."

        non_stat = self.results_df[~self.results_df["Stationary"]]
        stat     = self.results_df[self.results_df["Stationary"]]

        lines = [
            "=" * 62,
            "  Stationarity Test — Permutation Method",
            f"  Model       : {self.model_class.__name__}",
            f"  Permutations: {self.n_perms}",
            f"  Alpha level : {self.alpha_level}",
            "-" * 62,
            f"  NON-STATIONARY ({len(non_stat)} variables):",
        ]
        for _, row in non_stat.iterrows():
            lines.append(f"    {row['Variable']:<25} "
                         f"Vk={row['Vk_observed']:.6f}  "
                         f"p={row['p_value']:.4f}  ✗")
        lines.append(f"\n  STATIONARY ({len(stat)} variables):")
        for _, row in stat.iterrows():
            lines.append(f"    {row['Variable']:<25} "
                         f"Vk={row['Vk_observed']:.6f}  "
                         f"p={row['p_value']:.4f}  ✓")
        lines.append("=" * 62)
        return "\n".join(lines)

    def plot(self, figsize: tuple = (10, 4), save_path: str = None):
        """
        Plot observed Vk against the permutation distribution
        for each coefficient.

        Parameters
        ----------
        figsize   : tuple
        save_path : str or None   If given, saves the figure.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib required for plotting.")

        if self.results_df is None:
            raise RuntimeError("Run test first.")

        p = self._p
        ncols = min(3, p)
        nrows = int(np.ceil(p / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(figsize[0], figsize[1] * nrows))
        axes = np.array(axes).flatten()

        fig.suptitle(
            f"Stationarity Test — {self.model_class.__name__}\n"
            f"{self.n_perms} permutations  |  α={self.alpha_level}",
            fontsize=12
        )

        for j, name in enumerate(self._var_names):
            ax    = axes[j]
            vk_p  = self.vk_permuted[:, j]
            vk_o  = self.vk_observed[j]
            pval  = self.p_values[j]
            color = "#d7191c" if pval < self.alpha_level else "#2c7bb6"

            ax.hist(vk_p, bins=20, color="#cccccc",
                    edgecolor="white", alpha=0.8, label="Permuted")
            ax.axvline(vk_o, color=color, linewidth=2,
                       linestyle="--",
                       label=f"Observed (p={pval:.3f})")
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Vk (variance statistic)", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)

            status = "Non-stationary" if pval < self.alpha_level \
                     else "Stationary"
            ax.text(0.97, 0.95, status,
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    color=color, fontweight="bold")

        for k in range(j + 1, len(axes)):
            axes[k].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {save_path}")
        else:
            plt.show()

        return fig, axes
