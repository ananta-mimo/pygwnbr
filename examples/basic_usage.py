"""
basic_usage.py
----------------
Minimal end-to-end example for the gwnbr package using synthetic data.

This demonstrates the full standard workflow:
    1. Load data
    2. Find the optimal bandwidth (Golden Section Search, AICc)
    3. Fit GWNBRg (recommended starting model)
    4. View the summary
    5. Export local coefficients to a DataFrame / CSV
    6. (Optional) Compare against GWPR to see why NB matters

NOTE: synthetic_data.csv is randomly generated data for demonstration
only. It does not represent any real location, agency, or dataset.
See generate_synthetic_data.py for how it was created.

Run with:
    python basic_usage.py
"""

import sys
import os

# ── Import gwnbr ─────────────────────────────────────────────────────
# Works two ways:
#   1. If gwnbr is installed (pip install -e ".[viz]" from repo root)
#   2. If run directly from examples/ without installing first --
#      falls back to adding the repo root to sys.path.
try:
    from gwnbr.models import GWNBRg, GWPR
    from gwnbr.bandwidth import BandwidthSelector
except ImportError:
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, _repo_root)
    try:
        from gwnbr.models import GWNBRg, GWPR
        from gwnbr.bandwidth import BandwidthSelector
    except ImportError:
        sys.exit(
            "Error: could not import gwnbr.\n\n"
            "Install it first from the repository root:\n"
            "    pip install -e \".[viz]\"\n\n"
            "Then re-run this example from the examples/ folder."
        )

import numpy as np
import pandas as pd

# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = pd.read_csv("synthetic_data.csv")
print(f"Loaded {len(df)} synthetic spatial units")

coords = df[["longitude", "latitude"]].values
y      = df["crash_count"].values.astype(float)
offset = np.log(df["population"].values.astype(float))

predictor_cols   = ["median_income", "unemployment_rate"]
X = df[predictor_cols].values

overdispersion = y.var() / y.mean()
print(f"Overdispersion ratio: {overdispersion:.1f}x "
      f"(Poisson assumes 1.0 -- NB is justified here)")

# =============================================================================
# 2. FIND OPTIMAL BANDWIDTH
# =============================================================================

print("\nSearching for optimal bandwidth (GWNBRg, AICc)...")
selector = BandwidthSelector(
    GWNBRg,
    coords, y, X,
    offset=offset,
    variable_names=predictor_cols,
    kernel="gaussian",
    criterion="aicc",
    verbose=False,
)
optimal_bw = selector.search()
print(f"Optimal bandwidth: {optimal_bw:.4f}")

# =============================================================================
# 3. FIT GWNBRg
# =============================================================================

print("\nFitting GWNBRg...")
model = GWNBRg(
    coords, y, X,
    offset=offset,
    variable_names=predictor_cols
)
model.fit(bandwidth=optimal_bw, kernel="gaussian", verbose=False)

# =============================================================================
# 4. VIEW RESULTS
# =============================================================================

print("\n" + model.summary())

print("\nCoefficient summary (min / mean / max per variable):")
print(model.coefficient_summary().to_string(index=False))

lr2 = model.local_r2()
print(f"\nLocal R2:  min={lr2.min():.3f}  "
      f"mean={lr2.mean():.3f}  max={lr2.max():.3f}")

# =============================================================================
# 5. EXPORT RESULTS
# =============================================================================

results = model.to_dataframe()
results["tract_id"] = df["tract_id"].values
results.to_csv("synthetic_results.csv", index=False)
print("\nSaved: synthetic_results.csv")

# =============================================================================
# 6. OPTIONAL -- COMPARE AGAINST GWPR
# =============================================================================

print("\nFitting GWPR for comparison...")
gwpr = GWPR(coords, y, X, offset=offset, variable_names=predictor_cols)
gwpr.fit(bandwidth=optimal_bw, kernel="gaussian", verbose=False)

print(f"\n{'Model':<10}{'AICc':>12}{'Pseudo-R2':>12}")
print(f"{'GWPR':<10}{gwpr.AICc:>12.1f}{gwpr.pct_deviance:>12.4f}")
print(f"{'GWNBRg':<10}{model.AICc:>12.1f}{model.pct_deviance:>12.4f}")
print(f"\nGWNBRg AICc is lower than GWPR by "
      f"{gwpr.AICc - model.AICc:.1f} points, "
      f"reflecting the {overdispersion:.1f}x overdispersion "
      f"in this data.")
