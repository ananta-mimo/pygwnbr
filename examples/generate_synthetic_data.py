"""
generate_synthetic_data.py
----------------------------
Generates a small synthetic dataset for the gwnbr usage example.

This data is entirely artificial — randomly generated coordinates
within Maryland's bounding box, randomly generated socioeconomic
predictors, and Negative Binomial crash counts simulated from known
true parameters. It does not represent any real location, agency,
or dataset.

Run this once to regenerate synthetic_data.csv:
    python generate_synthetic_data.py
"""

import numpy as np
import pandas as pd

# Reproducible random generator
rng = np.random.default_rng(42)
n = 100   # small — keeps the example fast (~10 seconds to fit)

# ── Synthetic coordinates (within Maryland's bounding box) ────────────
lon = rng.uniform(-79.5, -75.0, n)
lat = rng.uniform(37.9, 39.7, n)

# ── Synthetic standardized predictors ──────────────────────────────────
median_income      = rng.standard_normal(n)
unemployment_rate  = rng.standard_normal(n)

# ── Synthetic population (used as exposure offset) ─────────────────────
population = rng.integers(1000, 50000, n)

# ── Simulate Negative Binomial crash counts from known true params ────
# True relationship: crashes decrease with income, increase with
# unemployment. True overdispersion alpha = 0.5 (i.e. NOT Poisson).
# Intercept calibrated so crash counts land in a realistic ~5-200 range
# relative to a per-1000-population exposure, matching typical tract-
# level crash frequency studies.
true_beta0 = -7.5
true_beta_income   = -0.30
true_beta_unemploy =  0.20
true_alpha = 0.5

mu = np.exp(
    true_beta0
    + true_beta_income * median_income
    + true_beta_unemploy * unemployment_rate
    + np.log(population)
)

# NB sampling via the standard (r, p) parameterisation
r_nb = 1.0 / true_alpha
p_nb = r_nb / (r_nb + mu)
crash_count = rng.negative_binomial(r_nb, p_nb)

# ── Assemble and save ───────────────────────────────────────────────────
df = pd.DataFrame({
    "tract_id":          [f"SYN{i:04d}" for i in range(n)],
    "longitude":         np.round(lon, 5),
    "latitude":          np.round(lat, 5),
    "crash_count":       crash_count,
    "population":        population,
    "median_income":     np.round(median_income, 4),
    "unemployment_rate": np.round(unemployment_rate, 4),
})

df.to_csv("synthetic_data.csv", index=False)

print(f"Generated {n} synthetic tracts -> synthetic_data.csv")
print(f"\nOverdispersion check:")
print(f"  mean(crash_count) = {df['crash_count'].mean():.1f}")
print(f"  var(crash_count)  = {df['crash_count'].var():.1f}")
print(f"  ratio             = "
      f"{df['crash_count'].var() / df['crash_count'].mean():.1f}x "
      f"(true alpha = {true_alpha})")
