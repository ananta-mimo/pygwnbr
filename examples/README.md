# gwnbr Examples

This folder contains a minimal, fully reproducible example demonstrating
the standard `gwnbr` workflow.

## Files

| File | Purpose |
|------|---------|
| `synthetic_data.csv` | Pre-generated synthetic dataset (100 spatial units) |
| `generate_synthetic_data.py` | Script that created `synthetic_data.csv` |
| `basic_usage.py` | End-to-end example: load data, fit GWNBRg, view results |

## About the data

`synthetic_data.csv` is **entirely artificial**. Coordinates are randomly
sampled within a geographic bounding box, predictors are randomly drawn
from a standard normal distribution, and crash counts are simulated from
a known Negative Binomial process with:

```
true_beta0          = -7.5
true_beta_income     = -0.30
true_beta_unemploy   =  0.20
true_alpha (overdispersion) = 0.5
```

This data does not represent any real location, agency, or published
dataset. It exists purely to demonstrate the package API with a fast,
reproducible example (fits in seconds, not minutes).

To regenerate it yourself:

```bash
python generate_synthetic_data.py
```

## Running the example

```bash
pip install -e "..[viz]"   # install gwnbr from the repo root first
cd examples
python basic_usage.py
```

Expected output includes:
- An overdispersion check (~20x, confirming NB is appropriate)
- Golden Section Search finding the optimal bandwidth
- A fitted `GWNBRg` model summary with AICc, Pseudo-R2, and local
  coefficient ranges
- A side-by-side AICc comparison against `GWPR`, showing why the
  Negative Binomial specification matters when data is overdispersed

The fitted coefficients should closely recover the true values used to
generate the data (median_income ≈ -0.30, unemployment_rate ≈ 0.20),
confirming the estimator is working correctly.

## Using your own data

Replace `synthetic_data.csv` with your own dataset, formatted as:

| Column | Type | Notes |
|--------|------|-------|
| `longitude`, `latitude` | float | Decimal degrees, centroid of each spatial unit |
| your response column | integer | Raw counts, not rates |
| your predictor columns | float | Standardize before fitting |
| `population` (or other exposure) | integer | Used as `offset = log(population)` |

Then update the column names in `basic_usage.py` accordingly. See the
main [README](../README.md) and [THEORY.md](../docs/THEORY.md) for full
details on data requirements and the underlying methodology.
