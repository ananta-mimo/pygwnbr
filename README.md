# gwnbr

**Geographically Weighted Negative Binomial Regression in Python**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

`gwnbr` fits spatially varying Negative Binomial regression models for count data — crash frequencies, disease counts, crime incidents — where the relationship between predictors and the outcome differs by location.

It is the first open-source Python implementation of GWNBR, translated from the SAS macro by Silva & Rodrigues (2014).

---

## When to use this

Use `gwnbr` when:
- Your response variable is a **count** (crashes, incidents, cases)
- Your data is **spatial** (census tracts, zip codes, counties)
- You suspect the effect of predictors **varies by location**
- Your data is **overdispersed** (variance > mean — almost always true for crash data)

If you are currently using a global Negative Binomial or a GWPR model, `gwnbr` is a direct upgrade.

---

## What it produces

For each spatial unit (e.g. each census tract), the model estimates:

- **Local beta coefficients** — how strongly each predictor relates to the outcome *at that location*
- **Standard errors and t-statistics** — for significance testing
- **Fitted values** — predicted counts per unit
- **Overdispersion parameter (alpha)** — global (GWNBRg) or local (GWNBR)

All results export to a tidy CSV ready to join back to a shapefile for mapping.

```
GEOID | y_obs | y_hat | alpha | beta_income | se_income | t_income | p_income | ...
```

---

## Models

| Model | Alpha | Bandwidth | Start here if... |
|-------|-------|-----------|-----------------|
| `GWPR` | Fixed at 0 | AICc or CV | You want a Poisson baseline |
| `GWNBRg` | One global value | **AICc** | **Recommended for most users** |
| `GWNBR` | Varies by location | CV | You need a full local alpha surface |

---

## Installation

```bash
git clone https://github.com/ananta-mimo/pygwnbr
cd gwnbr
pip install -e ".[viz]"
```

---

## Usage

### Step 1 - Prepare your data

```python
import numpy as np
import pandas as pd

df = pd.read_csv("maryland_tracts.csv") # Smaple dataset

coords = df[["longitude", "latitude"]].values   # centroid of each tract
y      = df["total_crashes"].values              # count response
X      = df[["income", "unemployment",
             "pct_black", "pop_density"]].values # predictors (standardized)
offset = np.log(df["population"].values)         # log population
```

### Step 2 - Find the optimal bandwidth

```python
from gwnbr.models import GWNBRg
from gwnbr.bandwidth import BandwidthSelector

selector = BandwidthSelector(
    GWNBRg, coords, y, X,
    offset=offset,
    variable_names=["income", "unemployment", "pct_black", "pop_density"],
    kernel="gaussian",
    criterion="aicc"
)
optimal_bw = selector.search()
```

### Step 3 - Fit the model

```python
model = GWNBRg(
    coords, y, X,
    offset=offset,
    variable_names=["income", "unemployment", "pct_black", "pop_density"]
)
model.fit(bandwidth=optimal_bw)
print(model.summary())
```

### Step 4 - Export results

```python
results = model.to_dataframe()
results["GEOID"] = df["GEOID"].values
results.to_csv("gwnbrg_results.csv", index=False)
```

### Step 5 - Map the output

```python
import geopandas as gpd
from gwnbr.viz import plot_coefficient_map, plot_significance_map

gdf = gpd.read_file("maryland_tracts.shp")
gdf = gdf.merge(results, on="GEOID")

# Coefficient surface for one variable
plot_coefficient_map(model, variable="income", gdf=gdf)

# Tracts where the effect is statistically significant
plot_significance_map(model, variable="unemployment", gdf=gdf)
```

---

## Comparing models

```python
from gwnbr.models import GWPR, GWNBRg, GWNBR

for ModelClass, name, bw in [
    (GWPR,   "GWPR",   50.0),
    (GWNBRg, "GWNBRg", optimal_bw),
]:
    m = ModelClass(coords, y, X, offset=offset)
    m.fit(bandwidth=bw, verbose=False)
    print(f"{name:8}  AICc={m.AICc:.1f}  R2={m.pct_deviance:.3f}")
```

---

## Input data requirements

| Column | Type | Notes |
|--------|------|-------|
| longitude, latitude | float | Decimal degrees. Centroid of each spatial unit. |
| response (y) | integer | Raw counts, not rates. |
| predictors (X) | float | Standardize before fitting for comparable coefficients. |
| population | integer | Used as offset via `log(population)`. Must be > 0. |
| GEOID | string | Tract identifier for joining results back to shapefile. |

---
For raw crash point data, use `geopandas` for spatial joining and 
aggregation to tracts, and `cenpy` or `pygris` for fetching ACS variables.

## Theory and methodology

For the full mathematical derivation - NB-2 log-likelihood, IRLS weight matrices, NR score and Hessian equations, kernel functions, bandwidth selection, and significance testing - see [docs/THEORY.md](docs/THEORY.md).

---

## Citation

If you use this package in your research, please cite both:

**Methodology:**
```
Silva, A. R. and Rodrigues, T. C. V. (2014).
Geographically Weighted Negative Binomial Regression — Incorporating Overdispersion.
Statistics and Computing, 24, 769–783.
```

**This package:**
```
Sinha, A. and Mathew, S. (2026). gwnbr: Geographically Weighted Negative Binomial Regression in Python.
https://github.com/ananta-mimo/pygwnbr
```

---

## License

MIT. See [LICENSE](LICENSE).
