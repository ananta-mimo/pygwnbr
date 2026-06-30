---
title: 'gwnbr: A Python Package for Geographically Weighted Negative Binomial Regression'
tags:
  - Python
  - spatial statistics
  - geographically weighted regression
  - negative binomial regression
  - overdispersion
  - count data
  - transportation safety
authors:
  - name: Ananta Sinha
    orcid: 0000-0003-1798-1266
    corresponding: true
    affiliation: 1
  - name: Sonu Mathew
    orcid: 0000-0002-0707-8332
    affiliation: 1
affiliations:
  - name: Kittelson and Associates, Inc., USA
    index: 1
date: 2026
bibliography: paper.bib
---

# Summary

`gwnbr` is an open-source Python package implementing Geographically
Weighted Negative Binomial Regression (GWNBR) and its global-alpha
variant (GWNBRg), alongside Geographically Weighted Poisson Regression
(GWPR) as a baseline. It is the first Python implementation of GWNBR,
translating the SAS macro of @silva2014 into a fully modular, tested,
and documented software package.

The package enables researchers working with spatially referenced count
data — crash frequencies, disease incidence, crime incidents — to
simultaneously address two pervasive modelling challenges: spatial
nonstationarity (relationships between predictors and outcomes vary by
location) and overdispersion (variance substantially exceeds the mean).
For each spatial unit, `gwnbr` estimates a separate set of local
regression coefficients and, optionally, a local overdispersion
parameter, using a kernel-weighted likelihood framework. Results are
exported as tidy DataFrames suitable for downstream spatial
visualisation and analysis.

# Statement of Need

Count data in spatial settings routinely violate two fundamental
assumptions of standard regression models. First, global models assume
stationarity — that the relationship between predictors and the outcome
is identical at every location. This assumption is rarely justified in
practice. For example, the association between median household income
and traffic crash frequency may differ substantially between dense urban
cores and sparse rural areas within the same state. Second, Poisson
regression, the standard model for count data, assumes equidispersion
— that variance equals the mean. Empirical count data almost universally
exhibit overdispersion, with variance exceeding the mean by factors of
ten or more [@cameron1990].

Geographically Weighted Poisson Regression (GWPR), introduced by
@nakaya2005, addresses nonstationarity but assumes equidispersion.
Global Negative Binomial (NB) regression handles overdispersion but
assumes stationarity. GWNBR, developed by @silva2014, resolves both
problems simultaneously by embedding a Negative Binomial likelihood
within the geographically weighted framework. Despite its publication
in 2014, GWNBR has seen limited adoption in applied research, partly
because no open-source implementation existed outside of the original
SAS macro [@silva2016]. The `R` ecosystem offers the `GWmodel` package
[@gollini2015] for standard GWR and GWPR, but no implementation of
GWNBR. The Python ecosystem similarly lacks any implementation.

`gwnbr` fills this gap. Researchers in transportation safety,
epidemiology, criminology, and urban analytics who work in Python can
now apply GWNBR without SAS access or manual implementation.

# Mathematics

## Negative Binomial Regression

The NB-2 parameterisation models the conditional mean as:

$$\log(\mu_j) = \mathbf{x}_j \boldsymbol{\beta} + \log(E_j)$$

where $E_j$ is an exposure offset (e.g. population) and the variance is:

$$\text{Var}(Y_j) = \mu_j + \alpha \mu_j^2$$

The overdispersion parameter $\alpha \geq 0$ captures excess variance
beyond Poisson. When $\alpha = 0$ the model reduces to Poisson regression.

## Geographically Weighted Framework

For each focal location $i$, GWNBR maximises a locally weighted
log-likelihood:

$$\mathcal{L}(\boldsymbol{\beta}_i, \alpha_i) =
\sum_j w_{ij} \cdot \ell_j(\boldsymbol{\beta}_i, \alpha_i)$$

where $w_{ij}$ is a spatial kernel weight and $\ell_j$ is the NB
log-likelihood contribution of observation $j$. The local coefficients
$\boldsymbol{\beta}_i$ are estimated via Iteratively Reweighted Least
Squares (IRLS) and the local overdispersion parameter $\alpha_i$ via
Newton-Raphson (NR), alternating until convergence [@silva2014].

## GWNBRg

In GWNBRg [@silva2014], $\alpha$ is estimated once globally from a
standard NB regression, then held fixed during local IRLS estimation
of $\boldsymbol{\beta}_i$. Because $\alpha$ contributes exactly one
effective parameter ($k_2 = 1$), bandwidth selection via the corrected
Akaike Information Criterion (AICc) is valid for GWNBRg but not for
full GWNBR, where the effective parameter count contributed by the
$\alpha$ surface is analytically intractable.

## Kernel Functions

Three kernel functions are implemented, matching the methods of the
original SAS macro [@silva2016]:

- **Gaussian** (fixed bandwidth): $w_{ij} = \exp\left(-\frac{d_{ij}^2}{2h^2}\right)$
- **Bisquare** (adaptive): $w_{ij} = \left(1 - \left(\frac{d_{ij}}{h}\right)^2\right)^2$ for $d_{ij} \leq h$
- **Adaptive k-NN** (bisquare with per-location bandwidth): $h_i$ set to the distance to the $k$-th nearest neighbour of location $i$

## Bandwidth Selection

The optimal bandwidth is identified via Golden Section Search
[@fotheringham2002], minimising AICc for GWNBRg and GWPR, or
cross-validation for full GWNBR:

$$\text{AICc} = -2\mathcal{L}(\hat{\boldsymbol{\beta}}, \hat{\alpha})
+ 2k + \frac{2k(k+1)}{n - k - 1}$$

where $k = \text{tr}(\mathbf{S}) + 1$ is the effective number of
parameters and $\mathbf{S}$ is the hat matrix.

## Stationarity Test

A permutation-based stationarity test [@silva2014] assesses whether
the spatial variation in each local coefficient is statistically
significant. For each coefficient $j$, the variance statistic is:

$$V_{k_j} = \frac{1}{n} \sum_i \left(\hat{\beta}_{ij} -
\overline{\hat{\beta}}_j\right)^2$$

The observed $V_{k_j}$ is compared against a permutation distribution
obtained by randomly shuffling the response variable $R$ times and
refitting the model. The p-value is the proportion of permuted
$V_{k_j}$ values exceeding the observed value.

# Package Design

`gwnbr` is structured as a modular Python package with six components:

- `models`: Three model classes (`GWNBR`, `GWNBRg`, `GWPR`) inheriting
  from a shared abstract base class that provides common diagnostics,
  summary statistics, and DataFrame export.
- `kernels`: Gaussian, bisquare, and adaptive k-NN kernel functions.
- `bandwidth`: `BandwidthSelector` implementing Golden Section Search
  with AICc, AIC, or cross-validation criteria.
- `utils`: Haversine and Euclidean distance calculators, Newton-Raphson
  solver for $\alpha$, and IRLS solver for $\boldsymbol{\beta}$.
- `stationarity`: Permutation-based stationarity test with visualisation.
- `viz`: Coefficient surface maps, significance maps, residual
  diagnostics, and bandwidth search plots, supporting both centroid
  scatter and polygon choropleth rendering via `geopandas`.

Parallel computation across spatial units is implemented via `joblib`,
making the package practical for datasets of 1,000+ spatial units.
All model classes expose a consistent API: `fit()`, `summary()`,
`to_dataframe()`, `coefficient_summary()`, `local_r2()`, and
`significant_betas()`.

# Example Application

We demonstrate `gwnbr` using fatal and injury crash counts for 1,460
Maryland census tracts (2021–2023) with five standardised ACS
socioeconomic predictors: log population density, percent female,
percent Black, median household income, and unemployment rate. The
response variable exhibits severe overdispersion (variance-to-mean
ratio = 42.7), motivating the use of GWNBR over GWPR.

Using an adaptive k-NN bisquare kernel with optimal bandwidth
$k = 215$ nearest neighbours (selected by AICc via Golden Section
Search), GWNBR achieves AICc = 14,147 compared to GWPR AICc = 38,664,
a reduction of 24,517 points providing decisive evidence in favour of
the Negative Binomial specification [@burnham2002]. The global
overdispersion parameter $\hat{\alpha} = 0.628$ confirms that Poisson
equidispersion is untenable for this data. A permutation-based
stationarity test (999 permutations) confirms that population density
and median income exhibit statistically significant spatial variation
($p < 0.05$), validating the geographically weighted approach.

# Availability

`gwnbr` is available on GitHub at
[github.com/ananta-mimo/gwnbr](https://github.com/ananta-mimo/gwnbr)
under the MIT license, with a permanent archived version at
Zenodo [@sinha2026].

# Acknowledgements

The authors thank Alan Ricardo da Silva and Thais Carvalho Valadares
Rodrigues for making their SAS macro publicly available [@silva2016],
which served as the reference implementation for this Python translation.

# References
