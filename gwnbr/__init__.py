"""
gwnbr: Geographically Weighted Negative Binomial Regression
============================================================

A Python implementation of Geographically Weighted Negative Binomial
Regression (GWNBR) and its global-alpha variant (GWNBRg), based on:

    Silva, A. R. and Rodrigues, T. C. V. (2014).
    "Geographically Weighted Negative Binomial Regression -
    Incorporating Overdispersion."
    Statistics and Computing, 24, 769-783.

Also implements Geographically Weighted Poisson Regression (GWPR) as a
special case, following:

    Nakaya, T., Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2005).
    "Geographically Weighted Poisson Regression for Disease Association Mapping."
    Statistics in Medicine, 24, 2695-2717.

Modules
-------
models      : GWNBR, GWNBRg, GWPR model classes
kernels     : Spatial kernel weighting functions
bandwidth   : Bandwidth selection (Golden Section Search, CV, AIC)
utils       : Distance calculations, IRLS solver, NR solver
viz         : Mapping and diagnostic plots

Example
-------
>>> from gwnbr.models import GWNBRg
>>> model = GWNBRg(coords=coords, y=y, X=X, offset=offset)
>>> model.fit(bandwidth=50.0, kernel='gaussian', method='fixed')
>>> print(model.summary())
"""

from gwnbr.models.gwnbrg import GWNBRg
from gwnbr.models.gwnbr import GWNBR
from gwnbr.models.gwpr import GWPR
from gwnbr.stationarity import StationarityTest

__version__ = "0.1.0"
__author__ = "Ananta Sinha, Sonu Mathew"
__license__ = "MIT"

__all__ = ["GWNBR", "GWNBRg", "GWPR", "StationarityTest"]
