"""
Custom type hints for use in various places of module
"""

# Author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2019-11-15


from __future__ import annotations

import sys

# import Protocol and self return type type hints depending on Python version
PYTHON_VERSION = float("{:d}.{:d}".format(*sys.version_info[:2]))
if PYTHON_VERSION < 3.8:
    from typing_extensions import Protocol  # type: ignore
else:  # Python 3.8+
    from typing import Protocol  # type: ignore

from typing import Any, Dict

from torch import tensor

# from typing import Protocol as Protocol  # type: ignore


class Predictor(Protocol):
    def predict(self, X: tensor) -> tensor:
        ...


class Model(Predictor, Protocol):
    def fit(self, X: tensor, y: tensor) -> Model:
        ...


class LinearModel(Model, Protocol):
    coef_: tensor
    intercept_: float

    def fit(self, X: tensor, y: tensor) -> LinearModel:
        ...


class ModelCallback(Protocol):
    def __call__(self, **kwargs: Dict[str, Any]) -> Model:
        ...


class WeightsCallback(Protocol):
    def __call__(self, yt: tensor, yp: tensor) -> tensor:
        ...
