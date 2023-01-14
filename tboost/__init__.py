"""tboost top-level module."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-14

from .boosted_model import BoostedModel
from .model_data_sets import ModelDataSets

__version__ = "0.0.0"

__all__ = ["BoostedModel", "ModelDataSets", "__version__"]
