"""
Meta-fixtures for unit tests
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2020-04-07


import pytest

from tboost import BoostedModel
from tboost.link_functions import IdentityLink
from tboost.loss_functions import LeastSquaresLoss
from tboost.weak_learners import SimplePLS


# fixture for generic boosted model object
@pytest.fixture(scope="function")
def boosted_model_instance():
    # setup
    model = BoostedModel(
        link=IdentityLink(),
        loss=LeastSquaresLoss(),
        model_callback=SimplePLS,
        model_callback_kwargs={"max_vars": 3, "filter_threshold": None},
    )
    # yield objects
    yield model

    # teardown
    del model
