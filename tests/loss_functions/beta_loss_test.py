"""
Unit tests for beta loss class implementations. Checks that the beta loss
implementation matches quasi-deviance with beta callback and that the leaky
beta loss matches intent.
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-18


import pytest
import torch
from torch import tensor

from tboost.loss_functions import BetaLoss, LeakyBetaLoss, QuasiLogLoss

# module pytest fixtures

# fixture for beta loss tests
@pytest.fixture(
    scope="function",
    params=[(2.0, 2.0), (4.0, 2.0), (3.0, 9.0)],
    ids=["alpha=2,beta=2", "alpha=4,beta=2", "alpha=3,beta=9"],
)
def beta_loss_objects(request):
    # setup
    alpha, beta = request.param
    beta_callback = BetaLoss.beta_callback(alpha, beta)
    ql_loss = QuasiLogLoss(beta_callback, d0_n=5000)
    beta_loss = BetaLoss(alpha, beta)

    # yield objects
    yield ql_loss, beta_loss

    # teardown
    del beta_callback, ql_loss, beta_loss


# fixture for leaky beta loss tests
@pytest.fixture(
    scope="function",
    params=[0.05, 0.1, 0.5, 1.0],
    ids=["gamma=0.05", "gamma=0.1", "gamma=0.5", "gamma=1.0"],
)
def leaky_beta_loss_objects(request):
    # setup
    alpha, beta = 3.0, 9.0
    gamma = request.param
    leaky_beta_loss = LeakyBetaLoss(alpha, beta, gamma)
    beta_loss = BetaLoss(alpha, beta)

    # yield objects
    yield leaky_beta_loss, beta_loss

    # teardown
    del leaky_beta_loss, beta_loss


# tests for BetaLoss class
def test_beta_loss(beta_loss_objects):
    # GIVEN a QuasiLogLoss instance with beta denominator callback and
    # BetaLoss instance with equivalent alpha/beta parameters
    ql_loss, beta_loss = beta_loss_objects

    # WHEN loss values are calculated at various points
    yt = tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    yp = tensor([0.1, 0.5, 0.9, 0.1, 0.5, 0.9], dtype=torch.float64)
    beta_calc = beta_loss(yt, yp)
    ql_calc = ql_loss(yt, yp)

    # THEN the two different instances should yield approximately equal values
    torch.testing.assert_close(beta_calc, ql_calc, atol=1e-3, rtol=1e-4)

    # WHEN d1 values are calculated at various points
    beta_calc_d1 = beta_loss.dldyp(yt, yp)
    ql_calc_d1 = ql_loss.dldyp(yt, yp)

    # THEN the two different instances should yield approximately equal values
    torch.testing.assert_close(beta_calc_d1, ql_calc_d1, atol=1e-4, rtol=1e-5)

    # WHEN d2 values are calculated at various points
    beta_calc_d2 = beta_loss.d2ldyp2(yt, yp)
    ql_calc_d2 = ql_loss.d2ldyp2(yt, yp)

    # THEN the two different instances should yield approximately equal values
    torch.testing.assert_close(beta_calc_d2, ql_calc_d2, atol=1e-3, rtol=1e-4)


# tests for LeakyBetaLoss class
def test_leaky_beta_loss(leaky_beta_loss_objects):
    # GIVEN a LeakyBetaLoss instance with sister BetaLoss instance
    leaky_loss, beta_loss = leaky_beta_loss_objects

    # WHEN loss values are calculated at various points
    yt = tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)  # mid-values
    yp = tensor([0.3, 0.5, 0.9, 0.1, 0.15, 0.2], dtype=torch.float64)  # mid-values
    leaky_calc = leaky_loss(yt, yp)
    beta_calc = beta_loss(yt, yp)

    # THEN the two different instances should yield equal values inside of the
    # ratio points
    torch.testing.assert_close(leaky_calc, beta_calc, atol=0.0, rtol=1e-8)

    # WHEN d1 values are calculated at various points inside the ratio points
    beta_calc_d1a = beta_loss.dldyp(yt, yp)
    leaky_calc_d1a = leaky_loss.dldyp(yt, yp)

    # THEN the two different instances should yield equal values
    torch.testing.assert_close(leaky_calc_d1a, beta_calc_d1a, atol=0.0, rtol=1e-8)

    # WHEN d1 values are calculated at various points outside the ratio points
    yt_shelf = tensor([1.0, 0.0, 0.0], dtype=torch.float64)  # shelf-values
    yp_shelf = tensor([0.001, 0.9, 0.99], dtype=torch.float64)  # shelf-values
    beta_calc_d1b = leaky_loss.gamma * beta_loss.dldyp(
        yt_shelf,
        tensor(0.25, dtype=torch.float64),  # 0.25 comes from alpha / (alpha + beta)
    )
    leaky_calc_d1b = leaky_loss.dldyp(yt_shelf, yp_shelf)

    # THEN the two different instances should yield equal values
    torch.testing.assert_close(
        leaky_calc_d1b, beta_calc_d1b, atol=0.0, rtol=1e-8
    )  # this will fail if gamma is too small (e.g., 0.01 close to flat shelves)

    # WHEN d2 values are calculated at various points inside ratio points
    beta_calc_d2a = beta_loss.d2ldyp2(yt, yp)
    leaky_calc_d2a = leaky_loss.d2ldyp2(yt, yp)

    # THEN the two sets should be equivalent
    torch.testing.assert_close(leaky_calc_d2a, beta_calc_d2a, atol=0.0, rtol=1e-8)

    # WHEN d2 values are calculated at various points outside ratio points
    leaky_calc_d2b = leaky_loss.d2ldyp2(yt_shelf, yp_shelf)

    # THEN the values should equal ZERO
    torch.testing.assert_close(
        leaky_calc_d2b,
        torch.zeros_like(leaky_calc_d2b, dtype=torch.float64),
        atol=0.0,
        rtol=1e-8,
    )


# test for LeakyBetaLoss class transition points
def test_leaky_beta_loss_transition(leaky_beta_loss_objects):
    # GIVEN a LeakyBetaLoss instance
    leaky_loss, _ = leaky_beta_loss_objects

    # WHEN loss values are calculated just to the left and just to the right of the
    # transition points (and first derivative)
    eps = 1e-8

    # yp/yt values
    yp_left = tensor([leaky_loss.xL - eps, leaky_loss.xR - eps], dtype=torch.float64)
    yp_right = tensor([leaky_loss.xL + eps, leaky_loss.xR + eps], dtype=torch.float64)
    yt = tensor([0.0, 1.0], dtype=torch.float64)

    # loss
    loss_left = leaky_loss(yt, yp_left)
    loss_right = leaky_loss(yt, yp_right)

    # dldyp
    dl_left = leaky_loss.dldyp(yt, yp_left)
    dl_right = leaky_loss.dldyp(yt, yp_right)

    # THEN the two calculations should be very close to one another for both
    # loss and derivatives
    torch.testing.assert_close(loss_left, loss_right, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(dl_left, dl_right, atol=1e-6, rtol=0.0)


# test for LeakyBetaLoss class end points
def test_leaky_beta_loss_transition(leaky_beta_loss_objects):
    # GIVEN a LeakyBetaLoss instance
    leaky_loss, _ = leaky_beta_loss_objects

    # WHEN loss values are calculated close to the end points and at the
    # transition points
    eps = 1e-8

    # yp/yt values
    yp_transition = tensor([leaky_loss.xL, leaky_loss.xR], dtype=torch.float64)
    yp_end = tensor([1.0 - eps, eps], dtype=torch.float64)
    yt = tensor([0.0, 1.0], dtype=torch.float64)

    # loss
    loss_transition = leaky_loss(yt, yp_transition)
    loss_end = leaky_loss(yt, yp_end)

    # THEN the end point calculations should be greater than or equal to the transition
    # point calculations
    assert torch.all(loss_end >= loss_transition)
