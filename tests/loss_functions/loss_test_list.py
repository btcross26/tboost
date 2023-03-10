"""
List of test params to import into loss_function_tests.py
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-18


import math

import torch
from torch import tensor

from tboost.loss_functions import (
    AbsoluteLoss,
    LeastSquaresLoss,
    LogCoshLoss,
    LogLoss,
    PoissonLoss,
    QuantileLoss,
    QuasiLogLoss,
    StudentTLoss,
)

# setup link test params to loop through tests
LOSS_TESTS = list()

# least squares test params
LOSS_TESTS.append(
    (
        "least_squares",
        LeastSquaresLoss(),
        tensor(
            [[2.0, 2.0, 0.0], [5.0, 4.0, 1.0 / 2.0], [10.0, 15.0, 25.0 / 2.0]],
            dtype=torch.float64,
        ),
        (0.002, 0.0001),
    )
)

# absolute loss test params
LOSS_TESTS.append(
    (
        "absolute_loss",
        AbsoluteLoss(),
        tensor(
            [[1.5, 2.0, 0.5], [5.0, 4.0, 1.0], [10.0, 15.0, 5.0]], dtype=torch.float64
        ),
        (0.001, 0.0001),
    )
)

# logcosh loss test params
LOSS_TESTS.append(
    (
        "logcosh_loss",
        LogCoshLoss(),
        tensor(
            [
                [1.5, 2.0, 0.120114507],
                [5.0, 4.0, 0.433780830],
                [10.0, 15.0, 4.30689822],
            ],
            dtype=torch.float64,
        ),
        (0.001, 0.0001),
    )
)

# log loss test params
LOSS_TESTS.append(
    (
        "log_loss",
        LogLoss(),
        tensor(
            [
                [1.0, 0.5, 0.693147180],
                [1.0, math.exp(-1), 1.0],
                [1.0, math.exp(-2), 2.0],
                [0.0, 0.5, 0.693147180],
                [0.0, 1.0 - math.exp(-1), 1.0],
                [0.0, 1.0 - math.exp(-2), 2.0],
            ],
            dtype=torch.float64,
        ),
        (0.001, 0.0001),
    )
)

# poisson loss test params
LOSS_TESTS.append(
    (
        "poisson_loss",
        PoissonLoss(),
        tensor(
            [
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, math.exp(-1), math.exp(-1) + 1.0],
                [1.0, math.exp(2), math.exp(2) - 2.0],
                [5.0, math.exp(1), math.exp(1) - 5.0],
            ],
            dtype=torch.float64,
        ),
        (0.001, 0.0001),
    )
)

# quantile loss test params - p = 0.05
LOSS_TESTS.append(
    (
        "quantile_loss_05",
        QuantileLoss(p=0.05),
        tensor(
            [[1.5, 2.0, 0.95 * 0.5], [5.0, 4.0, 0.05 * 1.0], [10.0, 15.0, 0.95 * 5.0]],
            dtype=torch.float64,
        ),
        (0.001, 0.0001),
    )
)

# quantile loss test params - p = 0.50 (absolute loss proportional)
LOSS_TESTS.append(
    (
        "quantile_loss_50",
        QuantileLoss(p=0.50),
        tensor(
            [[1.5, 2.0, 0.50 * 0.5], [5.0, 4.0, 0.50 * 1.0], [10.0, 15.0, 0.50 * 5.0]],
            dtype=torch.float64,
        ),
        (0.001, 0.0001),
    )
)

# quantile loss test params - p = 0.95
LOSS_TESTS.append(
    (
        "quantile_loss_95",
        QuantileLoss(p=0.95),
        tensor(
            [[1.5, 2.0, 0.05 * 0.5], [5.0, 4.0, 0.95 * 1.0], [10.0, 15.0, 0.05 * 5.0]],
            dtype=torch.float64,
        ),
        (0.001, 0.0001),
    )
)

# student-t loss test params
LOSS_TESTS.append(
    (
        "student_t_loss",
        StudentTLoss(dof=2),
        tensor(
            [
                [1.5, 2.0, 0.17667455348],
                [5.0, 4.0, 0.60819766216],
                [10.0, 15.0, 1.5 * math.log(1.0 + 25.0 / 2.0)],
            ],
            dtype=torch.float64,
        ),
        (0.001, 0.0001),
    )
)

# quantile loss test params
LOSS_TESTS.append(
    (
        "quasilog_loss",
        QuasiLogLoss(
            vt_callback=lambda mu: torch.maximum(mu * (1.0 - mu), tensor(1e-10)),
            d0_n=5000,
        ),
        tensor(
            [
                [1.0, 0.5, 0.693147180],
                [1.0, math.exp(-1), 1.0],
                [1.0, math.exp(-2), 2.0],
                [0.0, 0.5, 0.693147180],
                [0.0, 1.0 - math.exp(-1), 1.0],
                [0.0, 1.0 - math.exp(-2), 2.0],
            ],
            dtype=torch.float64,
        ),
        (0.001, 0.0001),
    )
)
