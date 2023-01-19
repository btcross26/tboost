"""
List of test params to import into link_function_tests.py
"""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-14

import math

import torch
from torch import tensor

from tboost.link_functions import (
    CLogLogLink,
    CubeRootLink,
    IdentityLink,
    LogitLink,
    LogLink,
    Logp1Link,
    PowerLink,
    ReciprocalLink,
    SqrtLink,
)

# setup link test params to loop through tests
LINK_TESTS = list()

# identity test params
LINK_TESTS.append(
    (
        "identity_test",
        IdentityLink(),
        tensor([[-5.0, -5.0], [0.0, 0.0], [10.0, 10.0]], dtype=torch.float64),
        torch.linspace(-50, 50, 101, dtype=torch.float64),
        (0.005, 0.0001),
    )
)

# logit test params
LINK_TESTS.append(
    (
        "logit_test",
        LogitLink(),
        tensor(
            [[0.5, 0.0], [0.25, -math.log(3)], [0.9, math.log(9)]], dtype=torch.float64
        ),
        torch.linspace(0.01, 0.99, 99, dtype=torch.float64),
        (0.001, 0.0001),
    )
)

# cloglog test params
LINK_TESTS.append(
    (
        "cloglog_test",
        CLogLogLink(),
        tensor(
            [
                [0.5, math.log(-math.log(0.5))],
                [0.25, math.log(-math.log(0.75))],
                [0.9, math.log(-math.log(0.1))],
            ],
            dtype=torch.float64,
        ),
        torch.linspace(0.01, 0.99, 99, dtype=torch.float64),
        (0.001, 0.0001),
    )
)

# log link test params
LINK_TESTS.append(
    (
        "log_test",
        LogLink(),
        tensor(
            [[1.0, 0.0], [math.exp(1), 1.0], [math.exp(10), 10.0]], dtype=torch.float64
        ),
        torch.logspace(0.0, 10.0, 111, base=math.exp(1), dtype=torch.float64),
        (0.001, 0.001),
    )
)

# logp1 link test params
LINK_TESTS.append(
    (
        "logp1_test",
        Logp1Link(),
        tensor(
            [[0.0, 0.0], [math.exp(1) - 1.0, 1.0], [math.exp(10) - 1.0, 10.0]],
            dtype=torch.float64,
        ),
        torch.logspace(0.0, 10.0, 111, base=math.exp(1), dtype=torch.float64),
        (0.001, 0.001),
    )
)

# power link (squared + 1) test params
LINK_TESTS.append(
    (
        "power_base_test",
        PowerLink(power=2, summand=1.0),
        tensor([[0.0, 1.0], [2.0, 9.0], [10.0, 121.0]], dtype=torch.float64),
        torch.linspace(0.0, 100.0, 101, dtype=torch.float64),
        (0.005, 0.001),
    )
)

# sqrt link test params
LINK_TESTS.append(
    (
        "sqrt_link_test",
        SqrtLink(),
        tensor([[0.01, 0.1], [4.0, 2.0], [100.0, 10.0]], dtype=torch.float64),
        torch.linspace(0.01, 10.0, 101, dtype=torch.float64),
        (0.001, 0.001),
    )
)

# power link (squared + 1) test params
LINK_TESTS.append(
    (
        "cuberoot_link_test",
        CubeRootLink(),
        tensor([[0.001, 0.1], [8.0, 2.0], [1000.0, 10.0]], dtype=torch.float64),
        torch.linspace(0.01, 10.0, 101, dtype=torch.float64),
        (0.001, 0.001),
    )
)

# power link (squared + 1) test params
LINK_TESTS.append(
    (
        "reciprocal_link_test",
        ReciprocalLink(),
        tensor([[1.0, 1.0], [2.0, 0.5], [10.0, 0.1]], dtype=torch.float64),
        torch.linspace(0.01, 10.0, 101, dtype=torch.float64),
        (0.001, 0.001),
    )
)
