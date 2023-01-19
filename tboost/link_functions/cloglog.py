"""Complementary log-log link function implementation."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-14


import torch
from torch import tensor

from .base_class import BaseLink


class CLogLogLink(BaseLink):
    """Complementary log-log link function implementation."""

    def __init__(self, eps: float = 1e-10):
        """
        Class initializer.

        Extends the BaseLink class initializer.

        Parameter
        ---------
        eps: float (default = 1e-10)
            A small constant float to prevent log from returning negative infinity.
        """
        super().__init__()
        self._eps = eps

    def _link(self, y: tensor) -> tensor:
        """
        Get the link, eta, as a function of y.

        Overrides BaseLink._link.
        """
        return torch.log(-torch.log(1.0 - y + self._eps))

    def _inverse_link(self, eta: tensor) -> tensor:
        """
        Get the target, y, as a function of the link, `eta`.

        Overrides BaseLink._inverse_link.
        """
        return 1.0 - torch.exp(-torch.exp(eta))

    def dydeta(self, y: tensor) -> tensor:
        """
        Get the derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.dydeta.
        """
        return -(1.0 - y) * torch.log(1.0 - y + self._eps)

    def d2ydeta2(self, y: tensor) -> tensor:
        """
        Get the second derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.d2ydeta2.
        """
        return self.dydeta(y) * (1.0 + torch.log(1.0 - y + self._eps))
