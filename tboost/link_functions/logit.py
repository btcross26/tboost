"""Logit link function implementation."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-14


import torch
from torch import tensor

from .base_class import BaseLink


class LogitLink(BaseLink):
    """Implementation of the logit link function."""

    def _link(self, y: tensor) -> tensor:
        """
        Get the link, eta, as a function of y.

        Overrides BaseLink._link.
        """
        return torch.log(y / (1.0 - y))

    def _inverse_link(self, eta: tensor) -> tensor:
        """
        Get the target, y, as a function of the link, `eta`.

        Overrides BaseLink._inverse_link.
        """
        return 1.0 / (1.0 + torch.exp(-eta))

    def dydeta(self, y: tensor) -> tensor:
        """
        Get the derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.dydeta.
        """
        return y * (1.0 - y)

    def d2ydeta2(self, y: tensor) -> tensor:
        """
        Get the second derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.d2ydeta2.
        """
        return self.dydeta(y) * (1.0 - 2.0 * y)
