"""Identity link function implementation."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-14


import torch
from torch import tensor

from .base_class import BaseLink


class IdentityLink(BaseLink):
    """Implementation of the IdentityLink function."""

    def _link(self, y: tensor) -> tensor:
        """
        Get the link, eta, as a function of y.

        Overrides BaseLink._link.
        """
        return 1.0 * y

    def _inverse_link(self, eta: tensor) -> tensor:
        """
        Get the target, y, as a function of the link, `eta`.

        Overrides BaseLink._inverse_link.
        """
        return 1.0 * eta

    def dydeta(self, y: tensor) -> tensor:
        """
        Get the derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.dydeta.
        """
        return torch.ones(y.shape, dtype=y.dtype)

    def d2ydeta2(self, y: tensor) -> tensor:
        """
        Get the second derivative of `y` with respect to the link as a function of y.

        Overrides BaseLink.d2ydeta2.
        """
        return torch.zeros(y.shape, dtype=y.dtype)
