"""Quantile regression loss function implementation."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-18

import warnings

import torch
from torch import tensor

from .base_class import BaseLoss


class QuantileLoss(BaseLoss):
    """Quantile regression loss function class."""

    def __init__(self, p: float):
        """
        Class initializer.

        Extends BaseLoss.__init__.

        Parameters
        ----------
        p: float
            Value in the interval (0, 1) representing the quantile to regress on.
        """
        super().__init__()
        self.p_ = p

    def _loss(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the per-observation loss as a function of `yt` and `yp`.

        Overrides BaseLoss._loss.
        """
        multiplier = torch.where(yt - yp < 0, 1.0 - self.p_, self.p_)
        return torch.abs(yt - yp) * multiplier

    def dldyp(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the first derivative of the loss with respect to `yp`.

        Overrides BaseLoss.dldyp.
        """
        return torch.where(yt - yp < 0, 1.0 - self.p_, -self.p_)

    def d2ldyp2(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the second derivative of the loss with respect to `yp`.

        Overrides BaseLoss.d2ldyp2.
        """
        warnings.warn(
            "second derivative of quantile value loss with respect to yp is zero"
        )
        return torch.zeros(yp.shape, dtype=yp.dtype)
