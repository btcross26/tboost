"""Least squares loss function implementation."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-13

import torch
from torch import tensor

from .base_class import BaseLoss


class LeastSquaresLoss(BaseLoss):
    """Least squares loss function class."""

    def _loss(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the per-observation loss as a function of `yt` and `yp`.

        Overrides BaseLoss._loss.
        """
        return 0.5 * (yt - yp) * (yt - yp)

    def dldyp(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the first derivative of the loss with respect to `yp`.

        Overrides BaseLoss.dldyp.
        """
        return yp - yt

    def d2ldyp2(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the second derivative of the loss with respect to `yp`.

        Overrides BaseLoss.d2ldyp2.
        """
        return torch.ones(*yp.shape, dtype=yp.dtype)
