"""Bernoulli loss function implementation (log-loss)."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-18


import torch
from torch import tensor

from .base_class import BaseLoss


class LogLoss(BaseLoss):
    """Log loss function class."""

    def __init__(self, eps: float = 1e-12):
        """
        Class initializer.

        Extends the BaseLoss class intializer.

        Parameter
        ---------
        eps: float (default = 1e-10)
            A small constant float to prevent log from returning negative infinity.
        """
        super().__init__()
        self._eps = eps

    def _loss(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the per-observation loss as a function of `yt` and `yp`.

        Overrides BaseLoss._loss.
        """
        return -yt * torch.log(yp + self._eps) - (1.0 - yt) * torch.log(
            1.0 - yp + self._eps
        )

    def dldyp(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the first derivative of the loss with respect to `yp`.

        Overrides BaseLoss.dldyp.
        """
        return -(yt / (yp + self._eps)) + (1.0 - yt) / (1.0 - yp + self._eps)

    def d2ldyp2(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the second derivative of the loss with respect to `yp`.

        Overrides BaseLoss.d2ldyp2.
        """
        return (yt / (yp**2 + self._eps)) + (1.0 - yt) / ((1.0 - yp) ** 2 + self._eps)
