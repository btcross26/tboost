"""Absolute value loss function implementation."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-18

import warnings

import torch
from torch import tensor

from .base_class import BaseLoss


class AbsoluteLoss(BaseLoss):
    """Absolute loss function class."""

    def _loss(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the per-observation loss as a function of `yt` and `yp`.

        Overrides BaseLoss._loss.
        """
        return torch.abs(yt - yp)

    def dldyp(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the first derivative of the loss with respect to `yp`.

        Overrides BaseLoss.dldyp.
        """
        return torch.where(
            yt - yp < 0, tensor(1.0, dtype=yp.dtype), tensor(-1.0, dtype=torch.float32)
        )

    def d2ldyp2(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the second derivative of the loss with respect to `yp`.

        Overrides BaseLoss.d2ldyp2.
        """
        warnings.warn(
            "second derivative of absolute value loss with respect to yp is zero"
        )
        return torch.zeros(yp.shape, dtype=yp.dtype)
