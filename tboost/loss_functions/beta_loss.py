"""Beta loss function implementation."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-18


import math
from typing import Callable

import torch
from scipy.optimize import bisect
from scipy.special import beta as beta_function
from scipy.special import betainc
from torch import tensor

from . import BaseLoss


class BetaLoss(BaseLoss):
    """BetaLoss class implementation."""

    def __init__(self, alpha: float, beta: float, eps: float = 1e-10):
        """
        Class initializer.

        Extends BaseLoss.__init__.

        Parameters
        ----------
        alpha: float
            Value of alpha in beta loss denominator, mu ** (1 - alpha)

        beta: float
            Value of beta in beta loss denominator, (1 - mu) ** (1 - beta)

        eps: float
            Small value to use in log calculations to avoid numerical error
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.scale = 1.0 / beta_function(alpha, beta)
        self._vt_callback = self.beta_callback(alpha, beta, eps)

    @staticmethod
    def beta_callback(
        alpha: float, beta: float, eps: float = 1e-10
    ) -> Callable[[tensor], tensor]:
        """
        Compute the beta callback function for quasi-deviance.

        Parameters
        ----------
        alpha: float
            Value of alpha in beta loss denominator, mu ** (1 - alpha)

        beta: float
            Value of beta in beta loss denominator, (1 - mu) ** (1 - beta)

        eps: float
            Small value to use in log calculations to avoid numerical error

        Returns
        -------
        callable
            A callable that takes an tensor and returns an tensor after
            calculating the beta quasi-deviance denominator
        """
        scale = 1.0 / beta_function(alpha, beta)

        def vt_callback(yp: tensor) -> tensor:
            return torch.exp(
                -math.log(scale)
                + (1.0 - alpha) * torch.log(yp + eps)
                + (1.0 - beta) * torch.log(1.0 - yp + eps)
            )

        return vt_callback

    def _loss(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the per-observation loss as a function of `yt` and `yp`.

        Overrides BaseLoss._loss.
        """
        c1 = betainc(self.alpha, self.beta, yt) - betainc(self.alpha, self.beta, yp)
        c2 = betainc(self.alpha + 1.0, self.beta, yt) - betainc(
            self.alpha + 1.0, self.beta, yp
        )
        b1 = beta_function(self.alpha, self.beta)
        b2 = beta_function(self.alpha + 1, self.beta)
        return (yt * c1 * b1 - c2 * b2) * self.scale

    def dldyp(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the first derivative of the loss with respect to `yp`.

        Overrides BaseLoss.dldyp.
        """
        return -(yt - yp) / self._vt_callback(tensor(yp))

    def d2ldyp2(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the second derivative of the loss with respect to `yp`.

        Overrides BaseLoss.d2ldyp2.
        """
        callback_values = self._vt_callback(yp)
        d_left = 1.0 / callback_values
        du = (1.0 - self.alpha) / (yp + self.eps) - (1.0 - self.beta) / (
            1.0 - yp + self.eps
        )
        d_right = du * (yt - yp) / callback_values
        return d_left + d_right


class LeakyBetaLoss(BetaLoss):
    """Class implementation of LeakyBetaLoss loss function."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float = 1.0,
        eps: float = 1e-10,
        xtol: float = 1e-8,
    ):
        """
        Class initializer.

        Extends BetaLoss.__init__.

        Parameters
        ----------
        alpha: float
            Passed as the alpha argument to the BetaLoss initializer

        beta: float
            Passed as the beta argument to the BetaLoss initializer

        eps: float, optional
            Passed as the eps argument to the BetaLoss initializer (the default value
            is 1e-10)

        gamma: float, optional
            A float in the range (0.0, 1.0] specifying ... (the default value is 1.0)

        xtol: float, optional
            Passed as the xtol argument to scipy.optimize.bisect (the default value is
            1e-8)
        """
        self.gamma = gamma
        super().__init__(alpha=alpha, beta=beta, eps=eps)

        # find leaky point values
        self.rL = self.alpha / (self.alpha + self.beta)  # left x
        self.vL = super().dldyp(tensor(0.0), tensor(self.rL))  # left slope
        self.rR = 1.0 - self.rL  # right x
        self.vR = super().dldyp(tensor(1.0), tensor(1.0 - self.rR))  # right slope

        # leaky slopes
        self.mL = self.gamma * self.vL
        self.mR = self.gamma * self.vR

        # transition pts
        floss = super().dldyp
        self.xL = bisect(
            lambda x: floss(tensor(0.0), x) - self.mL, self.rL, 1.0 - 1e-8, xtol=xtol
        )
        self.yL = super()._loss(0.0, self.xL)
        self.xR = bisect(
            lambda x: floss(tensor(1.0), x) - self.mR, 1e-8, 1.0 - self.rR, xtol=xtol
        )
        self.yR = super()._loss(1.0, self.xR)

    def _loss(self, yt: tensor, yp: tensor) -> tensor:
        # calculate loss function values from regular betaloss
        values = super()._loss(yt, yp)

        # modify left shelf
        values = torch.where(
            yt - yp < -self.xL, self.yL - self.mL * (-yp + self.xL), values
        )

        # modify right shelf
        values = torch.where(
            yt - yp > 1.0 - self.xR,
            self.yR - self.mR * (yt - yp - 1.0 + self.xR),
            values,
        )

        return values

    def dldyp(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the first derivative of the loss with respect to `yp`.

        Overrides BaseLoss.dldyp.
        """
        # calculate loss gradient values from regular betaloss
        values = super().dldyp(yt, yp)

        # modify left shelf
        values = torch.where(yt - yp < -self.xL, self.mL, values)

        # modify right shelf
        values = torch.where(yt - yp > 1.0 - self.xR, self.mR, values)

        return values

    def d2ldyp2(self, yt: tensor, yp: tensor) -> tensor:
        """
        Calculate the second derivative of the loss with respect to `yp`.

        Overrides BaseLoss.d2ldyp2.
        """
        # calculate loss gradient values from regular betaloss
        values = super().d2ldyp2(yt, yp)

        # modify shelves with 0.0 second derivative
        values = torch.where(
            (yt - yp < -self.xL) | (yt - yp > 1.0 - self.xR), 0.0, values
        )

        return values
