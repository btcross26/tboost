"""Loss function abstract base class."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-13

from abc import ABC, abstractmethod

from torch import tensor


class BaseLoss(ABC):
    """Base class for loss functions."""

    def __call__(self, yt: tensor, yp: tensor) -> tensor:
        """Call the instance object to calculate the loss function."""
        return self._loss(yt, yp)

    @abstractmethod
    def _loss(self, yt: tensor, yp: tensor) -> tensor:
        """Calculate the per-observation loss as a function of `yt` and `yp`."""
        ...

    @abstractmethod
    def dldyp(self, yt: tensor, yp: tensor) -> tensor:
        """Calculate the first derivative of the loss with respect to `yp`."""
        ...

    @abstractmethod
    def d2ldyp2(self, yt: tensor, yp: tensor) -> tensor:
        """Calculate the second derivative of the loss with respect to `yp`."""
        ...
