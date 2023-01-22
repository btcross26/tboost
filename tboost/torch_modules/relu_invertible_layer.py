"""Torch module LeakyReLUInvertible layer."""

# author: Benjamin Cross
# email: btcross26@yahoo.com


import torch
from torch import nn, tensor


class LeakyReLUInvertible(nn.LeakyReLU):
    """
    LeakyReLUInvertible module.

    Example
    -------
    >>> import torch
    >>> from tboost.torch_modules import LeakyReLUInvertible
    >>> t = torch.tensor([0.0, 1.0, 10.0, -10.0, -100.0], dtype=torch.float64)
    >>> lri = LeakyReLUInvertible(0.3)
    >>> out = lri(t)
    >>> out = lri.invert(out)
    >>> torch.testing.assert_close(t, out, atol=0.0, rtol=1e-8)
    """

    def invert(self, x: tensor) -> tensor:
        """Invert the operation of the leaky relu layer."""
        out = torch.where(x >= 0.0, x, x / self.negative_slope)
        return out
