"""Torch module SigmoidInvertible layer."""

# author: Benjamin Cross
# email: btcross26@yahoo.com


import torch
from torch import nn, tensor


class SigmoidInvertible(nn.Sigmoid):
    """
    SigmoidInvertible module.

    Example
    -------
    >>> import torch
    >>> from tboost.torch_modules import SigmoidInvertible
    >>> t = torch.linspace(-3.0, 3.0, 101, dtype=torch.float64)
    >>> si = SigmoidInvertible()
    >>> out = si(t)
    >>> out = si.invert(out)
    >>> torch.testing.assert_close(t, out, atol=1e-8, rtol=0.0)
    """

    def invert(self, x: tensor, eps: float = 1e-12) -> tensor:
        """
        Invert the operation of the sigmoid layer.

        Parameters
        ----------
        x: tensor
            The tensor to invert. Values should be between 0.0 and 1.0.

        eps: float (default: 1e-12)
            A small value to add to the numerator and denominator to avoid
            floating point errors in torch.log calculations.
        """
        out = torch.log((x + eps) / (1.0 - x + eps))
        return out
