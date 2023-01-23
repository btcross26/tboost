"""Torch module LinearInvertible layer."""

# author: Benjamin Cross
# email: btcross26@yahoo.com


import torch
from torch import nn, tensor


class LinearInvertible(nn.Linear):
    """
    LinearInvertible module.

    Example
    -------
    >>> import torch
    >>> from tboost.torch_modules import LinearInvertible
    >>> t = torch.randn(15, 20)
    >>> t.shape
    torch.Size([15, 20])
    >>> li = LinearInvertible(20, 10)
    >>> _ = torch.nn.init.xavier_uniform_(li.weight)
    >>> _ = li.bias.data.fill_(0.05)
    >>> out = li(t)
    >>> out.shape
    torch.Size([15, 10])
    >>> out = li.invert(out)
    >>> out.shape
    torch.Size([15, 20])
    """

    def invert(self, x: tensor) -> tensor:
        """
        Invert the operation of the linear layer.

        The output shape will be have the same dimension as the forward input
        operation, but the module must be trained to approximate equality between the
        input and the inverted output.
        """
        out = torch.matmul(x - self.bias.unsqueeze(0), self.weight)
        return out
