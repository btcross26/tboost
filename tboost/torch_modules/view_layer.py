"""Torch module View layer."""

# author: Benjamin Cross
# email: btcross26@yahoo.com


from typing import List

from torch import nn, tensor


class View(nn.Module):
    """
    Formal module to apply reshape/view as a layer.

    Example
    -------
    >>> import torch
    >>> from tboost.torch_modules import View
    >>> t = torch.randn(15, 20)
    >>> t.shape
    torch.Size([15, 20])
    >>> v = View(shape=(2, 10))
    >>> v(t).shape
    torch.Size([15, 2, 10])
    >>> v.forward(t).shape   # same as above
    torch.Size([15, 2, 10])
    """

    def __init__(self, shape: List[int]):
        """
        Class initializer.

        Parameters
        ----------
        shape: list-like
            Re-shaped dims of tensor, excluding the initial batch dimension.
        """
        super().__init__()
        self.shape = shape

    def forward(self, x: tensor) -> tensor:  # noqa: D102
        return x.view(-1, *self.shape)
