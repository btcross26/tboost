"""ModelDataSets class for splitting data into training and validation sets."""

# author: Benjamin Cross
# email: btcross26@yahoo.com
# created: 2023-01-18


from typing import Iterator, Optional

import numpy as np
import torch
from torch import tensor


class ModelDataSets:
    """ModelDataSets class for abstracting data set implementation from BoostedModel."""

    def __init__(
        self,
        X: tensor,
        yt: tensor,
        weights: Optional[tensor] = None,
        validation_fraction: float = 0.0,
        validation_stratify: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Class initializer.

        Parameters
        ----------
        X: tensor, shape (n_samples, n_features)
            Feature matrix of type float.

        yt: tensor, shape (n_samples,)
            Target vector.

        weights: tensor (optional, default=None), shape (n_samples,)
            Sample weights to be used in the fitting process.

        validation_fraction: float (optional, default=0.0)
            Fraction of dataset to use as validation set for early stopping.

        validation_stratify: bool (default=False)
            If True, stratify the validation sample and the training sample using the
            model target. This only makes sense for classification problems.

        random_state: int (optional, default=None)
            Set the random state of the instance so that the data set split can be
            reproduced.
        """
        # initialize attributes from init args
        self.X = X
        self.yt = yt
        self.weights = (
            torch.ones(yt.shape[0], dtype=yt.dtype) if weights is None else weights
        )
        self.validation_fraction = validation_fraction
        self.validation_stratify = validation_stratify
        self._rng = np.random.RandomState(random_state)

        # public vars to be created during class usage
        self.X_train: tensor
        self.yt_train: tensor
        self.weights_train: tensor
        self.X_val: tensor  # if validation_fraction > 0.0
        self.yt_val: tensor  # if validation_fraction > 0.0
        self.weights_val: tensor  # if validation_fraction > 0.0

        # private vars to be created during class usage
        self._tindex: tensor
        self._vindex: tensor  # if validation_fraction > 0.0

        # split data sets as necessary
        self._create_index(validation_fraction, validation_stratify)
        self._create_data_sets()

    def has_validation_set(self) -> bool:
        """Return True if the validation fraction is greater than zero."""
        return hasattr(self, "_vindex")

    def _create_data_sets(self) -> None:
        """Create the train/test datasets (private)."""
        self.X_train = self.X[self._tindex]
        self.yt_train = self.yt[self._tindex]
        self.weights_train = self.weights[self._tindex]

        if self.has_validation_set():
            self.X_val = self.X[self._vindex]
            self.yt_val = self.yt[self._vindex]
            self.weights_val = self.weights[self._vindex]

    def _create_index(
        self, validation_fraction: float, validation_stratify: bool
    ) -> None:
        """Get the sampled train/test indices (private method)."""
        # initialize training index
        self._tindex = np.arange(self.yt.shape[0]).astype(int)

        # create validation index if specified
        if validation_fraction > 0.0:
            if validation_stratify:
                self._vindex = np.hstack(
                    [
                        array
                        for array in self._stratify_groups_generator(
                            validation_fraction
                        )
                    ]
                )
            else:
                n = self.yt.shape[0]
                self._vindex = np.random.choice(
                    n, int(validation_fraction * n), replace=False
                )
            self._vindex.sort()
            self._tindex = np.setdiff1d(self._tindex, self._vindex, assume_unique=True)

            # convert to tensor
            self._vindex = torch.from_numpy(self._vindex.astype(int))
            self._tindex = torch.from_numpy(self._tindex.astype(int))

    def _stratify_groups_generator(
        self, validation_fraction: float
    ) -> Iterator[np.array]:
        """Stratify the dataset for classification problems (private method)."""
        full_index = np.arange(self.yt.shape[0])
        groups, group_index = np.unique(self.yt, return_inverse=True)
        for group in groups:
            mask = group_index == group
            n = mask.sum()
            g_index = self._rng.choice(
                full_index[mask], int(validation_fraction * n), replace=False
            )
            yield g_index
