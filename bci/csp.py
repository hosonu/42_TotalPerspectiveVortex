"""Common Spatial Patterns (CSP) — sklearn-compatible transformer shell.

You implement :meth:`CustomCSP._compute_spatial_filters` with NumPy/SciPy.
:meth:`transform` only applies the learned filters (matrix multiply per trial).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class CustomCSP(BaseEstimator, TransformerMixin):
    """Spatial filter bank for EEG trials.

    Parameters
    ----------
    n_components :
        Number of spatial filters (rows of ``W_``). Use an even count if
        :class:`LogVarianceTransformer` takes ``filters_per_end`` first/last
        pairs.

    Attributes
    ----------
    W_ :
        Shape ``(n_components, n_channels_)``. Row ``k`` is a spatial filter:
        ``z[k, t] = sum_c W_[k, c] * x[c, t]`` per trial.
    n_channels_ :
        Number of EEG channels seen during :meth:`fit`.
    """

    def __init__(self, n_components: int = 6) -> None:
        self.n_components = n_components

    def _compute_spatial_filters(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Return ``W_`` with shape ``(n_components, n_channels)``.

        Parameters
        ----------
        X :
            Shape ``(n_trials, n_channels, n_times)``.
        y :
            Shape ``(n_trials,)``, integer class labels (e.g. 0 and 1).
        """
        raise NotImplementedError(
            "Implement CSP: compute spatial filters from X and y, then return "
            "shape (n_components, n_channels). "
            "fit() stores the return value in self.W_."
        )

    def fit(self, X, y):
        X, y = self._validate_inputs(X, y)
        self.n_channels_ = X.shape[1]
        W = self._compute_spatial_filters(X, y)
        W = np.asarray(W, dtype=np.float64)
        if W.ndim != 2 or W.shape != (self.n_components, self.n_channels_):
            raise ValueError(
                f"_compute_spatial_filters must return shape "
                f"({self.n_components}, {self.n_channels_}), got {W.shape}"
            )
        self.W_ = W
        return self

    def transform(self, X):
        check_is_fitted(self, ("W_", "n_channels_"))
        X = check_array(X, allow_nd=True, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(
                "X must have shape (n_trials, n_channels, n_times); "
                f"got {X.shape}"
            )
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"Expected {self.n_channels_} channels, got {X.shape[1]}"
            )
        # W_[k, j] * X[n, j, t] -> Z[n, k, t]
        return np.einsum("kj,njt->nkt", self.W_, X, optimize=True)

    def _validate_inputs(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        X = check_array(X, allow_nd=True, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(
                "X must have shape (n_trials, n_channels, n_times); "
                f"got {X.shape}"
            )
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y must be 1-D with len {X.shape[0]}, got shape {y.shape}"
            )
        return X, y
