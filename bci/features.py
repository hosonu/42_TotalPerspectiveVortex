"""Feature extraction after spatial filtering."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class LogVarianceTransformer(BaseEstimator, TransformerMixin):
    """Log variance across time for each spatial filter output.

    Typical CSP usage: keep the first ``m`` and last ``m`` filters, then
    concatenate log-variances → ``2m`` features.

    Parameters
    ----------
    epsilon :
        Small constant for numerical stability inside ``log``.
    filters_per_end :
        If set to ``m``, take ``X[:, :m, :]`` and ``X[:, -m:, :]`` along the
        filter axis before computing variances. If ``None``, use all filters.
    """

    def __init__(
        self,
        *,
        epsilon: float = 1e-12,
        filters_per_end: int | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.filters_per_end = filters_per_end

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = check_array(X, allow_nd=True, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(
                f"Expected shape (n_trials, n_filters, n_times); got {X.shape}"
            )
        if self.filters_per_end is not None:
            m = self.filters_per_end
            if 2 * m > X.shape[1]:
                raise ValueError(
                    f"filters_per_end={m} needs at least {2 * m} filters, "
                    f"got {X.shape[1]}"
                )
            X = np.concatenate([X[:, :m, :], X[:, -m:, :]], axis=1)
        var = X.var(axis=-1)
        return np.log(var + self.epsilon)
