"""Common Spatial Patterns (CSP) — sklearn-compatible transformer shell.

You implement :meth:`CustomCSP._compute_spatial_filters` with NumPy/SciPy.
:meth:`transform` only applies the learned filters (matrix multiply per trial).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
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
        # Get unique class labels (assuming binary classification)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly two classes.")

        covs = []
        # 1 & 2: Separate trials by class and compute the average spatial covariance matrix
        for c in classes:
            X_c = X[y == c]
            trial_covs = []

            for i in range(X_c.shape[0]):
                E = X_c[i]
                # Covariance matrix of a single trial (E * E^T)
                C = np.dot(E, E.T)
                # Normalize by its trace
                trace = np.trace(C)

                if trace > 1e-12:
                    C = C / trace
                else:
                    C = np.eye(C.shape[0])

                trial_covs.append(C)

            # Append the mean covariance matrix for the current class
            covs.append(np.mean(trial_covs, axis=0))

        C_0 = covs[0]
        C_1 = covs[1]

        # Add a tiny value to the diagonal to make the matrices positive-definite
        epsilon = 1e-6
        n_channels = C_0.shape[0]

        # You can scale epsilon by the trace to make it proportional to the data variance
        C_0_reg = C_0 + epsilon * np.trace(C_0) * np.eye(n_channels)
        C_1_reg = C_1 + epsilon * np.trace(C_1) * np.eye(n_channels)

        # 3. Solve the generalized eigenvalue problem: C_0 * w = lambda * (C_0 + C_1) * w
        # scipy.linalg.eigh takes A and B to solve A*w = lambda*B*w
        evals, evecs = scipy.linalg.eigh(C_0_reg, C_0_reg + C_1_reg)

        # 4. Sort eigenvalues in descending order and rearrange eigenvectors accordingly
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]

        # 5. Select n_components (extract extreme components from both ends)
        half_n = self.n_components // 2

        # Get the first half_n and last half_n eigenvectors
        top_filters = evecs[:, :half_n]
        bottom_filters = evecs[:, -half_n:]

        # Horizontally stack the chosen filter (shape: n_channels, n_components)
        W_cols = np.hstack((top_filters, bottom_filters))

        # Transpose to return a matrix of shape (n_components, n_channels) as expected by fit()
        W = W_cols.T

        return (W)

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
