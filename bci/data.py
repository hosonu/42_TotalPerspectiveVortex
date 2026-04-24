"""Helpers to convert MNE objects to NumPy arrays for sklearn."""

from __future__ import annotations

import mne
import numpy as np
from sklearn.preprocessing import LabelEncoder


def epochs_to_Xy(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """Return trial tensor ``X`` and integer labels ``y`` (0 .. n_classes-1).

    Parameters
    ----------
    epochs :
        Must contain at least one event type. Labels use ``events[:, 2]``.

    Returns
    -------
    X :
        Shape ``(n_epochs, n_channels, n_times)``, from ``epochs.get_data()``.
    y :
        Shape ``(n_epochs,)``, integer labels 0 .. K-1 (``LabelEncoder``).
    """
    X = epochs.get_data()
    raw_labels = epochs.events[:, 2].astype(int)
    y = LabelEncoder().fit_transform(raw_labels)
    return X, y
