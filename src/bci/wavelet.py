import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin


class WaveletTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts time–frequency features using the Morlet wavelet
    transform (bonus assignment). Takes CSP-filtered signals of shape
    (n_trials, n_components, n_times).
    """

    def __init__(self, freqs=np.arange(8, 32, 2), fs=160.0):
        # Frequency grid covering Mu (8–12 Hz) and Beta (13–30 Hz), bands important for motor imagery
        self.freqs = freqs
        self.fs = fs  # PhysioNet EEGBCI sampling rate is 160 Hz

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def transform(self, X):
        n_trials, n_components, n_times = X.shape
        features = []

        # Morlet wavelet parameter (wavelet width)
        wavelet = 'cmor1.5-1.0'

        # Map target frequencies (Hz) to PyWavelets scale values
        scales = pywt.central_frequency(wavelet) / (self.freqs / self.fs)

        for i in range(n_trials):
            trial_features = []
            for c in range(n_components):
                sig = X[i, c, :]

                # Continuous wavelet transform via PyWavelets
                cwtmatr, _ = pywt.cwt(sig, scales, wavelet)

                # Power (squared magnitude)
                power = np.abs(cwtmatr) ** 2

                # Average over time
                mean_power_per_freq = np.mean(power, axis=1)

                # Log transform for normalization
                log_power = np.log(mean_power_per_freq + 1e-7)

                trial_features.extend(log_power)
            features.append(trial_features)

        return np.array(features)
