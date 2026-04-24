"""Shared PhysioNet EEGBCI loading helpers for project scripts."""

import os

import mne
from mne.datasets import eegbci

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MNE_DATA_DIR = os.path.join(_PROJECT_ROOT, "mne_data")

IMAGERY_RUNS = [4, 8, 12]
L_FREQ = 8.0
H_FREQ = 30.0


def configure_eegbci_data_path() -> None:
    os.makedirs(MNE_DATA_DIR, exist_ok=True)
    mne.set_config("MNE_DATASETS_EEGBCI_PATH", MNE_DATA_DIR, set_env=True)


def fetch_raw_eegbci(subject: int, runs: list[int] | None = None) -> mne.io.Raw:
    """Fetch EDF files from PhysioNet and return a concatenated Raw object."""
    configure_eegbci_data_path()
    if runs is None:
        runs = IMAGERY_RUNS
    fnames = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in fnames]
    raw = mne.concatenate_raws(raws)

    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.set_eeg_reference("average", projection=True)
    return raw


def bandpass_filter_raw(
    raw: mne.io.Raw,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
) -> mne.io.Raw:
    """Apply bandpass on continuous data (in-place). Prefer before epoching."""
    raw.filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge")
    return raw
