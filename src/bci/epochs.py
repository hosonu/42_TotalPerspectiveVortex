"""Build MNE :class:`Epochs` from a supported EEG dataset.

Default dataset is PhysioNet EEGBCI (motor-imagery T1/T2).
Pass ``dataset="bcic4_2a"`` to use BCI Competition IV Dataset 2a instead.
"""

from __future__ import annotations

import mne

from bci.eegbci import H_FREQ, IMAGERY_RUNS, L_FREQ, bandpass_filter_raw, fetch_raw_eegbci

# T1 / T2 for standard motor tasks (annotation convention on EEGBCI runs)
EVENT_ID = {"T1": 1, "T2": 2}

TMIN = 0.0
TMAX = 4.0
BASELINE = (0.0, 0.5)


def build_epochs(
    subject: int,
    runs: list[int] | None = None,
    *,
    dataset: str = "eegbci",
    data_path: str | None = None,
    classes: list[str] | None = None,
) -> mne.Epochs:
    """Return bandpass-filtered, epoched EEG data for the requested dataset.

    Parameters
    ----------
    subject :
        Subject ID.  For EEGBCI: 1-109.  For bcic4_2a: 1-9.
    runs :
        Run numbers (EEGBCI only).  Ignored for other datasets.
    dataset :
        ``"eegbci"`` (default) or ``"bcic4_2a"``.
    data_path :
        Local directory that contains the dataset files.
        For ``bcic4_2a``, defaults to ``mne_data/BCICIV_2a_gdf``; ignored for ``eegbci``.
    classes :
        Two class names for ``bcic4_2a``
        (e.g. ``["left_hand", "right_hand"]``).  Ignored for ``eegbci``.
    """
    if dataset == "bcic4_2a":
        from bci.datasets.bcic4_2a import build_epochs as _build_2a
        return _build_2a(subject, data_path=data_path, classes=classes)

    if dataset != "eegbci":
        from bci.datasets import AVAILABLE
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {AVAILABLE}"
        )

    # --- EEGBCI (default) ---
    if runs is None:
        runs = list(IMAGERY_RUNS)
    raw = fetch_raw_eegbci(subject, runs)
    bandpass_filter_raw(raw, L_FREQ, H_FREQ)

    events, event_id = mne.events_from_annotations(raw, event_id=EVENT_ID)
    if len(events) == 0:
        raise ValueError(
            f"No T1/T2 events found in runs {runs} for subject {subject}. "
            "These runs might be baseline recordings (e.g., Run 1 or 2)."
        )

    picks = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin=TMIN,
        tmax=TMAX,
        proj=True,
        picks=picks,
        baseline=BASELINE,
        preload=True,
        verbose="WARNING",
    )


def print_epoch_summary(epochs: mne.Epochs, subject: int, runs: list[int] | None = None) -> None:
    """Print counts and shapes for a quick sanity check (CLI / notebooks)."""
    r = IMAGERY_RUNS if runs is None else runs
    print("=" * 60)
    print(f"Subject {subject:03d} | Runs {r}")
    print(f"Event codes : {epochs.event_id}")
    print(f"Total epochs: {len(epochs)}")
    for name, code in sorted(epochs.event_id.items(), key=lambda x: x[1]):
        n = int((epochs.events[:, 2] == code).sum())
        print(f"  {name} (code {code}): {n}")
    print(f"Time window : {TMIN} – {TMAX} s (baseline {BASELINE})")
    print(f"Bandpass    : {L_FREQ} – {H_FREQ} Hz (on continuous data)")
    print(f"Channels    : {epochs.info['nchan']} EEG")
    print(
        f"Epoch shape : {epochs.get_data().shape}  (epochs × channels × time samples)")
    print("=" * 60)
