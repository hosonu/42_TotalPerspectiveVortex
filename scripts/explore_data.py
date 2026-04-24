"""Task 1: Fetch, filter, and visualize PhysioNet motor imagery EEG data.

Usage:
    uv run python scripts/explore_data.py [subject_number]

    subject_number: integer 1-109 (default: 1)

Fetches runs [4, 8, 12] (motor imagery: left fist vs right fist),
applies an 8-30 Hz bandpass filter, and shows before/after visualizations.

EEGBCI files are downloaded to ``<project_root>/mne_data/`` (not committed).
"""

import sys

import mne

from eegbci_load import H_FREQ, IMAGERY_RUNS, L_FREQ, fetch_raw_eegbci


def parse_subject(argv: list[str]) -> int:
    if len(argv) < 2:
        return 1
    try:
        subject = int(argv[1])
    except ValueError:
        print(f"Error: '{argv[1]}' is not a valid subject number.")
        sys.exit(1)
    if not 1 <= subject <= 109:
        print(f"Error: subject must be between 1 and 109, got {subject}.")
        sys.exit(1)
    return subject


def show_info(raw: mne.io.Raw, subject: int) -> None:
    """Print a summary of the loaded data."""
    print("=" * 60)
    print(f"Subject {subject:03d} | Runs {IMAGERY_RUNS}")
    print(f"Channels : {raw.info['nchan']}")
    print(f"Sfreq    : {raw.info['sfreq']} Hz")
    print(f"Duration : {raw.times[-1]:.1f} s")
    print("=" * 60)


def visualize(raw: mne.io.Raw, title_suffix: str) -> None:
    """Show time-series and PSD plots for the given Raw object."""
    raw.plot(
        title=f"Raw EEG — {title_suffix}",
        n_channels=10,
        scalings="auto",
        show=True,
        block=False,
    )
    raw.compute_psd(fmax=60.0).plot(show=True)


def main() -> None:
    subject = parse_subject(sys.argv)

    print(f"\n>>> Fetching data for subject {subject}, runs {IMAGERY_RUNS} …")
    raw = fetch_raw_eegbci(subject, IMAGERY_RUNS)
    show_info(raw, subject)

    events, event_id = mne.events_from_annotations(raw)
    print(f"Events found : {len(events)}")
    print(f"Event mapping : {event_id}")

    print("\n>>> Showing PRE-filter plots …")
    visualize(raw, "Before filtering")

    print(f"\n>>> Applying bandpass filter ({L_FREQ}–{H_FREQ} Hz) …")
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")

    print(">>> Showing POST-filter plots …")
    visualize(raw, f"After {L_FREQ}–{H_FREQ} Hz bandpass")

    raw.plot_sensors(show_names=True, show=True)

    print("\n>>> Close all plot windows to exit.")
    import matplotlib.pyplot as plt
    plt.show(block=True)


if __name__ == "__main__":
    main()
