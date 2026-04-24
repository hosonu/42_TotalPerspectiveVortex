"""Cut motor-imagery epochs (left vs right fist) from PhysioNet EEGBCI runs 4, 8, 12.

Usage:
    uv run python scripts/extract_epochs.py [subject_number] [--plot] [--save PATH]

    subject_number: integer 1-109 (default: 1)

For runs 4 / 8 / 12, annotation codes T1 = imagined left fist, T2 = imagined right fist.
Continuous data are bandpass-filtered (8–30 Hz) before epoching.

EEGBCI files are stored under ``<project_root>/mne_data/`` (not committed).
"""

from __future__ import annotations

import argparse
import os
import sys

import mne

from eegbci_load import (
    H_FREQ,
    IMAGERY_RUNS,
    L_FREQ,
    bandpass_filter_raw,
    fetch_raw_eegbci,
)

# T1 / T2 for runs [4, 8, 12]: left vs right fist motor imagery
EVENT_ID = {"T1": 1, "T2": 2}

TMIN = 0.0
TMAX = 4.0
BASELINE = (0.0, 0.5)

# MNE writes small files next to raw; keep under mne_data when using --save
DEFAULT_SAVE_TEMPLATE = "mne_data/epochs_s{subject:03d}-epo.fif"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build motor-imagery Epochs from EEGBCI.")
    p.add_argument(
        "subject",
        nargs="?",
        type=int,
        default=1,
        help="Subject index 1–109 (default: 1)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Open interactive epoch and evoked plots (blocks until closed).",
    )
    p.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Write epochs to this .fif path (default with --commit-save: under mne_data/).",
    )
    p.add_argument(
        "--commit-save",
        action="store_true",
        help="Save to mne_data/epochs_sNNN-epo.fif (NNN = subject number).",
    )
    return p.parse_args(argv)


def validate_subject(subject: int) -> None:
    if not 1 <= subject <= 109:
        print(f"Error: subject must be between 1 and 109, got {subject}.", file=sys.stderr)
        sys.exit(1)


def build_epochs(subject: int) -> mne.Epochs:
    raw = fetch_raw_eegbci(subject, IMAGERY_RUNS)
    bandpass_filter_raw(raw, L_FREQ, H_FREQ)

    events, event_id = mne.events_from_annotations(raw, event_id=EVENT_ID)
    if len(events) == 0:
        print(
            "Warning: no T1/T2 events found after filtering annotations. "
            "Check run list and raw.annotations.",
            file=sys.stderr,
        )

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    epochs = mne.Epochs(
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
    return epochs


def print_epoch_summary(epochs: mne.Epochs, subject: int) -> None:
    print("=" * 60)
    print(f"Subject {subject:03d} | Runs {IMAGERY_RUNS}")
    print(f"Event codes : {epochs.event_id}")
    print(f"Total epochs: {len(epochs)}")
    for name, code in sorted(epochs.event_id.items(), key=lambda x: x[1]):
        n = int((epochs.events[:, 2] == code).sum())
        print(f"  {name} (code {code}): {n}")
    print(f"Time window : {TMIN} – {TMAX} s (baseline {BASELINE})")
    print(f"Bandpass    : {L_FREQ} – {H_FREQ} Hz (on continuous data)")
    print(f"Channels    : {epochs.info['nchan']} EEG")
    print(f"Epoch shape : {epochs.get_data().shape}  (epochs × channels × time samples)")
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    validate_subject(args.subject)

    save_path = args.save
    if args.commit_save:
        save_path = DEFAULT_SAVE_TEMPLATE.format(subject=args.subject)

    print(f"\n>>> Loading & filtering subject {args.subject}, runs {IMAGERY_RUNS} …")
    epochs = build_epochs(args.subject)
    print_epoch_summary(epochs, args.subject)

    if save_path:
        out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", save_path))
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        epochs.save(out, overwrite=True)
        print(f"\n>>> Saved epochs to {out}")

    if args.plot:
        print("\n>>> Plotting epochs (close windows to continue) …")
        epochs.plot(block=True, scalings="auto")
        evoked_l = epochs["T1"].average()
        evoked_r = epochs["T2"].average()
        mne.viz.plot_compare_evokeds(
            {"T1 (left MI)": evoked_l, "T2 (right MI)": evoked_r},
            picks="eeg",
            invert_y=True,
        )


if __name__ == "__main__":
    main()
