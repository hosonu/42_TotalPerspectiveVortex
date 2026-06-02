"""BCI Competition IV Dataset 2a — GDF loader (subjects 1-9).

Dataset description
-------------------
4-class motor imagery: left hand, right hand, feet, tongue.
22 EEG channels (10-20 layout) + 3 EOG.  Sampling rate: 250 Hz.
Files: ``A0{subject}T.gdf`` (training) / ``A0{subject}E.gdf`` (evaluation).

Download
--------
https://www.bbci.de/competition/iv/#dataset2a

Usage
-----
Place all .gdf files in ``mne_data/BCICIV_2a_gdf/`` (default).
Override with ``--data-path`` if needed.

    python mybci.py 0 train --subject 1 --dataset bcic4_2a

Classes (``--classes``)
-----------------------
Choose any 2 of: ``left_hand``, ``right_hand``, ``feet``, ``tongue``.
Default: ``left_hand right_hand``.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

# src/bci/datasets/bcic4_2a.py → repo root is four levels up
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_PATH = _PROJECT_ROOT / "mne_data" / "BCICIV_2a_gdf"

TMIN = 0.0
TMAX = 4.0
BASELINE = (0.0, 0.5)
L_FREQ = 8.0
H_FREQ = 30.0

# GDF integer event codes  →  human-readable class name
EVENT_CODES: dict[int, str] = {
    769: "left_hand",
    770: "right_hand",
    771: "feet",
    772: "tongue",
}

DEFAULT_CLASSES = ["left_hand", "right_hand"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_data_path(data_path: str | Path | None) -> Path:
    if data_path is None:
        return DEFAULT_DATA_PATH
    return Path(data_path)


def _gdf_path(data_path: Path, subject: int, phase: str) -> Path:
    fname = data_path / f"A0{subject}{phase}.gdf"
    if not fname.exists():
        raise FileNotFoundError(
            f"GDF file not found: {fname}\n"
            "Download from https://www.bbci.de/competition/iv/#dataset2a\n"
            "and pass its directory with --data-path."
        )
    return fname


def _annotation_event_id(raw: mne.io.Raw, wanted_codes: set[int]) -> dict[str, int]:
    """Map annotation descriptions to integer event codes.

    GDF files from BCI Competition IV store events as integers; MNE converts
    them to annotation descriptions that may be "769" or "769.0" depending on
    the MNE version.  This helper resolves either form.
    """
    descriptions = {a["description"] for a in raw.annotations}
    mapping: dict[str, int] = {}
    for code in wanted_codes:
        for desc in descriptions:
            try:
                if int(float(desc)) == code:
                    mapping[desc] = code
                    break
            except (ValueError, TypeError):
                continue
    return mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_raw(
    subject: int,
    *,
    data_path: str | Path | None = None,
    phase: str = "T",
) -> mne.io.Raw:
    """Read and preprocess continuous Raw (no epoching).

    Applies average reference and bandpass filter in place.
    """
    fname = _gdf_path(_resolve_data_path(data_path), subject, phase)
    raw = mne.io.read_raw_gdf(str(fname), preload=True, verbose="WARNING")

    raw.pick("eeg")

    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    raw.set_eeg_reference("average", projection=True)

    raw.filter(L_FREQ, H_FREQ, fir_design="firwin",
               skip_by_annotation="edge", verbose="WARNING")
    return raw


def build_epochs(
    subject: int,
    *,
    data_path: str | Path | None = None,
    classes: list[str] | None = None,
    phase: str = "T",
) -> mne.Epochs:
    """Load GDF, bandpass-filter, and return Epochs for the requested 2 classes.

    Parameters
    ----------
    subject :
        Subject index 1-9.
    data_path :
        Directory containing the ``.gdf`` files.
        Defaults to ``mne_data/BCICIV_2a_gdf`` under the project root.
    classes :
        Exactly 2 class names from ``["left_hand", "right_hand", "feet",
        "tongue"]``.  Defaults to ``["left_hand", "right_hand"]``.
    phase :
        ``"T"`` for training file, ``"E"`` for evaluation file.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    valid_names = set(EVENT_CODES.values())
    if len(classes) != 2:
        raise ValueError(
            f"Exactly 2 classes required, got {len(classes)}: {classes}"
        )
    for c in classes:
        if c not in valid_names:
            raise ValueError(
                f"Unknown class '{c}'. Valid names: {sorted(valid_names)}"
            )

    name_to_code = {v: k for k, v in EVENT_CODES.items()}
    wanted_codes = {name_to_code[c] for c in classes}

    raw = fetch_raw(subject, data_path=data_path, phase=phase)

    event_id_map = _annotation_event_id(raw, wanted_codes)
    if not event_id_map:
        present = {a["description"] for a in raw.annotations}
        raise ValueError(
            f"No events found for classes {classes} in subject {subject} "
            f"({phase} file).\n"
            f"Annotation descriptions present: {present}\n"
            "Expected codes: "
            + ", ".join(f"{c} ({name_to_code[c]})" for c in classes)
        )

    events, _ = mne.events_from_annotations(
        raw, event_id=event_id_map, verbose="WARNING"
    )

    # Remap integer codes → class names for the Epochs event_id dict
    code_to_name = {v: k for k, v in name_to_code.items()}
    event_id = {
        code_to_name[code]: code
        for code in np.unique(events[:, 2])
        if code in code_to_name
    }

    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin=TMIN,
        tmax=TMAX,
        proj=True,
        baseline=BASELINE,
        preload=True,
        verbose="WARNING",
    )
