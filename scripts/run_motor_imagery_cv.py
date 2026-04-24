"""Cross-validate CSP + log-variance + LDA (CSP lives in ``bci.csp``).

Usage (from repository root):
    uv run python scripts/run_motor_imagery_cv.py [subject]

Implement :meth:`bci.csp.CustomCSP._compute_spatial_filters` or ``fit`` exits
with code 2 (NotImplementedError).

EEGBCI cache: ``mne_data/`` (see ``eegbci_load``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Repository root and ``scripts/`` for ``bci`` and ``extract_epochs``
_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (_ROOT, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bci.data import epochs_to_Xy  # noqa: E402
from bci.pipeline import make_motor_imagery_pipeline  # noqa: E402
from extract_epochs import build_epochs  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified CV for CSP, log-variance, and LDA.",
    )
    p.add_argument(
        "subject",
        nargs="?",
        type=int,
        default=1,
        help="EEGBCI subject index 1–109 (default: 1)",
    )
    p.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of stratified folds (default: 5)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffled CV splits",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not 1 <= args.subject <= 109:
        print("Error: subject must be 1–109.", file=sys.stderr)
        sys.exit(1)

    print(f">>> Building epochs for subject {args.subject:03d} …")
    epochs = build_epochs(args.subject)
    X, y = epochs_to_Xy(epochs)
    print(
        f"    X shape: {X.shape}, y shape: {y.shape}, classes: {np.unique(y)}",
        flush=True,
    )

    pipe = make_motor_imagery_pipeline()
    print(
        ">>> Checking pipeline fit (CSP must be implemented in bci/csp.py) …",
        flush=True,
    )
    try:
        clone(pipe).fit(X, y)
    except NotImplementedError as exc:
        print("\nCSP is not implemented yet:\n", exc, file=sys.stderr)
        sys.exit(2)

    cv = StratifiedKFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=args.seed,
    )
    print(
        f">>> cross_val_score (StratifiedKFold n={args.cv}, "
        f"seed={args.seed}) …",
    )
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )

    print(f"Fold accuracies: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")


if __name__ == "__main__":
    main()
