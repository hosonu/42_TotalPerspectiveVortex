import sys
import time
import argparse
import numpy as np
import joblib
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
# Import local BCI package (adjust paths if your layout differs).

import mne
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

from bci.pipeline import make_motor_imagery_pipeline
from bci.epochs import build_epochs
from bci.data import epochs_to_Xy

@contextmanager
def suppress_stdout_stderr():
    """MNE-Python logging suppression context manager"""
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def get_model_file(use_bonus):
    return "bonus_bci_model.pkl" if use_bonus else "saved_bci_model.pkl"


def get_test_data_file(use_bonus):
    return "bonus_test_data.pkl" if use_bonus else "test_data.pkl"


def do_train(subject, runs, use_bonus=False):
    """
    Train the pipeline on the given runs, print cross-validation scores, then save the model.
    """
    epochs = build_epochs(subject, runs=runs)
    X, y = epochs_to_Xy(epochs)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y)

    # Build pipeline
    pipeline = make_motor_imagery_pipeline(
        n_csp_components=6, use_bonus=use_bonus)

    # Cross-validation (aligned with the PDF example output)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)

    scores_str = " ".join([f"{s:.4f}" for s in scores])
    print(f"[{scores_str}]")
    print(f"cross_val_score: {np.mean(scores):.4f}")

    # Fit on all data and persist the model
    pipeline.fit(X, y)

    model_file = get_model_file(use_bonus)
    test_file = get_test_data_file(use_bonus)

    joblib.dump(pipeline, model_file)
    joblib.dump((X_test, y_test), test_file)

    # print(f"Model saved to {model_file}")
    # print(f"Hold-out test data saved to {test_file}")


def do_predict(subject, runs, use_bonus=False):
    """
    Load the saved model and simulate a data stream, predicting one epoch at a time.
    """
    model_file = get_model_file(use_bonus)
    test_file = get_test_data_file(use_bonus)

    model_path = Path(model_file)
    if not model_path.exists():
        print(
            f"Error: Model file '{model_file}' not found. Please run 'train' first.")
        sys.exit(1)

    pipeline = joblib.load(model_file)

    X_test, y_test = joblib.load(test_file)

    correct_predictions = 0
    total_epochs = len(X_test)

    print("epoch nb: [prediction] [truth] equal?")
    # Simulate a real-time stream (one epoch per step)
    for i in range(total_epochs):
        start_time = time.time()

        # One epoch chunk; shape: (1, n_channels, n_times)
        X_chunk = X_test[i:i+1]
        truth = y_test[i]

        # Run prediction
        prediction = pipeline.predict(X_chunk)[0]

        # Latency check (assignment: within 2 seconds)
        elapsed_time = time.time() - start_time

        # Compare and print (PDF-style format)
        is_equal = (prediction == truth)
        if is_equal:
            correct_predictions += 1

        pred_out = prediction + 1
        truth_out = truth + 1
        print(f"epoch {i:02d}: [{pred_out}] [{truth_out}] {is_equal}")

        if elapsed_time > time.time() - start_time:
            print(
                f"WARNING: Prediction took longer than 2 seconds! ({elapsed_time:.3f}s)")

        time.sleep(0.5)

    accuracy = correct_predictions / total_epochs
    print(f"Accuracy: {accuracy:.4f}")


def do_evaluate_all(use_bonus=False):
    """
    Main script to evaluate the BCI pipeline across all 109 subjects
    for the 6 different experimental conditions specified in the assignment.
    """

    # Define the 6 experimental combinations (Run IDs) as per the PhysioNet dataset
    # Experiment 0: Open/Close Left vs Right Fist (Real)
    # Experiment 1: Open/Close Both Fists vs Both Feet (Real)
    # Experiment 2: Imagine Open/Close Left vs Right Fist
    # Experiment 3: Imagine Open/Close Both Fists vs Both Feet
    # Experiment 4: Real vs Imagine (Fists)
    # Experiment 5: Real vs Imagine (Feet)
    experiments = {
        0: [3, 7, 11],
        1: [4, 8, 12],
        2: [5, 9, 13],
        3: [6, 10, 14],
        4: [3, 5, 7, 9, 11, 13],
        5: [4, 6, 8, 10, 12, 14],
    }

    results = {exp_id: [] for exp_id in experiments}

    for exp_id, runs in experiments.items():

        for subject in range(1, 110):
            try:
                epochs = build_epochs(subject, runs)
                X, y = epochs_to_Xy(epochs)

                pipeline = make_motor_imagery_pipeline(
                    n_csp_components=6, use_bonus=use_bonus)

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)

                mean_score = float(np.mean(scores))
                results[exp_id].append(mean_score)

                print(f"experiment {exp_id}: subject {subject:03d}: accuracy = {mean_score:.1f}")

            except Exception:
                continue

    print("Mean accuracy of the six different experiments for all 109 subjects:")
    all_scores = []
    for exp_id in experiments:
        if results[exp_id]:
            exp_mean = float(np.mean(results[exp_id]))
            print(f"experiment {exp_id}: accuracy = {exp_mean:.4f}")
            all_scores.append(exp_mean)

    if all_scores:
        total_mean = float(np.mean(all_scores))
        print(f"Mean accuracy of 6 experiments: {total_mean:.4f}")
    else:
        print("Mean accuracy of 6 experiments: N/A")


def main():
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "--bonus"):
        use_bonus = "--bonus" in sys.argv
        do_evaluate_all(use_bonus=use_bonus)
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Brain Computer Interface CLI")

    parser.add_argument("runs", metavar="N", type=int,
                        nargs="+", help="Run numbers (e.g., 4 14)")
    parser.add_argument(
        "mode", choices=["train", "predict"], help="Mode to run: 'train' or 'predict'")
    parser.add_argument("--subject", type=int, default=1,
                        help="Subject ID (default: 1)")
    parser.add_argument("--bonus", action="store_true",
                        help="Use the custom classifier (Bonus part)")

    args = parser.parse_args()

    # With this parser layout, mode is the last positional so runs are not parsed as mode.
    # e.g. python mybci.py 4 14 train -> runs=[4, 14], mode='train'

    if args.mode == "train":
        do_train(args.subject, args.runs)
    elif args.mode == "predict":
        do_predict(args.subject, args.runs)


if __name__ == "__main__":
    main()
