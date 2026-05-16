import sys
import time
import argparse
import numpy as np
import threading
import queue
import joblib
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

def do_train(subject, runs, use_bonus=False):
    """
    Train the pipeline on the given runs, print cross-validation scores, then save the model.
    """
    epochs = build_epochs(subject, runs=runs)
    X, y = epochs_to_Xy(epochs)

    # Build pipeline
    pipeline = make_motor_imagery_pipeline(
        n_csp_components=6, use_bonus=use_bonus)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)

    scores_str = " ".join([f"{s:.4f}" for s in scores])
    print(f"[{scores_str}]")
    print(f"cross_val_score: {np.mean(scores):.4f}")

    # Fit on all data and persist the model
    pipeline.fit(X, y)

    model_file = get_model_file(use_bonus)

    joblib.dump(pipeline, model_file)

def do_predict(subject, runs, use_bonus=False):
    """
    Load the saved model and simulate a data stream, predicting one epoch at a time.
    """
    model_file = get_model_file(use_bonus)
    model_path = Path(model_file)
    if not model_path.exists():
        print(
            f"Error: Model file '{model_file}' not found. Please run 'train' first.")
        sys.exit(1)

    pipeline = joblib.load(model_file)

    epochs = build_epochs(subject, runs=runs)
    X,y = epochs_to_Xy(epochs)

    data_queue = queue.Queue()
    total_epochs = len(X)

    def eeg_producer():
        for i in range(total_epochs):
            data_queue.put({
                'epoch_nb': i,
                'X_chunk': X[i:i+1],
                'y_truth': y[i]
            })
            time.sleep(0.5)
        
        data_queue.put(None)

    producer_thread = threading.Thread(target=eeg_producer)
    producer_thread.daemon = True
    producer_thread.start()

    correct_predictions = 0
    print("epoch nb: [prediction] [truth] equal?")
    try:
        while True:
            item = data_queue.get()
            if item is None:
                break

            epoch_nb = item['epoch_nb']
            X_chunk = item['X_chunk']
            y_truth = item['y_truth']

            # Run prediction
            prediction = pipeline.predict(X_chunk)[0]

            # Compare and print (PDF-style format)
            is_equal = (prediction == y_truth)
            if is_equal:
                correct_predictions += 1

            pred_out = prediction + 1
            truth_out = y_truth + 1
            print(f"epoch {epoch_nb:02d}: [{pred_out}] [{truth_out}] {is_equal}")
    
    except KeyboardInterrupt:
        print("\n[Predict] Stopped by user.")
    finally:
        if total_epochs > 0:
            accuracy = correct_predictions / total_epochs
            print(f"Accuracy: {accuracy:.4f}")
        
    producer_thread.join()


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

    try:
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
    except KeyboardInterrupt:
        print("\n[Evaluation] Interrupted by user. Calculating partial results...")

    print("Mean accuracy of the six different experiments for all 109 subjects:")
    all_scores = []
    for exp_id in experiments:
        if results[exp_id]:
            exp_mean = float(np.mean(results[exp_id]))
            evaluated_subjects_count = len(results[exp_id])
            print(f"experiment {exp_id}: accuracy = {exp_mean:.4f} (based on {evaluated_subjects_count} subjects)")
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
        do_train(args.subject, args.runs, use_bonus=args.bonus)
    elif args.mode == "predict":
        do_predict(args.subject, args.runs, use_bonus=args.bonus)


if __name__ == "__main__":
    main()
