import sys
import time
import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Import local BCI package (adjust paths if your layout differs).
try:
    from bci.pipeline import make_motor_imagery_pipeline
    from bci.epochs import build_epochs  # or from bci.data
    from bci.data import epochs_to_Xy
except ImportError:
    print("Error: bci module not found. Check PYTHONPATH.")
    sys.exit(1)

MODEL_FILE = "saved_bci_model.pkl"

def do_train(subject, runs):
    """
    Train the pipeline on the given runs, print cross-validation scores, then save the model.
    """
    print(f"Loading data for subject {subject}, runs {runs}...")
    epochs = build_epochs(subject, runs=runs)
    X, y = epochs_to_Xy(epochs)

    # Build pipeline
    pipeline = make_motor_imagery_pipeline(n_csp_components=6)

    # Cross-validation (aligned with the PDF example output)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)
    
    print(f"cross_val_score: {np.mean(scores):.4f}")
    print(f"Fold accuracies: {scores}")

    # Fit on all data and persist the model
    print("\nTraining final model on all provided data...")
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")


def do_predict(subject, runs):
    """
    Load the saved model and simulate a data stream, predicting one epoch at a time.
    """
    model_path = Path(MODEL_FILE)
    if not model_path.exists():
        print(f"Error: Model file '{MODEL_FILE}' not found. Please run 'train' first.")
        sys.exit(1)

    print(f"Loading model from {MODEL_FILE}...")
    pipeline = joblib.load(MODEL_FILE)

    print(f"Loading data for subject {subject}, runs {runs} for playback...")
    epochs = build_epochs(subject, runs=runs)
    X, y = epochs_to_Xy(epochs)

    print("\nStarting playback simulation...\n")
    correct_predictions = 0
    total_epochs = len(X)

    # Simulate a real-time stream (one epoch per step)
    for i in range(total_epochs):
        start_time = time.time()
        
        # One epoch chunk; shape: (1, n_channels, n_times)
        X_chunk = X[i:i+1]
        truth = y[i]

        # Run prediction
        prediction = pipeline.predict(X_chunk)[0]
        
        # Latency check (assignment: within 2 seconds)
        elapsed_time = time.time() - start_time
        
        # Compare and print (PDF-style format)
        is_equal = (prediction == truth)
        if is_equal:
            correct_predictions += 1

        print(f"epoch {i:02d}:")
        print(f"[{prediction}]")
        print(f"[{truth}] {is_equal}")
        
        # Optional: slow down if runs finish too fast for a streaming feel
        # time.sleep(0.5)
        
        if elapsed_time > 2.0:
            print(f"WARNING: Prediction took longer than 2 seconds! ({elapsed_time:.3f}s)")

        time.sleep(0.5)

    accuracy = correct_predictions / total_epochs
    print(f"\nAccuracy: {accuracy:.4f}")

def do_evaluate_all():
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
        print(f"Processing Experiment {exp_id}...")

        for subject in range(1, 110):
            try:
                epochs = build_epochs(subject, runs)
                X, y = epochs_to_Xy(epochs)

                pipeline = make_motor_imagery_pipeline(n_csp_components=6)

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)

                mean_score = float(np.mean(scores))
                results[exp_id].append(mean_score)

            except Exception as e:
                print(f"  Warning - Subject {subject:03d}: Error - {e}")
                continue

        if results[exp_id]:
            exp_mean = float(np.mean(results[exp_id]))
            print(f"experiment {exp_id}:\naccuracy = {exp_mean:.4f}\n")
        else:
            print(f"experiment {exp_id}:\naccuracy = N/A (No valid subjects)\n")

    all_scores = [float(np.mean(res)) for res in results.values() if res]
    if all_scores:
        total_mean = float(np.mean(all_scores))
        print(f"Mean accuracy of 6 experiments: {total_mean:.4f}")
    else:
        print("Mean accuracy of 6 experiments: N/A")


def main():
    if len(sys.argv) == 1:
        do_evaluate_all()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Brain Computer Interface CLI")
    # Args match the PDF example: python mybci.py 4 14 train
    parser.add_argument("runs", metavar="N", type=int, nargs="+", help="Run numbers (e.g., 4 14)")
    parser.add_argument("mode", choices=["train", "predict"], help="Mode to run: 'train' or 'predict'")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (default: 1)")

    args = parser.parse_args()

    # With this parser layout, mode is the last positional so runs are not parsed as mode.
    # e.g. python mybci.py 4 14 train -> runs=[4, 14], mode='train'

    if args.mode == "train":
        do_train(args.subject, args.runs)
    elif args.mode == "predict":
        do_predict(args.subject, args.runs)

if __name__ == "__main__":
    main()