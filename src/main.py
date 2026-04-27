import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

from bci.data import epochs_to_Xy
from bci.epochs import build_epochs
from bci.pipeline import make_motor_imagery_pipeline


def main():
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


if __name__ == "__main__":
    main()
