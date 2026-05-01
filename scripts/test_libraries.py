"""Simple test to demonstrate MNE-Python and scikit-learn functionality."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def test_sklearn():
    """Test basic scikit-learn functionality."""
    print("Testing scikit-learn:")
    print("-" * 50)

    # Create a simple dataset
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train a simple model
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(
        f"✓ Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"✓ Trained LogisticRegression model")
    print(f"✓ Model accuracy on test set: {accuracy:.2%}")
    print()


def test_mne():
    """Test basic MNE-Python functionality."""
    print("Testing MNE-Python:")
    print("-" * 50)

    import mne

    # Create some synthetic raw data
    sfreq = 100  # Sampling frequency
    times = np.arange(0, 10, 1 / sfreq)  # 10 seconds of data
    n_channels = 5

    # Create synthetic data
    data = np.random.randn(n_channels, len(times))

    # Create channel information
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    print(f"✓ Created synthetic EEG data with {n_channels} channels")
    print(f"✓ Sampling frequency: {sfreq} Hz")
    print(f"✓ Duration: {len(times) / sfreq:.1f} seconds")
    print(f"✓ Created MNE Raw object: {raw}")
    print()


if __name__ == "__main__":
    test_sklearn()
    test_mne()
    print("=" * 50)
    print("All tests passed successfully!")
