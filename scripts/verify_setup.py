"""Verification script to confirm MNE-Python and scikit-learn are correctly installed."""

import sys


def verify_installation():
    """Verify that MNE-Python and scikit-learn are correctly installed."""
    print("Python Version:", sys.version)
    print("-" * 50)

    # Verify MNE-Python
    try:
        import mne

        print("✓ MNE-Python is installed")
        print(f"  Version: {mne.__version__}")
    except ImportError as e:
        print("✗ MNE-Python is NOT installed")
        print(f"  Error: {e}")
        return False

    # Verify scikit-learn
    try:
        import sklearn

        print("✓ scikit-learn is installed")
        print(f"  Version: {sklearn.__version__}")
    except ImportError as e:
        print("✗ scikit-learn is NOT installed")
        print(f"  Error: {e}")
        return False

    print("-" * 50)
    print("All required libraries are successfully installed!")
    return True


if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
