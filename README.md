# 42 Total Perspective Vortex

Motor-imagery EEG classification in Python using MNE, scikit-learn, a custom Common Spatial Patterns (CSP) transformer, and optional wavelet-based spectral features.

## Overview

This project implements a Brain-Computer Interface (BCI) classification pipeline for public EEG motor-imagery datasets. It focuses on classical EEG signal processing and machine learning rather than deep learning or foundation models.

The default pipeline decodes two-class EEG trials with:

1. MNE-based EEG loading, montage setup, filtering, and epoch extraction
2. A from-scratch CSP spatial filter implemented as a scikit-learn transformer
3. Log-variance feature extraction from CSP-filtered signals
4. Feature scaling and Linear Discriminant Analysis (LDA)
5. Stratified cross-validation with `cross_val_score`

A bonus path adds Morlet continuous wavelet transform features and a custom logistic regression classifier trained with gradient descent.

## Pipeline

```text
EEG files (EDF/GDF)
    -> MNE preprocessing
    -> band-pass filtering (8-30 Hz)
    -> epoch extraction
    -> Custom CSP spatial filtering
    -> log-variance features or Morlet wavelet features
    -> StandardScaler
    -> LDA or custom logistic regression
    -> cross-validation / simulated streaming inference
```

The real-time mode is a playback simulation: previously epoched EEG trials are streamed over a local socket and classified one epoch at a time. It is not connected to live EEG hardware.

## Features

- Custom CSP implementation using NumPy/SciPy and the generalized eigenvalue problem.
- scikit-learn-compatible transformers built with `BaseEstimator` and `TransformerMixin`.
- End-to-end `Pipeline` integration for reproducible training and evaluation.
- Support for PhysioNet EEG Motor Movement/Imagery (EEGBCI) data.
- Support for BCI Competition IV Dataset 2a (`.gdf`) with selectable two-class tasks.
- MNE-based EEG parsing, filtering, montage setup, annotation handling, and visualization.
- Simulated real-time inference with a producer/client playback flow.
- Bonus pipeline with Morlet CWT wavelet features and custom logistic regression.

## Datasets

### PhysioNet EEGBCI

The default dataset is the [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/) from PhysioNet. MNE downloads the required EDF files automatically on first use.

Typical motor-imagery runs:

- `4 8 12`: imagined left/right fist or both fists/feet tasks depending on the experiment setup
- `3 7 11`, `5 9 13`, `6 10 14`: other real/imagined motor task combinations used by the full evaluation script

### BCI Competition IV Dataset 2a

This project also supports [BCI Competition IV Dataset 2a](https://www.bbci.de/competition/iv/#dataset2a). Place the `.gdf` files under:

```text
mne_data/BCICIV_2a_gdf/
```

The supported class names are:

- `left_hand`
- `right_hand`
- `feet`
- `tongue`

The CLI uses two-class classification for this dataset.

## Requirements

- Python 3.11
- `make`
- `curl` for bootstrapping `uv`

Core Python dependencies:

- `mne`
- `scikit-learn`
- `numpy`
- `scipy`
- `pywavelets`

Development tools:

- `flake8`
- `autopep8`

## Quick Start

Install the project dependencies:

```bash
make install
```

Train the default EEGBCI pipeline:

```bash
make train SUBJECT=1 RUNS="4 8 12"
```

Run playback-style prediction with the saved model:

```bash
make predict SUBJECT=1 RUNS="4 8 12"
```

Visualize raw and filtered EEG signals:

```bash
make visualize SUBJECT=1 RUNS="4 8 12"
```

Run the full EEGBCI evaluation across the predefined experiments:

```bash
make run
```

## Usage

All commands should be run from the repository root.

### Train

```bash
make train
make train SUBJECT=3 RUNS="3 7 11"
```

This builds the pipeline, evaluates it with stratified 5-fold cross-validation, fits on all selected epochs, and saves the model to `saved_bci_model.pkl`.

### Predict

```bash
make predict
make predict SUBJECT=3 RUNS="3 7 11"
```

This loads the saved model and simulates a streamed inference loop over epoched EEG trials.

### BCI Competition IV 2a

```bash
make train DATASET=bcic4_2a SUBJECT=1
make train DATASET=bcic4_2a SUBJECT=1 CLASSES="left_hand right_hand"
make predict DATASET=bcic4_2a SUBJECT=1 CLASSES="left_hand right_hand"
```

The `.gdf` files must already exist under `mne_data/BCICIV_2a_gdf/`, unless a custom data path is passed directly to the Python CLI.

### Bonus Pipeline

```bash
make run BONUS=1
make train BONUS=1 SUBJECT=1 RUNS="4 8 12"
```

The bonus pipeline swaps the default log-variance/LDA setup for wavelet features and a custom logistic regression classifier.

### Development Commands

```bash
make lint
make format
make check
make clean
make fclean
```

See `CONTRIBUTING.md` for development workflow details.

## Project Structure

```text
42_TotalPerspectiveVortex/
├── src/
│   ├── bci/
│   │   ├── classifier.py        # Custom logistic regression classifier
│   │   ├── csp.py               # Custom CSP implementation
│   │   ├── data.py              # Conversion helpers for MNE Epochs
│   │   ├── datasets/
│   │   │   └── bcic4_2a.py      # BCI Competition IV 2a GDF loader
│   │   ├── eegbci.py            # PhysioNet EEGBCI EDF loader
│   │   ├── epochs.py            # Dataset-agnostic epoch builder
│   │   ├── features.py          # Log-variance feature extraction
│   │   ├── pipeline.py          # scikit-learn Pipeline assembly
│   │   └── wavelet.py           # Morlet CWT feature transformer
│   ├── bci_client.py            # Socket client for streamed inference
│   ├── mybci.py                 # Main CLI entry point
│   ├── sensor_server.py         # Mock EEG stream server
│   └── visualize.py             # EEG visualization entry point
├── scripts/
│   ├── explore_data.py
│   ├── extract_epochs.py
│   ├── run_motor_imagery_cv.py
│   └── verify_setup.py
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

## Implementation Notes

The CSP transformer computes class-wise normalized covariance matrices, solves a generalized eigenvalue problem with SciPy, and applies the selected spatial filters to each trial. The default feature extractor then computes log-variance over time, which is a standard pairing for CSP-based motor-imagery classification.

The wavelet transformer uses PyWavelets with a complex Morlet wavelet over the 8-30 Hz motor-imagery frequency range, then averages power over time to produce spectral features.

## Evaluation

Training commands print fold-level scores and the mean `cross_val_score`. The full evaluation mode iterates over predefined PhysioNet EEGBCI experiment/run combinations and subjects, then reports mean accuracies.

Example output shape:

```text
[0.7333 0.6667 0.8000 0.7333 0.7000]
cross_val_score: 0.7267
```

Actual scores depend on the subject, selected runs, dataset, and pipeline mode.

## Limitations

- Real-time inference is simulated from preloaded EEG epochs; this repository does not interface with live EEG hardware.
- The default pipeline is classical ML, not a neural network or foundation model.
- The wavelet and custom logistic regression path is optional bonus functionality, not the default configuration.
- BCI Competition IV 2a files must be obtained separately and placed locally.

## Data and License Notes

This repository does not vendor the EEG datasets. Please follow the usage and citation requirements of PhysioNet EEGBCI and BCI Competition IV Dataset 2a when using their data.

No project license has been selected yet. If you plan to reuse this code, check the repository license status first.

## Acknowledgments

- [MNE-Python](https://mne.tools/) for EEG data handling and preprocessing utilities.
- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/).
- [BCI Competition IV Dataset 2a](https://www.bbci.de/competition/iv/#dataset2a).
