# 42_TotalPerspectiveVortex

## Project Overview

The Total Perspective Vortex project aims to develop a Brain-Computer Interface (BCI) that utilizes machine learning algorithms to interpret electroencephalographic (EEG) data.

**Key Objectives**
- **Signal Interface**: The system must infer whether a subject is performing or imagining specific movements based on EEG readings within a defined timeframe.
- **Dimensionality Reduction**: A core requirement is the custom implementation of a dimensionality reduction algorithm (Common Spatial Patterns — CSP) to extract the most meaningful features from cerebral signals.
- **Pipeline Integration**: The data processing workflow must be integrated into a scikit-learn `Pipeline` object, utilizing `BaseEstimator` and `TransformerMixin` classes.
- **Real-time Classification**: The system should be capable of classifying data streams in "real time", providing predictions within a 2-second delay after receiving a data chunk.

**Technical Stack**
- **Language**: Python (3.11.*)
- **Libraries**: **MNE** for EEG data parsing, visualization, and filtering, and **scikit-learn** for machine learning and classification tasks.

**Performance Standards**
- **Evaluation**: The entire processing pipeline must be evaluated using `cross_val_score`.
- **Target Accuracy**: Achieve a minimum mean accuracy of 60% across all test subjects and experiment runs using never-learned data.

---

## Project Structure

```
42_TotalPerspectiveVortex/
├── .github/
├── src/
│   ├── bci/
│   │   ├── __init__.py
│   │   ├── classifier.py   # LDA / SVM classifier wrapper
│   │   ├── csp.py          # Custom CSP implementation
│   │   ├── data.py         # Data loading helpers
│   │   ├── eegbci.py       # MNE EEGBCI dataset interface
│   │   ├── epochs.py       # Epoch extraction
│   │   ├── features.py     # Feature extraction
│   │   ├── pipeline.py     # sklearn Pipeline assembly
│   │   └── wavelet.py      # Wavelet-based feature transformer
│   ├── mybci.py            # CLI entry point
│   ├── bci_client.py       # Real-time BCI client
│   ├── sensor_server.py    # Simulated sensor server
│   └── visualize.py        # EEG visualization
├── .gitignore
├── .python-version
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management with Python 3.11. All setup is handled through the `Makefile`.

### Prerequisites

- `make`
- `curl` (to bootstrap uv locally)

### Installation

```bash
make install
```

This will automatically:
- Download `uv` locally into `./bin/` (no system-wide install required)
- Create a Python 3.11 virtual environment under `./.venv/`
- Install all dependencies from `uv.lock`

> **MNE data**: By default the Makefile looks for a dataset on external storage at `/run/media/$USER/F0E9-334E/mne_data` and creates a symlink at `./mne_data`. If that path does not exist, MNE will download the EEG data automatically on first run.

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| `mne` | ≥ 1.11.0 | EEG data parsing, filtering, visualization |
| `scikit-learn` | ≥ 1.8.0 | Machine learning pipeline & classification |
| `pywavelets` | ≥ 1.9.0 | Wavelet feature extraction |
| `flake8` | ≥ 7.3.0 | Linting |
| `autopep8` | ≥ 2.3.2 | Code formatting |

---

## Usage

All commands are run from the **project root**.

### Full Evaluation (all subjects)

```bash
make run
```

Runs `src/mybci.py` across all subjects and prints cross-validated accuracy metrics.

### Train a Model

```bash
make train                          # default: SUBJECT=1, RUNS="4 8 12"
make train SUBJECT=3 RUNS="3 7 11"
```

Trains the pipeline and saves the model to `saved_bci_model.pkl`.

### Predict (Playback Simulation)

```bash
make predict                        # default: SUBJECT=1, RUNS="4 8 12"
make predict SUBJECT=3 RUNS="3 7 11"
```

Loads a saved model and runs a real-time playback simulation via `sensor_server` / `bci_client`.

### Visualize EEG Data

```bash
make visualize                      # default: SUBJECT=1, RUNS="4 8 12"
make visualize SUBJECT=2 RUNS="6 10 14"
```

Displays raw and bandpass-filtered EEG signals using MNE.

### Bonus Pipeline

```bash
make bonus
# or
make run BONUS=1
```

---

## Development

### Naming Conventions

This project follows standard Python naming conventions (PEP 8):

| Target | Convention | Example |
|---|---|---|
| Classes | `PascalCase` | `CommonSpatialPattern` |
| Functions & variables | `snake_case` | `get_epochs`, `fit_pipeline` |
| Constants | `UPPER_CASE` | `MAX_ITERATIONS` |
| Files & directories | `snake_case` | `data_loader.py` |

- **Verb-Noun pattern**: Functions should follow `verb_noun` (e.g., `get_data`, `fit_model`).
- **Execution root**: All scripts must be executed from the project root directory.

### Code Quality Tools

```bash
make lint           # Run flake8 on src/
make format         # Run autopep8 formatter on src/ (alias: make fmt)
make check          # lint + environment verification
```

### Cleanup

```bash
make clean          # Remove __pycache__ and saved model
make fclean         # Full cleanup: also removes .venv, ./bin, uv caches, mne_data symlink
make re             # fclean + install
```

### Git Workflow

#### Commit Messages (Conventional Commits)

Format: `type(scope): subject`

| Type | Use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, whitespace |
| `refactor` | No feature change, no bug fix |
| `test` | Adding or correcting tests |
| `chore` | Build process or tooling changes |

**Example**: `feat(csp): implement custom CSP transformer`

#### Branch Naming

Format: `type/description`

```
feat/wavelet-features
fix/epoch-extraction
docs/update-readme
```
