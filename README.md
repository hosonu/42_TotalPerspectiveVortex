# 42_TotalPerspectiveVortex

## Project Overview

The Total Perspective Vortex project aims to develop a Brain-Computer interface (BCI) that utilizes machine learning algorithms to interpret electroencephalographic (EGG) data.

**Key Objectives**
- **Signal Interface**: The system must infer whether a subject is performing or imaging specific movements based on EEG readings within a defined timeframe.
- **Dimensionality Reduction**: A core requirement is the custom implementation of a dimensionality reduction algorithm (such as Common Spatial Patterns(CSP) or Principal Component Analysis (PCA)) to extract the most meaningful features from cerebral signals.
- **Pipeline Integration**: The data processing workflow must be integrated into a scikit-learn `Pipeline` object, utilizing `BaseRstimator` and `TransformerMixin` classes.
- **Real-time Classification**: The system should be capable of classifying data streams in "real time", providing predicitons within a 2-second delay after receiving a data chunk.

**Technical Stack**
- **Language**: Python (3.11.*)
- **Libraries**: **MNE** for EEG data parsing, visualization, and filtering, and **scikit-learn** for machine learning and classfication tasks.

**Perfomance Standards**
- **Evaluation**: The entire proccesing pipeline must be evaulated using `cross_val_socre`.
- **Target Accuracy**: Achieve a minmum mean accuracy of 60% across all test subjects and experiment runs using never-learned data.

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management with Python 3.11.

### Prerequisites

- Python 3.11 or later
- uv (will be installed automatically if not present)

### Installation

1. Install uv (if not already installed):
```bash
pip install uv
```

2. Create the Python environment and install dependencies:
```bash
uv sync
```

This will automatically:
- Create a Python 3.11 virtual environment
- Install MNE-Python and scikit-learn along with their dependencies

### Running Scripts

Use `uv run` to execute Python scripts with the project environment. The `bci` package lives under `src/bci` and is installed in editable mode with `uv sync`.

```bash
# Verify the setup
uv run python scripts/verify_setup.py

# Test library functionality
uv run python scripts/test_libraries.py

# Main evaluation (project root: metrics across subjects / experiments)
uv run python src/main.py
```

### Installed Libraries

- **MNE-Python** (v1.11.0): A package for exploring, visualizing, and analyzing human neurophysiological data
- **scikit-learn** (v1.8.0): Machine learning library for Python

## Development

### Naming Conventions

This project follows standard Python naming conventions (PEP 8):

- **Classes**: `PascalCase` (e.g., `LogisticRegression`)
- **Functions & Variables**: `snake_case` (e.g., `train_model`, `learning_rate`)
- **Constants**: `UPPER_CASE` (e.g., `MAX_ITERATIONS`)
- **Files & Directories**: `snake_case` (e.g., `data_processing.py`)

**Consistency is key**: Once a rule is decided, it must be followed strictly.
- **Verb-Noun Pattern**: Functions should follow the `verb_noun` pattern (e.g., `get_data`, `calc_mean`).

### Execution Directory & Paths

- **Execution Root**: All scripts must be executed from the **project root directory**.
- **Relative Paths**: Use relative paths starting from the project root.
    - **Command Line**: `python main.py` (Run from root)
    - **Data Access**: `pd.read_csv("data/raw/dataset_train.csv")`
    - **Saving Files**: `plt.savefig("plots/scatter_plot.png")`
    - **Bad Practice**: Avoid using `../` to go up directories or absolute paths.

### Git Workflow

#### Commit Messages (Conventional Commits)
Format: `type(scope): subject`

- **Types**:
  - `feat`: New feature
  - `fix`: Bug fix
  - `docs`: Documentation only
  - `style`: Formatting, missing semi-colons, etc.
  - `refactor`: Code change that neither fixes a bug nor adds a feature
  - `test`: Adding or correcting tests
  - `chore`: Build process or auxiliary tool changes

**Example**: `feat(model): implement gradient descent`

#### Branch Naming
Format: `type/description`
(ex)
- `feat/logistic-regression`
- `fix/parsing-error`
- `docs/update-readme`

### Code Quality Tools

This project uses the following tools to maintain code quality:
