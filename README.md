# 42_TotalPerspectiveVortex

## Project Overview

This project uses MNE-Python and scikit-learn for EEG data analysis and machine learning.

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

Use `uv run` to execute Python scripts with the project environment:

```bash
# Verify the setup
uv run python verify_setup.py

# Test library functionality
uv run python test_libraries.py

# Run the main script
uv run python main.py
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
