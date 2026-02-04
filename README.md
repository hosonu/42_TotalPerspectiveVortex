# 42_Template

## Project Overview


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
