# Coding Standards

## General

Keep code readable, small, and tied to the statistical question.
Prefer simple scripts for early experiments.
Move stable reusable logic into `src/` only after it has been used at least once.

## Python

Use:

- NumPy and pandas for data handling
- matplotlib for plots
- scikit-learn for simple baselines
- PyTorch for neural components when needed

Avoid adding complex frameworks unless they directly support the research
question.

## Experiment Scripts

Every experiment script should make clear:

- research question
- simulator setup
- method
- baseline
- metrics
- output folder
- random seed

Prefer CLI arguments for configurable experiment settings.
Save the run configuration in a machine-readable file when possible.

## Results

Each result bundle should include:

- `RESULTS.md`
- one machine-readable metrics file such as CSV or JSON
- generated plots if useful
- enough configuration to rerun the experiment

State whether the result is canonical, exploratory, or temporary.

## Verification

For code changes:

- run the relevant script or a quick mode when possible
- compile Python files with `python -m py_compile` if runtime is too high
- inspect output files
- report limitations honestly

## Refactoring

Refactor only for readability and reuse.
Do not change statistical behavior unless the task explicitly asks for it.
After refactoring, state what changed and what stayed equivalent.
