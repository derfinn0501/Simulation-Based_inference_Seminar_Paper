# Active FMPE-SBI Prototype

This folder contains the first implementation prototype for the project idea:

> BO-guided simulation design for SBI, with a simple NPE-style baseline first and FMPE intended as the later posterior estimator.

The current implementation intentionally starts small:

- benchmark: Lotka-Volterra predator-prey simulator,
- inferred physical parameters: growth/interactions rates `theta`,
- BO-controlled design variables: initial populations and observation window `psi`,
- posterior estimator: simple neural Gaussian posterior baseline using `sklearn`,
- BO reward: improvement in validation log posterior, with optional coverage tracking.

The important design constraint from the notes is preserved:

```text
BO optimizes psi, not theta.
theta is still sampled from the prior.
```

## Run

From the repository root:

```bash
.venv/bin/python experiments/active_fmpe_sbi/run_lotka_volterra.py --quick
```

For a somewhat larger run:

```bash
.venv/bin/python experiments/active_fmpe_sbi/run_lotka_volterra.py \
  --initial 160 \
  --batch 24 \
  --rounds 8 \
  --validation 300 \
  --seed 7
```

To test the sharper diagnostic setup where BO only controls the observation
window and the random baseline often sees weakly informative windows:

```bash
.venv/bin/python experiments/active_fmpe_sbi/run_lotka_volterra.py \
  --design-space hard_window \
  --initial 100 \
  --batch 15 \
  --rounds 6 \
  --validation 200 \
  --repeats 5 \
  --seed 21 \
  --output-dir experiments/active_fmpe_sbi/outputs_hard_window
```

Then create the summary plot:

```bash
.venv/bin/python experiments/active_fmpe_sbi/plot_experiment_results.py \
  --input-dir experiments/active_fmpe_sbi/outputs_hard_window
```

Outputs are written to `experiments/active_fmpe_sbi/outputs/`.

## What To Look At

- `metrics.csv`: posterior quality over simulator calls.
- `posterior_quality.png`: random design vs BO-guided design.
- `bo_trace.csv`: BO-selected design variables and rewards.

The main first question is whether BO-guided design improves validation log posterior faster than random design for the same simulation budget.

## FMPE-Style Experiment

The first FMPE-style experiment is implemented in:

```text
run_lotka_volterra_fmpe.py
```

It uses the same simulator and BO design interface, but replaces the Gaussian posterior baseline with a lightweight rectified-flow posterior estimator. It trains a conditional vector field and samples by Euler integration.

Run:

```bash
.venv/bin/python experiments/active_fmpe_sbi/run_lotka_volterra_fmpe.py \
  --design-space hard_window \
  --initial 80 \
  --batch 12 \
  --rounds 5 \
  --validation 120 \
  --repeats 3 \
  --bo-candidates 128 \
  --posterior-samples 64 \
  --ode-steps 16 \
  --seed 404 \
  --output-dir experiments/active_fmpe_sbi/outputs_fmpe_hard_window
```

Main output:

```text
outputs_fmpe_hard_window/posterior_rmse.png
```

The FMPE metric is posterior mean RMSE, not validation log posterior, because this lightweight implementation produces posterior samples but does not yet compute exact FMPE densities.

## Later Full FMPE Slot

The later full FMPE implementation should replace the lightweight `RectifiedFMPE` class with a stronger conditional vector-field posterior estimator while keeping:

- the simulator,
- the design variable interface,
- the BO loop,
- the validation reward.

## Standalone FMPE Quality Diagnostic

Before interpreting BO-vs-random results, evaluate whether FMPE itself is good under random design:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_fmpe_quality.py \
  --design-space hard_window \
  --budgets 60 100 140 \
  --validation 90 \
  --repeats 2 \
  --seed 515 \
  --output-dir experiments/results/approach_1_2_fmpe_quality_check
```

This diagnostic compares:

- prior-mean prediction,
- the Gaussian NPE-style baseline,
- the lightweight rectified-flow FMPE estimator.

It writes:

```text
diagnostics.csv
summary_by_budget.csv
fmpe_quality_summary.png
RESULTS.md
```

The point is to answer whether FMPE is good enough before judging whether BO improves simulation design.

## Result Folder Convention

Longer-lived results should be stored under:

```text
experiments/results/
```

with approach-named subfolders such as:

```text
approach_1_1_active_design_snapshot/
approach_1_2_fmpe_quality_check/
```
