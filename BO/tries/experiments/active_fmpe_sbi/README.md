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

To make the observation-window choice more consequential, use `wide_window`.
This keeps the number of observations fixed but lets BO choose between short
dense windows and longer trend windows:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_bo_design_effect.py \
  --design-space wide_window \
  --output-dir experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check
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

## Four-Method Suitability Diagnostic

To test whether the simulator setting is suitable for the neural methods at all, run:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_four_method_suitability.py \
  --design-space hard_window \
  --budgets 100 250 500 1000 \
  --validation 200 \
  --repeats 3 \
  --seed 616 \
  --output-dir experiments/results/lotka_volterra/approach_1_2_four_method_suitability_check
```

This compares:

- `prior_mean`,
- `abc_knn`,
- `gaussian_npe`,
- `rectified_fmpe`.

It writes results live to `diagnostics.csv`, one row per completed `method x budget x replicate` unit, so interrupted runs can be resumed.

It writes:

```text
diagnostics.csv
summary_by_budget.csv
four_method_suitability_summary.png
RESULTS.md
simulated_data/replicate_*/
```

Each `simulated_data/replicate_*` folder contains `train_full.npz`, `train_full.csv`, `validation.npz`, and `validation.csv`. The neural methods use prefixes of `train_full` for each simulation budget.

## BO Design Effect Diagnostic

To test whether BO improves parameter estimation through adaptive design choices, run:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_bo_design_effect.py
```

This compares:

- `random`: uniformly random designs,
- `fixed_dumb`: one fixed naive design after the shared initial data,
- `bo`: adaptive BO-selected designs,
- `bo_marginal_random`: random designs sampled from BO's empirical marginal design distribution.

The BO-marginal random control asks whether BO found a useful design region or whether the adaptive sequence itself is necessary.
The fixed dumb control asks whether BO at least beats an intentionally poor non-adaptive design.

The default diagnostic uses `wide_window`:

```text
psi = (t_span, t_start)
t_span in [2, 24]
t_end <= 40
n_obs fixed
```

This tests whether BO can exploit the choice between short dense observations and longer observation windows.

The current default estimator is the lightweight rectified FMPE model:

```text
estimator = rectified_fmpe
```

The current default BO reward is sample-based and calibration-aware:

```text
reward_objective = -range_normalized_rmse - 0.25 * coverage_error - 0.1 * posterior_mean_predictive_rmse
```

This replaces the earlier Gaussian-NPE log-posterior reward, because the lightweight FMPE estimator produces posterior samples but does not expose exact posterior densities. The predictive term asks whether posterior mean parameters reproduce validation trajectories through the simulator.

The current default budget allocation uses more reward rounds and a modestly
larger final budget:

```text
initial = 80
batch = 5
rounds = 30
final_budget = 230
```

This gives the BO surrogate more sequential reward observations than the earlier
few-large-batches setup while only adding 50 simulations beyond the previous
20-round run.

To optimize the design loop for one synthetic observed time series `x0`, run:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_bo_design_effect.py \
  --target-mode fixed_x0 \
  --output-dir experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check_x0
```

By default this uses:

```text
target_theta = 1.0 0.05 1.0 0.05
target_psi_unit = 0.5 0.5
```

The script also scores simple structured fixed-window policies at the final budget:

```text
short_early
short_late
long_early
long_late
medium_mid
```

These are written to:

```text
design_choice_scores.csv
design_choice_summary.csv
design_choice_scores.png
```

This is the diagnostic for checking whether BO is truly adding adaptive value or whether a simple non-adaptive observation window already explains the gain. In `fixed_x0` mode, simulating at or near the known target design `psi0` is an especially important baseline.

The default fixed dumb design is:

```text
dumb_psi_unit = 0.0 0.0
t_start = 0
t_span = 2
```

Default output path:

```text
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check/
```

It writes posterior metrics and design-structure traces so the selected `psi` values can be inspected directly.

## Result Folder Convention

Longer-lived results should be stored under:

```text
experiments/results/
```

with approach-named subfolders such as:

```text
approach_1_1_active_design_snapshot/
approach_1_3_bo_design_effect_check/
```
