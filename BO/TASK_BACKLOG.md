# Task Backlog

This file is for actionable tasks only.
Methodological decisions go in `DECISION_LOG.md`.
Stable insights go in `project_summary/LEARNINGS.md`.

## Task: Create Repository Documentation Skeleton

### Goal

Add root-level research-agent files and reusable `docs/` templates so future
Codex work follows a consistent research workflow.

### Status

Done on 2026-05-24.

### Next Action

Use this structure for the next non-trivial implementation or experiment.

## Task: Condense Workflow Instructions And Move Project Summary

### Goal

Keep root instructions tight and put stable project-summary material into a
dedicated folder.

### Status

Done on 2026-05-24.

### Next Action

Treat `AGENTS.md` as the agreed instruction contract and use
`project_summary/` for project state.

## Task: Rebuild Approach 1.2 Suitability Test

### Goal

Restore the test that checks whether complex posterior methods work on the
chosen Lotka-Volterra simulation design.

### Status

Done on 2026-05-24.

### Next Action

Use `experiments/results/lotka_volterra/approach_1_2_four_method_suitability_check/` as the
canonical evidence that the simulation design is learnable by NPE/FMPE.

## Task: Run Larger BO Design-Effect Check

### Goal

Check whether the negative BO result in Approach 1.3 is stable across more
repeats or a stronger design setting.

### Status

Done on 2026-05-24 for the current diagnostic level.

### Next Action

The stronger `wide_window` design, coverage-aware Gaussian-NPE reward, FMPE
sample-based reward, posterior-mean predictive FMPE reward, many small BO reward
rounds, fixed-`x0` evaluation, structured fixed-window scores, and a `10`-repeat
budget-500 paired check have now been tested. The repeat check supports the
negative interpretation: current BO does not robustly beat uniform random or
BO-marginal random. The next useful task is to sweep several `x0` values and
check whether simulating near the known target design `psi0` remains strong or
whether a simpler structured policy can replace BO.

## Task: Sweep Target Observations For Structured Window Baselines

### Goal

Check whether simulating near the known target design `psi0` is generally
strong, or only strong for the current synthetic `x0`.

### Status

Planned.

### Next Action

Create a small grid of target `theta`/`psi0` settings, run the target-`x0` BO
diagnostic for each, and compare BO, BO-marginal random, random, and structured
fixed windows.

## Task: Repeat Categorical BO Design Check

### Goal

Check whether the promising two-repeat categorical BO signal survives more
paired seeds.

### Status

Planned.

### Next Action

Repeat `approach_1_3_bo_design_effect_check_budget500_categorical_grid_repeats2`
with more paired repeats before treating categorical BO as a robust improvement.

## Task: Run Wide-Window FMPE Budget Saturation Sweep

### Goal

Find the simulation-budget region where rectified FMPE starts improving and
where its RMSE curve begins to saturate under the same `wide_window` setting
used by the BO design-effect diagnostic.

### Status

Done on 2026-05-24.

### Next Action

Use `experiments/results/lotka_volterra/approach_1_2_wide_window_budget_saturation/` as the
budget-translation baseline for BO, and
`experiments/results/lotka_volterra/approach_1_2_wide_window_budget_saturation_extended/` for
the larger-budget curve. The next BO budget test should either move toward
`500+` final simulations or explicitly remain a low-budget diagnostic.

## Task: Test FMPE Training Iteration Budget

### Goal

Check whether repeated `MLPRegressor` non-convergence warnings mean the current
FMPE estimator is under-trained at larger simulation budgets.

### Status

Done for budget `500`.

### Next Action

The `wide_window` BO diagnostic at budget `500` was repeated with
`fmpe_max_iter = 500` while keeping the architecture fixed at `(128, 128)`.
Final-round metrics were unchanged from `fmpe_max_iter = 220`, so do not spend
more BO wall time on this cap alone.

If this question comes back, instrument `RectifiedFMPE` to record
`MLPRegressor.n_iter_` and compare `500` against `1000` only after that
diagnostic shows fits are still hitting the cap.

## Task: Create Minimal Gaussian SBI Experiment

### Goal

Create a toy SBI setting with an analytic posterior before adding neural NPE or
flow matching.

### Status

Planned.

### Next Action

Create an experiment plan using `docs/EXPERIMENT_TEMPLATE.md`.

## Task: Refactor Reusable Experiment Code Into src

### Goal

Move stable, reusable utilities out of one-off experiment scripts once the toy
case is clear.

### Status

Planned.

### Next Action

Wait until at least one minimal Gaussian SBI experiment exists.
