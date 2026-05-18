# Prompt: Implement Approach 1.3 Four-Method Suitability Check

You are working in the repository:

```text
/home/finn/Documents/1-projects/2026-seminar
```

## Goal

Implement `Approach 1.3: Four-Method Suitability Check`.

The purpose is to test whether the current Lotka-Volterra simulation setting is suitable for neural posterior estimators at all. The experiment should compare simple and neural posterior-estimation methods across simulation budgets.

The four methods to compare are:

```text
prior_mean
abc_knn
gaussian_npe
rectified_fmpe
```

The main research question is:

> Do simple methods already solve this simulation setting better than the neural models?

If simple methods outperform the NN-based methods, that should be treated as useful diagnostic evidence, not as a failure. The setting can then be used to calibrate and understand the complex models.

## Existing Code To Reuse

Reuse the existing simulator and estimators from:

```text
experiments/active_fmpe_sbi/run_lotka_volterra.py
experiments/active_fmpe_sbi/run_lotka_volterra_fmpe.py
experiments/active_fmpe_sbi/evaluate_fmpe_quality.py
```

Important existing components:

```text
sample_theta
sample_psi
simulate_batch
simulate_one
THETA_BOUNDS
GaussianPosteriorRegressor
RectifiedFMPE
```

Use the existing metric style from `evaluate_fmpe_quality.py` where possible:

```text
raw_rmse
per-parameter RMSE
range_normalized_rmse
prior_std_normalized_rmse
coverage_error
posterior_mean_predictive_rmse
validation_log_posterior where available
```

## Required New Script

Create a new script:

```text
experiments/active_fmpe_sbi/evaluate_four_method_suitability.py
```

Default output folder:

```text
experiments/results/approach_1_3_four_method_suitability_check/
```

## ABC-kNN Baseline

Implement an ABC nearest-neighbor posterior-sample baseline.

For each query `(x_val, psi_val)`:

1. Fit a feature scaler on training features:

```text
feature_train = concat(x_train, psi_train)
```

2. Transform training and validation features with the same scaler.

3. Find the `K` nearest training examples to each validation example.

4. Use the corresponding `theta_train` values as posterior samples.

The ABC posterior is:

```text
q_abc(theta | x, psi) = empirical distribution of theta values from K nearest neighbors
```

Posterior mean:

```text
mean(theta_neighbors)
```

Coverage:

```text
quantiles of theta_neighbors
```

Use Euclidean distance in standardized `(x, psi)` feature space.

Add a CLI option:

```text
--abc-k
```

If `--abc-k` is not provided, use:

```text
K = int(sqrt(N_train))
```

clipped to:

```text
10 <= K <= 100
```

Make sure `K <= N_train`.

## Budget Sweep

The script should support a budget sweep:

```text
--budgets 100 250 500 1000 2000
```

Use a smaller default if runtime is a concern:

```text
100 250 500 1000
```

For each replicate:

1. Generate one validation set.
2. Generate one random-design training set at `max(budgets)`.
3. For each budget, use the prefix of the training set:

```text
train_budget = train[:budget]
```

This ensures methods are compared on the same data for the same budget.

## Live / Interrupt-Safe Writing

The experiment must write results live so interrupted runs keep completed results.

Unit of completion:

```text
method x budget x replicate
```

After each completed unit:

1. Append one row to:

```text
diagnostics.csv
```

2. Flush the file.

3. On restart, skip rows that already exist with the same:

```text
method
budget
replicate
seed
design_space
```

This makes the experiment restartable.

Do not allow multiple processes to append to the same CSV at the same time. For the first implementation, prefer sequential execution with robust live writing. If adding parallelization, use worker-local files or ensure only the main process writes.

## CLI Options

The script should include at least:

```text
--design-space hard_window
--budgets 100 250 500 1000
--validation 200
--repeats 3
--seed 616
--n-obs 10
--noise-std 0.06
--abc-k
--flow-samples-per-pair 4
--posterior-samples 48
--ode-steps 12
--fmpe-max-iter 350
--output-dir experiments/results/approach_1_3_four_method_suitability_check
--quick
--force
```

`--quick` should use a tiny smoke-test configuration.

`--force` should ignore existing rows and recompute from scratch.

## Output Files

Write:

```text
diagnostics.csv
summary_by_budget.csv
four_method_suitability_summary.png
RESULTS.md
```

`diagnostics.csv` should contain one row per completed method-budget-replicate unit.

`summary_by_budget.csv` should aggregate mean and std by:

```text
method
budget
```

`RESULTS.md` should include:

- run configuration,
- method list,
- final-budget summary table,
- current interpretation,
- note that lower-is-better metrics ideally follow:

```text
prior_mean > abc_knn > gaussian_npe > rectified_fmpe
```

or reveal where simple methods remain stronger.

## Plot

Create a summary plot with at least these panels:

```text
range_normalized_rmse
prior_std_normalized_rmse
coverage_error
posterior_mean_predictive_rmse
```

X-axis:

```text
simulation budget
```

Lines:

```text
prior_mean
abc_knn
gaussian_npe
rectified_fmpe
```

## Documentation Updates

Update:

```text
experiments/active_fmpe_sbi/README.md
experiments/active_fmpe_sbi/IMPLEMENTATION_NOTES.md
experiments/results/README.md
documentation/approach_log/README.md
```

Add the new result folder name:

```text
approach_1_3_four_method_suitability_check/
```

In `documentation/approach_log/README.md`, mark Approach 1.3 as implemented after running at least a quick smoke test.

## Verification

Run:

```bash
.venv/bin/python -m py_compile experiments/active_fmpe_sbi/evaluate_four_method_suitability.py
```

Then run a quick smoke test:

```bash
.venv/bin/python experiments/active_fmpe_sbi/evaluate_four_method_suitability.py --quick
```

Inspect:

```text
diagnostics.csv
summary_by_budget.csv
RESULTS.md
four_method_suitability_summary.png
```

Do not run the full long experiment unless explicitly requested.

## Engineering Constraints

- Keep the implementation close to existing project style.
- Do not rewrite unrelated experiment scripts.
- Do not remove existing result folders.
- Avoid unsafe concurrent writes.
- Preserve restartability.
- Keep output deterministic with seeds.
- Use clear method names exactly:

```text
prior_mean
abc_knn
gaussian_npe
rectified_fmpe
```
