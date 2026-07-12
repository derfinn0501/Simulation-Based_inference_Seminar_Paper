# Implementation Notes

This prototype follows the project documentation in `my_paper/meta/research_guide_BO_FMPE.md` and `my_paper/personal_summary/3_contribution/3_possible_contribution_BO.md`.

## Current Scope

This is the initial design, not the final FMPE implementation.

It implements:

1. a simple physical benchmark,
2. a simulator with inferred physical parameters `theta`,
3. BO-controlled design variables `psi`,
4. a simple NPE-style posterior baseline,
5. a posterior-quality reward for BO,
6. random-design vs BO-design comparison.

## Benchmark

The simulator is Lotka-Volterra predator-prey dynamics.

The inferred physical parameters are:

```text
theta = (alpha, beta, gamma, delta)
```

where:

- `alpha`: prey growth,
- `beta`: predation rate,
- `gamma`: predator death,
- `delta`: predator reproduction from prey.

## BO Design Variables

BO does not choose `theta`.

BO chooses:

```text
psi = (prey0, pred0, t_start, t_span)
```

implemented as a normalized 4D vector in `[0, 1]^4`.

This means BO controls the simulation/experiment design:

- initial prey population,
- initial predator population,
- start of the observation window,
- length of the observation window.

For every BO-selected design, physical parameters are still sampled from the prior:

```text
theta ~ p(theta)
```

This preserves the project constraint that BO should improve simulation design, not directly optimize inferred physical parameters.

For the focused BO diagnostic, `wide_window` fixes initial populations and lets
BO choose only:

```text
psi = (t_span, t_start)
```

with:

```text
t_span in [2, 24]
t_end <= 40
n_obs fixed
```

This is meant to expose the short-dense-window versus long-trend-window tradeoff.

## Posterior Model

The current posterior estimator is `GaussianPosteriorRegressor`.

It is a deliberately simple NPE-style baseline:

```text
q(theta | x, psi)
```

It predicts the posterior mean with an MLP and uses a residual Gaussian covariance to evaluate log posterior density.

This gives a scalar posterior-quality metric before implementing FMPE.

## BO Reward

The first BO reward was validation log-posterior improvement:

```text
reward = J[q_new] - J[q_old]
```

where:

```text
J[q] = mean log q(theta_val | x_val, psi_val)
```

on held-out simulator-generated validation pairs.

This reward does not push the model toward one predefined parameter. Each validation example has its own simulator-known `theta`.

The earlier Gaussian-NPE Approach 1.3 diagnostic used a calibration-aware objective:

```text
objective = validation_log_posterior - coverage_weight * coverage_error
reward = objective_new - objective_old
```

with:

```text
coverage_weight = 5.0
```

This tests whether BO can avoid selecting designs that improve point/density fit
while producing overconfident or miscalibrated posteriors.

The first FMPE Approach 1.3 diagnostic used a sample-based objective:

```text
objective = -range_normalized_rmse - coverage_weight * coverage_error
reward = objective_new - objective_old
```

with:

```text
coverage_weight = 0.25
```

This is compatible with the lightweight rectified-FMPE estimator, which produces posterior samples but no exact posterior density.

The current FMPE reward also includes a simulator-facing predictive term:

```text
objective = -range_normalized_rmse
            - coverage_weight * coverage_error
            - predictive_weight * posterior_mean_predictive_rmse
```

with:

```text
coverage_weight = 0.25
predictive_weight = 0.1
```

This asks BO to prefer designs that improve parameter recovery, coverage, and posterior-mean trajectory reproduction.

## Later FMPE Replacement

The next major model step is to replace `GaussianPosteriorRegressor` with an FMPE model that learns a conditional vector field:

```text
v_phi(t, theta, x, psi)
```

The simulator, design variables, BO loop, validation reward, and comparison plots can stay the same.

## Four-Method Suitability Check

The broader diagnostic script is:

```text
evaluate_four_method_suitability.py
```

It compares:

```text
prior_mean
abc_knn
gaussian_npe
rectified_fmpe
```

across simulation budgets.

The new non-neural baseline is `abc_knn`: for each validation observation, it finds the nearest simulated observations in standardized `(x, psi)` feature space and treats their `theta` values as posterior samples.

The script writes rows live to:

```text
diagnostics.csv
```

after each completed `method x budget x replicate` unit. This makes long runs interrupt-safe and restartable.

It also writes the generated synthetic datasets to:

```text
simulated_data/replicate_*/
```

Each replicate stores:

```text
train_full.npz
train_full.csv
validation.npz
validation.csv
metadata.json
```

For every budget, `gaussian_npe` and `rectified_fmpe` are trained on the first `N` rows of `train_full`.

Default output path:

```text
experiments/results/lotka_volterra/approach_1_2_four_method_suitability_check/
```

This diagnostic asks whether the simulator setting is learnable and whether simple methods outperform the neural estimators.

## BO Design Effect Check

The active-design diagnostic script is:

```text
evaluate_bo_design_effect.py
```

It compares three simulation-design strategies with the same validation set, initial data, estimator, and final simulation budget:

```text
random
bo
bo_marginal_random
```

`bo_marginal_random` samples random designs from BO's empirical marginal `psi` distribution. It is a control for whether BO found a useful design region, rather than whether the adaptive sequence itself matters.

Default output path:

```text
experiments/results/lotka_volterra/approach_1_3_bo_design_effect_check/
```

Key output files:

```text
metrics.csv
summary_by_round.csv
design_trace.csv
design_summary_by_round.csv
bo_trace.csv
RESULTS.md
```

The design traces are part of the diagnostic. They should be inspected to see whether BO-selected windows have structure in `t_start`, `t_span`, or both.
