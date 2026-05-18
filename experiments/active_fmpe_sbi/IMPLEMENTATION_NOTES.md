# Implementation Notes

This prototype follows the project documentation in `my_paper/meta/research_guide_BO_FMPE.md` and `my_paper/personal_summary/3_possible_contribution_BO.md`.

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

## Posterior Model

The current posterior estimator is `GaussianPosteriorRegressor`.

It is a deliberately simple NPE-style baseline:

```text
q(theta | x, psi)
```

It predicts the posterior mean with an MLP and uses a residual Gaussian covariance to evaluate log posterior density.

This gives a scalar posterior-quality metric before implementing FMPE.

## BO Reward

The main BO reward is validation log-posterior improvement:

```text
reward = J[q_new] - J[q_old]
```

where:

```text
J[q] = mean log q(theta_val | x_val, psi_val)
```

on held-out simulator-generated validation pairs.

This reward does not push the model toward one predefined parameter. Each validation example has its own simulator-known `theta`.

## Later FMPE Replacement

The next major model step is to replace `GaussianPosteriorRegressor` with an FMPE model that learns a conditional vector field:

```text
v_phi(t, theta, x, psi)
```

The simulator, design variables, BO loop, validation reward, and comparison plots can stay the same.

## Standalone FMPE Quality Check

Before evaluating whether BO helps, the current FMPE-style estimator should be evaluated under random design only.

The diagnostic script is:

```text
evaluate_fmpe_quality.py
```

It compares:

```text
prior_mean
gaussian_npe
rectified_fmpe
```

and reports:

- raw RMSE,
- per-parameter RMSE,
- prior-range normalized RMSE,
- prior-std normalized RMSE,
- coverage error,
- posterior-mean predictive RMSE.

Default output path:

```text
experiments/results/approach_4_2_fmpe_quality_check/
```

This separates the posterior-estimator quality question from the active-design question.
