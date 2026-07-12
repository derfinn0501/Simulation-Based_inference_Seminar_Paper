# Evaluation Principles

Posterior samples are not automatically valid.
Evaluation must depend on the experimental setting.

## When Ground Truth Is Available

Use:

- posterior mean error
- posterior variance comparison
- KL divergence if tractable
- Wasserstein distance
- coverage
- simulation-based calibration
- posterior plots

## When Ground Truth Is Unavailable

Use:

- posterior predictive checks
- prior predictive sanity checks
- sensitivity to prior
- sensitivity to simulator assumptions
- repeated simulation consistency
- qualitative diagnostics
- domain plausibility checks

## Critical Questions

Always ask:

1. Does the true `theta` fall into credible regions with expected frequency?
2. Does the posterior collapse?
3. Are all posterior modes captured?
4. Does sampling `theta` from the posterior reproduce `x_0` through the simulator?
5. Does the method fail when `x_0` lies in a low-density region of simulated `x`?
6. Does the method depend strongly on the prior?
7. Is uncertainty real posterior uncertainty or model/network error?

## Current Project Metrics

For the current Lotka-Volterra experiments, use:

- raw RMSE
- per-parameter RMSE
- prior-range-normalized RMSE
- prior-std-normalized RMSE
- coverage error
- posterior-mean predictive RMSE
- validation log posterior where available

Posterior plots are useful for communication, but they are not sufficient
evidence of posterior validity.
