# Approach 1.2 Four-Method Suitability Check Results

This diagnostic evaluates whether the Lotka-Volterra setting is suitable for increasingly complex posterior estimators.

## Run Configuration

```text
design_space = wide_window
budgets = 80 180 230 500 1000 2000
validation = 100
repeats = 2
seed = 717
abc_k = sqrt(N), clipped to [10, 100]
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
reward_mode = rmse_coverage_predictive
coverage_weight = 0.25
predictive_weight = 0.1
save_sim_data = True
```

The synthetic training and validation data are saved under:

```text
simulated_data/replicate_*/
```

## Methods

```text
prior_mean
abc_knn
gaussian_npe
rectified_fmpe
```

Final evaluated budget: `2000` simulator calls.

## Final-Budget Summary

| Method | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: |
| prior_mean | 0.2901 | -0.4013 | 0.0200 | 1.0624 |
| abc_knn | 0.2195 | -0.3185 | 0.0192 | 0.9420 |
| gaussian_npe | 0.1312 | -0.2325 | 0.0667 | 0.8460 |
| rectified_fmpe | 0.1328 | -0.2456 | 0.1567 | 0.7364 |

## Current Interpretation

For lower-is-better metrics, the ideal error pattern is:

```text
prior_mean > abc_knn > gaussian_npe > rectified_fmpe
```

Observed final-budget ordering by range-normalized RMSE:

```text
gaussian_npe < rectified_fmpe < abc_knn < prior_mean
```

The final-budget ordering is mixed. Use per-metric and per-parameter diagnostics before deciding whether the setting is suitable.
