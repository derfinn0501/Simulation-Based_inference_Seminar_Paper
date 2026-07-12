# Approach 1.2 Four-Method Suitability Check Results

This diagnostic evaluates whether the Lotka-Volterra setting is suitable for increasingly complex posterior estimators.

## Run Configuration

```text
design_space = hard_window
budgets = 100 250 500 1000
validation = 200
repeats = 3
seed = 616
abc_k = sqrt(N), clipped to [10, 100]
flow_samples_per_pair = 4
posterior_samples = 48
ode_steps = 12
fmpe_max_iter = 350
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

Final evaluated budget: `1000` simulator calls.

## Final-Budget Summary

| Method | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: |
| prior_mean | 0.2867 | -0.3950 | 0.0106 | 1.0560 |
| abc_knn | 0.2149 | -0.3082 | 0.0250 | 0.8705 |
| gaussian_npe | 0.1396 | -0.2353 | 0.0757 | 0.7685 |
| rectified_fmpe | 0.1307 | -0.2443 | 0.1635 | 0.7277 |

## Current Interpretation

For lower-is-better metrics, the ideal error pattern is:

```text
prior_mean > abc_knn > gaussian_npe > rectified_fmpe
```

Observed final-budget ordering by range-normalized RMSE:

```text
rectified_fmpe < gaussian_npe < abc_knn < prior_mean
```

Rectified FMPE is strongest on the final-budget point-estimate metric. Check coverage before treating it as a well-calibrated posterior.
