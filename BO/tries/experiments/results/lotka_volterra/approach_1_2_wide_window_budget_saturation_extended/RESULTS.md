# Approach 1.2 Four-Method Suitability Check Results

This diagnostic evaluates whether the Lotka-Volterra setting is suitable for increasingly complex posterior estimators.

## Run Configuration

```text
design_space = wide_window
budgets = 80 180 230 500 1000 2000 3000 5000
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

Final evaluated budget: `5000` simulator calls.

## Final-Budget Summary

| Method | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: |
| prior_mean | 0.2901 | -0.4013 | 0.0200 | 1.0624 |
| abc_knn | 0.2122 | -0.3075 | 0.0129 | 0.9206 |
| gaussian_npe | 0.1098 | -0.1903 | 0.0250 | 0.7430 |
| rectified_fmpe | 0.0978 | -0.1993 | 0.1625 | 0.6094 |

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
