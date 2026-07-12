# Approach 1.4 Reward Landscape Results

This diagnostic tests whether categorical observation-window choices `psi` are visibly coupled to posterior-quality rewards before asking BO to optimize them.

For each replicate, the same initial training set is scored once. Then one batch is added at each categorical `psi`, one fresh estimator is trained per category, and each reward is computed as both an absolute value and a delta from the shared initial baseline.

All rewards in this report are higher-is-better. Delta rewards are the clearest BO signal because BO receives feedback from improvement after choosing the next batch.

## Run Configuration

```text
design_space = wide_window
initial = 80
batch = 14
validation = 100
repeats = 3
initial_design_mode = categorical
batch_theta_mode = paired
target_mode = validation_set
estimator = rectified_fmpe
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
coverage_weight = 0.25
predictive_weight = 0.1
predictive_timeout_seconds = 0.5
seed = 717
category_policies = all
```

## Reward Linkage

| Reward | Paired eta^2 | Rank stability | Best category | Best mean delta | Policy range |
| --- | ---: | ---: | --- | ---: | ---: |
| neg_range_rmse_alpha | 0.492 | 0.233 | long late | -0.0326 | 0.0462 |
| neg_raw_rmse | 0.477 | 0.256 | long late | -0.0084 | 0.0300 |
| rmse_predictive | 0.468 | 0.222 | long late | -0.0145 | 0.0884 |
| neg_range_rmse_gamma | 0.454 | 0.183 | long late | 0.0069 | 0.0518 |
| rmse_coverage_predictive | 0.444 | 0.117 | long late | -0.0306 | 0.0922 |
| neg_range_rmse | 0.388 | 0.222 | long late | -0.0127 | 0.0329 |
| neg_prior_std_rmse | 0.388 | 0.222 | long late | -0.0440 | 0.1138 |
| neg_predictive_rmse | 0.385 | 0.089 | long late | -0.0178 | 0.5549 |
| rmse_coverage | 0.328 | 0.194 | short late | -0.0286 | 0.0370 |
| neg_coverage_error | 0.263 | -0.233 | short late | -0.0244 | 0.0967 |
| neg_range_rmse_delta | 0.226 | -0.139 | long late | 0.0103 | 0.0277 |
| neg_range_rmse_beta | 0.177 | -0.222 | medium late | -0.0255 | 0.0290 |

## Interpretation

At least one reward shows moderate category-level signal, but stability should be checked before trusting BO.

Use `reward_delta_landscape.png` to see the denoised 3x3 category grid, and `reward_snr_landscape.png` to see where the mean delta is large relative to repeat noise.

## Reward Definitions

| Reward | Meaning |
| --- | --- |
| `neg_coverage_error` | -coverage error |
| `neg_predictive_rmse` | -posterior-mean predictive RMSE |
| `neg_prior_std_rmse` | -prior-std-normalized RMSE |
| `neg_range_rmse` | -range-normalized RMSE |
| `neg_range_rmse_alpha` | -range-normalized RMSE alpha |
| `neg_range_rmse_beta` | -range-normalized RMSE beta |
| `neg_range_rmse_delta` | -range-normalized RMSE delta |
| `neg_range_rmse_gamma` | -range-normalized RMSE gamma |
| `neg_raw_rmse` | -raw RMSE |
| `rmse_coverage` | -RMSE - coverage penalty |
| `rmse_coverage_predictive` | -RMSE - coverage - predictive penalties |
| `rmse_predictive` | -RMSE - predictive penalty |

## Output Files

```text
base_metrics.csv
reward_landscape.csv
reward_policy_summary.csv
reward_linkage_summary.csv
reward_delta_landscape.png
reward_snr_landscape.png
reward_linkage_summary.png
RESULTS.md
```
