# Approach 1.4 Reward Landscape Results

This diagnostic tests whether categorical observation-window choices `psi` are visibly coupled to posterior-quality rewards before asking BO to optimize them.

For each replicate, the same initial training set is scored once. Then one batch is added at each categorical `psi`, one fresh estimator is trained per category, and each reward is computed as both an absolute value and a delta from the shared initial baseline.

When multiple batch sizes are configured, larger batches are nested prefixes of the same generated category batch. This makes the sweep ask how the same category intervention strengthens as the added simulation budget grows.

All rewards in this report are higher-is-better. Delta rewards are the clearest BO signal because BO receives feedback from improvement after choosing the next batch.

## Run Configuration

```text
design_space = wide_window
initial = 500
batch = 14
batch_sizes = 14 56 112 224
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

| Batch | Reward | Paired eta^2 | Rank stability | Best category | Best mean delta | Positive categories | Policy range |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 14 | neg_range_rmse_beta | 0.463 | 0.189 | long late | 0.0275 | 0.78 | 0.0357 |
| 14 | rmse_coverage_predictive | 0.423 | 0.028 | medium late | 0.0153 | 0.67 | 0.0412 |
| 14 | rmse_predictive | 0.408 | -0.017 | medium early | 0.0178 | 0.89 | 0.0251 |
| 14 | neg_range_rmse_delta | 0.394 | 0.117 | long middle | 0.0124 | 0.56 | 0.0336 |
| 14 | neg_coverage_error | 0.383 | 0.078 | medium late | 0.0000 | 0.11 | 0.0744 |
| 14 | rmse_coverage | 0.360 | 0.017 | medium late | 0.0108 | 0.56 | 0.0367 |
| 14 | neg_predictive_rmse | 0.336 | -0.094 | medium early | 0.0567 | 0.89 | 0.0567 |
| 14 | neg_range_rmse | 0.317 | -0.050 | medium early | 0.0122 | 0.89 | 0.0195 |
| 14 | neg_prior_std_rmse | 0.317 | -0.050 | medium early | 0.0422 | 0.89 | 0.0674 |
| 14 | neg_raw_rmse | 0.227 | -0.217 | medium late | 0.0134 | 1.00 | 0.0133 |
| 14 | neg_range_rmse_alpha | 0.207 | -0.256 | medium late | 0.0286 | 1.00 | 0.0215 |
| 14 | neg_range_rmse_gamma | 0.196 | -0.128 | medium early | 0.0199 | 0.78 | 0.0320 |
| 56 | rmse_predictive | 0.702 | 0.661 | medium middle | 0.0233 | 0.78 | 0.0312 |
| 56 | rmse_coverage_predictive | 0.675 | 0.678 | medium middle | 0.0179 | 0.56 | 0.0391 |
| 56 | neg_range_rmse_delta | 0.661 | 0.600 | medium middle | 0.0166 | 0.22 | 0.0389 |
| 56 | rmse_coverage | 0.585 | 0.489 | medium middle | 0.0134 | 0.56 | 0.0371 |
| 56 | neg_range_rmse | 0.570 | 0.356 | medium middle | 0.0188 | 0.89 | 0.0292 |
| 56 | neg_prior_std_rmse | 0.570 | 0.356 | medium middle | 0.0651 | 0.89 | 0.1012 |
| 56 | neg_raw_rmse | 0.482 | 0.311 | medium middle | 0.0129 | 0.78 | 0.0201 |
| 56 | neg_range_rmse_alpha | 0.477 | 0.183 | medium middle | 0.0271 | 0.89 | 0.0407 |
| 56 | neg_coverage_error | 0.417 | 0.283 | medium early | -0.0136 | 0.00 | 0.0514 |
| 56 | neg_predictive_rmse | 0.394 | -0.039 | short middle | 0.0656 | 0.89 | 0.1523 |
| 56 | neg_range_rmse_beta | 0.360 | -0.194 | medium middle | 0.0228 | 0.89 | 0.0293 |
| 56 | neg_range_rmse_gamma | 0.341 | -0.089 | long late | 0.0148 | 0.56 | 0.0216 |
| 112 | neg_range_rmse_beta | 0.820 | 0.839 | medium late | 0.0358 | 0.67 | 0.0629 |
| 112 | neg_prior_std_rmse | 0.718 | 0.572 | medium late | 0.0798 | 0.78 | 0.1499 |
| 112 | neg_range_rmse | 0.718 | 0.572 | medium late | 0.0230 | 0.78 | 0.0433 |
| 112 | neg_raw_rmse | 0.701 | 0.522 | long late | 0.0183 | 0.78 | 0.0322 |
| 112 | rmse_coverage | 0.692 | 0.444 | medium late | 0.0190 | 0.33 | 0.0578 |
| 112 | rmse_predictive | 0.632 | 0.311 | medium late | 0.0301 | 0.78 | 0.0487 |
| 112 | rmse_coverage_predictive | 0.618 | 0.283 | medium late | 0.0260 | 0.33 | 0.0632 |
| 112 | neg_range_rmse_alpha | 0.604 | 0.422 | long late | 0.0388 | 0.78 | 0.0564 |
| 112 | neg_coverage_error | 0.593 | 0.217 | short late | -0.0133 | 0.00 | 0.0611 |
| 112 | neg_range_rmse_gamma | 0.515 | 0.389 | medium late | 0.0183 | 0.56 | 0.0394 |
| 112 | neg_range_rmse_delta | 0.434 | 0.050 | medium late | 0.0087 | 0.33 | 0.0249 |
| 112 | neg_predictive_rmse | 0.386 | -0.072 | medium late | 0.0707 | 0.89 | 0.0926 |
| 224 | neg_range_rmse_delta | 0.901 | 0.811 | long late | 0.0170 | 0.44 | 0.0561 |
| 224 | neg_prior_std_rmse | 0.894 | 0.778 | long late | 0.1018 | 0.67 | 0.1549 |
| 224 | neg_range_rmse | 0.894 | 0.778 | long late | 0.0294 | 0.67 | 0.0447 |
| 224 | rmse_coverage | 0.858 | 0.750 | long late | 0.0234 | 0.44 | 0.0532 |
| 224 | rmse_predictive | 0.856 | 0.661 | long late | 0.0339 | 0.67 | 0.0493 |
| 224 | rmse_coverage_predictive | 0.825 | 0.711 | long late | 0.0279 | 0.67 | 0.0572 |
| 224 | neg_raw_rmse | 0.781 | 0.467 | long late | 0.0222 | 0.67 | 0.0329 |
| 224 | neg_range_rmse_gamma | 0.776 | 0.444 | long middle | 0.0249 | 0.78 | 0.0609 |
| 224 | neg_range_rmse_alpha | 0.625 | 0.394 | long late | 0.0422 | 0.78 | 0.0486 |
| 224 | neg_range_rmse_beta | 0.564 | 0.256 | long late | 0.0382 | 1.00 | 0.0380 |
| 224 | neg_coverage_error | 0.499 | 0.217 | long late | -0.0242 | 0.00 | 0.0719 |
| 224 | neg_predictive_rmse | 0.374 | 0.111 | medium early | 0.0583 | 0.89 | 0.0586 |

## Interpretation

At least one larger batch produces positive reward deltas with detectable category-level structure. This is the first condition BO needs before surrogate tuning is meaningful.

Use `reward_delta_landscape.png` to see the denoised 3x3 category grid at the largest configured batch, and `reward_snr_landscape.png` to see where that mean delta is large relative to repeat noise. Use `reward_linkage_summary.png` to see how linkage and positivity change across batch sizes.

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
