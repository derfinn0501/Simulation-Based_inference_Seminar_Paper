# Approach 1.3 BO Design Effect Check Results

This diagnostic asks whether adaptive BO-selected simulator designs improve parameter estimation, or whether gains mainly come from adding more data.

## Run Configuration

```text
design_space = wide_window
initial = 80
batch = 5
rounds = 30
final_budget = 230
validation = 100
repeats = 2
n_obs = 10
seed = 717
target_mode = validation_set
bo_candidates = 192
bo_random_fraction = 0.2
marginal_jitter_std = 0.04
estimator = rectified_fmpe
reward_mode = rmse_coverage_predictive
coverage_weight = 0.25
predictive_weight = 0.1
flow_samples_per_pair = 3
posterior_samples = 32
ode_steps = 8
fmpe_max_iter = 220
dumb_psi_unit = 0.0 0.0
```

## Design Focus

BO controls design variables `psi`, not the inferred physical parameters `theta`.

For `wide_window`, `psi = (t_span, t_start)`: initial populations and `n_obs` are fixed, while BO can choose between short dense observation windows and longer trend windows.

When `target_mode = fixed_x0`, the reward is evaluated only on one synthetic observed time series. This mimics the real SBI use case more closely than averaging over many validation observations, but it is noisier and should not be interpreted as global posterior quality.

The BO surrogate is fit to round-to-round changes in the configured posterior-quality objective:

```text
negative range-normalized RMSE minus coverage and predictive penalties
```

The fixed dumb baseline uses the same shared initial random data, then adds every later batch at one fixed design:

```text
prey0 = 40.000
pred0 = 15.000
t_start = 0.000
t_span = 2.000
t_end = 2.000
```

## Methods

| Method | Meaning |
| --- | --- |
| random | Uniform random design at every round. |
| fixed_dumb | One fixed naive design at every round after the shared initial data. |
| bo | Adaptive BO-selected design with a small random-design fraction. |
| bo_marginal_random | Random designs sampled from BO's empirical marginal psi distribution after BO finishes. |

Final evaluated round: `30`.

## Final-Round Summary

| Method | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: |
| random | 0.2475 | -0.4196 | 0.3033 | 0.9624 |
| fixed_dumb | 0.2904 | -0.4785 | 0.3596 | 0.9820 |
| bo | 0.2342 | -0.4041 | 0.3129 | 0.9163 |
| bo_marginal_random | 0.2504 | -0.4308 | 0.3479 | 0.9347 |

## Structured Fixed Design Scores

These policies add all post-initial simulations at one interpretable fixed window. They are not adaptive, but they test whether simple design structure can explain or beat the BO choices.

| Design | t_start | t_span | Range-norm RMSE | Objective | Coverage error | Predictive RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short early | 0.000 | 2.000 | 0.2998 | -0.4912 | 0.3825 | 0.9578 |
| short late | 38.000 | 2.000 | 0.2766 | -0.4577 | 0.3329 | 0.9794 |
| long early | 0.000 | 24.000 | 0.2688 | -0.4582 | 0.3900 | 0.9182 |
| long late | 16.000 | 24.000 | 0.2667 | -0.4529 | 0.3679 | 0.9426 |
| medium middle | 13.500 | 13.000 | 0.2539 | -0.4353 | 0.3383 | 0.9687 |

## Current Interpretation

At the final round, BO changes range-normalized RMSE by `5.4%` relative to uniform random design.
BO changes range-normalized RMSE by `19.4%` relative to the fixed dumb design.
The BO-marginal random control changes range-normalized RMSE by `-1.1%` relative to uniform random design.
BO changes range-normalized RMSE by `6.5%` relative to the BO-marginal random control.

BO is strongest at the final budget. This is initial evidence that adaptive design can help beyond simply sampling the BO-favored design region.

## Structured-Design Interpretation

The best fixed structured window by RMSE is `medium middle` with RMSE `0.2539`.
The best fixed structured window by objective is `medium middle` with objective `-0.4353`.

BO beats the structured fixed windows on these final diagnostics, which is stronger evidence for adaptive design than a comparison to random alone.

## Output Files

```text
metrics.csv
summary_by_round.csv
design_trace.csv
design_summary_by_round.csv
bo_trace.csv
design_choice_scores.csv
design_choice_summary.csv
design_choice_trace.csv
design_choice_scores.png
bo_design_effect_summary.png
design_structure_summary.png
```
